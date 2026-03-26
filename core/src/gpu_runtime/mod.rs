pub mod array;
pub mod dataflow;
pub mod expr;
pub mod ffi;
pub mod optimize;
pub mod relation;
pub mod stratum;
pub mod tag;
pub mod tuple;
pub mod value;

use crate::{
  common::{output_option::OutputOption, tuple_type::TupleType},
  compiler::ram,
  runtime::{
    database::intentional::*,
    dynamic::{CommunicationContext, DynamicExecutionContext, RuntimeChannel},
    provenance::Provenance,
    provenance::{diff_add_mult_prob::*, diff_min_max_prob::*, diff_top_k_proofs::*, min_max_prob::*, *},
  },
  utils::*,
};
use array::{C_Array, C_String};
use dataflow::C_Dataflow;
use expr::{C_BinaryExpr, C_Expr};
use ffi::execute_stratum_device;
use graphviz_rust::dot_structures::{Attribute, Edge, EdgeTy, Graph, Id, Node, NodeId, Stmt, Subgraph, Vertex};
use relation::IDB;
use std::{collections::*, ffi::c_void, fs::File};
use stratum::C_Stratum;
use tag::{
  C_DiffAddMultProb, C_DiffMinMaxProb, C_DiffTopKProofs,
  {C_DualNumber, C_Gradient, C_MinMaxProb, C_Proof, C_Provenance, C_Tag},
};
use tuple::C_TupleType;
use value::{C_Value, C_ValueType};

/// Constructs a graph of the RAM program separated into sub-graphs that correspond to strata.
/// Returns a (node_sets, edges) tuple.
pub fn construct_ram_graph(strata: &Vec<ram::Stratum>) -> (Vec<HashSet<String>>, HashSet<(String, String)>) {
  let mut graphs = vec![];
  strata.iter().for_each(|s| {
    let mut targets: HashSet<String> = s.updates.iter().map(|u| u.target.clone()).collect();
    let facts: HashSet<String> = s.relations.iter().map(|(r, _)| r.clone()).collect();
    for rel in facts {
      targets.insert(rel);
    }
    graphs.push(targets);
  });

  let get_sources = |u: &ram::Update| -> HashSet<String> {
    let mut sources = HashSet::new();
    u.dataflow.source_relations().iter().for_each(|s| {
      sources.insert(s.to_string());
    });
    sources
  };
  let mut edges = HashSet::new();
  strata.iter().for_each(|s| {
    s.updates.iter().for_each(|u| {
      let sources = get_sources(u);
      sources.iter().for_each(|s| {
        edges.insert((s.clone(), u.target.clone()));
      });
    })
  });
  return (graphs, edges);
}

/// Prints the RAM graph returned from `construct_ram_graph`
pub fn print_ram_graph(graphs: &Vec<HashSet<String>>, edges: &HashSet<(String, String)>) {
  let str_filt = |s: String| format!("\"{}\"", s);

  let mut graph = Graph::DiGraph {
    id: Id::Plain("G".to_string()),
    strict: false,
    stmts: vec![],
  };
  for (i, subgraph) in graphs.iter().enumerate() {
    let mut nodes: Vec<_> = subgraph
      .iter()
      .map(|n| Stmt::Node(Node::new(NodeId(Id::Plain(str_filt(n.to_string())), None), vec![])))
      .collect();
    nodes.push(Stmt::Attribute(Attribute(
      Id::Plain("label".to_string()),
      Id::Plain(format!("\"Stratum {}\"", i)),
    )));
    let subgraph = Subgraph {
      id: Id::Plain(format!("cluster_{}", i)),
      stmts: nodes,
    };
    graph.add_stmt(Stmt::Subgraph(subgraph));
  }
  for (src, tgt) in edges.iter() {
    graph.add_stmt(Stmt::Edge(Edge {
      ty: EdgeTy::Pair(
        Vertex::N(NodeId(Id::Plain(str_filt(src.to_string())), None)),
        Vertex::N(NodeId(Id::Plain(str_filt(tgt.to_string())), None)),
      ),
      attributes: vec![],
    }));
  }

  use std::io::Write;
  let mut file = std::fs::File::create("graph.dot").unwrap();
  let as_string = graphviz_rust::print(graph, &mut graphviz_rust::printer::PrinterContext::default());
  file.write_all(as_string.as_bytes()).unwrap();
}

// TODO: represent the set of strata to execute on the GPU as a list of ranges
// instead of individual integers
pub struct GpuExecutionContext {
  pub stratum_set: std::ops::Range<usize>,
  pub check_cuda: bool,
}
impl GpuExecutionContext {
  /// Parse the environment variable STRATUM as a comma separated list of integers that indicate
  /// which strata should be executed on the GPU
  pub fn new_from_env(strata: &Vec<ram::Stratum>) -> Option<Self> {
    let stratum_set = std::env::var("STRATUM")
      .ok()?
      .split(",")
      .map(|s| {
        s.parse::<usize>()
          .expect("Failed to parse STRATUM environment variable as a comma separated list of integers")
      })
      .collect::<Vec<_>>();
    assert_eq!(
      stratum_set.len(),
      2,
      "libsclgpu error: STRATUM environment variable must be a range of integers"
    );
    assert!(
      stratum_set[0] <= stratum_set[1],
      "libsclgpu error: STRATUM environment variable must be a range of integers"
    );

    let stratum_set = stratum_set[0]..stratum_set[1] + 1;

    if stratum_set.start >= strata.len() || stratum_set.end > strata.len() {
      panic!(
        "libsclgpu error: can't execute stratum set {:?} on GPU (only {} strata)",
        stratum_set,
        strata.len()
      );
    }

    let check_cuda = !std::env::var("NO_CHECK").is_ok();

    Some(Self {
      stratum_set,
      check_cuda,
    })
  }

  pub fn skip_cpu_strata(&self, strata_index: usize) -> bool {
    self.stratum_set.contains(&strata_index) && !self.check_cuda
  }

  pub fn range(&self) -> std::ops::Range<usize> {
    self.stratum_set.clone()
  }

  /// Returns a tuple of (input_relations, output_relations), where input_relations are
  /// relations that must be entirely populated before running the stratum set, and
  /// output_relations are relations that are produced by the stratum set.
  ///
  /// Note that intermediate relations may also be computed, but they will not be in
  /// output_relations unless they are needed in a subsequent stratum.
  fn boundary_relations(&self, strata: &Vec<ram::Stratum>) -> (Vec<String>, Vec<String>) {
    let stratum_set_relations = self
      .stratum_set
      .clone()
      .filter_map(|i| Some(strata.get(i)?.relations.keys().cloned().collect::<HashSet<_>>()))
      .flatten()
      .collect::<HashSet<_>>();

    let (_graphs, edges) = construct_ram_graph(strata);

    let mut input_relations = HashSet::new();
    let mut output_relations = HashSet::new();
    edges.iter().for_each(|(src, tgt)| {
      if !stratum_set_relations.contains(src) && stratum_set_relations.contains(tgt) {
        input_relations.insert(src.clone());
      }
      if stratum_set_relations.contains(src) && !stratum_set_relations.contains(tgt) {
        output_relations.insert(src.clone());
      }
    });
    if stratum_set_relations.contains("result") {
      output_relations.insert("result".to_string());
    }
    // Output any relations that have no outgoing edges (safe to assume
    // these are used for output)
    //self.stratum_set.clone().for_each(|i| {
    //  strata
    //    .get(i)
    //    .unwrap()
    //    .relations
    //    .iter()
    //    .for_each(|(name, rel)| match rel.output {
    //      OutputOption::Default => {
    //        output_relations.insert(name.clone());
    //      }
    //      _ => {}
    //    })
    //});

    let input_relations = input_relations.into_iter().collect::<Vec<_>>();
    let output_relations = output_relations.into_iter().collect::<Vec<_>>();
    (input_relations, output_relations)
  }

  pub fn optimize_program<Prov: Provenance, Ptr: PointerFamily>(
    &mut self,
    execution_context: &DynamicExecutionContext<Prov, Ptr>,
    _ctx: &Prov,
    program: &ram::Program,
  ) -> ram::Program {
    let new_strata = if std::env::var("NO_OPTIMIZE").is_ok() {
      program.strata.clone()
    } else {
      if std::env::var("PRINT_OPTIMIZE").is_ok() && execution_context.is_leader() {
        println!("Stratum before optimization");
        for i in self.stratum_set.clone() {
          println!("Stratum #{}", i);
          println!("{}", program.strata[i]);
        }
      }
      //let _guard = flame::start_guard("optimize");
      let optimized_strata = optimize::gpu_optimize(self, &program.strata);
      if std::env::var("PRINT_OPTIMIZE").is_ok() && execution_context.is_leader() {
        println!("Stratum after optimization");
        for i in self.stratum_set.clone() {
          println!("Stratum #{}", i);
          println!("{}", optimized_strata[i]);
        }
      }
      optimized_strata
    };
    ram::Program {
      strata: new_strata,
      ..program.clone()
    }
  }

  /// Executes the stratum set on the GPU, returning the resulting intentional database
  pub fn execute<Prov: Provenance, Ptr: PointerFamily>(
    &mut self,
    execution_context: &DynamicExecutionContext<Prov, Ptr>,
    ctx: &Prov,
    program: &ram::Program,
    current_idb: &IntentionalDatabase<Prov, Ptr>,
  ) -> IntentionalDatabase<Prov, Ptr> {
    let log_stratum = std::env::var("log_stratum").is_ok();
    let (input_relations, output_relations) = self.boundary_relations(&program.strata);

    let is_leader = execution_context.is_leader();
    if is_leader {
      println!("Offloading stratum set {:?} to the GPU", self.stratum_set);
      println!("Input relations: {:?}", input_relations);
      println!("Output relations: {:?}", output_relations);
    }

    let mut cuda_idb = None;
    unsafe {
      if is_leader {
        flame::start("libsclgpu");
      }

      let mut relation_types = HashMap::<String, TupleType>::new();
      for strata in program.strata.iter() {
        for (rel_name, rel) in strata.relations.iter() {
          relation_types.insert(rel_name.clone(), rel.tuple_type.clone());
        }
      }
      if is_leader {
        flame::start("ffi_db_conversion");
      }
      let edb = {
        relation::EDB::from_extensional_database_with_filter(&execution_context.edb, &relation_types, &input_relations)
      };
      let current_idb =
        { relation::IDB::from_intentional_database_with_filter(&current_idb, &relation_types, &input_relations) };
      if is_leader {
        flame::end("ffi_db_conversion");
      }

      let program_strata = &program.strata;
      let strata = C_Array::new(
        self
          .stratum_set
          .clone()
          .map(|i| C_Stratum::from_stratum(&program_strata[i]))
          .collect::<Vec<_>>(),
      );
      let output_relations = C_Array::new(
        output_relations
          .iter()
          .map(|s| C_String::new(s.clone()))
          .collect::<Vec<_>>(),
      );

      match &execution_context.communication {
        // Case 1: parallel execution context where this is a follower thread
        Some(CommunicationContext {
          id: _,
          thread_count: _,
          channel: RuntimeChannel::Follower(tx, rx),
        }) => {
          tx.send((edb, current_idb, ctx.clone()))
            .expect("Leader thread dropped channel early");
          let output: *const IDB<Prov> = rx.recv().expect("Leader thread dropped channel early").into();
          //println!("Receieved IDB from leader thread");
          cuda_idb = Some(
            output
              .as_ref()
              .expect("IDB returned by libsclgpu was null")
              .to_intentional_database(ctx),
          );
        }
        // Case 2: parallel execution context where this is the leader thread
        Some(CommunicationContext {
          id: _,
          thread_count: _,
          channel: RuntimeChannel::Leader(rxs, txs),
        }) => {
          let mut edbs = vec![edb];
          let mut idbs = vec![current_idb];
          let mut ctxes = vec![ctx.clone()];
          for rx in rxs {
            let (edb, idb, new_ctx) = rx.recv().expect("Follower thread dropped channel early");
            edbs.push(edb);
            idbs.push(idb);
            ctxes.push(new_ctx);
          }
          let edbs = C_Array::new(edbs);
          let idbs = C_Array::new(idbs);
          let provenance = Prov::to_ffi_provenance(&ctxes);
          let output_idbs = execute_stratum_device(&provenance, &strata, &output_relations, &edbs, &idbs);

          for (i, idb) in (*output_idbs).as_slice().iter().enumerate() {
            match i {
              0 => {
                let _guard = flame::start_guard("ffi_idb_returned");
                cuda_idb = Some(idb.to_intentional_database(ctx));
              }
              i => {
                txs[i - 1]
                  .send(idb.into())
                  .expect("Follower thread dropped channel early");
              }
            }
          }
        }
        // Case 3: single-threaded execution context
        None => {
          if log_stratum {
            println!("Rust edb length: {}", execution_context.edb.extensional_relations.len());
            println!("C++ edb length: {}", edb.relations.length);
            println!("stratum: {}", strata);
            println!("current_idb: {}", current_idb);
            println!("edb: {}", edb);
          }

          let edbs = C_Array::new(vec![edb]);
          let idbs = C_Array::new(vec![current_idb]);
          let provenance = Prov::to_ffi_provenance(&vec![ctx.clone()]);
          let output_idbs = execute_stratum_device(&provenance, &strata, &output_relations, &edbs, &idbs);

          let output_idb = (*output_idbs).get(0).unwrap();

          if log_stratum {
            println!("Successfully executed stratum set");
            for i in 0..(*output_idb).relations.length {
              let relation = (*output_idb).relations.get(i).unwrap();
              let length = relation.tuples.get(0).unwrap().length;
              println!("Relation: {} (length {})", relation.predicate.to_string(), length);

              let mut tuples = Vec::new();
              for row in 0..length {
                let mut tuple = Vec::new();
                for col in 0..(relation.tuples.length) {
                  let value = relation.tuples.get(col).unwrap().get(row).unwrap();
                  tuple.push(value);
                }
                tuples.push(tuple);
              }

              for tuple in tuples {
                println!("{:?}", tuple);
              }
            }
          }
          let _guard = flame::start_guard("ffi_idb_returned");
          cuda_idb = Some(output_idb.to_intentional_database(ctx));
        }
      }
      if is_leader {
        flame::end("libsclgpu");
      }
    }
    cuda_idb.unwrap()
  }

  /// Compares GPU output to CPU output to check for correctness
  pub fn check_gpu_output<Prov: Provenance, Ptr: PointerFamily>(
    &self,
    cuda_idb: IntentionalDatabase<Prov, Ptr>,
    program: &ram::Program,
    execution_context: &DynamicExecutionContext<Prov, Ptr>,
  ) {
    let (_input_relations, output_relations) = self.boundary_relations(&program.strata);
    let mut incorrect_output = false;
    for output_relation in output_relations {
      let expected_output = match execution_context.idb.get_internal_collection(&output_relation) {
        Some(collection) => collection,
        None => match execution_context.edb.get_dynamic_collection(&output_relation) {
          Some(collection) => collection,
          None => {
            println!("ERROR: Output relation {} not found in IDB or EDB", output_relation);
            continue;
          }
        },
      };
      let actual_output = cuda_idb
        .get_internal_collection(&output_relation)
        .expect(format!("Output relation {} not found in CUDA IDB", output_relation).as_str());

      if expected_output != actual_output {
        println!("Output relation {} does not match", output_relation);
        println!(
          "Expected length: {}, Actual length: {}",
          expected_output.len(),
          actual_output.len()
        );
        println!("Output relation {} does not match", output_relation);
        println!("Expected: {}", expected_output);
        println!("Actual: {}", actual_output);
        incorrect_output = true;
      }

      let check_tags: bool = std::env::var("CHECK_TAGS").is_ok();
      if check_tags {
        let actual_tags = actual_output
          .elements()
          .iter()
          .map(|Tagged { tag, tuple: _tuple }| tag)
          .collect::<Vec<_>>();
        let expected_tags = expected_output
          .elements()
          .iter()
          .map(|Tagged { tag, tuple: _tuple }| tag)
          .collect::<Vec<_>>();
        let tuples = actual_output
          .elements()
          .iter()
          .map(|Tagged { tag: _, tuple }| tuple)
          .collect::<Vec<_>>();
        println!("Tuples: \n{:?}", tuples);
        println!("Actual tags: \n{:?}", actual_tags);
        println!("Expected tags: \n{:?}", expected_tags);
      }
    }
    if incorrect_output {
      panic!("GPU output does not match CPU output. See prior logging.");
    }
  }
}

extern "C" {
  /// cbindgen only generates bindings for types used in extern
  /// function definitions, so this is a dummy function that takes
  /// every type we want to export as an argument
  pub fn rust_cbindgen_exports(
    //<Prov: Provenance>(
    //_: C_Relation<Prov>,
    _: C_TupleType,
    _: C_Value,
    _: C_ValueType,
    _: C_Provenance,
    _: C_MinMaxProb,
    _: C_Expr,
    _: C_BinaryExpr,
    _: C_Dataflow,
    // _: C_Stratum<Prov>,
    // _: IDB<Prov>,
    // _: EDB<Prov>,
    _: C_String,
    _: C_DualNumber,
    _: C_Gradient,
    _: C_Proof,
  );
}

use std::collections::*;
use std::fs::File;

use super::*;
use crate::common::tuple::*;
use crate::compiler::ram;
use crate::gpu_runtime::relation::{EDB, IDB};
use crate::runtime::database::extensional::*;
use crate::runtime::database::intentional::*;
use crate::runtime::database::*;
use crate::runtime::env::*;
use crate::runtime::error::*;
use crate::runtime::monitor::Monitor;
use crate::runtime::provenance::*;
use crate::utils::*;
use std::sync::mpsc;

#[derive(Clone, Debug)]
pub struct ExecutionOptions {
  pub type_check: bool,
  pub incremental_maintain: bool,
  pub retain_internal_when_recover: bool,
}

impl Default for ExecutionOptions {
  fn default() -> Self {
    Self {
      type_check: true,
      incremental_maintain: false,
      retain_internal_when_recover: true,
    }
  }
}

pub struct SendablePtr<T>(*const T);
impl<T> From<*const T> for SendablePtr<T> {
  fn from(ptr: *const T) -> Self {
    SendablePtr(ptr)
  }
}
impl<T> From<&T> for SendablePtr<T> {
  fn from(ptr: &T) -> Self {
    SendablePtr(ptr)
  }
}
impl<T> Into<*const T> for SendablePtr<T> {
  fn into(self) -> *const T {
    self.0
  }
}
// SAFETY: only accessed immutably
unsafe impl<T> Send for SendablePtr<T> {}
// SAFETY: only accessed immutably
unsafe impl<T> Sync for SendablePtr<T> {}

pub enum RuntimeChannel<Prov: Provenance> {
  Follower(
    mpsc::Sender<(EDB<Prov>, IDB<Prov>, Prov)>,
    mpsc::Receiver<SendablePtr<IDB<Prov>>>,
  ),
  Leader(
    Vec<mpsc::Receiver<(EDB<Prov>, IDB<Prov>, Prov)>>,
    Vec<mpsc::Sender<SendablePtr<IDB<Prov>>>>,
  ),
}
// TODO: can the following unsafe impls be avoided?
unsafe impl<Prov: Provenance> Send for RuntimeChannel<Prov> {}
unsafe impl<Prov: Provenance> Sync for RuntimeChannel<Prov> {}

pub struct CommunicationContext<Prov: Provenance> {
  pub id: usize,
  pub thread_count: usize,
  pub channel: RuntimeChannel<Prov>,
}

impl<Prov: Provenance> Clone for CommunicationContext<Prov> {
  fn clone(&self) -> Self {
    panic!("Cannot clone CommunicationContext due to channels")
  }
}

pub struct DynamicExecutionContext<Prov: Provenance, Ptr: PointerFamily = RcFamily> {
  pub options: ExecutionOptions,
  // TODO: ask Ziyang where to store this data
  pub communication: Option<CommunicationContext<Prov>>,
  pub program: ram::Program,
  pub edb: ExtensionalDatabase<Prov>,
  pub idb: IntentionalDatabase<Prov, Ptr>,
}

impl<Prov: Provenance, Ptr: PointerFamily> Clone for DynamicExecutionContext<Prov, Ptr> {
  fn clone(&self) -> Self {
    Self {
      options: self.options.clone(),
      communication: self.communication.clone(),
      program: self.program.clone(),
      edb: self.edb.clone(),
      idb: self.idb.clone(),
    }
  }
}

impl<Prov: Provenance, Ptr: PointerFamily> DynamicExecutionContext<Prov, Ptr> {
  pub fn is_leader(&self) -> bool {
    !matches!(
      &self.communication,
      Some(CommunicationContext {
        id: _,
        thread_count: _,
        channel: RuntimeChannel::Follower(_, _),
      })
    )
  }
  pub fn new() -> Self {
    let options = ExecutionOptions::default();
    Self::new_with_options(options)
  }

  pub fn new_with_options(options: ExecutionOptions) -> Self {
    let edb = ExtensionalDatabase::new_with_options(options.type_check.clone());
    let idb = IntentionalDatabase::default();
    let program = ram::Program::new();
    Self {
      options,
      communication: None,
      program,
      edb,
      idb,
    }
  }

  pub fn new_with_program_and_options(program: ram::Program, options: ExecutionOptions) -> Self {
    let edb =
      ExtensionalDatabase::with_relation_types_and_options(program.relation_types(), options.type_check.clone());
    let idb = IntentionalDatabase::default();
    Self {
      options,
      communication: None,
      program,
      edb,
      idb,
    }
  }

  pub fn clone_with_new_provenance<Prov2: Provenance>(&self) -> DynamicExecutionContext<Prov2, Ptr>
  where
    Prov2::InputTag: ConvertFromInputTag<Prov::InputTag>,
  {
    DynamicExecutionContext {
      options: self.options.clone(),
      communication: None,
      program: self.program.clone(),
      edb: self.edb.clone_with_new_provenance::<Prov2>(),
      idb: self.idb.clone_with_new_provenance::<Prov2>(),
    }
  }

  pub fn set_non_incremental(&mut self) {
    self.options.incremental_maintain = false;
  }

  /// Directly execute the program stored in the file
  pub fn execute(&mut self, runtime: &RuntimeEnvironment, ctx: &mut Prov) -> Result<(), RuntimeError> {
    self.incremental_execute_helper(None, runtime, ctx)
  }

  pub fn incremental_execute(
    &mut self,
    program: ram::Program,
    runtime: &RuntimeEnvironment,
    ctx: &Prov,
  ) -> Result<(), RuntimeError> {
    self.incremental_execute_helper(Some(program), runtime, ctx)
  }

  fn incremental_execute_helper(
    &mut self,
    maybe_new_program: Option<ram::Program>,
    runtime: &RuntimeEnvironment,
    ctx: &Prov,
  ) -> Result<(), RuntimeError> {
    // Pull the IDB
    let mut incremental_result = IntentionalDatabase::default();
    std::mem::swap(&mut self.idb, &mut incremental_result);

    // Persistent relations
    let mut temp_program = ram::Program::new();
    std::mem::swap(&mut self.program, &mut temp_program);
    let program_ref = if let Some(new_program) = &maybe_new_program {
      // Process the EDB; populate using program facts
      self.edb.populate_program_facts(runtime, new_program)?;

      // If need to incrementalize, remove such computed results
      let edb_need_update_relations = self.edb.need_update_relations();

      // If a new program is provided, compute the persistent relations that we can re-use
      let pers = if edb_need_update_relations.is_empty() {
        self.program.persistent_relations(new_program)
      } else {
        self
          .program
          .persistent_relations_with_need_update_relations(new_program, &edb_need_update_relations)
      };

      // Keep the relations only in the set of persistent relations
      incremental_result.retain_relations(&pers);

      // Return this new program
      &new_program
    } else {
      self.edb.populate_program_facts(runtime, &temp_program)?;

      // If there is no new program, we directly take our current program
      &temp_program
    };

    // Internalize EDB relations
    self.edb.internalize(runtime, ctx);

    // Generate stratum information
    let strata_info = stratum_inputs_outputs(program_ref);

    // Make sure that all immutable relations are existed in the edb
    for relation in program_ref.relations() {
      if relation.immutable {
        self.edb.tap_relation(&relation.predicate);
      }
    }

    if self.is_leader() && std::env::var("TIME").is_ok() {
      println!("{}", "=".repeat(40));
    }
    let start = std::time::Instant::now();

    // print a graphviz file representing this RAM program
    //let (graphs, edges) = crate::gpu_runtime::construct_ram_graph(&program_ref.strata);
    //crate::gpu_runtime::print_ram_graph(&graphs, &edges);

    unsafe {
      crate::gpu_runtime::ffi::libsclgpu_init();
    }
    let mut gpu_ctx = crate::gpu_runtime::GpuExecutionContext::new_from_env(&program_ref.strata);
    let mut cuda_idb = None;

    let unoptimized_stratum_set = gpu_ctx.as_ref().map(|x| x.stratum_set.clone());

    // Go through each stratum
    for (i, stratum) in program_ref.strata.iter().enumerate() {
      // Check if this stratum should be offloaded to the GPU
      if let Some(gpu_ctx) = gpu_ctx.as_mut() {
        if gpu_ctx.stratum_set.start == i {
          let now = std::time::Instant::now();
          if self.is_leader() {
            flame::start("incremental_execute_helper_for");
            //for (k, v) in incremental_result.intentional_relations.iter() {
            //  println!("{}: {}", k, v);
            //}
            //println!("Extensional relations:");
            //for (k,v) in self.edb.extensional_relations.iter() {
            //  println!("{}: {}", k, v);
            //}
          }

          let optimized_program = gpu_ctx.optimize_program(self, ctx, program_ref);
          let result = gpu_ctx.execute(self, ctx, &optimized_program, &incremental_result);
          //println!("GPU computed relations {:?}", result.intentional_relations.keys().collect::<Vec<_>>());
          if gpu_ctx.check_cuda {
            cuda_idb = Some((optimized_program, result));
          } else {
            incremental_result.extend(result.into_iter());
          }

          if self.is_leader() {
            flame::end("incremental_execute_helper_for");
            let duration = now.elapsed();
            if std::env::var("TIME").is_ok() {
              println!(
                "Stratum set: {:?}\t Time in run(): {:?}us",
                gpu_ctx.stratum_set,
                duration.as_micros()
              );
            }
            if std::env::var("TRACE").is_ok() {
              crate::tracing::flamescope::dump(&mut File::create("flamescope.json").unwrap()).unwrap();
            }
            flame::clear();
            if std::env::var("EARLY_EXIT").is_ok() {
              std::process::exit(0);
            }
          }
        };

        let unoptimized_stratum_set = unoptimized_stratum_set.clone().unwrap();
        if unoptimized_stratum_set.contains(&i) && !gpu_ctx.check_cuda {
          continue;
        }
      }

      // Run the stratum to get the result
      let result = self.execute_stratum(i, stratum, &incremental_result, program_ref, &strata_info, runtime, ctx)?;

      // Extend the incremental result with the relations within their lifespan
      incremental_result.extend(result.into_iter().filter(|(name, _)| {
        // Check if the relation is a hidden one
        if program_ref.relation_unchecked(&name).output.is_hidden() {
          // If it is hidden, additionally check if it is going to be used later
          if let Some(last_stratum) = strata_info.relation_lifespan.get(name) {
            // If the last stratum that this relation is used is later than current, then keep; otherwise drop
            return *last_stratum > i;
          }
        }
        true
      }));
    }

    // Store the result
    self.idb = incremental_result;

    let total_duration = start.elapsed();
    if self.is_leader() {
      if std::env::var("TIME").is_ok() {
        println!("Total sample time: {:?}", total_duration.as_micros());
      }
    }

    // compare scallop-gpu result to scallop-cpu
    if let Some((optimized_program, cuda_idb)) = cuda_idb {
      gpu_ctx.unwrap().check_gpu_output(cuda_idb, &optimized_program, self);
    }

    // See if any EDB relations need to directly go into IDB
    for relation in program_ref.relations() {
      if relation.output.is_not_hidden() && !self.idb.has_relation(&relation.predicate) {
        if self.options.incremental_maintain {
          if let Some(edb_collection) = self.edb.get_dynamic_collection(&relation.predicate) {
            self
              .idb
              .insert_dynamic_collection(relation.predicate.clone(), edb_collection.clone_internal());
          }
        } else {
          if let Some(edb_collection) = self.edb.pop_dynamic_collection(&relation.predicate) {
            self
              .idb
              .insert_dynamic_collection(relation.predicate.clone(), edb_collection.clone());
          }
        }
      }
    }

    // Update the program
    if let Some(new_program) = maybe_new_program {
      self.program = new_program;
    } else {
      self.program = temp_program;
    }

    // Success!
    Ok(())
  }

  fn execute_stratum(
    &mut self,
    stratum_id: usize,
    stratum: &ram::Stratum,
    current_idb: &IntentionalDatabase<Prov, Ptr>,
    ram_program: &ram::Program,
    strata_info: &StrataInformation,
    runtime: &RuntimeEnvironment,
    ctx: &Prov,
  ) -> Result<IntentionalDatabase<Prov, Ptr>, RuntimeError> {
    let dyn_relas = stratum
      .relations
      .iter()
      .filter(|(r, _)| !current_idb.has_relation(*r))
      .map(|(r, _)| r.clone())
      .collect::<HashSet<_>>();

    // Check if we need to compute anything new
    if dyn_relas.is_empty() {
      return Ok(IntentionalDatabase::new());
    }

    // Otherwise, do computation
    let mut iter = DynamicIteration::<Prov>::new();

    // Add input collections
    if self.options.incremental_maintain {
      for (rel, col) in &self.edb.extensional_relations {
        if ram_program.relation(rel).map(|r| r.immutable).unwrap_or(false) {
          iter.add_input_dynamic_collection(&rel, col.internal.as_ref());
        }
      }
      for (rel, col) in current_idb {
        iter.add_input_dynamic_collection(&rel, col.internal_facts.as_ref());
      }
    } else {
      // Non-incremental version:

      // First add dynamic collection for every immutable relations in the stratum
      for (predicate, relation) in &stratum.relations {
        if relation.immutable {
          iter.add_input_dynamic_collection(&predicate, self.edb.get_dynamic_collection(predicate).unwrap());
        }
      }

      // Then add dynamic collection for every input relations in the stratum
      if let Some(stratum_inputs) = strata_info.stratum_inputs.get(&stratum_id) {
        for (rel, _) in stratum_inputs {
          if ram_program.relation_unchecked(rel).immutable {
            // The collection could be immutable and thus will be from EDB
            iter.add_input_dynamic_collection(&rel, self.edb.get_dynamic_collection(rel).unwrap_or_else(|| panic!("Relation {} is not in EDB", rel)));
          } else {
            // Otherwise it will be computed by previous stratum and thus from IDB
            iter.add_input_dynamic_collection(&rel, current_idb.get_internal_collection(rel).unwrap_or_else(|| panic!("Relation {} is not in IDB", rel)));
          }
        }
      }
    }

    // Create dynamic relations; all of them will be in the output
    // Note: Unwrap is ok since the relation in stratum must be in the ram program
    for rela in dyn_relas
      .iter()
      .filter(|r| !ram_program.relation_unchecked(r).immutable)
    {
      iter.create_dynamic_relation(rela);

      // Check if we need it to be output
      if self.options.incremental_maintain
        || strata_info
          .stratum_outputs
          .get(&stratum_id)
          .map_or(false, |o| o.contains(rela))
        || ram_program.relation_unchecked(rela).output.is_not_hidden()
      {
        let storage = &ram_program.relation_unchecked(rela).storage;
        iter.add_output_relation(rela, storage);
      }

      // Check if we need it to be a goal
      if ram_program.relation(rela).unwrap().is_goal {
        iter.add_goal_relation(rela);
      }

      // Add relation specific scheduler
      if let Some(scheduler) = &ram_program.relation(rela).unwrap().scheduler {
        iter.add_relation_scheduler(rela, scheduler.clone());
      }

      // Load external facts
      if let Some(facts) = self.edb.get_dynamic_collection(rela) {
        // Mutable relations need their EDB facts to go into dynamic relation
        let dataflow = dataflow::DynamicDataflow::dynamic_recent_collection(facts);
        iter
          .get_dynamic_relation_unsafe(rela)
          .insert_dataflow_recent(ctx, &dataflow, runtime);
      }
    }

    // Add updates
    for update in &stratum.updates {
      if dyn_relas.contains(&update.target) {
        iter.add_update(update.clone());
      }
    }

    // Run!
    let now = std::time::Instant::now();
    let result = iter.run(ctx, runtime);
    let duration = now.elapsed();
    if (self.is_leader() || std::env::var("ALL_PRINT").is_ok())  && std::env::var("TIME").is_ok() {
      println!("t{:?}: Stratum: {}\t Time in run(): {:?}us", self.communication.as_ref().map(|x| x.id).unwrap_or(0), stratum_id, duration.as_micros());
    }

    let result_idb = IntentionalDatabase::from_dynamic_collections(result.into_iter());

    // Success!
    Ok(result_idb)
  }

  /// Directly execute the program stored in the file
  pub fn execute_with_monitor<M>(&mut self, runtime: &RuntimeEnvironment, ctx: &Prov, m: &M) -> Result<(), RuntimeError>
  where
    M: Monitor<Prov>,
  {
    self.incremental_execute_with_monitor_helper(None, runtime, ctx, m)?;
    m.observe_finish_execution();
    Ok(())
  }

  pub fn incremental_execute_with_monitor<M>(
    &mut self,
    program: ram::Program,
    runtime: &RuntimeEnvironment,
    ctx: &Prov,
    m: &M,
  ) -> Result<(), RuntimeError>
  where
    M: Monitor<Prov>,
  {
    self.incremental_execute_with_monitor_helper(Some(program), runtime, ctx, m)
  }

  fn incremental_execute_with_monitor_helper<M>(
    &mut self,
    maybe_new_program: Option<ram::Program>,
    runtime: &RuntimeEnvironment,
    ctx: &Prov,
    m: &M,
  ) -> Result<(), RuntimeError>
  where
    M: Monitor<Prov>,
  {
    // Pull the IDB
    let mut incremental_result = IntentionalDatabase::default();
    std::mem::swap(&mut self.idb, &mut incremental_result);

    // Persistent relations
    let mut temp_program = ram::Program::new();
    std::mem::swap(&mut self.program, &mut temp_program);
    let program_ref = if let Some(new_program) = &maybe_new_program {
      // Process the EDB; populate using program facts
      self.edb.populate_program_facts(runtime, new_program)?;

      // If need to incrementalize, remove such computed results
      let edb_need_update_relations = self.edb.need_update_relations();

      // If a new program is provided, compute the persistent relations that we can re-use
      let pers = if edb_need_update_relations.is_empty() {
        self.program.persistent_relations(new_program)
      } else {
        self
          .program
          .persistent_relations_with_need_update_relations(new_program, &edb_need_update_relations)
      };

      // Keep the relations only in the set of persistent relations
      incremental_result.retain_relations(&pers);

      // Return this new program
      &new_program
    } else {
      self
        .edb
        .populate_program_facts(runtime, &temp_program)
        .expect("Since there is no new program, no error should be raised during program facts population");

      // If there is no new program, we directly take our current program
      &temp_program
    };

    // Internalize EDB relations
    // !SPECIAL MONITORING!
    self.edb.internalize_with_monitor(runtime, ctx, m);

    // Generate stratum information
    let strata_info = stratum_inputs_outputs(program_ref);

    // Make sure that all immutable relations are existed in the edb
    for relation in program_ref.relations() {
      if relation.immutable {
        self.edb.tap_relation(&relation.predicate);
      }
    }

    // Go through each stratum
    for (i, stratum) in program_ref.strata.iter().enumerate() {
      // Run the stratum to get the result
      let result = self.execute_stratum_with_monitor(
        i,
        stratum,
        &incremental_result,
        program_ref,
        &strata_info,
        runtime,
        ctx,
        m,
      )?;

      // Extend the incremental result with the relations within their lifespan
      incremental_result.extend(result.into_iter().filter(|(name, _)| {
        // Check if the relation is a hidden one
        if program_ref.relation_unchecked(&name).output.is_hidden() {
          // If it is hidden, additionally check if it is going to be used later
          if let Some(last_stratum) = strata_info.relation_lifespan.get(name) {
            // If the last stratum that this relation is used is later than current, then keep; otherwise drop
            return *last_stratum > i;
          }
        }
        true
      }));
    }

    // Store the result
    self.idb = incremental_result;

    // See if any EDB relations need to directly go into IDB
    for relation in program_ref.relations() {
      if relation.output.is_not_hidden() && !self.idb.has_relation(&relation.predicate) {
        if self.options.incremental_maintain {
          if let Some(edb_collection) = self.edb.get_dynamic_collection(&relation.predicate) {
            self
              .idb
              .insert_dynamic_collection(relation.predicate.clone(), edb_collection.clone_internal());
          }
        } else {
          if let Some(edb_collection) = self.edb.pop_dynamic_collection(&relation.predicate) {
            self
              .idb
              .insert_dynamic_collection(relation.predicate.clone(), edb_collection.clone());
          }
        }
      }
    }

    // Update the program
    if let Some(new_program) = maybe_new_program {
      self.program = new_program;
    } else {
      self.program = temp_program;
    }

    // Success!
    Ok(())
  }

  fn execute_stratum_with_monitor<M>(
    &mut self,
    stratum_id: usize,
    stratum: &ram::Stratum,
    current_idb: &IntentionalDatabase<Prov, Ptr>,
    ram_program: &ram::Program,
    strata_info: &StrataInformation,
    runtime: &RuntimeEnvironment,
    ctx: &Prov,
    m: &M,
  ) -> Result<IntentionalDatabase<Prov, Ptr>, RuntimeError>
  where
    M: Monitor<Prov>,
  {
    // !SPECIAL MONITORING!
    m.observe_executing_stratum(stratum_id);

    let dyn_relas = stratum
      .relations
      .iter()
      .filter(|(r, _)| !current_idb.has_relation(*r))
      .map(|(r, _)| r.clone())
      .collect::<HashSet<_>>();

    // Check if we need to compute anything new
    if dyn_relas.is_empty() {
      return Ok(IntentionalDatabase::new());
    }

    // Otherwise, do computation
    let mut iter = DynamicIteration::<Prov>::new();

    // Add input collections
    if self.options.incremental_maintain {
      for (rel, col) in &self.edb.extensional_relations {
        if ram_program.relation(rel).map(|r| r.immutable).unwrap_or(false) {
          // !SPECIAL MONITOR!
          m.observe_loading_relation(rel);
          m.observe_loading_relation_from_edb(rel);

          iter.add_input_dynamic_collection(&rel, col.internal.as_ref());
        }
      }
      for (rel, col) in current_idb {
        // !SPECIAL MONITOR!
        m.observe_loading_relation(rel);
        m.observe_loading_relation_from_idb(rel);

        iter.add_input_dynamic_collection(&rel, col.internal_facts.as_ref());
      }
    } else {
      // Non-incremental version:

      // First add dynamic collection for every immutable relations in the stratum
      for (predicate, relation) in &stratum.relations {
        if relation.immutable {
          // !SPECIAL MONITOR!
          m.observe_loading_relation(predicate);
          m.observe_loading_relation_from_edb(predicate);

          iter.add_input_dynamic_collection(&predicate, self.edb.get_dynamic_collection(predicate).unwrap());
        }
      }

      // Then add dynamic collection for every input relations in the stratum
      if let Some(stratum_inputs) = strata_info.stratum_inputs.get(&stratum_id) {
        for (rel, _) in stratum_inputs {
          if ram_program.relation_unchecked(rel).immutable {
            // !SPECIAL MONITOR!
            m.observe_loading_relation(rel);
            m.observe_loading_relation_from_edb(rel);

            // The collection could be immutable and thus will be from EDB
            iter.add_input_dynamic_collection(&rel, self.edb.get_dynamic_collection(rel).unwrap());
          } else {
            // !SPECIAL MONITOR!
            m.observe_loading_relation(rel);
            m.observe_loading_relation_from_idb(rel);

            // Otherwise it will be computed by previous stratum and thus from IDB
            iter.add_input_dynamic_collection(&rel, current_idb.get_internal_collection(rel).unwrap());
          }
        }
      }
    }

    // Create dynamic relations; all of them will be in the output
    // Note: Unwrap is ok since the relation in stratum must be in the ram program
    for rela in dyn_relas
      .iter()
      .filter(|r| !ram_program.relation_unchecked(r).immutable)
    {
      iter.create_dynamic_relation(rela);

      // Check if we need it to be output
      if self.options.incremental_maintain
        || strata_info.stratum_outputs[&stratum_id].contains(rela)
        || ram_program.relation_unchecked(rela).output.is_not_hidden()
      {
        let storage = &ram_program.relation_unchecked(rela).storage;
        iter.add_output_relation(rela, storage);
      }

      // Check if we need it to be a goal
      if ram_program.relation(rela).unwrap().is_goal {
        iter.add_goal_relation(rela);
      }

      // Load external facts
      if let Some(facts) = self.edb.get_dynamic_collection(rela) {
        // !SPECIAL MONITOR!
        m.observe_loading_relation(rela);
        m.observe_loading_relation_from_edb(rela);

        // Mutable relations need their EDB facts to go into dynamic relation
        let dataflow = dataflow::DynamicDataflow::dynamic_recent_collection(facts);
        iter
          .get_dynamic_relation_unsafe(rela)
          .insert_dataflow_recent(ctx, &dataflow, runtime);
      }
    }

    // Add updates
    for update in &stratum.updates {
      if dyn_relas.contains(&update.target) {
        iter.add_update(update.clone());
      }
    }

    // Run!
    // !SPECIAL MONITORING!
    let result = iter.run_with_monitor(ctx, runtime, m);

    // Success!
    Ok(IntentionalDatabase::from_dynamic_collections(result.into_iter()))
  }

  pub fn add_facts(
    &mut self,
    relation: &str,
    facts: Vec<(Option<InputTagOf<Prov>>, Tuple)>,
  ) -> Result<(), DatabaseError> {
    // For incremental: when new fact is added, we need to recompute the relation
    if !facts.is_empty() {
      self.idb.remove_relation(relation);
    }

    // Add facts back
    self.edb.add_static_input_facts(relation, facts)
  }

  pub fn internal_relation(&self, r: &str) -> Option<DynamicCollectionRef<Prov>> {
    self.idb.get_internal_collection(r)
  }

  pub fn recover(&mut self, r: &str, runtime: &RuntimeEnvironment, ctx: &Prov) {
    if self.idb.has_relation(r) {
      self
        .idb
        .recover(r, runtime, ctx, !self.options.retain_internal_when_recover);
    } else if self.edb.has_relation(r) {
      self
        .idb
        .recover_from_edb(r, runtime, ctx, &self.edb.extensional_relations[r]);
    }
  }

  pub fn recover_with_monitor<M: Monitor<Prov>>(&mut self, r: &str, runtime: &RuntimeEnvironment, ctx: &Prov, m: &M) {
    self
      .idb
      .recover_with_monitor(r, runtime, ctx, m, !self.options.retain_internal_when_recover)
  }

  pub fn relation_ref(&self, r: &str) -> Option<&DynamicOutputCollection<Prov>> {
    self.idb.get_output_collection_ref(r)
  }

  pub fn relation(&self, r: &str) -> Option<Ptr::Rc<DynamicOutputCollection<Prov>>> {
    self.idb.get_output_collection(r)
  }

  pub fn is_computed(&self, r: &str) -> bool {
    self.idb.has_relation(r)
  }

  pub fn num_relations(&self) -> usize {
    self.program.relation_to_stratum.len()
  }

  pub fn relations(&self) -> Vec<String> {
    self
      .program
      .relation_to_stratum
      .iter()
      .map(|(n, _)| n.clone())
      .collect()
  }
}

type StratumInputs = HashMap<usize, HashSet<(String, usize)>>;

type StratumOutputs = HashMap<usize, HashSet<String>>;

type RelationLifespan = HashMap<String, usize>;

struct StrataInformation {
  stratum_inputs: StratumInputs,
  stratum_outputs: StratumOutputs,
  relation_lifespan: RelationLifespan,
}

fn stratum_inputs_outputs(ram: &ram::Program) -> StrataInformation {
  // Cache stratum input output
  let mut stratum_inputs = HashMap::<usize, HashSet<(String, usize)>>::new();
  let mut stratum_outputs = HashMap::<usize, HashSet<String>>::new();

  // Relation start
  let mut relation_lifespan = HashMap::<String, usize>::new();

  // Iterate through the strata
  for (i, stratum) in ram.strata.iter().enumerate() {
    // With dependency, add to input/output
    for dep in stratum.dependency() {
      let dep_stratum = ram.relation_to_stratum[&dep];

      // The `dep_stratum` needs to have an output `dep` relation
      stratum_outputs.entry(dep_stratum).or_default().insert(dep.clone());

      // The `i`-th stratum needs to have `dep` as an input
      stratum_inputs.entry(i).or_default().insert((dep.clone(), dep_stratum));

      // The `i`-th stratum is (currently) the last stratum to use `dep` relation
      relation_lifespan.insert(dep.clone(), i);
    }

    // Regular user requested outputs
    for (predicate, relation) in &stratum.relations {
      if relation.output.is_not_hidden() {
        stratum_outputs.entry(i).or_default().insert(predicate.clone());
      }
    }
  }

  // Generate information
  StrataInformation {
    stratum_inputs,
    stratum_outputs,
    relation_lifespan,
  }
}

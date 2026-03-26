use std::collections::*;
use std::ffi::c_void;

use super::*;
use crate::common::output_option::OutputOption;
use crate::common::tuple::*;
use crate::common::value_type::FromType;
use crate::compiler::ram;
use crate::gpu_runtime::{
  array::C_Array,
  relation::{EDB, IDB},
};
use crate::runtime::database::extensional::*;
use crate::runtime::database::intentional::*;
use crate::runtime::database::*;
use crate::runtime::env::*;
use crate::runtime::error::*;
use crate::runtime::monitor::Monitor;
use crate::runtime::provenance::{
  diff_add_mult_prob::*, diff_min_max_prob::*, diff_top_k_proofs::*, min_max_prob::*, *,
};
use crate::utils::*;
use std::sync::mpsc;

pub fn gpu_optimize(gpu_ctx: &mut GpuExecutionContext, strata: &Vec<ram::Stratum>) -> Vec<ram::Stratum> {
  let mut predicate_types = HashMap::new();
  for strata in strata.iter() {
    for (name, relation) in strata.relations.iter() {
      predicate_types.insert(name.clone(), relation.tuple_type.clone());
    }
  }

  let gpu_strata = gpu_ctx.stratum_set.clone().map(|i| strata[i].clone()).collect();
  let used_in_later_strata = strata
    .iter()
    .enumerate()
    .map(|(i, s)| {
      s.relations
        .keys()
        .cloned()
        .filter(|relation_name| ((i + 1)..strata.len()).any(|j| strata[j].source_relations().contains(relation_name)))
        .collect::<HashSet<_>>()
    })
    .collect::<Vec<_>>();
  let mut optimized = gpu_strata;
  inline(gpu_ctx, &mut optimized, &used_in_later_strata);
  delay(gpu_ctx, &mut optimized);
  //extract(gpu_ctx, &mut optimized, &predicate_types);
  remove_empty_strata(gpu_ctx, &mut optimized);
  optimize_join_index(gpu_ctx, &mut optimized);

  // Make rule order deterministic
  for strata in optimized.iter_mut() {
    strata.updates.sort_by(|a, b| a.target.cmp(&b.target));
  }

  let optimized_size = optimized.len();

  let mut result = strata.clone();
  result.splice(gpu_ctx.range(), optimized);
  gpu_ctx.stratum_set = gpu_ctx.stratum_set.start..(gpu_ctx.stratum_set.start + optimized_size);
  result
}

fn remove_empty_strata(_gpu_ctx: &GpuExecutionContext, gpu_strata: &mut Vec<ram::Stratum>) {
  gpu_strata.retain(|stratum| stratum.updates.len() != 0);
}

pub fn delay(_gpu_ctx: &GpuExecutionContext, gpu_strata: &mut Vec<ram::Stratum>) {
  let mut new_strata = Vec::new();

  for (i, strata) in gpu_strata.iter_mut().enumerate() {
    if !strata.is_recursive {
      continue;
    }
    let (delayed_updates, delayed_relations) = delay_unused_relation(strata);
    let new_stratum = ram::Stratum {
      is_recursive: false,
      relations: delayed_relations,
      updates: delayed_updates,
    };
    new_strata.push((i + 1, new_stratum));
  }

  for (location, stratum) in new_strata.into_iter().rev() {
    gpu_strata.insert(location, stratum);
  }
}

/// Delay relation R til after a stratum if the stratum is recursive but R is not used as a source
/// in the stratum.
pub fn delay_unused_relation(stratum: &mut ram::Stratum) -> (Vec<ram::Update>, BTreeMap<String, ram::Relation>) {
  assert!(stratum.is_recursive);
  let dependencies = stratum.source_relations();
  let delayed_updates: Vec<_> = stratum
    .updates
    .extract_if(..,|update| {
      let target = &update.target;
      !dependencies.contains(target)
    })
    .collect();
  let delayed_updates_names = delayed_updates
    .iter()
    .map(|update| update.target.clone())
    .collect::<HashSet<_>>();
  let delayed_relations = stratum
    .relations
    .extract_if(|name, _| delayed_updates_names.contains(name))
    .collect();
  (delayed_updates, delayed_relations)
}

/// Finds relations which are only computed onces in a stratum and are not used outside that
/// relation and replaces their use with the dataflow that computes them
pub fn inline(
  gpu_ctx: &GpuExecutionContext,
  gpu_strata: &mut Vec<ram::Stratum>,
  used_in_later_strata: &Vec<HashSet<String>>,
) {
  gpu_strata.iter_mut().zip(gpu_ctx.range()).for_each(|(stratum, i)| {
    let removed_relations = inline_single_use_relations(&mut stratum.updates, &used_in_later_strata[i]);
    stratum.relations.retain(|name, _| !removed_relations.contains(name));
  });
}

/// inline relation R iff R is only targeted by one update and R is only used as a source in one
/// location
pub fn inline_single_use_relations(
  updates: &mut Vec<ram::Update>,
  used_in_later_strata: &HashSet<String>,
) -> Vec<String> {
  let mut inlined_relations = Vec::new();

  // the number of updates that target each relation in this strata
  let mut target_counts = HashMap::new();
  updates.iter().for_each(|update| {
    *target_counts.entry(update.target.clone()).or_insert(0) += 1;
  });
  // the relations that only have one update which targets them
  let single_update_targets = target_counts
    .iter()
    .filter(|(_, count)| **count == 1)
    .map(|(relation, _)| relation.clone())
    .collect::<HashSet<_>>();

  for to_inline in single_update_targets {
    // Only inline relations that aren't used later and that are auto-generated by Scallop
    // (relations Scallop generates will have a '#' character)
    if used_in_later_strata.contains(&to_inline) || !to_inline.contains("#") {
      continue;
    }

    let mut use_sites = Vec::new();
    // Find the use sites of this relation
    for ram::Update { target: _, dataflow } in updates.iter() {
      use_sites.extend(dataflow.find_subtree_all(&ram::Dataflow::Relation(to_inline.clone())));
    }

    // If this relation is only used in one place, we can inline it
    if use_sites.len() <= 2 {
      let inline_body = updates.iter().find(|u| u.target == to_inline).unwrap().dataflow.clone();
      updates.retain(|u| u.target != to_inline);
      let mut found = false;
      for ram::Update { target: _, dataflow } in updates.iter_mut() {
        if let Some(inline_location) = dataflow.find_subtree_mut(&ram::Dataflow::Relation(to_inline.clone())) {
          found = true;
          *inline_location = inline_body.clone();
        }
      }
      if !found {
        panic!("Failed to find use site for relation {}", to_inline);
      }
      inlined_relations.push(to_inline.clone());
    }
  }

  inlined_relations
}

pub fn optimize_join_index(_gpu_ctx: &GpuExecutionContext, strata: &mut Vec<ram::Stratum>) {
  fn optimize_join_index_on_dataflow(dataflow: &mut ram::Dataflow, active_relations: &HashSet<&String>) {
    match dataflow {
      ram::Dataflow::Join(d1, d2, index_on_right) => {
        let left_dependencies = d1.source_relations();
        let right_dependencies = d2.source_relations();
        let left_is_constant = left_dependencies.intersection(active_relations).count() == 0;
        let right_is_constant = right_dependencies.intersection(active_relations).count() == 0;
        let left_is_source = matches!(**d1, ram::Dataflow::Relation(_));
        let right_is_source = matches!(**d2, ram::Dataflow::Relation(_));

        match (left_is_source, right_is_source, left_is_constant, right_is_constant) {
          (true, true, false, false) => {
            *index_on_right = false;
          }
          (true, true, true, false) => {
            *index_on_right = false;
          }
          (true, true, false, true) => {
            *index_on_right = true;
          }
          (true, true, true, true) => {
            *index_on_right = false;
          }
          (true, false, _, _) => {
            *index_on_right = false;
          }
          (false, true, _, _) => {
            *index_on_right = true;
          }
          (false, false, false, true) => {
            *index_on_right = true;
          }
          (false, false, _, _) => {
            *index_on_right = false;
          }
        };

        optimize_join_index_on_dataflow(d1, active_relations);
        optimize_join_index_on_dataflow(d2, active_relations);
      }
      ram::Dataflow::Unit(_) => {}
      ram::Dataflow::Union(d1, d2)
      | ram::Dataflow::Intersect(d1, d2)
      | ram::Dataflow::Product(d1, d2)
      | ram::Dataflow::Antijoin(d1, d2)
      | ram::Dataflow::Difference(d1, d2) => {
        optimize_join_index_on_dataflow(d1, active_relations);
        optimize_join_index_on_dataflow(d2, active_relations);
      }
      ram::Dataflow::Project(d, _)
      | ram::Dataflow::Filter(d, _)
      | ram::Dataflow::Find(d, _)
      | ram::Dataflow::OverwriteOne(d)
      | ram::Dataflow::ForeignPredicateConstraint(d, _, _)
      | ram::Dataflow::ForeignPredicateJoin(d, _, _)
      | ram::Dataflow::Sorted(d)
      | ram::Dataflow::JoinIndexedVec(d, _)
      | ram::Dataflow::Exclusion(d, _) => optimize_join_index_on_dataflow(d, active_relations),
      ram::Dataflow::Reduce(_) => unimplemented!(),
      ram::Dataflow::Relation(_) => {}
      ram::Dataflow::ForeignPredicateGround(_, _) | ram::Dataflow::UntaggedVec(_) => unimplemented!(),
    }
  }
  for stratum in strata.iter_mut() {
    let active_relations = stratum.relations.keys().collect();
    for update in &mut stratum.updates {
      optimize_join_index_on_dataflow(&mut update.dataflow, &active_relations);
    }
  }
}

/// If a branch of a recursive update is constant w.r.t. the stratum, extract that branch into a
/// separate relation and update
pub fn extract(
  _gpu_ctx: &GpuExecutionContext,
  gpu_strata: &mut Vec<ram::Stratum>,
  predicate_types: &HashMap<String, TupleType>,
) {
  gpu_strata.iter_mut().for_each(|stratum| {
    let target_relations = stratum.relations.keys().cloned().collect::<Vec<_>>();
    let target_relations_ref = target_relations.iter().collect::<HashSet<&_>>();
    let new_updates = stratum
      .updates
      .iter_mut()
      .filter_map(|update| {
        let is_recursive = update
          .dependency()
          .iter()
          .collect::<HashSet<_>>()
          .intersection(&target_relations_ref)
          .count()
          != 0;
        if is_recursive {
          extract_constant_branch(update, &target_relations_ref)
        } else {
          None
        }
      })
      .collect::<Vec<_>>();
    // Extend relations
    stratum.relations.extend(new_updates.iter().map(|update| {
      let schema: TupleType = update.dataflow.result_type(predicate_types);
      (
        update.target.clone(),
        ram::Relation::hidden_relation(update.target.clone(), schema.clone()),
      )
    }));
    // Extend updates
    stratum.updates.splice(0..0, new_updates);
  });
}

static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub fn gensym(name: &str) -> String {
  let value = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
  format!("{}_{}", name, value)
}

pub fn extract_constant_branch(update: &mut ram::Update, target_relations: &HashSet<&String>) -> Option<ram::Update> {
  let mut frontier: VecDeque<&_> = VecDeque::new();
  frontier.push_front(&update.dataflow);

  let branch_to_extract = loop {
    if let Some(branch) = frontier.pop_front() {
      // Don't extract if the branch is just a leaf
      match branch {
        ram::Dataflow::Relation(_) => {
          continue;
        }
        _ => {}
      };
      if branch.source_relations().intersection(target_relations).count() == 0 {
        break branch.clone();
      } else {
        match branch {
          ram::Dataflow::Unit(_) => {}
          ram::Dataflow::Union(d1, d2)
          | ram::Dataflow::Join(d1, d2, _)
          | ram::Dataflow::Intersect(d1, d2)
          | ram::Dataflow::Product(d1, d2)
          | ram::Dataflow::Antijoin(d1, d2)
          | ram::Dataflow::Difference(d1, d2) => {
            frontier.push_back(d1);
            frontier.push_back(d2);
          }
          ram::Dataflow::Project(d, _)
          | ram::Dataflow::Filter(d, _)
          | ram::Dataflow::Find(d, _)
          | ram::Dataflow::OverwriteOne(d)
          | ram::Dataflow::ForeignPredicateConstraint(d, _, _)
          | ram::Dataflow::ForeignPredicateJoin(d, _, _)
          | ram::Dataflow::Sorted(d)
          | ram::Dataflow::JoinIndexedVec(d, _)
          | ram::Dataflow::Exclusion(d, _) => {
            frontier.push_back(d);
          }
          ram::Dataflow::Relation(_) => {}
          ram::Dataflow::Reduce(_) => unimplemented!(),
          ram::Dataflow::ForeignPredicateGround(_, _) | ram::Dataflow::UntaggedVec(_) => unimplemented!(),
        }
      }
    } else {
      return None;
    }
  };

  let new_relation_name = gensym("extracted");
  let new_branch = ram::Dataflow::Relation(new_relation_name.clone());

  let cursor = update.dataflow.find_subtree_mut(&branch_to_extract).unwrap();
  *cursor = new_branch.clone();

  Some(ram::Update {
    target: new_relation_name,
    dataflow: branch_to_extract,
  })
}

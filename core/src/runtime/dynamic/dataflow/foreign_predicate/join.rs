use std::vec::IntoIter as IntoIter;

use crate::common::expr::*;
use crate::common::foreign_predicate::*;
use crate::common::input_tag::*;
use crate::common::tuple::*;
use crate::common::value::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

pub struct ForeignPredicateJoinDataflow<'a, Prov: Provenance> {
  pub left: DynamicDataflow<'a, Prov>,

  /// The foreign predicate
  pub foreign_predicate: String,

  /// The already bounded constants (in order to make this Dataflow free)
  pub args: Vec<Expr>,

  /// Provenance context
  pub ctx: &'a Prov,

  /// Runtime environment
  pub runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> Clone for ForeignPredicateJoinDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      left: self.left.clone(),
      foreign_predicate: self.foreign_predicate.clone(),
      args: self.args.clone(),
      ctx: self.ctx,
      runtime: self.runtime,
    }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for ForeignPredicateJoinDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::new(ForeignPredicateJoinBatches {
      batches: self.left.iter_stable(),
      foreign_predicate: self
        .runtime
        .predicate_registry
        .get(&self.foreign_predicate)
        .expect("Foreign predicate not found")
        .clone(),
      args: self.args.clone(),
      env: self.runtime,
      ctx: self.ctx,
    })
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    DynamicBatches::new(ForeignPredicateJoinBatches {
      batches: self.left.iter_recent(),
      foreign_predicate: self
        .runtime
        .predicate_registry
        .get(&self.foreign_predicate)
        .expect("Foreign predicate not found")
        .clone(),
      args: self.args.clone(),
      env: self.runtime,
      ctx: self.ctx,
    })
  }
}

#[derive(Clone)]
pub struct ForeignPredicateJoinBatches<'a, Prov: Provenance> {
  pub batches: DynamicBatches<'a, Prov>,
  pub foreign_predicate: DynamicForeignPredicate,
  pub args: Vec<Expr>,
  pub env: &'a RuntimeEnvironment,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Batches<'a, Prov> for ForeignPredicateJoinBatches<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    // First, try to get a batch from the set of batches
    self.batches.next_batch().map(|batch| {
      // Generate a new batch
      DynamicBatch::new(ForeignPredicateJoinBatch {
        batch: batch,
        foreign_predicate: self.foreign_predicate.clone(),
        args: self.args.clone(),
        current_batch: None,
        cached_batches: None,
        env: self.env,
        ctx: self.ctx,
      })
    })
  }
}

type FPOutputBatch = IntoIter<(DynamicInputTag, Vec<Value>)>;

#[derive(Clone)]
pub struct ForeignPredicateJoinBatch<'a, Prov: Provenance> {
  pub batch: DynamicBatch<'a, Prov>,
  pub foreign_predicate: DynamicForeignPredicate,
  pub args: Vec<Expr>,
  pub current_batch: Option<(DynamicElement<Prov>, FPOutputBatch)>,
  pub cached_batches: Option<IntoIter<(DynamicElement<Prov>, FPOutputBatch)>>,
  pub env: &'a RuntimeEnvironment,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> ForeignPredicateJoinBatch<'a, Prov> {
  fn eval_input(&self, elem: &DynamicElement<Prov>) -> Vec<Value> {
    self.args
      .iter()
      .map(|arg| match arg {
        Expr::Access(a) => elem.tuple[a].as_value(),
        Expr::Constant(c) => c.clone(),
        _ => panic!("Foreign predicate join only supports constant and access arguments"),
      })
      .collect()
  }

  /// Evaluate the foreign predicate on the given element
  fn eval_foreign_predicate_once(&self, elem: DynamicElement<Prov>) -> (DynamicElement<Prov>, FPOutputBatch) {
    // First get the arguments to pass to the foreign predicate
    let args_to_fp = self.eval_input(&elem);

    // Then evaluate the foreign predicate on these arguments
    let outputs: Vec<_> = self.foreign_predicate.evaluate_with_env(self.env, &args_to_fp);

    // Return the input element and output elements pair
    (elem, outputs.into_iter())
  }

  /// Populate the next batch assuming that the foreign predicate is non-batched
  ///
  /// Return bool indicating wether the next batch has been populated
  pub fn populate_next_batch_non_batched(&mut self) -> bool {
    if let Some(next_left_elem) = self.batch.next_elem() {
      self.current_batch = Some(self.eval_foreign_predicate_once(next_left_elem));
      return true;
    } else {
      return false;
    }
  }

  /// Populate the next batch assuming that the foreign predicate is batched
  ///
  /// Return bool indicating wether the next batch has been populated
  pub fn populate_next_batch_batched(&mut self) -> bool {
    // Check if there is a cached batch
    if let Some(list_of_batches) = &mut self.cached_batches {
      if let Some(next_batch) = list_of_batches.next() {
        self.current_batch = Some(next_batch);
        return true;
      }
    }

    // Need to generate a new cached batch
    let input_elems = if let Some(batch_size) = self.foreign_predicate.batch_size() {
      self.batch.take_n_elems(batch_size)
    } else {
      self.batch.take_all_elems()
    };

    // Check termination condition: if there is no new input, we will stop
    if input_elems.is_empty() {
      return false;
    }

    // Evaluate a set of inputs for the foreign predicate
    let batched_input = input_elems
      .iter()
      .map(|e| self.eval_input(&e))
      .collect::<Vec<_>>();

    // Pass the inputs to the foreign predicate in a batch
    let batched_output = self.foreign_predicate.evaluate_batch_with_env(
      self.env,
      batched_input.iter().map(|dp| dp.as_slice()).collect(),
    );

    // Sanity check on the batch size: input and output should be the same
    assert_eq!(
      batched_input.len(),
      batched_output.len(),
      "[Error] Output batch size ({}) is different from the input batch size ({}) when evaluating the foreign predicate `{}`. This is likely an error introduced by the foreign predicate.",
      batched_input.len(),
      batched_output.len(),
      self.foreign_predicate.name(),
    );

    // Process the output and generate the current batch
    let mut cache = input_elems
      .into_iter()
      .zip(batched_output.into_iter())
      .map(|(i, o)| (i, o.into_iter()))
      .collect::<Vec<_>>()
      .into_iter();
    let current_batch = cache.next();

    // Populate the current batch and cached batches
    self.current_batch = current_batch;
    self.cached_batches = Some(cache);

    // There is progress, so we return true
    return true;
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for ForeignPredicateJoinBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    loop {
      if let Some((left_elem, current_right_elems)) = &mut self.current_batch {
        if let Some((tag, values)) = current_right_elems.next() {
          // First work on the tag
          let input_tag = Prov::InputTag::from_dynamic_input_tag(&tag);
          let right_tag = self.ctx.tagging_optional_fn(input_tag);

          // Generate a tuple from the values produced by the foreign predicate
          let right_tuple = self.env.internalize_tuple(&Tuple::from(values))?;
          let tuple = (left_elem.tuple.clone(), right_tuple);
          let new_tag = self.ctx.mult(&left_elem.tag, &right_tag);

          // Generate the output element
          return Some(DynamicElement::new(tuple, new_tag));
        } else {
          self.current_batch = None;
        }
      } else {
        // Try populate the next batch depending on whether the foreign predicate is batched or not
        let populated = if self.foreign_predicate.batched() {
          self.populate_next_batch_batched()
        } else {
          self.populate_next_batch_non_batched()
        };

        // If there is nothing getting populated, then we terminate
        if !populated {
          return None;
        }
      }
    }
  }
}

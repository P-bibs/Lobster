use crate::common::expr::*;
use crate::common::foreign_predicate::*;
use crate::common::input_tag::*;
use crate::common::value::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct ForeignPredicateConstraintDataflow<'a, Prov: Provenance> {
  /// Sub-dataflow
  pub dataflow: DynamicDataflow<'a, Prov>,

  /// The foreign predicate
  pub foreign_predicate: String,

  /// The arguments to the foreign predicate
  pub args: Vec<Expr>,

  /// Provenance context
  pub ctx: &'a Prov,

  /// Runtime environment
  pub runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for ForeignPredicateConstraintDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    let fp = self
      .runtime
      .predicate_registry
      .get(&self.foreign_predicate)
      .expect("Foreign predicate not found");
    DynamicBatches::new(ForeignPredicateConstraintBatches {
      batches: self.dataflow.iter_stable(),
      foreign_predicate: fp.clone(),
      args: self.args.clone(),
      env: self.runtime,
      ctx: self.ctx,
    })
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    let fp = self
      .runtime
      .predicate_registry
      .get(&self.foreign_predicate)
      .expect("Foreign predicate not found");
    DynamicBatches::new(ForeignPredicateConstraintBatches {
      batches: self.dataflow.iter_recent(),
      foreign_predicate: fp.clone(),
      args: self.args.clone(),
      env: self.runtime,
      ctx: self.ctx,
    })
  }
}

#[derive(Clone)]
pub struct ForeignPredicateConstraintBatches<'a, Prov: Provenance> {
  pub batches: DynamicBatches<'a, Prov>,
  pub foreign_predicate: DynamicForeignPredicate,
  pub args: Vec<Expr>,
  pub env: &'a RuntimeEnvironment,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Batches<'a, Prov> for ForeignPredicateConstraintBatches<'a, Prov> {
  fn next_batch(&mut self) -> Option<DynamicBatch<'a, Prov>> {
    self.batches.next_batch().map(|batch| {
      DynamicBatch::new(ForeignPredicateConstraintBatch {
        batch: batch,
        foreign_predicate: self.foreign_predicate.clone(),
        cached_result: None,
        args: self.args.clone(),
        env: self.env,
        ctx: self.ctx,
      })
    })
  }
}

#[derive(Clone)]
pub struct ForeignPredicateConstraintBatch<'a, Prov: Provenance> {
  pub batch: DynamicBatch<'a, Prov>,
  pub foreign_predicate: DynamicForeignPredicate,
  pub cached_result: Option<std::vec::IntoIter<Option<DynamicElement<Prov>>>>,
  pub args: Vec<Expr>,
  pub env: &'a RuntimeEnvironment,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> ForeignPredicateConstraintBatch<'a, Prov> {
  fn process_one_output(&self, input: DynamicElement<Prov>, result: Vec<(DynamicInputTag, Vec<Value>)>) -> Option<DynamicElement<Prov>> {
    let Tagged { tuple, tag } = input;

    // Check if the foreign predicate returned a tag
    if !result.is_empty() {
      // Check the sanity of the result
      assert_eq!(
        result.len(),
        1,
        "Constraint foreign predicate should return at most one element per evaluation, however {} is found; this is likely a bug from the foreign predicate implementation of `{}`",
        result.len(),
        self.foreign_predicate.name(),
      );
      assert_eq!(
        result[0].1.len(),
        0,
        "Constraint foreign predicate's result should be 0-tuples; but we found the result to have arity {}; this is likely a bug from the foreign predicate implementation of `{}`",
        result[0].1.len(),
        self.foreign_predicate.name(),
      );

      // Generate an output dynamic element
      let input_tag = Prov::InputTag::from_dynamic_input_tag(&result[0].0);
      let new_tag = self.ctx.tagging_optional_fn(input_tag);
      let combined_tag = self.ctx.mult(&tag, &new_tag);
      Some(DynamicElement::new(tuple, combined_tag))
    } else {
      None
    }
  }

  fn process_one_non_batched_item(&mut self) -> Option<DynamicElement<Prov>> {
    while let Some(elem) = self.batch.next_elem() {
      // Try evaluate the arguments; if failed, continue to the next element in the batch
      let values = self
        .args
        .iter()
        .map(|arg| match arg {
          Expr::Access(a) => elem.tuple[a].as_value(),
          Expr::Constant(c) => c.clone(),
          _ => panic!("Invalid argument to bounded foreign predicate"),
        })
        .collect::<Vec<_>>();

      // Evaluate the foreign predicate to produce a list of output tags
      // Note that there will be at most one output tag since the foreign predicate is bounded
      let result = self.foreign_predicate.evaluate_with_env(self.env, &values);

      // Check if the foreign predicate returned a tag
      if let Some(postproc_result) = self.process_one_output(elem, result) {
        return Some(postproc_result);
      }
    }
    None
  }

  fn process_one_batched_item(&mut self) -> Option<DynamicElement<Prov>> {
    loop {
      // A.
      // Check if we have cached result iterator
      if let Some(cached_result) = &mut self.cached_result {
        // A.1.
        // Check if the iterator has a next element
        if let Some(next_result) = cached_result.next() {
          // Check if that next element satisfies the Constraint.
          // If so, return; otherwise it is okay and we move on to the next element
          // in the iterator
          if let Some(next_result) = next_result {
            return Some(next_result);
          } else {
            continue;
          }
        } else {
          // A.2.
          // The iterator is empty now, meaning that we want to grab a new batch.
          // We do that by setting the cached result iterator to None, so that
          // in the next iteration it starts from the branch B.
          self.cached_result = None;
        }
      } else {
        // B.
        // If not, take a batch from the input and process
        let batch = if let Some(batch_size) = self.foreign_predicate.batch_size() {
          self.batch.take_n_elems(batch_size)
        } else {
          self.batch.take_all_elems()
        };

        // Check if there is still some element in the batch
        if batch.is_empty() {
          // B.1.
          // There is no more element to draw from the input; we terminate the whole loop
          return None;
        } else {
          // B.2.
          // Otherwise, prepare the batch of inputs to be sent to the foreign predicate
          let batched_input = batch
            .iter()
            .map(|elem| {
              let proc_tuple = self
                .args
                .iter()
                .map(|arg| match arg {
                  Expr::Access(a) => elem.tuple[a].as_value(),
                  Expr::Constant(c) => c.clone(),
                  _ => panic!("Invalid argument to bounded foreign predicate"),
                })
                .collect::<Vec<_>>();
              proc_tuple
            })
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

          // Process the output
          let cache = batch
            .into_iter()
            .zip(batched_output.into_iter())
            .map(|(i, o)| self.process_one_output(i, o))
            .collect::<Vec<_>>()
            .into_iter();

          // Store the cache and wait for the next
          self.cached_result = Some(cache);
        }
      }
    }
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for ForeignPredicateConstraintBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    if self.foreign_predicate.batched() {
      self.process_one_batched_item()
    } else {
      self.process_one_non_batched_item()
    }
  }
}

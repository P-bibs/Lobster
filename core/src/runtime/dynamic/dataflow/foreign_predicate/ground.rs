use crate::common::foreign_predicate::*;
use crate::common::input_tag::*;
use crate::common::tuple::*;
use crate::common::value::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct ForeignPredicateGroundDataflow<'a, Prov: Provenance> {
  /// The foreign predicate
  pub foreign_predicate: String,

  /// The already bounded constants (in order to make this Dataflow free)
  pub bounded_constants: Vec<Value>,

  /// Whether this Dataflow is running on first iteration
  pub first_iteration: bool,

  /// Provenance context
  pub ctx: &'a Prov,

  /// Runtime environment
  pub runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> ForeignPredicateGroundDataflow<'a, Prov> {
  /// Turn one output into optional dynamic element for the ground dataflow
  fn process_one_output(&self, input_tag: DynamicInputTag, values: Vec<Value>) -> Option<DynamicElement<Prov>> {
    let input_tag = StaticInputTag::from_dynamic_input_tag(&input_tag);
    let tag = self.ctx.tagging_optional_fn(input_tag);
    let tuple = Tuple::from(values);
    let internal_tuple = self.runtime.internalize_tuple(&tuple)?;
    Some(DynamicElement::new(internal_tuple, tag))
  }

  /// Generate a batch from the foreign predicate
  fn generate_batch(&self) -> ElementsBatch<Prov> {
    // Fetch the foreign predicate
    let foreign_predicate = self.runtime
      .predicate_registry
      .get(&self.foreign_predicate)
      .expect("Foreign predicate not found");

    // Check if the foreign predicate supports batching
    let elements = if foreign_predicate.batched() {
      // Evaluate the foreign predicate
      foreign_predicate
        .evaluate_batch_with_env(self.runtime, vec![&self.bounded_constants])
        .concat() // Concatenation here means that we join all the outputs because there should be only one
        .into_iter()
        .filter_map(|(input_tag, values)| self.process_one_output(input_tag, values))
        .collect::<Vec<_>>()
    } else {
      // Evaluate the foreign predicate
      foreign_predicate
        .evaluate_with_env(self.runtime, &self.bounded_constants)
        .into_iter()
        .filter_map(|(input_tag, values)| self.process_one_output(input_tag, values))
        .collect::<Vec<_>>()
    };

    ElementsBatch::new(elements)
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for ForeignPredicateGroundDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    if self.first_iteration {
      DynamicBatches::empty()
    } else {
      DynamicBatches::single(self.generate_batch())
    }
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    if self.first_iteration {
      DynamicBatches::single(self.generate_batch())
    } else {
      DynamicBatches::empty()
    }
  }
}

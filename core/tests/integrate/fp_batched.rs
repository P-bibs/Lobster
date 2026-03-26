use scallop_core::common::foreign_predicate::ForeignPredicate;
use scallop_core::runtime::env::RuntimeEnvironment;
use scallop_core::common::value_type::ValueType;
use scallop_core::common::value::Value;
use scallop_core::common::input_tag::DynamicInputTag;

use scallop_core::integrate;
use scallop_core::runtime::*;
use scallop_core::utils::*;

use scallop_core::testing::*;

#[derive(Clone, Debug)]
struct BatchedRange;

impl ForeignPredicate for BatchedRange {
  fn name(&self) -> String {
    "my_range".to_string()
  }

  fn arity(&self) -> usize {
    3
  }

  fn num_bounded(&self) -> usize {
    2
  }

  fn batched(&self) -> bool {
    true
  }

  fn batch_size(&self) -> Option<usize> {
    Some(10)
  }

  fn argument_type(&self, i: usize) -> ValueType {
    match i {
      0 => ValueType::I32,
      1 => ValueType::I32,
      2 => ValueType::I32,
      _ => panic!("Should not happen"),
    }
  }

  fn evaluate(&self, _: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    panic!("Should not happen")
  }

  fn evaluate_with_env(&self, _: &RuntimeEnvironment, _: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    panic!("Should not happen")
  }

  fn evaluate_batch_with_env(&self, _: &RuntimeEnvironment, batched_input: Vec<&[Value]>) -> Vec<Vec<(DynamicInputTag, Vec<Value>)>> {
    batched_input
      .into_iter()
      .map(|input| {
        match (&input[0], &input[1]) {
          (Value::I32(start), Value::I32(end)) => {
            (*start..*end)
              .map(|i| (DynamicInputTag::None, vec![Value::I32(i)])).collect()
          }
          _ => panic!("should not happen")
        }
      })
      .collect()
  }
}

#[test]
fn batched_ground_fp_range() {
  // Initialize a context
  let prov_ctx = provenance::unit::UnitProvenance::default();
  let mut ctx = integrate::IntegrateContext::<_, RcFamily>::new(prov_ctx);

  // Register the foreign predicate
  ctx.register_foreign_predicate(BatchedRange).unwrap();

  // Add a program
  ctx
    .add_program(
      r#"
      rel result(i) = my_range(0, 3, i)
      "#,
    )
    .unwrap();

  // Run the context
  ctx.run().unwrap();

  // Test the result
  expect_output_collection(
    "result",
    ctx.computed_relation_ref("result").unwrap(),
    vec![(0i32,), (1,), (2,)],
  );
}

#[test]
fn batched_join_fp_range() {
  // Initialize a context
  let prov_ctx = provenance::unit::UnitProvenance::default();
  let mut ctx = integrate::IntegrateContext::<_, RcFamily>::new(prov_ctx);

  // Register the foreign predicate
  ctx.register_foreign_predicate(BatchedRange).unwrap();

  // Add a program
  ctx
    .add_program(
      r#"
      rel pair = {(0, 10), (20, 30), (40, 50)}
      rel result(k) = pair(i, j) and my_range(i, j, k)
      "#,
    )
    .unwrap();

  // Run the context
  ctx.run().unwrap();

  // Test the result
  expect_output_collection(
    "result",
    ctx.computed_relation_ref("result").unwrap(),
    (0i32..10).chain(20..30).chain(40..50).map(|i| (i,)).collect::<Vec<_>>(),
  );
}

use pyo3::types::*;
use pyo3::*;

use scallop_core::common::foreign_predicate::*;
use scallop_core::common::input_tag::*;
use scallop_core::common::tuple_type::*;
use scallop_core::common::value::*;
use scallop_core::common::value_type::*;
use scallop_core::runtime::env::*;

use super::tag::*;
use super::tuple::*;

#[derive(Clone)]
pub struct PythonForeignPredicate {
  fp: PyObject,
  name: String,
  type_params: Vec<ValueType>,
  types: Vec<ValueType>,
  tag_type: String,
  num_bounded: usize,
  batched: bool,
  batch_size: Option<usize>,
  suppress_warning: bool,
}

impl PythonForeignPredicate {
  pub fn new(fp: PyObject) -> Self {
    Python::with_gil(|py| {
      let name = fp
        .getattr(py, "name")
        .expect("Cannot get foreign predicate name")
        .extract(py)
        .expect("Foreign predicate name cannot be extracted into String");

      let suppress_warning = fp
        .getattr(py, "suppress_warning")
        .expect("Cannot get foreign predicate `suppress_warning` property")
        .extract(py)
        .expect("Foreign predicate `suppress_warning` cannot be extracted into bool");

      let batched = fp
        .getattr(py, "batched")
        .expect("Cannot get foreign predicate `batched` property")
        .extract(py)
        .expect("Foreign predicate `batched` cannot be extracted into bool");

      let batch_size = fp
        .getattr(py, "batch_size")
        .expect("Cannot get foreign predicate `batch_size` property")
        .extract(py)
        .expect("Foreign predicate `batch_size` cannot be extracted into Option<usize>");

      let type_params = {
        let type_param_pyobjs: Vec<PyObject> = fp
          .getattr(py, "type_params")
          .expect("Cannot get all_argument_types function")
          .extract(py)
          .expect("Cannot extract function into PyObject");

        // Convert the Python types into Scallop types
        type_param_pyobjs
          .into_iter()
          .map(|py_type| py_param_type_to_fp_param_type(py_type, py))
          .collect()
      };

      let types = {
        // Call `all_argument_types` function of the Python object
        let func: PyObject = fp
          .getattr(py, "all_argument_types")
          .expect("Cannot get all_argument_types function")
          .extract(py)
          .expect("Cannot extract function into PyObject");

        // Invoke the function
        let py_types: Vec<PyObject> = func
          .call0(py)
          .expect("Cannot call function")
          .extract(py)
          .expect("Cannot extract into PyList");

        // Convert the Python types into Scallop types
        py_types
          .into_iter()
          .map(|py_type| py_param_type_to_fp_param_type(py_type, py))
          .collect()
      };

      let tag_type: String = fp
        .getattr(py, "tag_type")
        .expect("Cannot get tag_type")
        .extract(py)
        .expect("tag_type is not a string");

      let num_bounded: usize = {
        let func: PyObject = fp
          .getattr(py, "num_bounded")
          .expect("Cannot get num_bounded function")
          .extract(py)
          .expect("Cannot extract function into PyObject");

        // Invoke the function
        func
          .call0(py)
          .expect("Cannot call function")
          .extract(py)
          .expect("Cannot extract into usize")
      };

      Self {
        fp,
        name,
        type_params,
        types,
        tag_type,
        num_bounded,
        batched,
        batch_size,
        suppress_warning,
      }
    })
  }

  fn output_tuple_type(&self) -> TupleType {
    self.types.iter().skip(self.num_bounded).cloned().collect()
  }
}

impl ForeignPredicate for PythonForeignPredicate {
  fn name(&self) -> String {
    self.name.clone()
  }

  fn generic_type_parameters(&self) -> Vec<ValueType> {
    self.type_params.clone()
  }

  fn arity(&self) -> usize {
    self.types.len()
  }

  fn argument_type(&self, i: usize) -> ValueType {
    self.types[i].clone()
  }

  fn num_bounded(&self) -> usize {
    self.num_bounded
  }

  fn batched(&self) -> bool {
    self.batched
  }

  fn batch_size(&self) -> Option<usize> {
    self.batch_size.clone()
  }

  fn evaluate_with_env(&self, env: &RuntimeEnvironment, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    Python::with_gil(|py| {
      // Construct the arguments
      let args: Vec<Py<PyAny>> = bounded.iter().filter_map(|v| to_python_value(v, &env.into())).collect();
      let args_tuple = PyTuple::new(py, args);

      // Invoke the function
      let maybe_result = match self.fp.call1(py, args_tuple) {
        Ok(result) => Some(result),
        Err(err) => {
          if !self.suppress_warning {
            eprintln!("[Foreign Predicate Error] {}", err);
            err.print(py);
          }
          None
        }
      };

      // Turn the result back to Scallop values
      if let Some(result) = maybe_result {
        let output_tuple_type = self.output_tuple_type();
        let elements: Vec<(&PyAny, &PyAny)> = result.extract(py).expect(&format!(
          "Cannot extract into list of elements during evaluation of {}",
          self.name
        ));
        let internal: Vec<_> = elements
          .into_iter()
          .filter_map(|(py_tag, py_tup)| {
            let tag = match from_python_input_tag(&self.tag_type, py_tag) {
              Ok(tag) => tag,
              Err(err) => {
                if !self.suppress_warning {
                  eprintln!("Error when parsing tag: {}", err);
                }
                return None;
              }
            };
            let tuple = match from_python_tuple(py_tup, &output_tuple_type, &env.into()) {
              Ok(tuple) => tuple,
              Err(err) => {
                if !self.suppress_warning {
                  eprintln!("Error when parsing tuple: {}", err);
                }
                return None;
              }
            };
            Some((tag, tuple.as_values()))
          })
          .collect();
        internal
      } else {
        vec![]
      }
    })
  }

  fn evaluate_batch_with_env(&self, env: &RuntimeEnvironment, batched_input: Vec<&[Value]>) -> Vec<Vec<(DynamicInputTag, Vec<Value>)>> {
    Python::with_gil(|py| {
      let pyenv = env.into();

      // Construct the batch of arguments
      let batched_py_facts = batched_input
        .into_iter()
        .map(|values| {
          let pyvalues = values.iter().filter_map(|v| to_python_value(v, &pyenv)).collect::<Box<[_]>>();
          let pytuple = PyTuple::new(py, pyvalues);
          pytuple
        });
      let batched_py_list = PyList::new(py, batched_py_facts);
      let batched_py_input = PyTuple::new(py, std::iter::once(batched_py_list));

      // Actually call the python foreign predicate
      match self.fp.call1(py, batched_py_input) {
        Ok(result) => {
          // First parse the outputs into the rust elements
          let batched_elements: Vec<Vec<(&PyAny, &PyAny)>> = result.extract(py).expect(&format!(
            "Cannot extract into batched list of elements during evaluation of `{}`",
            self.name
          ));

          // Based on the output tuple type, parse the python outputs into Scallop values
          let output_tuple_type = self.output_tuple_type();
          batched_elements
            .into_iter() // batch level iteration
            .map(|elements| {
              elements
                .into_iter() // output list level iteration
                .filter_map(|(py_tag, py_tup)| {
                  let tag = match from_python_input_tag(&self.tag_type, py_tag) {
                    Ok(tag) => tag,
                    Err(err) => {
                      if !self.suppress_warning {
                        eprintln!("Error when parsing tag: {}", err);
                      }
                      return None;
                    }
                  };
                  let tuple = match from_python_tuple(py_tup, &output_tuple_type, &env.into()) {
                    Ok(tuple) => tuple,
                    Err(err) => {
                      if !self.suppress_warning {
                        eprintln!("Error when parsing tuple: {}", err);
                      }
                      return None;
                    }
                  };
                  Some((tag, tuple.as_values()))
                })
                .collect()
            })
            .collect()
        },
        Err(err) => {
          if !self.suppress_warning {
            eprintln!("[Foreign Predicate Error] {}", err);
            err.print(py);
          }
          vec![]
        }
      }
    })
  }
}

fn py_param_type_to_fp_param_type(obj: PyObject, py: Python<'_>) -> ValueType {
  let param_type: String = obj
    .getattr(py, "type")
    .expect("Cannot get param type")
    .extract(py)
    .expect("Cannot extract into String");
  match param_type.as_str() {
    "i8" => ValueType::I8,
    "i16" => ValueType::I16,
    "i32" => ValueType::I32,
    "i64" => ValueType::I64,
    "i128" => ValueType::I128,
    "isize" => ValueType::ISize,
    "u8" => ValueType::U8,
    "u16" => ValueType::U16,
    "u32" => ValueType::U32,
    "u64" => ValueType::U64,
    "u128" => ValueType::U128,
    "usize" => ValueType::USize,
    "f32" => ValueType::F32,
    "f64" => ValueType::F64,
    "bool" => ValueType::Bool,
    "char" => ValueType::Char,
    "String" => ValueType::String,
    "DateTime" => ValueType::DateTime,
    "Duration" => ValueType::Duration,
    "Entity" => ValueType::Entity,
    "Tensor" => ValueType::Tensor,
    _ => panic!("Unknown type {}", param_type),
  }
}

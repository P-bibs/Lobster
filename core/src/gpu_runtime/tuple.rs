use crate::common::{tuple::Tuple, tuple_type::TupleType, value_type::ValueType};

use super::{array::C_Array, value::{C_ValueType, C_Value}};

#[repr(C)]
#[derive(Debug, Clone)]
pub enum C_TupleType {
  Tuple(C_Array<C_TupleType>),
  Unit(C_ValueType),
}
impl std::fmt::Display for C_TupleType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      C_TupleType::Tuple(tuple_types) => write!(f, "({})", tuple_types),
      C_TupleType::Unit(value_type) => write!(f, "{}", value_type),
    }
  }
}

pub fn value_list_to_tuple(value_list: Vec<C_Value>, schema: &C_TupleType) -> Tuple {
  fn to_tuple(value_list: &mut std::collections::VecDeque<C_Value>, schema: &C_TupleType) -> Tuple {
    match schema {
      C_TupleType::Tuple(tuple_types) => {
        let mut tuples = Vec::new();
        for tuple_type in tuple_types.as_slice() {
          let tuple = to_tuple(value_list, &tuple_type);
          tuples.push(tuple);
        }
        Tuple::Tuple(tuples.into())
      }
      C_TupleType::Unit(_) => {
        let elt = value_list.pop_front().unwrap();
        Tuple::Value(elt.to_value())
      }
    }
  }
  let mut deq = value_list.into();
  to_tuple(&mut deq, schema)
}

impl C_TupleType {
  pub fn from_tuple_type(tuple_type: &TupleType) -> Self {
    match tuple_type {
      TupleType::Tuple(tuple_types) => {
        let tuple_types = tuple_types
          .iter()
          .map(|tuple_type| C_TupleType::from_tuple_type(tuple_type))
          .collect::<Vec<_>>();
        C_TupleType::Tuple(C_Array::new(tuple_types))
      }
      TupleType::Value(value_type) => C_TupleType::Unit(C_ValueType::from_value_type(value_type)),
    }
  }

  pub fn to_tuple_type(&self) -> TupleType {
    match self {
      C_TupleType::Tuple(tuple_types) => TupleType::Tuple(tuple_types.as_slice().iter().map(|tuple_type| tuple_type.to_tuple_type()).collect()),
      C_TupleType::Unit(value_type) => TupleType::Value(value_type.to_value_type()),
    }
  }

  pub fn from_tuple(tuple: &Tuple) -> Self {
    match tuple {
      Tuple::Tuple(tuples) => {
        let tuples = tuples
          .iter()
          .map(|tuple| C_TupleType::from_tuple(tuple))
          .collect::<Vec<_>>();
        C_TupleType::Tuple(C_Array::new(tuples))
      }
      Tuple::Value(value) => C_TupleType::Unit(C_ValueType::from_value_type(&ValueType::type_of(&value))),
    }
  }

  pub fn width(&self) -> usize {
    match self {
      C_TupleType::Tuple(tuple_types) => tuple_types.as_slice().iter().map(|tuple_type| tuple_type.width()).sum(),
      C_TupleType::Unit(_) => 1,
    }
  }
}

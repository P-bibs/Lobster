use crate::common::{value_type::ValueType, value::Value};

use super::array::C_String;
use ordered_float::OrderedFloat;

#[repr(C)]
#[derive(Debug, Clone)]
pub enum C_Value {
  None,
  I8(i8),
  I16(i16),
  I32(i32),
  I64(i64),
  //I128(i128),
  ISize(isize),
  U8(u8),
  U16(u16),
  U32(u32),
  U64(u64),
  //U128(u128),
  USize(usize),
  F32(f32),
  F64(f64),
  Char(u32),
  Bool(bool),
  //Str(&'static CStr),
  // TODO: only need to copy if this stratum actually has string operations
  String(C_String),
  Symbol(usize),
  //SymbolString(CString),
  //DateTime(DateTime<Utc>),
  //Duration(Duration),
  Entity(u64),
  //EntityString(String)
}
impl std::fmt::Display for C_Value {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      C_Value::None => write!(f, "None"),
      C_Value::I8(value) => write!(f, "{}", value),
      C_Value::I16(value) => write!(f, "{}", value),
      C_Value::I32(value) => write!(f, "{}", value),
      C_Value::I64(value) => write!(f, "{}", value),
      //C_Value::I128(value) => write!(f, "{}", value),
      C_Value::ISize(value) => write!(f, "{}", value),
      C_Value::U8(value) => write!(f, "{}", value),
      C_Value::U16(value) => write!(f, "{}", value),
      C_Value::U32(value) => write!(f, "{}", value),
      C_Value::U64(value) => write!(f, "{}", value),
      //C_Value::U128(value) => write!(f, "{}", value),
      C_Value::USize(value) => write!(f, "{}", value),
      C_Value::F32(value) => write!(f, "{}", value),
      C_Value::F64(value) => write!(f, "{}", value),
      C_Value::Char(value) => write!(f, "{}", value),
      C_Value::Bool(value) => write!(f, "{}", value),
      //C_Value::Str(value) => write!(f, "{}", value),
      C_Value::String(value) => write!(f, "{}", value),
      C_Value::Symbol(value) => write!(f, "{}", value),
      //C_Value::SymbolString(value) => write!(f, "{}", value),
      //C_Value::DateTime(value) => write!(f, "{}", value),
      //C_Value::Duration(value) => write!(f, "{}", value),
      C_Value::Entity(value) => write!(f, "{}", value),
      //C_Value::EntityString(value) => write!(f, "{}", value),
    }
  }
}
impl C_Value {
  pub fn from_value(value: &Value) -> Self {
    match value {
      Value::I8(value) => C_Value::I8(*value),
      Value::I16(value) => C_Value::I16(*value),
      Value::I32(value) => C_Value::I32(*value),
      Value::I64(value) => C_Value::I64(*value),
      //Value::I128(value) => C_Value::I128(*value),
      Value::ISize(value) => C_Value::ISize(*value),
      Value::U8(value) => C_Value::U8(*value),
      Value::U16(value) => C_Value::U16(*value),
      Value::U32(value) => C_Value::U32(*value),
      Value::U64(value) => C_Value::U64(*value),
      //Value::U128(value) => C_Value::U128(*value),
      Value::USize(value) => C_Value::USize(*value),
      Value::F32(value) => C_Value::F32(**value),
      Value::F64(value) => C_Value::F64(*value),
      Value::Char(value) => C_Value::Char(value.to_digit(10).unwrap()),
      Value::Bool(value) => C_Value::Bool(*value),
      //Value::Str(value) => C_Value::Str(CStr::from_bytes_with_nul(value).unwrap()),
      Value::String(value) => C_Value::String(C_String::new(value.clone())),
      Value::Symbol(value) => C_Value::Symbol(*value),
      //Value::SymbolString(value) => C_Value::SymbolString(CString::new(value.clone()).unwrap()),
      //Value::DateTime(value) => C_Value::DateTime(*value),
      //Value::Duration(value) => C_Value::Duration(*value),
      _ => unimplemented!(),
    }
  }

  pub fn to_value(&self) -> Value {
    match self {
      C_Value::I8(value) => Value::I8(*value),
      C_Value::I16(value) => Value::I16(*value),
      C_Value::I32(value) => Value::I32(*value),
      C_Value::I64(value) => Value::I64(*value),
      //C_Value::I128(value) => Value::I128(*value),
      C_Value::ISize(value) => Value::ISize(*value),
      C_Value::U8(value) => Value::U8(*value),
      C_Value::U16(value) => Value::U16(*value),
      C_Value::U32(value) => Value::U32(*value),
      C_Value::U64(value) => Value::U64(*value),
      //C_Value::U128(value) => Value::U128(*value),
      C_Value::USize(value) => Value::USize(*value),
      C_Value::F32(value) => Value::F32(OrderedFloat(*value)),
      C_Value::F64(value) => Value::F64(*value),
      //C_Value::Char(value) => Value::Char(value.to_string()),
      C_Value::Bool(value) => Value::Bool(*value),
      //C_Value::Str(value) => Value::Str(value.to_str().unwrap().to_string()),
      C_Value::String(value) => Value::String(value.to_string()),
      C_Value::Symbol(value) => Value::Symbol(*value),
      //C_Value::SymbolString(value) => Value::SymbolString(value.to_str().unwrap().to_string()),
      //C_Value::DateTime(value) => Value::DateTime(*value),
      //C_Value::Duration(value) => Value::Duration(*value),
      _ => unimplemented!(),
    }
  }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub enum C_ValueType {
  None,
  I8,
  I16,
  I32,
  I64,
  //I128,
  ISize,
  U8,
  U16,
  U32,
  U64,
  //U128,
  USize,
  F32,
  F64,
  Char,
  Bool,
  Str,
  String,
  Symbol,
  //DateTime,
  //Duration,
  Entity,
  //Tensor,
}
impl std::fmt::Display for C_ValueType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      C_ValueType::None => write!(f, "none"),
      C_ValueType::I8 => write!(f, "i8"),
      C_ValueType::I16 => write!(f, "i16"),
      C_ValueType::I32 => write!(f, "i32"),
      C_ValueType::I64 => write!(f, "i64"),
      //C_ValueType::I128 => write!(f, "i128"),
      C_ValueType::ISize => write!(f, "isize"),
      C_ValueType::U8 => write!(f, "u8"),
      C_ValueType::U16 => write!(f, "u16"),
      C_ValueType::U32 => write!(f, "u32"),
      C_ValueType::U64 => write!(f, "u64"),
      //C_ValueType::U128 => write!(f, "u128"),
      C_ValueType::USize => write!(f, "usize"),
      C_ValueType::F32 => write!(f, "f32"),
      C_ValueType::F64 => write!(f, "f64"),
      C_ValueType::Char => write!(f, "char"),
      C_ValueType::Bool => write!(f, "bool"),
      C_ValueType::Str => write!(f, "str"),
      C_ValueType::String => write!(f, "string"),
      C_ValueType::Symbol => write!(f, "symbol"),
      //C_ValueType::DateTime => write!(f, "datetime"),
      //C_ValueType::Duration => write!(f, "duration"),
      C_ValueType::Entity => write!(f, "entity"),
      //C_ValueType::Tensor => write!(f, "tensor"),
    }
  }
}

impl C_ValueType {
  pub fn from_value_type(value_type: &ValueType) -> Self {
    match value_type {
      ValueType::I8 => C_ValueType::I8,
      ValueType::I16 => C_ValueType::I16,
      ValueType::I32 => C_ValueType::I32,
      ValueType::I64 => C_ValueType::I64,
      //ValueType::I128 => C_ValueType::I128,
      ValueType::ISize => C_ValueType::ISize,
      ValueType::U8 => C_ValueType::U8,
      ValueType::U16 => C_ValueType::U16,
      ValueType::U32 => C_ValueType::U32,
      ValueType::U64 => C_ValueType::U64,
      //ValueType::U128 => C_ValueType::U128,
      ValueType::USize => C_ValueType::USize,
      ValueType::F32 => C_ValueType::F32,
      ValueType::F64 => C_ValueType::F64,
      ValueType::Char => C_ValueType::Char,
      ValueType::Bool => C_ValueType::Bool,
      //ValueType::Str => C_ValueType::Str,
      ValueType::String => C_ValueType::String,
      ValueType::Symbol => C_ValueType::Symbol,
      //ValueType::DateTime => C_ValueType::DateTime,
      //ValueType::Duration => C_ValueType::Duration,
      ValueType::Entity => C_ValueType::Entity,
      //ValueType::Tensor => C_ValueType::Tensor,
      _ => unimplemented!(),
    }
  }

  pub fn to_value_type(&self) -> ValueType {
    match self {
      C_ValueType::I8 => ValueType::I8,
      C_ValueType::I16 => ValueType::I16,
      C_ValueType::I32 => ValueType::I32,
      C_ValueType::I64 => ValueType::I64,
      //C_ValueType::I128 => ValueType::I128,
      C_ValueType::ISize => ValueType::ISize,
      C_ValueType::U8 => ValueType::U8,
      C_ValueType::U16 => ValueType::U16,
      C_ValueType::U32 => ValueType::U32,
      C_ValueType::U64 => ValueType::U64,
      //C_ValueType::U128 => ValueType::U128,
      C_ValueType::USize => ValueType::USize,
      C_ValueType::F32 => ValueType::F32,
      C_ValueType::F64 => ValueType::F64,
      C_ValueType::Char => ValueType::Char,
      C_ValueType::Bool => ValueType::Bool,
      //C_ValueType::Str => ValueType::Str,
      C_ValueType::String => ValueType::String,
      C_ValueType::Symbol => ValueType::Symbol,
      //C_ValueType::DateTime => ValueType::DateTime,
      //C_ValueType::Duration => ValueType::Duration,
      C_ValueType::Entity => ValueType::Entity,
      //C_ValueType::Tensor => ValueType::Tensor,
      _ => unimplemented!(),
    }
  }
}

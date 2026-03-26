from typing import TypeVar, ForwardRef

from .. import torch_importer
from .. import syntax

ALIASES = {
  "I8": "i8",
  "I16": "i16",
  "I32": "i32",
  "I64": "i64",
  "I128": "i128",
  "ISize": "isize",
  "U8": "u8",
  "U16": "u16",
  "U32": "u32",
  "U64": "u64",
  "U128": "u128",
  "USize": "usize",
  "F32": "f32",
  "F64": "f64",
  "Char": "char",
  "Bool": "bool",
}


# Predicate Data Type
class Type:
  def __init__(self, value):
    if isinstance(value, ForwardRef):
      self.type = Type.sanitize_type_str(value.__forward_arg__)
    elif isinstance(value, syntax.AstTypeNode):
      self.type = value.name()
    elif isinstance(value, TypeVar):
      self.type = Type.sanitize_type_str(value.__name__)
    elif isinstance(value, str):
      self.type = Type.sanitize_type_str(value)
    elif value == float:
      self.type = "f32"
    elif value == int:
      self.type = "i32"
    elif value == bool:
      self.type = "bool"
    elif value == str:
      self.type = "String"
    elif value == torch_importer.Tensor:
      self.type = "Tensor"
    else:
      raise Exception(f"Unknown scallop predicate type annotation `{value}`")

  def __repr__(self):
    return self.type

  @staticmethod
  def sanitize_type_str(value):
    if value == "i8" or value == "i16" or value == "i32" or value == "i64" or value == "i128" or value == "isize" or \
      value == "u8" or value == "u16" or value == "u32" or value == "u64" or value == "u128" or value == "usize" or \
      value == "f32" or value == "f64" or \
      value == "bool" or value == "char" or value == "String" or value == "Symbol" or value == "Tensor" or \
      value == "DateTime" or value == "Duration" or value == "Entity":
      return value
    elif value in ALIASES:
      return ALIASES[value]
    else:
      raise Exception(f"Unknown scallop predicate type annotation `{value}`")

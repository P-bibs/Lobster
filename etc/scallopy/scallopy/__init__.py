from .attribute import foreign_attribute, ForeignAttributeProcessor
from .context import ScallopContext
from .provenance import ScallopProvenance
from .function import GenericTypeParameter, foreign_function, ForeignFunction
from .predicate import foreign_predicate, ForeignPredicate
from .predicate import Facts, BatchedFacts, BatchedTuples
from .value_types import i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64, bool, char, String, Symbol, DateTime, Duration, Entity, Tensor, Any, Number, Integer, SignedInteger, UnsignedInteger, Float
from .tag_types import none, natural, prob, exclusion, boolean, exclusive_prob, diff_prob, exclusive_diff_prob
from .input_mapping import InputMapping
from .plugin import PluginRegistry, Plugin
from .syntax import NodeLocation, AstNode, AstConstantNode, AstEnumNode, AstStructNode, AstTypeNode, AstVariantNode
from .scallopy import torch_tensor_enabled

from . import input_output as io

# Provide a few aliases
Context = ScallopContext
Provenance = ScallopProvenance
Generic = GenericTypeParameter
ff = foreign_function
fp = foreign_predicate
Map = InputMapping

# Loading
def __getattr__(name: str):
  forward_alias = ["ScallopForwardFunction", "ForwardFunction", "Module"]
  if name in forward_alias:
    from .forward import ScallopForwardFunction
    return ScallopForwardFunction
  elif name in globals():
    return globals()[name]
  else:
    raise AttributeError(f"Attribute `{name}` not found inside `scallopy`")

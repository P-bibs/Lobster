import collections
import typing
from typing import TypeVar, Generic, Union, Tuple, Callable, List, Optional, Any, ForwardRef, ClassVar, TypeVarTuple, Unpack
import inspect

from .. import utils
from .. import tag_types

from .types import Type
from .predicate import ForeignPredicate
from .utils import _extract_return_tuple_type, TyUtils as Ty

ERROR_HEADER = "[Scallopy Foreign Predicate Typing Error]"

@utils.doublewrap
def foreign_predicate(
  func: Callable,
  name: Optional[str] = None,
  type_params: Optional[List] = None,
  input_arg_types: Optional[List] = None,
  output_arg_types: Optional[List] = None,
  tag_type: Optional[Any] = None,
  batched: bool = False,
  batch_size: Optional[int] = None,
  suppress_warning: bool = False,
):
  """
  A decorator to create a Scallop foreign predicate, for example

  ``` python
  @scallopy.foreign_predicate
  def string_chars(s: str) -> scallopy.Facts[Tuple[int, char]]:
    for (i, c) in enumerate(s):
      yield (i, c)
  ```

  A foreign predicate can be batched.
  In this case, there could only be one single argument that must have the type
  `BatchedTuples[...]`.
  The return type must be noted as `BatchedFacts[TagType, TupleType]`.

  ``` python
  @scallopy.foreign_predicate(batched=True)
  def batched_plus(
    inputs: scallopy.BatchedTuples[int, int]]
  ) -> scallopy.BatchedFacts[None, Tuple[int]]:
    return [[(i + j,)] for (i, j) in inputs]
  ```
  """

  # Get the function name
  func_name = func.__name__ if not name else name

  # Get the function signature
  signature = inspect.signature(func)

  # Store all the type params
  if type_params is None:
    processed_type_params = []
  else:
    processed_type_params = [Type(type_param) for type_param in type_params]

  # Store all the argument types
  if input_arg_types is None:
    params = signature.parameters
    argument_types = _extract_arg_types(params, batched)
  else:
    argument_types = [Type(t) for t in input_arg_types]

  # Find return type
  if output_arg_types is None:
    if signature.return_annotation is None:
      raise Exception(f"Return type annotation not provided")
    else:
      ret_anno = signature.return_annotation
      (return_tag_type, return_tuple_type) = _extract_return_tag_tuple_type(ret_anno, batched)
  else:
    return_tag_type = tag_types.get_tag_type(tag_type)
    return_tuple_type = [Type(t) for t in output_arg_types]

  # Create the foreign predicate
  return ForeignPredicate(
    func=func,
    name=func_name,
    type_params=processed_type_params,
    input_arg_types=argument_types,
    output_arg_types=return_tuple_type,
    tag_type=return_tag_type,
    batched=batched,
    batch_size=batch_size,
    suppress_warning=suppress_warning,
  )


def _extract_arg_types(parameters, batched):
  if batched:
    argument_types = _extract_batched_arg_types(parameters)
  else:
    argument_types = _extract_non_batched_arg_types(parameters)
  return argument_types


def _extract_non_batched_arg_types(parameters):
  # Find argument types
  argument_types = []
  for (arg_name, item) in parameters.items():
    optional = item.default != inspect.Parameter.empty
    if item.annotation is None or 'inspect._empty' in str(item.annotation):
      raise Exception(f"Argument `{arg_name}`'s type annotation not provided in the foreign predicate")
    if item.kind == inspect.Parameter.VAR_POSITIONAL:
      raise Exception(f"Cannot have variable arguments in foreign predicate")
    elif not optional:
      ty = Type(item.annotation)
      argument_types.append(ty)
    else:
      raise Exception(f"Cannot have optional argument in foreign predicate")
  return argument_types


def _extract_batched_arg_types(parameters):
  argument_types = []

  # Expect there to be only one single argument
  parameters = list(parameters.items())
  assert len(parameters) == 1, f"batched and explicitly typed foreign predicate expect only 1 parameter; but found {len(parameters)}"
  the_param = parameters[0][1] # Get the 0th parameter

  # Make sure that there is annotation
  assert the_param.default == inspect.Parameter.empty, f"parameter for batched foreign predicate should not have default value"
  assert the_param.annotation is not None and 'inspect._empty' not in str(the_param.annotation), \
    f"batched and explicitly typed foreign predicate's parameter must be typed"

  # Make sure that it is a type List[Tuple[...]]
  assert the_param.annotation.__origin__ == list, f"batched foreign predicate's parameter should have BatchedTuples type"
  assert len(the_param.annotation.__args__) == 1 and the_param.annotation.__args__[0].__origin__ == tuple, f"batched foreign predicate's parameter should have BatchedTuples type"

  # Passed all the test, directly extract the annotations
  tuple_annotations = the_param.annotation.__args__[0].__args__
  for anno in tuple_annotations:
    argument_types.append(Type(anno))

  return argument_types


def _extract_return_tag_tuple_type(ret_anno, batched):
  # First, depending on whether the FP is batched, extract corresponding return value information
  if batched:
    (variant_1_fact_tuple, variant_2_value_tuple) = _extract_batched_ret_type_variants(ret_anno)
  else:
    (variant_1_fact_tuple, variant_2_value_tuple) = _extract_non_batched_ret_type_variants(ret_anno)

  # More fine-grained information and whether the return type indicates that the FP is tagged
  (variant_1_tag, variant_1_value_tuple) = _extract_ret_type_variant_1_tag_and_value_tuple(variant_1_fact_tuple)
  (maybe_variant_1_value_nested_tuple, is_tagged) = _extract_ret_type_is_tagged(variant_1_value_tuple)

  # Derive the actual return tag and tuple types
  if is_tagged:
    # The first parameter is the tag type
    return_tag_type = tag_types.get_tag_type(Ty.get_arg(variant_1_fact_tuple, 0))

    # The second parameter has to be a tuple type
    return_tuple_type = _extract_return_tuple_type(maybe_variant_1_value_nested_tuple)
  else:
    # There is no tag type
    return_tag_type = tag_types.get_tag_type(None)

    # All parameters are tuple types
    # NOTE: This is a hack, the type would not check
    if Ty.is_tuple(variant_1_tag):
      return_tuple_type = _extract_return_tuple_type(variant_1_tag)
    else:
      return_tuple_type = _extract_return_tuple_type((variant_1_tag, *variant_2_value_tuple))

  return (return_tag_type, return_tuple_type)


def _extract_batched_ret_type_variants(ret_anno):
  # Check if the typed annotation is `BatchedFacts`
  assert Ty.is_union(ret_anno) and Ty.num_args(ret_anno) == 2, f"{ERROR_HEADER} Return type must be `scallopy.BatchedFacts`"

  # Check the first union variant; should be tagged batched facts
  variant_1 = Ty.get_arg(ret_anno, 0)
  assert Ty.is_list(variant_1) and Ty.is_list(Ty.child(variant_1)), f"{ERROR_HEADER} Return type must be `scallopy.BatchedFacts`: the first variant should be a 2-dimensional List, but is found to be `{variant_1}`"
  assert Ty.is_tuple(Ty.child(Ty.child(variant_1))), f"{ERROR_HEADER} Return type must be `scallopy.BatchedFacts`: the first variant should be a 2-dimensional List of tuples, but is found to be `{variant_1}`"
  variant_1_fact_tuple = Ty.child(Ty.child(variant_1))

  # Check the second union variant; should be untagged batched facts
  variant_2 = Ty.get_arg(ret_anno, 1)
  assert Ty.is_list(variant_2) and Ty.is_list(Ty.child(variant_2)), f"{ERROR_HEADER} Return type must be `scallopy.BatchedFacts`: the second variant should be a 2-dimensional List, but is found to be `{variant_2}`"
  assert Ty.is_tuple(Ty.child(Ty.child(variant_2))), f"{ERROR_HEADER} Return type must be `scallopy.BatchedFacts`: the second variant should be a 2-dimensional List of tuples, but is found to be `{variant_2}`"
  variant_2_value_tuple = Ty.args(Ty.child(Ty.child(variant_2)))

  return (variant_1_fact_tuple, variant_2_value_tuple)


def _extract_non_batched_ret_type_variants(ret_anno):
  # Check if the typed annotation is `Facts`
  assert Ty.is_union(ret_anno) and Ty.num_args(ret_anno) == 2, f"{ERROR_HEADER} Return type must be `scallopy.Facts`"

  # Check the first union variant; should be tagged facts
  variant_1 = Ty.get_arg(ret_anno, 0)
  assert Ty.is_generator(variant_1), f"{ERROR_HEADER} Return type must be `scallopy.Facts`: the first variant should be a Generator, but is found to be `{variant_1}`"
  assert Ty.is_tuple(Ty.get_arg(variant_1, 0)), f"{ERROR_HEADER} Return type must be `scallopy.Facts`: the first variant should be generator of tuple, but is found to be `{variant_1}`"
  variant_1_fact_tuple = Ty.get_arg(variant_1, 0)

  # Check the second union variant; should be untagged facts
  variant_2 = Ty.get_arg(ret_anno, 1)
  assert Ty.is_generator(variant_2), f"{ERROR_HEADER} Return type must be `scallopy.Facts`: the second variant should be a Generator, but is found to be `{variant_2}`"
  assert Ty.is_tuple(Ty.child(variant_2)), f"{ERROR_HEADER} Return type must be `scallopy.Facts`: the second variant should be generator of tuple, but is found to be `{variant_2}`"
  variant_2_value_tuple = Ty.args(Ty.child(variant_2))

  return (variant_1_fact_tuple, variant_2_value_tuple)


def _extract_ret_type_variant_1_tag_and_value_tuple(variant_1_fact_tuple):
  assert Ty.num_args(variant_1_fact_tuple) == 2, f"{ERROR_HEADER} Return type must be `scallopy.Facts`: the first variant should be generator of 2-tuple"
  variant_1_tag = Ty.get_arg(variant_1_fact_tuple, 0)
  variant_1_value_tuple = Ty.get_arg(variant_1_fact_tuple, 1)
  assert Ty.is_tuple(variant_1_value_tuple), f"{ERROR_HEADER} Return type must be `scallopy.Facts`: the first variant should be generator of tuple where the second argument is a tuple"
  return (variant_1_tag, variant_1_value_tuple)


def _extract_ret_type_is_tagged(variant_1_value_tuple):
  if Ty.num_args(variant_1_value_tuple) == 0:
    maybe_variant_1_value_nested_tuple = ()
    is_tagged = False
  else:
    maybe_variant_1_value_nested_tuple = Ty.child(variant_1_value_tuple)
    is_tagged = Ty.is_tuple(maybe_variant_1_value_nested_tuple) # If the second argument is a nested tuple, it means that we have a tagged fact
  return (maybe_variant_1_value_nested_tuple, is_tagged)

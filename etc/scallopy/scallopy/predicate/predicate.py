from typing import Callable, List, Optional

from .. import tag_types

from .types import Type

class ForeignPredicate:
  """
  Scallop foreign predicate class
  """
  def __init__(
    self,
    func: Callable,
    name: str,
    type_params: List[Type],
    input_arg_types: List[Type],
    output_arg_types: List[Type],
    tag_type: str,
    batched: bool = False,
    batch_size: Optional[int] = None,
    suppress_warning: bool = False,
  ):
    self.func = func
    self.name = name
    self.type_params = type_params
    self.input_arg_types = input_arg_types
    self.output_arg_types = output_arg_types
    self.tag_type = tag_type
    self.batched = batched
    self.batch_size = batch_size
    self.suppress_warning = suppress_warning

  def __repr__(self):
    r = f"extern pred {self.name}"
    if len(self.type_params) > 0:
      r += "<"
      for (i, type_param) in enumerate(self.type_params):
        if i > 0:
          r += ", "
        r += f"{type_param}"
      r += ">"
    r += f"[{self.pattern()}]("
    first = True
    for arg in self.input_arg_types:
      if first:
        first = False
      else:
        r += ", "
      r += f"{arg}"
    for arg in self.output_arg_types:
      if first:
        first = False
      else:
        r += ", "
      r += f"{arg}"
    r += ")"
    return r

  def __call__(self, *args):
    # Check if the foreign predicate is batched
    if not self.batched:
      # if not, we assume that the function will generate (yield) a bunch of facts
      if self.does_output_tag():
        return [f for f in self.func(*args)]
      else:
        return [(None, f) for f in self.func(*args)]
    else:
      if self.does_output_tag():
        # if batched, we directly ask the predicate to do everything by returning
        # a 2-dimensional list, where each element is a tuple of tag and tuple
        return self.func(*args)
      else:
        # If there is no output tag, we will help the user to pad the missing `None` tag
        result = self.func(*args)
        for row in result:
          for i in range(len(row)):
            row[i] = (None, row[i])
        return result

  def arity(self):
    return len(self.input_arg_types) + len(self.output_arg_types)

  def num_bounded(self):
    return len(self.input_arg_types)

  def all_argument_types(self):
    return self.input_arg_types + self.output_arg_types

  def pattern(self):
    return "b" * len(self.input_arg_types) + "f" * len(self.output_arg_types)

  def does_output_tag(self):
    return self.tag_type is not tag_types.none

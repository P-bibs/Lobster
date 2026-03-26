import collections
import typing
from typing import List

from .types import Type

def _extract_return_tuple_type(tuple_type) -> List[Type]:
  # First check if it is a None type (i.e. returning zero-tuple)
  if tuple_type == None:
    return []

  # Then try to convert it to a base type
  try:
    ty = Type(tuple_type)
    return [ty]
  except: pass

  # If not, it must be a tuple of base types
  if isinstance(tuple_type, tuple):
    return [Type(t) for t in tuple_type]
  elif hasattr(tuple_type, '__dict__') and "__origin__" in tuple_type.__dict__ and tuple_type.__dict__["__origin__"] == tuple:
    if "__args__" in tuple_type.__dict__:
      return [Type(t) for t in tuple_type.__dict__["__args__"]]
    else:
      return []
  else:
    raise Exception(f"Return tuple type must be a base type or a tuple of base types")


class TyUtils:
  def is_union(ty) -> bool:
    return ty.__origin__ == typing.Union

  def is_list(ty) -> bool:
    return ty.__origin__ == list

  def is_generator(ty) -> bool:
    return ty.__origin__ == collections.abc.Generator

  def is_tuple(ty) -> bool:
    if isinstance(ty, tuple): return True
    elif hasattr(ty, '__dict__') and "__origin__" in ty.__dict__ and ty.__origin__ == tuple: return True
    else: return False

  def args(ty) -> List:
    return list(ty.__args__)

  def num_args(ty) -> int:
    if isinstance(ty, tuple): return len(ty)
    elif hasattr(ty, '__args__'): return len(ty.__args__)
    else: raise Exception(f"Cannot parse type {ty} into arguments")

  def has_num_args(ty, num: int) -> bool:
    return len(ty.__args__) == num

  def get_arg(ty, id: int):
    return ty.__args__[id]

  def child(ty):
    return TyUtils.get_arg(ty, 0)

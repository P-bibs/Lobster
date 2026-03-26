from typing import Tuple

from .value_types import Tensor
from .predicate import foreign_predicate, Facts

@foreign_predicate(name="soft_eq", type_params=[Tensor])
def soft_eq_tensor(x: Tensor, y: Tensor) -> Facts[Tensor, Tuple]:
  import torch
  cs = torch.cosine_similarity(x, y, dim=0)
  prob = cs + 1.0 / 2.0
  yield (prob, ())

@foreign_predicate(name="soft_neq", type_params=[Tensor])
def soft_neq_tensor(x: Tensor, y: Tensor) -> Facts[Tensor, ()]:
  import torch
  cs = torch.cosine_similarity(x, y, dim=0)
  prob = 1.0 - (cs + 1.0 / 2.0)
  yield (prob, ())

STDLIB = {
  "functions": [
    # TODO
  ],
  "predicates": [
    soft_eq_tensor,
    soft_neq_tensor,
  ],
  "attributes": [
    # TODO
  ],
}

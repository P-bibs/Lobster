from typing import Tuple

import scallopy

@scallopy.foreign_predicate(batched=True)
def my_plus(inputs: scallopy.BatchedTuples[int, int]) -> scallopy.BatchedFacts[None, Tuple[int]]:
  return [[(i + j,)] for (i, j) in inputs]

ctx = scallopy.Context()
ctx.register_foreign_predicate(my_plus)
ctx.add_program("""
  rel pair = {(0, 5), (10, 15)}
  rel result(i, j, k) = pair(i, j) and my_plus(i, j, k)
""")
ctx.run()
print(list(ctx.relation("result")))

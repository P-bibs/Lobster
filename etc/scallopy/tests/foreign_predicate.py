from typing import *
import unittest

import torch
import scallopy

@scallopy.foreign_predicate
def my_range(a: int, b: int) -> scallopy.Facts[int]:
  for i in range(a, b):
    yield (i,)


@scallopy.foreign_predicate
def my_dummy_semantic_parser(s: str) -> scallopy.Facts[scallopy.Entity]:
  if s == "If I have 3 apples and 2 pears, how many fruits do I have?":
    yield ("Add(Const(3), Const(2))",)


@scallopy.foreign_predicate(suppress_warning=True)
def dummy(s: str) -> scallopy.Facts[str]:
  raise Exception("always false")


@scallopy.foreign_predicate
def my_eq(i1: int, i2: int) -> scallopy.Facts[scallopy.Tensor, ()]:
  if i1 == i2:
    yield (torch.tensor(1.0), ())
  elif abs(i1 - i2) <= 1:
    yield (torch.tensor(0.5), ())


@scallopy.foreign_predicate
def my_gpt(s1: str) -> scallopy.Facts[float, (bool,)]:
  if s1 == "toss a coin and it faces up":
    yield (0.5, (True,))
    yield (0.5, (False,))
  elif s1 == "1 + 1 = 2?":
    yield (1.0, (True,))


@scallopy.foreign_predicate
def my_check(i1: int, i2: int) -> scallopy.Facts[Tuple]:
  if i1 == i2: yield ()


@scallopy.foreign_predicate(batched=True, batch_size=3)
def batched_similarity(
  num_pairs: scallopy.BatchedTuples[float, float]
) -> scallopy.BatchedFacts[float, ()]:
  x = torch.tensor([i for (i, j) in num_pairs])
  y = torch.tensor([j for (i, j) in num_pairs])
  sim = (-((x - y).abs())).exp()
  return [[(s, ())] for s in sim]


class TestForeignPredicate(unittest.TestCase):
  def test_foreign_predicate_range(self):
    ctx = scallopy.Context()
    ctx.register_foreign_predicate(my_range)
    ctx.add_rule("r(x) = my_range(1, 5, x)")
    ctx.run()
    self.assertEqual(list(ctx.relation("r")), [(1,), (2,), (3,), (4,)])

  def test_fp_entity(self):
    ctx = scallopy.Context()

    # Register the semantic parser
    ctx.register_foreign_predicate(my_dummy_semantic_parser)

    # Add a program
    ctx.add_program("""
      type Expr = Const(i32) | Add(Expr, Expr)
      rel eval(e, v)       = case e is Const(v)
      rel eval(e, v1 + v2) = case e is Add(e1, e2) and eval(e1, v1) and eval(e2, v2)
      rel prompt = {"If I have 3 apples and 2 pears, how many fruits do I have?"}
      rel result(v) = prompt(p) and my_dummy_semantic_parser(p, e) and eval(e, v)
    """)

    # Run the context
    ctx.run()

    # The result should be 5
    self.assertEqual(list(ctx.relation("result")), [(5,)])

  def test_fp_suppress_warning(self):
    ctx = scallopy.Context()
    ctx.register_foreign_predicate(dummy)
    ctx.add_program("""rel result(y) = dummy("hello", y)""")
    ctx.run()
    self.assertEqual(list(ctx.relation("result")), [])

  def test_prob_prompt(self):
    ctx = scallopy.Context(provenance="minmaxprob")
    ctx.register_foreign_predicate(my_gpt)
    ctx.add_program("""
      rel coin_face_up(b) = my_gpt("toss a coin and it faces up", b)
      rel one_plus_one_is_two(b) = my_gpt("1 + 1 = 2?", b)
    """)
    ctx.run()
    self.assertEqual(list(ctx.relation("coin_face_up")), [(0.5, (False,)), (0.5, (True,))])
    self.assertEqual(list(ctx.relation("one_plus_one_is_two")), [(1.0, (True,))])

  def test_diff_eq(self):
    ctx = scallopy.Context(provenance="diffminmaxprob")
    ctx.register_foreign_predicate(my_eq)
    ctx.add_program("""
      rel pair = {(0, 0), (1, 0), (3, 0)}
      rel result(i, j) = pair(i, j) and my_eq(i, j)
    """)
    ctx.run()
    self.assertEqual(list(ctx.relation("result")), [(1.0, (0, 0)), (0.5, (1, 0))])


class TestBatchedForeignPredicate(unittest.TestCase):
  def test_batched_sim(self):
    ctx = scallopy.Context(provenance="minmaxprob")
    ctx.register_foreign_predicate(batched_similarity)
    ctx.add_program("""
      rel pair = {(0.0, -2.0), (0.0, 0.0), (1.0, 0.0)}
      rel result(i, j) = pair(i, j) and batched_similarity(i, j)
    """)
    ctx.run()
    result = {(i, j): p for (p, (i, j)) in ctx.relation("result")}
    answer = {(0.0, -2.0): 0.135335, (0.0, 0.0): 1.0, (1.0, 0.0): 0.367879}
    for (key, val) in answer.items():
      self.assertAlmostEqual(result[key], val, places=3)


if __name__ == '__main__':
  unittest.main()

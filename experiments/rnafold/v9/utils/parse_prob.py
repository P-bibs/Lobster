from typing import List, Generator, Tuple, Optional
import torch

HelixLeft = 0
HelixRight = 1
LoopLeft = 2
LoopRight = 3
LoopUnpaired = 4
ExtUnpaired = 5

class SubStructure: pass


class SubStructure:
  def __init__(self, seq: List[int], prob: float):
    self.seq = seq
    self.prob = prob

  def join(self, other: SubStructure) -> SubStructure:
    seq = self.seq + other.seq
    prob = self.prob * other.prob
    return SubStructure(seq, prob)

  def token_to_dot_bracket(self, t: int) -> str:
    if t == 0: return "("
    elif t == 1: return ")"
    elif t == 2: return "("
    elif t == 3: return ")"
    elif t == 4: return "."
    elif t == 5: return "."
    else: raise Exception(f"Unknown token {t}")

  def __repr__(self) -> str:
    dot_bracket = "".join([self.token_to_dot_bracket(t) for t in self.seq])
    return f"{self.prob}::\"{dot_bracket}\""


class Cache:
  def __init__(self):
    self._cache = {}

  def insert(self, i: int, j: int, substr: SubStructure) -> bool:
    if i in self._cache:
      if j in self._cache[i]:
        existing_substr = self._cache[i][j]
        if existing_substr.prob > substr.prob - 0.001:
          return False
    else:
      self._cache[i] = {}

    # Need to update
    self._cache[i][j] = substr
    return True

  def __iter__(self) -> Generator[Tuple[Tuple[int, int], SubStructure], None, None]:
    for (i, row) in self._cache.items():
      for (j, substr) in row.items():
        yield ((i, j), substr)

  def find_row(self, i) -> Generator[Tuple[Tuple[int, int], SubStructure], None, None]:
    if i in self._cache:
      for (j, substr) in self._cache[i].items():
        yield ((i, j), substr)

  def __getitem__(self, key: Tuple[int, int]) -> Optional[SubStructure]:
    (i, j) = key
    if i in self._cache:
      if j in self._cache[i]:
        return self._cache[i][j]
    return None


class ParsingContext:
  def __init__(self, prob_seq: torch.Tensor):
    self._prob_seq = prob_seq
    self._length = prob_seq.shape[0]

    self._paired_substring = Cache()
    self._ext_unpaired = Cache()
    self._int_unpaired = Cache()
    self._int_loop = Cache()
    self._loop = Cache()
    self._helix_stack = Cache()
    self._rna_ss = Cache()

  def __len__(self) -> int:
    return self._length

  def parse(self):
    # Base cases
    #   \text{(Ext. Unpaired)} & \mathcal{E} & ::= & E_u
    #   \text{(Int. Unpaired)} & \mathcal{U} & ::= & L_u
    for i in range(self._length):
      self._ext_unpaired.insert(i, i, SubStructure([ExtUnpaired], self._prob_seq[i, ExtUnpaired]))
      self._int_unpaired.insert(i, i, SubStructure([LoopUnpaired], self._prob_seq[i, LoopUnpaired]))

    # Enter iterative loop
    iterator_count = 0
    has_update = True
    while has_update:
      print(f"Iteration {iterator_count}")
      has_update = False

      # \text{(Paired Substr.)} & \mathcal{P} & ::= & \mathcal{H}
      for ((i, j), h) in self._helix_stack:
        has_update = self._paired_substring.insert(i, j, h) or has_update

      # \text{(Paired Substr.)} & \mathcal{P} & ::= & \mathcal{L}
      for ((i, j), l) in self._loop:
        has_update = self._paired_substring.insert(i, j, l) or has_update

      # \text{(Ext. Unpaired)} & \mathcal{E} & ::= & \mathcal{E} \cdot E_u
      for ((i, j), e) in list(self._ext_unpaired):
        if j >= len(self) - 1: continue
        has_update = self._ext_unpaired.insert(i, j + 1, SubStructure(e.seq + [ExtUnpaired], e.prob * self._prob_seq[j + 1, ExtUnpaired])) or has_update
        # print("3", has_update, (i, j), e)

      # \text{(Int. Unpaired)} & \mathcal{U} & ::= & \mathcal{U} \cdot L_u
      for ((i, j), u) in list(self._int_unpaired):
        if j >= len(self) - 1: continue
        has_update = self._int_unpaired.insert(i, j + 1, SubStructure(u.seq + [LoopUnpaired], u.prob * self._prob_seq[j + 1, LoopUnpaired])) or has_update

      # \text{(Int. Loop)} & \mathcal{I} & ::= & \mathcal{U}
      for ((i, j), u) in self._int_unpaired:
        has_update = self._int_loop.insert(i, j, u) or has_update

      # \text{(Int. Loop)} & \mathcal{I} & ::= & \mathcal{P} \cdot \mathcal{U}
      for ((i, j), p) in self._paired_substring:
        for ((k, l), u) in self._int_unpaired.find_row(j + 1):
          assert k == j + 1
          has_update = self._int_loop.insert(i, l, p.join(u)) or has_update

      # \text{(Int. Loop)} & \mathcal{I} & ::= & \mathcal{I} \cdot \mathcal{P}
      for ((i, j), l1) in list(self._int_loop):
        for ((k, l), l2) in self._paired_substring.find_row(j + 1):
          assert k == j + 1
          has_update = self._int_loop.insert(i, l, l1.join(l2)) or has_update

      # \text{(Int. Loop)} & \mathcal{I} & ::= & \mathcal{I} \cdot \mathcal{U}
      for ((i, j), l1) in list(self._int_loop):
        for ((k, l), l2) in self._int_unpaired.find_row(j + 1):
          assert k == j + 1
          has_update = self._int_loop.insert(i, l, l1.join(l2)) or has_update

      # \text{(Loop)} & \mathcal{L} & ::= & L_l \cdot \mathcal{I} \cdot L_r
      for ((i, j), l1) in self._int_loop:
        if i == 0 or j == len(self) - 1: continue
        prev_prob = self._prob_seq[i - 1, LoopLeft]
        next_prob = self._prob_seq[j + 1, LoopRight]
        has_update = self._loop.insert(i - 1, j + 1, SubStructure([LoopLeft] + l1.seq + [LoopRight], l1.prob * prev_prob * next_prob)) or has_update

      # \text{(Helix Stack)} & \mathcal{H} & ::= & H_l \cdot \mathcal{P} \cdot H_r
      for ((i, j), l1) in self._paired_substring:
        if i == 0 or j >= len(self) - 1: continue
        prev_prob = self._prob_seq[i - 1, HelixLeft]
        next_prob = self._prob_seq[j + 1, HelixRight]
        has_update = self._helix_stack.insert(i - 1, j + 1, SubStructure([HelixLeft] + l1.seq + [HelixRight], l1.prob * prev_prob * next_prob)) or has_update

      # \text{(RNA SS)} & \mathcal{R} & ::= & \mathcal{P}
      for ((i, j), l) in self._paired_substring:
        has_update = self._rna_ss.insert(i, j, l) or has_update

      # \text{(RNA SS)} & \mathcal{R} & ::= & \mathcal{E}
      for ((i, j), l) in self._ext_unpaired:
        has_update = self._rna_ss.insert(i, j, l) or has_update

      # \text{(RNA SS)} & \mathcal{R} & ::= & \mathcal{R} \cdot \mathcal{R}
      for ((i, j), l1) in list(self._rna_ss):
        for ((k, l), l2) in list(self._rna_ss.find_row(j + 1)):
          assert k == j + 1
          has_update = self._rna_ss.insert(i, j, l1.join(l2)) or has_update

      iterator_count += 1

    parsed_seq = self._rna_ss[(0, len(self) - 1)]
    return parsed_seq

prob_tokens = torch.tensor([
  [0.886, 0.004, 0.027, 0.002, 0.051, 0.012,],
  [0.906, 0.003, 0.022, 0.001, 0.044, 0.009,],
  [0.890, 0.003, 0.036, 0.001, 0.044, 0.009,],
  [0.826, 0.003, 0.082, 0.001, 0.064, 0.010,],
  [0.905, 0.002, 0.034, 0.001, 0.043, 0.007,],
  [0.817, 0.003, 0.102, 0.001, 0.059, 0.008,],
  [0.091, 0.007, 0.383, 0.006, 0.486, 0.009,],
  [0.015, 0.003, 0.012, 0.002, 0.948, 0.003,],
  [0.193, 0.003, 0.112, 0.002, 0.667, 0.007,],
  [0.880, 0.002, 0.032, 0.001, 0.064, 0.007,],
  [0.877, 0.002, 0.040, 0.001, 0.061, 0.007,],
  [0.853, 0.002, 0.067, 0.001, 0.059, 0.008,],
  [0.550, 0.004, 0.334, 0.002, 0.094, 0.008,],
  [0.543, 0.003, 0.192, 0.002, 0.240, 0.007,],
  [0.023, 0.003, 0.013, 0.002, 0.947, 0.003,],
  [0.014, 0.003, 0.010, 0.002, 0.955, 0.003,],
  [0.011, 0.003, 0.004, 0.002, 0.961, 0.002,],
  [0.014, 0.003, 0.003, 0.002, 0.954, 0.003,],
  [0.023, 0.004, 0.007, 0.005, 0.947, 0.003,],
  [0.018, 0.006, 0.023, 0.006, 0.933, 0.003,],
  [0.006, 0.009, 0.003, 0.007, 0.957, 0.003,],
  [0.006, 0.017, 0.003, 0.019, 0.934, 0.004,],
  [0.010, 0.052, 0.006, 0.083, 0.821, 0.009,],
  [0.008, 0.169, 0.006, 0.203, 0.572, 0.013,],
  [0.009, 0.611, 0.012, 0.241, 0.084, 0.016,],
  [0.007, 0.756, 0.009, 0.133, 0.052, 0.016,],
  [0.004, 0.798, 0.005, 0.111, 0.050, 0.011,],
  [0.596, 0.008, 0.047, 0.003, 0.317, 0.010,],
  [0.910, 0.003, 0.024, 0.001, 0.044, 0.007,],
  [0.898, 0.003, 0.025, 0.001, 0.052, 0.007,],
  [0.892, 0.003, 0.027, 0.001, 0.052, 0.008,],
  [0.873, 0.004, 0.048, 0.002, 0.051, 0.009,],
  [0.381, 0.013, 0.488, 0.007, 0.084, 0.015,],
  [0.036, 0.026, 0.145, 0.019, 0.753, 0.007,],
  [0.007, 0.007, 0.003, 0.005, 0.961, 0.003,],
  [0.008, 0.007, 0.003, 0.006, 0.959, 0.003,],
  [0.008, 0.014, 0.003, 0.013, 0.942, 0.003,],
  [0.009, 0.020, 0.002, 0.012, 0.931, 0.004,],
  [0.011, 0.009, 0.004, 0.007, 0.955, 0.003,],
  [0.008, 0.014, 0.002, 0.013, 0.941, 0.003,],
  [0.012, 0.465, 0.018, 0.340, 0.120, 0.017,],
  [0.003, 0.855, 0.003, 0.072, 0.036, 0.011,],
  [0.003, 0.872, 0.004, 0.054, 0.031, 0.012,],
  [0.006, 0.848, 0.007, 0.056, 0.038, 0.017,],
  [0.014, 0.561, 0.013, 0.126, 0.249, 0.013,],
  [0.020, 0.012, 0.006, 0.005, 0.937, 0.004,],
  [0.029, 0.004, 0.009, 0.003, 0.944, 0.002,],
  [0.010, 0.004, 0.002, 0.003, 0.965, 0.002,],
  [0.011, 0.003, 0.003, 0.003, 0.966, 0.002,],
  [0.371, 0.004, 0.018, 0.002, 0.578, 0.005,],
  [0.890, 0.002, 0.025, 0.001, 0.058, 0.007,],
  [0.896, 0.003, 0.023, 0.001, 0.054, 0.008,],
  [0.878, 0.003, 0.032, 0.001, 0.063, 0.008,],
  [0.852, 0.002, 0.045, 0.001, 0.081, 0.007,],
  [0.120, 0.010, 0.644, 0.007, 0.200, 0.009,],
  [0.007, 0.007, 0.003, 0.004, 0.954, 0.003,],
  [0.008, 0.007, 0.003, 0.004, 0.955, 0.003,],
  [0.006, 0.021, 0.003, 0.019, 0.926, 0.004,],
  [0.008, 0.015, 0.003, 0.010, 0.946, 0.003,],
  [0.008, 0.010, 0.003, 0.006, 0.952, 0.004,],
  [0.007, 0.006, 0.002, 0.006, 0.959, 0.003,],
  [0.009, 0.043, 0.005, 0.080, 0.827, 0.008,],
  [0.006, 0.643, 0.008, 0.221, 0.079, 0.015,],
  [0.002, 0.871, 0.002, 0.052, 0.034, 0.012,],
  [0.004, 0.836, 0.003, 0.045, 0.034, 0.027,],
  [0.002, 0.874, 0.003, 0.052, 0.033, 0.012,],
  [0.004, 0.781, 0.005, 0.125, 0.052, 0.012,],
  [0.003, 0.808, 0.004, 0.103, 0.048, 0.012,],
  [0.003, 0.805, 0.004, 0.039, 0.036, 0.038,],
  [0.005, 0.709, 0.005, 0.038, 0.040, 0.068,],
  [0.006, 0.578, 0.005, 0.040, 0.048, 0.110,],
  [0.006, 0.713, 0.006, 0.049, 0.040, 0.073,],
  [0.007, 0.678, 0.007, 0.047, 0.040, 0.091,],
  [0.020, 0.122, 0.013, 0.031, 0.047, 0.283,],
  [0.028, 0.065, 0.015, 0.024, 0.050, 0.262,],
])

print(prob_tokens.shape)

ctx = ParsingContext(prob_tokens)
result = ctx.parse()
print(result)

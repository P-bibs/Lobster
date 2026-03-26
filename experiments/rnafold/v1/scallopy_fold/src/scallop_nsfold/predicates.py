from typing import Tuple, List
import numpy as np
import math

from scallopy import i32, usize, Facts, fp as foreign_predicate

from . import param_turner2004

complementary_pair_mapping = np.array([
  # _, A, C, G, U
  [ 0, 0, 0, 0, 0 ], # _
  [ 0, 0, 0, 0, 5 ], # A
  [ 0, 0, 0, 1, 0 ], # C
  [ 0, 0, 2, 0, 3 ], # G
  [ 0, 6, 0, 4, 0 ], # U
])

@foreign_predicate
def extern_score_hairpin(
  i: usize,
  x_i: i32,
  x_ip1: i32,
  x_jm1: i32,
  j: usize,
  x_j: i32,
) -> Facts[float, Tuple]:
  # https://github.com/mxfold/mxfold2/blob/master/mxfold2/src/param/turner.cpp#L158-L179
  #
  # i:     the index of the start of the hairpin
  # x_i:   x[i]
  # x_ip1: x[i + 1]
  # x_jm1: x[j - 1]
  # j:     the index of the end of the hairpin
  # x_j:   x[j]

  length = (j - 1) - (i + 1) + 1
  energy = 0.0

  # Length penalty
  if length <= 30:
    energy += param_turner2004.score_hairpin[length]
  else:
    energy += param_turner2004.score_hairpin[30] + param_turner2004.score_lxc[0] * math.log(length / 30.)

  # Conditioned on different lengths, we get more energies
  pair_id = complementary_pair_mapping[x_i, x_j]
  if length == 3:
    if pair_id > 2:
      energy += param_turner2004.score_terminalAU[0]

  elif length > 3:
    energy += param_turner2004.score_mismatch_external[pair_id][x_ip1][x_jm1]

  # Return the actual exponential value
  yield (math.exp(energy), ())


@foreign_predicate
def extern_score_single_loop(
  i: usize, x_i: i32, x_ip1: i32,
  x_jm1: i32, j: usize, x_j: i32,
  x_km1: i32, k: usize, x_k: i32,
  l: usize, x_l: i32, x_lp1: i32,
) -> Facts[float, Tuple]:
  # https://github.com/mxfold/mxfold2/blob/master/mxfold2/src/param/turner.cpp#L224-L283

  type1 = complementary_pair_mapping[x_i][x_j]
  type2 = complementary_pair_mapping[x_l][x_k]
  l1 = (k - 1) - (i + 1) + 1 # k - i - 1
  l2 = (j - 1) - (l + 1) + 1 # j - l - 1
  ls, ll = min(l1, l2), max(l1, l2)

  e = 0

  if ll == 0: # Stack
    return param_turner2004.score_stack[type1, type2]
  elif ls == 0: # Bulge
    if ll <= 30:
      e = param_turner2004.score_bulge[ll]
    else:
      e = param_turner2004.score_bulge[30] + param_turner2004.score_lxc[0] * math.log(ll / 30.)
    if ll == 1:
      e += param_turner2004.score_stack[type1, type2]
    else:
      if type1 > 2:
        e += param_turner2004.score_terminalAU[0]
      if type2 > 2:
        e += param_turner2004.score_terminalAU[0]
  else: # Internal
    if ll == 1 and ls == 1: # 1x1 loop
      e = param_turner2004.score_int11[type1, type2, x_ip1, x_jm1]
    elif l1 == 2 and l2 == 1: # 2x1 loop
      e = param_turner2004.score_int21[type2, type1, x_lp1, x_ip1, x_km1]
    elif l1 == 1 and l2 == 2: # 1x2 loop
      e = param_turner2004.score_int21[type1, type2, x_ip1, x_lp1, x_jm1]
    elif ls == 1: # 1xn loop
      if ll + 1 <= 30:
        e = param_turner2004.score_internal[ll + 1]
      else:
        e = param_turner2004.score_internal[30] + param_turner2004.score_lxc[0] * math.log((ll + 1)/ 30.)
      e += max(param_turner2004.score_max_ninio[0], (ll - ls) * param_turner2004.score_ninio[0])
      e += param_turner2004.score_mismatch_internal_1n[type1, x_ip1, x_jm1]
      e += param_turner2004.score_mismatch_internal_1n[type2, x_lp1, x_km1]
    elif ls == 2 and ll == 2: # 2x2 loop
      e = param_turner2004.score_int22[type1, type2, x_ip1, x_km1, x_lp1, x_jm1]
    elif ls == 2 and ll == 3: # 2x3 loop
      e = param_turner2004.score_internal[ls + ll] + param_turner2004.score_ninio[0]
      e += param_turner2004.score_mismatch_internal_23[type1, x_ip1, x_jm1]
      e += param_turner2004.score_mismatch_internal_23[type2, x_lp1, x_km1]
    else: # general internal loop
      if ls + ll <= 30:
        e = param_turner2004.score_internal[ls + ll]
      else:
        e = param_turner2004.score_internal[30] + param_turner2004.score_lxc[0] * math.log((ls + ll) / 30)
      e += max(param_turner2004.score_max_ninio[0], (ll - ls) * param_turner2004.score_ninio[0])
      e += param_turner2004.score_mismatch_internal[type1, x_ip1, x_jm1]
      e += param_turner2004.score_mismatch_internal[type2, x_lp1, x_km1]

  yield (math.exp(e), ())


@foreign_predicate
def extern_score_multi_loop(
  x_i: i32, x_ip1: i32, x_jm1: i32, x_j: i32,
) -> Facts[float, Tuple]:
  # https://github.com/mxfold/mxfold2/blob/master/mxfold2/src/param/turner.cpp#L452-L465
  e = 0.0
  ty = complementary_pair_mapping[x_j][x_i]
  e += param_turner2004.score_mismatch_multi[ty, x_jm1, x_ip1]
  if ty > 2:
    e += param_turner2004.score_terminalAU[0]
  e += param_turner2004.score_ml_intern[0]
  e += param_turner2004.score_ml_closing[0]
  yield (math.exp(e), ())


@foreign_predicate
def extern_score_multi_paired(
  x_im1: i32, i: usize, x_i: i32,
  j: usize, x_j: i32, x_jp1: i32,
  total_len: usize,
) -> Facts[float, Tuple]:
  # https://github.com/mxfold/mxfold2/blob/master/mxfold2/src/param/turner.cpp#L479-L497
  l = total_len - 2
  e = 0.0
  ty = complementary_pair_mapping[x_i][x_j]
  if i - 1 >= 1 and j + 1 <= l:
    e += param_turner2004.score_mismatch_multi[ty, x_im1, x_jp1]
  elif i - 1 >= 1:
    e += param_turner2004.score_dangle5[ty, x_im1]
  elif j + 1 <= l:
    e += param_turner2004.score_dangle3[ty, x_jp1]
  if ty > 2:
    e += param_turner2004.score_terminalAU[0]
  e += param_turner2004.score_ml_intern[0]
  yield (math.exp(e), ())


@foreign_predicate
def extern_score_multi_unpaired(
  i: usize, j: usize,
) -> Facts[float, Tuple]:
  # https://github.com/mxfold/mxfold2/blob/master/mxfold2/src/param/turner.cpp#L516-L521
  e = param_turner2004.score_ml_base[0] * (j - i + 1)
  yield (math.exp(e), ())

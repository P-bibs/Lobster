import numpy as np

import param_turner2004 as params

# Defining constants for A, C, G, U
A = 1
C = 2
G = 3
U = 4

NEUCLEOTIDES_MAP = {"A": A, "C": C, "G": G, "U": U}
NEUCLEOTIDES = {A, C, G, U}

# Defining constants for Complementary pairs
CG = 1
GC = 2
GU = 3
UG = 4
AU = 5
UA = 6

PAIR_TYPES = {CG, GC, GU, UG, AU, UA}
PAIR_TYPES_MAP = {"CG": CG, "GC": GC, "GU": GU, "UG": UG, "AU": AU, "UA": UA}

def dump_type_decl():
  neucleotides = [f"{name}={val}" for (name, val) in sorted([item for item in NEUCLEOTIDES_MAP.items()], key=lambda x: x[1])]
  neucleotides_ty = f"type Neucleotide = {' | '.join(neucleotides)}"

  pair_types = [f"{name}={val}" for (name, val) in sorted([item for item in PAIR_TYPES_MAP.items()], key=lambda x: x[1])]
  pair_types_ty = f"type PairType = {' | '.join(pair_types)}"

  return neucleotides_ty + "\n" + pair_types_ty

def is_plausible_value(dim_ty, val):
  if dim_ty == "Neucleotide":
    return val in NEUCLEOTIDES
  elif dim_ty == "PairType":
    return val in PAIR_TYPES
  elif dim_ty == "f32" or dim_ty == "i32":
    return True
  else:
    raise Exception(f"Unknown dimension type `{dim_ty}`")

def generate_facts(dims, prev_values, np_array):
  if len(dims) > 0:
    dim_ty = dims[0]
    facts = []
    for (i, row) in enumerate(np_array):
      if not is_plausible_value(dim_ty, i):
        continue
      next_tuple = tuple([*prev_values, i])
      next_dims = dims[1:]
      facts += generate_facts(next_dims, next_tuple, row)
    return facts
  else:
    if np.isinf(np_array):
      return []
    else:
      prob = -np_array
      tup = ",".join([str(v) for v in prev_values])
      fact = f"{prob}::({tup})"
      return [fact]

def dump_facts_of_relation(scores, name, arg_names, dims):
  ty_args = [f"{arg_name}: {ty}" for (arg_name, ty) in zip(arg_names, dims)]
  ty = f"type {name}({', '.join(ty_args)})"
  facts = ",\n  ".join(generate_facts(dims, (), scores))
  rel = f"rel {name} = {{\n  {facts}\n}}"
  return f"{ty}\n{rel}"

def dump_score_stack():
  return dump_facts_of_relation(params.score_stack, "score_stack", ["i", "j"], ["PairType", "PairType"])

def dump_score_mismatch(scores, name):
  return dump_facts_of_relation(scores, name, ["i", "j", "k"], ["PairType", "PairType", "PairType"])

def dump_score_mismatch_hairpin():
  return dump_score_mismatch(params.score_mismatch_hairpin, "score_mismatch_hairpin")

def dump_score_mismatch_internal():
  return dump_score_mismatch(params.score_mismatch_internal, "score_mismatch_internal")

def dump_score_mismatch_internal_1n():
  return dump_score_mismatch(params.score_mismatch_internal_1n, "score_mismatch_internal_1n")

def dump_score_mismatch_internal_23():
  return dump_score_mismatch(params.score_mismatch_internal_23, "score_mismatch_internal_23")

def dump_score_mismatch_multi():
  return dump_score_mismatch(params.score_mismatch_multi, "score_mismatch_multi")

def dump_score_mismatch_external():
  return dump_score_mismatch(params.score_mismatch_external, "score_mismatch_external")

def dump_score_dangle(scores, name):
  return dump_facts_of_relation(scores, name, ["t", "i"], ["PairType", "Neucleotide"])

def dump_score_dangle5():
  return dump_score_dangle(params.score_dangle5, "score_dangle5")

def dump_score_dangle3():
  return dump_score_dangle(params.score_dangle3, "score_dangle3")

def dump_score_int11():
  return dump_facts_of_relation(params.score_int11, "score_int11", ["t1", "t2", "i", "j"], ["PairType", "PairType", "Neucleotide", "Neucleotide"])

def dump_score_int21():
  return dump_facts_of_relation(params.score_int21, "score_int21", ["t1", "t2", "i", "j", "k"], ["PairType", "PairType", "Neucleotide", "Neucleotide", "Neucleotide"])

def dump_score_int22():
  return dump_facts_of_relation(params.score_int22, "score_int22", ["t1", "t2", "i", "j", "k", "l"], ["PairType", "PairType", "Neucleotide", "Neucleotide", "Neucleotide", "Neucleotide"])

def dump_score_hairpin():
  return dump_facts_of_relation(params.score_hairpin, "score_hairpin", ["len"], ["i32"])

def dump_score_bulge():
  return dump_facts_of_relation(params.score_bulge, "score_bulge", ["len"], ["i32"])

def dump_score_internal():
  return dump_facts_of_relation(params.score_internal, "score_internal", ["len"], ["i32"])

def dump_singleton(score, name):
  return dump_facts_of_relation(score[0], name, [], [])

def dump_score_ml_base():
  return dump_singleton(params.score_ml_base, "score_ml_base")

def dump_score_ml_closing():
  return dump_singleton(params.score_ml_closing, "score_ml_closing")

def dump_score_ml_intern():
  return dump_singleton(params.score_ml_intern, "score_ml_intern")

def dump_score_ninio():
  return dump_singleton(params.score_ninio, "score_ninio")

def dump_score_max_ninio():
  return dump_singleton(params.score_max_ninio, "score_max_ninio")

def dump_score_duplex_init():
  return dump_singleton(params.score_duplex_init, "score_duplex_init")

def dump_score_terminalAU():
  return dump_singleton(params.score_terminalAU, "score_terminalAU")

def dump_score_lxc():
  return dump_singleton(params.score_lxc, "score_lxc")

ALL_SCORE_DUMPERS = [
  dump_type_decl,
  dump_score_stack,
  dump_score_mismatch_hairpin,
  dump_score_mismatch_internal,
  dump_score_mismatch_internal_1n,
  dump_score_mismatch_internal_23,
  dump_score_mismatch_multi,
  dump_score_mismatch_external,
  dump_score_dangle5,
  dump_score_dangle3,
  dump_score_int11,
  dump_score_int21,
  dump_score_int22,
  dump_score_hairpin,
  dump_score_bulge,
  dump_score_internal,
  dump_score_ml_base,
  dump_score_ml_closing,
  dump_score_ml_intern,
  dump_score_ninio,
  dump_score_max_ninio,
  dump_score_duplex_init,
  dump_score_terminalAU,
  dump_score_lxc,
]

def dump_param_scl_file():
  params = []
  for score_dumper in ALL_SCORE_DUMPERS:
    params.append(score_dumper())
  file_content = "\n\n".join(params)
  print(file_content)

dump_param_scl_file()

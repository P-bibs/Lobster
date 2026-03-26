import math
from typing import Tuple

from scallopy import Context as ScallopContext, char, usize, Facts, fp as foreign_predicate

from . import param_turner2004
from . import predicates

def load_into_context(ctx: ScallopContext):
  ctx.register_foreign_predicate(predicates.extern_score_hairpin)
  ctx.register_foreign_predicate(predicates.extern_score_single_loop)
  ctx.register_foreign_predicate(predicates.extern_score_multi_loop)
  ctx.register_foreign_predicate(predicates.extern_score_multi_paired)
  ctx.register_foreign_predicate(predicates.extern_score_multi_unpaired)

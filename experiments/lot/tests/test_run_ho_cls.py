import os

import torch
import scallopy

#                                                // | rela | sub  obj |
# "context": "
#   A chestnut is not a flower.                  // |    3 |   0    6 |
#   A chestnut is a fruit.                       // |    2 |   1    7 |
#   A chestnut does not have a brow.             // |    5 |   2    8 |
#   A flower is part of angiosperm.              // |    0 |   3    9 |
#   A fruit is not part of angiosperm.           // |    1 |   4   10 |
#   A chestnut is a woody plant.                 // |    2 |   5   11 |
# ",
# "phrase": "A chestnut is part of angiosperm.", // |    0 |  12   13 |
# "answer": 0

seed = 1357

num_context = 6
num_sentences = num_context + 1
num_entities = num_sentences * 2

torch.manual_seed(seed)

context = [(torch.rand((), requires_grad=True), (k, i, num_context + i, True)) for i in range(num_context) for k in range(10)]
question = [(torch.rand((), requires_grad=True), (k, num_context * 2, num_context * 2 + 1)) for k in range(10)]
synonym = [(torch.rand((), requires_grad=True), (i, j)) for i in range(num_entities) for j in range(i + 1, num_entities)]

ctx = scallopy.ScallopContext(provenance="difftopbottomkclauses", k=5)
ctx.import_file(os.path.abspath(os.path.join(os.path.abspath(__file__), "../../scl/lot_ho_cls.scl")))

print("context", context)
ctx.add_facts("context", context)
print("question", question)
ctx.add_facts("question", question)
print("synonym", synonym)
ctx.add_facts("synonym", synonym)

ctx.run(iter_limit=5)

print(list(ctx.relation("answer")))

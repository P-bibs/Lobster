import os
import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser

import scallopy

class RandNet(nn.Module):
  def __init__(self, SIZE):
    super(RandNet, self).__init__()
    self.fc1 = nn.Linear(1024, 1024)
    self.fc2 = nn.Linear(1024, SIZE)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.softmax(x, dim=1)

DATA = set([
(0, 0, 1),
(0, 1, 2),
(0, 2, 3),
(0, 3, 4),
(0, 4, 5),
#(0, 5, 6),
#(0, 6, 7),
#(0, 7, 8),
#(0, 8, 9),
#(0, 9, 10),
#(0, 10, 11),
#(0, 11, 12),
#(0, 12, 13),
#(0, 13, 14),
#(0, 14, 15),
#(0, 15, 16),
#(1, 0, 1),
#(1, 1, 2),
#(1, 2, 3),
#(1, 3, 4),
#(1, 4, 5),
#(1, 5, 6),
#(1, 6, 7),
#(1, 7, 8),
#(1, 8, 9),
#(1, 9, 10),
#(1, 10, 11),
#(1, 11, 12),
#(1, 12, 13),
#(1, 13, 14),
#(1, 14, 15),
#(1, 15, 16),
#(2, 0, 1),
#(2, 1, 2),
#(2, 2, 3),
#(2, 3, 4),
#(2, 4, 5),
#(2, 5, 6),
#(2, 6, 7),
#(2, 7, 8),
#(2, 8, 9),
#(2, 9, 10),
#(2, 10, 11),
#(2, 11, 12),
#(2, 12, 13),
#(2, 13, 14),
#(2, 14, 15),
#(2, 15, 16),
#(3, 0, 1),
#(3, 1, 2),
#(3, 2, 3),
#(3, 3, 4),
#(3, 4, 5),
#(3, 5, 6),
#(3, 6, 7),
#(3, 7, 8),
#(3, 8, 9),
#(3, 9, 10),
#(3, 10, 11),
#(3, 11, 12),
#(3, 12, 13),
#(3, 13, 14),
#(3, 14, 15),
#(3, 15, 16)
])

SIZE = 16
in_mapping = [(i, j, k) for i in range(SIZE) for j in range(SIZE) for k in range(SIZE)]
out_mapping = [(i, j, k, l) for i in range(SIZE) for j in range(SIZE) for k in range(SIZE) for l in range(SIZE)]

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()

    self.model_a = RandNet(SIZE ** 3)

    # Scallop Context
    self.scl = scallopy.Module(
      program="""
        type single_match(t0: u32, v0: u32, v1: u32)
        type sub_match(t0: u32, t1: u32, v0: u32, v1: u32)

        rel sub_match(t0, t0, v0, v1) = single_match(t0, v0, v1)
        rel sub_match(t0, t0, v0, v2) = sub_match(t0, t0, v0, v1), single_match(t0, v1, v2)
        rel sub_match(t0, t1, v0, v2) = sub_match(t0, t1 - 1, v0, v1), single_match(t1, v1, v2)

      """,
      provenance="diffminmaxprob",
      k=1,
      input_mappings={
        "single_match": in_mapping
        },
      output_mappings={"sub_match": out_mapping},
      retain_graph=True,dispatch='single')

  def forward(self, x):
    a = self.model_a(x)

    x = torch.tensor([[0.5 if t in DATA else 0 for t in in_mapping]])
    print(x)

    return self.scl(single_match=x)
def main():
  os.environ["STRATUM"] = "1"
  model = Model()

  x = torch.rand(1, 1024)
  y = model(x)

  #[print(l) for l in zip(out_mapping, y[0].tolist()) if l[1] != 0]

main()


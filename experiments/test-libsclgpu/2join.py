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
  def __init__(self, size):
    super(RandNet, self).__init__()
    self.fc1 = nn.Linear(1024, 1024)
    self.fc2 = nn.Linear(1024, size)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.softmax(x, dim=1)


class Model(nn.Module):
  def __init__(self, size):
    super(Model, self).__init__()
    self.size = size

    self.model_a = RandNet(size ** 3)
    self.model_b = RandNet(size ** 3)

    # Scallop Context
    self.scl = scallopy.Module(
      program="""
        type digit1(x: u32, y: u32, s: u32)
        type digit2(x: u32, y: u32, s: u32)
        type sum(x: u32)

        rel sum(s) = digit1(x, y, _), digit2(x, y, s)
      """,
      #provenance="diffminmaxprob",
      provenance="diffaddmultprob",
      #provenance="difftopkproofs",
      k=1,
      input_mappings={
        "digit1": [(i, j, s) for i in range(size) for j in range(size) for s in range(size)],
        "digit2": [(i, j, s) for i in range(size) for j in range(size) for s in range(size)],
        },
      output_mappings={"sum": list(range(size))},
      retain_graph=True,dispatch='single')

  def forward(self, x):
    a = self.model_a(x)
    b = self.model_b(x)
    
    return self.scl(digit1=a, digit2=b)

def main():
  os.environ["STRATUM"] = "2"
  model = Model(3)

  x = torch.rand(1, 1024)
  y = model(x)
  print(y)


main()


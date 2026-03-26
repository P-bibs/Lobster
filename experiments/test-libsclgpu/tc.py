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

    self.model_a = RandNet(size ** 2)

    mapping = [(i, j) for i in range(size) for j in range(size)]
    # Scallop Context
    self.scl = scallopy.Module(
      program="""
        type edge(x: u32, y: u32)
        type path(x: u32, y: u32)

        rel path(x, y) = edge(x, y)
        rel path(x, y) = edge(x, z), path(z, y)
      """,
      provenance="diffminmaxprob",
      k=1,
      input_mappings={
        "edge": mapping,
        },
      output_mappings={"path": mapping},
      retain_graph=True,dispatch='single')

  def forward(self, x):
    a = self.model_a(x)
    
    return self.scl(edge=a)

def main():
  os.environ["STRATUM"] = "1"
  model = Model(3)

  x = torch.rand(1, 1024)
  y = model(x)


main()


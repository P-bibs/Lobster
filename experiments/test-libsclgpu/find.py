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

    self.model = RandNet(size ** 2)

    # Scallop Context
    mapping = [(i, j) for i in range(size) for j in range(size)]
    self.scl = scallopy.Module(
      program="""
        type digit1(x: u64, y: u64)
        type filtered(x: u64, y: u64)

        rel filtered(0,y) = digit1(0, y)
      """,
      provenance="diffminmaxprob",
      k=1,
      input_mappings={
        "digit1": mapping,
        },
      output_mappings={"filtered": mapping},
      retain_graph=True,dispatch='single')

  def forward(self, x):
    a = self.model(x)
    
    return self.scl(digit1=a)

def main():
  os.environ["STRATUM"] = "1"
  model = Model(3)

  x = torch.rand(1, 1024)
  y = model(x)


main()


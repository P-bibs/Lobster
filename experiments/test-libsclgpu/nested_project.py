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
    in_mapping = [(i, j, s) for i in range(size) for j in range(size) for s in range(size)]
    out_mapping = [(i, j) for i in range(size) for j in range(size)]
    self.scl = scallopy.Module(
      program="""
        type in1(x: u32, y: u32, z: u32)
        type in2(x: u32, y: u32, z: u32)
        type out(x: u32, y: u32)

        rel out(a, b) = in1(x, a, b), in2(x, c, d)
      """,
      provenance="diffminmaxprob",
      k=1,
      input_mappings={
        "in1": in_mapping,
        "in2": in_mapping,
        },
      output_mappings={"out": out_mapping},
      retain_graph=True,dispatch='single')

  def forward(self, x):
    a = self.model_a(x)
    b = self.model_b(x)
    
    return self.scl(in1=a, in2=b)

def main():
  os.environ["STRATUM"] = "2"
  model = Model(3)

  x = torch.rand(1, 1024)
  y = model(x)


main()

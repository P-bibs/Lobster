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

    self.model1 = RandNet(size)
    self.model2 = RandNet(size)

    # Scallop Context
    in_mapping = [i for i in range(size)]
    out_mapping = [i for i in range(size)]
    self.scl = scallopy.Module(
      program="""
        type digit1(x: u32)
        type out(x: u32)

        rel out(x - 1) = digit1(x)
      """,
      provenance="diffminmaxprob",
      k=1,
      input_mappings={
        "digit1": in_mapping,
        },
      output_mappings={"out": out_mapping},
      retain_graph=True,dispatch='single')

  def forward(self, x):
    a = self.model1(x)
    
    return self.scl(digit1=a)

def main():
  os.environ["STRATUM"] = "1"
  model = Model(3)

  x = torch.rand(1, 1024)
  y = model(x)


main()


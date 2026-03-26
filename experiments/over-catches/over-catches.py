import os
import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm

import scallopy

def foo():
    with open("experiments/over-catches/over_catches_ziyang.scl", "r") as f:
    #with open("experiments/over-catches/over-catches-simple.scl", "r") as f:
        code = f.read()

    path_planner = scallopy.Module(
      program=code,
      provenance="diffminmaxprob",
      #facts={"grid_node": [(torch.tensor(args.attenuation, requires_grad=False), c) for c in self.cells]},
      #input_mappings={"actor": self.cells, "goal": self.cells, "enemy": self.cells},
      #retain_topk={"actor": 3, "goal": 3, "enemy": 7},
      #output_mappings={"next_action": list(range(4)), "violation": ()},
      output_relation="over_catches",
      #retain_graph=True,
      dispatch='single')

    result = path_planner(dummy=[])
    print("Result: ", result)

if __name__ == "__main__":
  print(os.getpid())
  # Argument parser
  parser = ArgumentParser("over-catches")
  args = parser.parse_args()

  foo()


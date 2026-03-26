import math
import os
import random
import json
from argparse import ArgumentParser
from tqdm import tqdm

def generate_image_paths(data_root):
  symbols = {
      "+": "+",
      "-": "-",
      "*": "times",
      "/": "div",
      0: "0",
      1: "1",
      2: "2",
      3: "3",
      4: "4",
      5: "5",
      6: "6",
      7: "7",
      8: "8",
      9: "9"
  }

  paths = {}
  for (symbol, dir_name) in symbols.items():
    path = os.path.join(data_root, "Handwritten_Math_Symbols", dir_name)
    image_paths = os.listdir(path)
    paths[symbol] = [os.path.join(dir_name, image_path) for image_path in image_paths]

  return paths

class ExpressionGenerator:
  def __init__(self, length, min_length, split, distribution, batch_size):
    self.digits = list(range(10))
    self.operators = ["+", "-", "*", "/"]
    self.length = length
    self.min_length = min_length
    self.split = split
    self.distribution = distribution
    self.batch_size = batch_size
    self.image_paths = generate_image_paths(data_root)

  def generate_length(self):
    choices = list(range(self.min_length // 2, self.length // 2 + 1))
    if self.distribution == "uniform":
      weights = [1] * len(choices)
    elif self.distribution == "linear":
      weights = [1 + 2 * choice for choice in choices]
    elif self.distribution == "quadratic":
      weights = [1 + 2 * choice ** 2 for choice in choices]

    choice = random.choices(choices, weights=weights)[0]
    length = 1 + 2 * choice
    return length

  def generate_expr(self, length):
    expr = []
    for i in range(length):
      if i % 2 == 0:
        digit = self.digits[random.randint(0, len(self.digits) - 1)]
        expr.append(digit)
      else:
        operator = self.operators[random.randint(0, len(self.operators) - 1)]
        expr.append(operator)
    return expr

  def generate_datapoints(self, size):
    pad_size = math.ceil(math.log10(size))
    datapoints = []
    i = 0

    has_max_length = False
    while len(datapoints) < size:
      if i % self.batch_size == self.batch_size - 1 and not has_max_length:
        length = self.length
      else:
        length = self.generate_length()
      expr = self.generate_expr(length)
      expr_image_paths = []
      for symbol in expr:
        image_path = self.image_paths[symbol][random.randint(0, len(self.image_paths[symbol]) - 1)]
        expr_image_paths.append(image_path)
      expression = "".join(map(str, expr))

      try:
        result = eval(expression)
      except ZeroDivisionError:
        continue
      datapoints.append(
        {"id": f"{self.split}_{i:0{pad_size}}", "img_paths": expr_image_paths, "expr": expression, "res": result}
      )
      i += 1
      if len(expr) == self.length:
        has_max_length = True
      if i % self.batch_size == 0:
        has_max_length = False
    return datapoints

def generate_and_dump(args, data_root, split):
  generator = ExpressionGenerator(args.length, args.min_length, split, args.distribution, args.batch_size)
  data = generator.generate_datapoints(args.num_datapoints if split == "train" else args.num_datapoints // 10)
  lengths = [len(d["expr"]) for d in data]

  histogram = {length: lengths.count(length) for length in set(lengths)}
  print(f"Generated {split} data. Lengths:")
  for length, count in histogram.items():
    print(f"{length}: {count}")

  json.dump(data, open(os.path.join(data_root, split + "_" + args.output), "w"))

if __name__ == "__main__":
  parser = ArgumentParser("hwf/generate")
  parser.add_argument("--num-datapoints", type=int, default=10000)
  parser.add_argument("--length", type=int)
  parser.add_argument("--min-length", type=int, default=1)
  parser.add_argument("--seed", type=int)
  parser.add_argument("--output", type=str)
  parser.add_argument("--distribution", type=str, default="uniform")
  parser.add_argument("--batch-size", type=int, default=4)
  args = parser.parse_args()

  if args.length % 2 == 0:
    print("Length must be odd")
    exit(1)
  if args.min_length % 2 == 0:
    print("Min length must be odd")
    exit(1)

  if args.seed is None:
    import time
    args.seed = int(time.time())

  # Parameters
  random.seed(args.seed)
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/HWF"))

  generate_and_dump(args, data_root, "train")
  generate_and_dump(args, data_root, "test")


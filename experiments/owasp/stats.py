import argparse
import json
from collections import defaultdict

class Stat:
  def __init__(self):
    self.tp = 0
    self.tn = 0
    self.fp = 0
    self.fn = 0

  def total(self):
    return self.tp + self.fp + self.fn + self.tn

  def accuracy(self):
    if self.total() == 0: return float("nan")
    else: return (self.tp + self.tn) / self.total()

  def precision(self):
    if self.tp + self.fp == 0: return float("nan")
    else: return self.tp / (self.tp + self.fp)

  def recall(self):
    if self.tp + self.fn == 0: return float("nan")
    else: return self.tp / (self.tp + self.fn)

  def record(self, y_pred: bool, y: bool):
    if y_pred and y: self.tp += 1
    elif y_pred: self.fp += 1
    elif y: self.fn += 1
    else: self.tn += 1

  def __repr__(self) -> str:
    return f"Accu: {self.accuracy():.3f}, Rec: {self.recall():.3f}, Prec: {self.precision():.3f}, #TP: {self.tp}, #FP: {self.fp}, #FN: {self.fn}, Total: {self.total()}"


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file")
  args = parser.parse_args()

  content = json.load(open(args.file))
  stat = Stat()
  cwe_stats = defaultdict(lambda: Stat())
  for item in content:
    cwe_id = item["cwe_id"]
    y_pred = item["result"]
    y = item["gt"]
    stat.record(y_pred, y)
    cwe_stats[cwe_id].record(y_pred, y)

  print(f"Overall: {stat}")
  print(f"Individual CWE:")
  for (cwe_id, cwe_stats) in cwe_stats.items():
    print(f"- CWE {cwe_id}: {cwe_stats}")

import os
import sys
import argparse
import random
import json
import time

from typing import Optional, List
from collections import defaultdict
import xml.etree.ElementTree as ET
from tqdm import tqdm

import torch
import scallopy
import scallop_gpt

class OWASPJavaDataset(torch.utils.data.Dataset):
  def __init__(
      self,
      percentage: float = 1.0,
      randomize: bool = False,
      cwe: List[str] = [],
      filter: List[str] = [],
  ):
    self.root = os.path.join(os.path.dirname(__file__), "..", "data", "OWASP", "BenchmarkJava", "src", "main", "java", "org", "owasp", "benchmark", "testcode")
    self.tests = []
    for file_dir in os.listdir(self.root):
      if file_dir.endswith(".xml"):
        xml_content = ET.parse(os.path.join(self.root, file_dir))
        test_number = xml_content.find("test-number").text
        cwe_id = xml_content.find("cwe").text
        is_vulnerable = xml_content.find("vulnerability").text == "true"
        self.tests.append((test_number, cwe_id, is_vulnerable))
    if randomize:
      random.shuffle(self.tests)
    if percentage < 1.0:
      num_to_keep = int(len(self.tests) * percentage)
      self.tests = self.tests[:num_to_keep]
    if cwe is not None and len(cwe) > 0:
      self.tests = [t for t in self.tests if t[1] in cwe]
    if filter is not None and len(filter) > 0:
      self.tests = [t for t in self.tests if any(f in t[0] for f in filter)]

  def __len__(self):
    return len(self.tests)

  def get_test_number(self, i):
    (test_number, cwe_id, is_vulnerable) = self.tests[i]
    return test_number

  def get_cwe_id(self, i):
    (test_number, cwe_id, is_vulnerable) = self.tests[i]
    return cwe_id

  def __getitem__(self, i):
    (test_number, cwe_id, is_vulnerable) = self.tests[i]

    # Read the java file
    java_file_dir = f"{self.root}/BenchmarkTest{test_number}.java"
    java_file_lines = list(open(java_file_dir))

    # Remove the lines up to */
    java_file_lines = java_file_lines[java_file_lines.index(" */\n") + 2:]
    java_file_lines = [l for l in java_file_lines if not l.startswith("@WebServlet(") and not l.startswith("import java")]

    # Remove the comments which might spoil the result
    java_file_lines = [l[:l.index(" // ")] if " // " in l else l for l in java_file_lines]

    java_file_content = "".join(java_file_lines)
    return (java_file_content, is_vulnerable)


class VulDetector:
  def __call__(self, java_file_content: str):
    raise Exception("Not implemented")


class ScallopVulDetector(VulDetector):
  def scallop_file(self): raise Exception("Not implemented")

  def __call__(self, java_file_content: str):
    # Create a context
    ctx = scallopy.Context()

    # Load gpt plugin
    plugin_registry = scallopy.PluginRegistry()
    plugin_registry.load_plugin(scallop_gpt.ScallopGPTPlugin())
    plugin_registry.configure()
    plugin_registry.load_into_ctx(ctx)

    # Add information into context
    ctx.import_file(self.scallop_file())
    ctx.add_facts("program", [(java_file_content,)])

    # Run the context
    ctx.run()

    # Get the output
    output = list(ctx.relation("output"))
    if (True,) in output: result = True
    else: result = False

    # Get other log
    log = {}
    if ctx.has_relation("reason"): log["reason"] = list(ctx.relation("reason"))
    if ctx.has_relation("dataflow_edge"): log["dataflow_edge"] = list(ctx.relation("dataflow_edge"))
    if ctx.has_relation("source"): log["source"] = list(ctx.relation("source"))
    if ctx.has_relation("sanitizer"): log["sanitizer"] = list(ctx.relation("sanitizer"))
    if ctx.has_relation("sink"): log["sink"] = list(ctx.relation("sink"))

    # Return result and log
    return (result, log)


class GPTVulDetector(ScallopVulDetector):
  def scallop_file(self):
    return os.path.join(os.path.dirname(__file__), "detect_vul_gpt.scl")


class GPTCotVulDetector(ScallopVulDetector):
  def scallop_file(self):
    return os.path.join(os.path.dirname(__file__), "detect_vul_gpt_cot.scl")


class NeuroSymVulDetector(ScallopVulDetector):
  def scallop_file(self):
    return os.path.join(os.path.dirname(__file__), "detect_vul_neurosym.scl")


class RandomVulDetector(VulDetector):
  def __call__(self, java_file_content: str):
    return bool(random.randint(0, 1))


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


class TestRunner:
  def __init__(self, dataset: OWASPJavaDataset, vul_detector: VulDetector, method: str, log_dir: str):
    self.dataset = dataset
    self.vul_detector = vul_detector
    self.method = method
    self.log_dir = log_dir
    self.logs = []
    self.stat = Stat()
    self.stats_per_cwe = defaultdict(lambda: Stat())

  def run(self):
    iterator = tqdm(range(len(self.dataset)))
    for i in iterator:
      (x, y) = self.dataset[i]
      (y_pred, log) = self.vul_detector(x)

      # Record statistics
      test_number = self.dataset.get_test_number(i)
      cwe_id = self.dataset.get_cwe_id(i)
      self.logs.append({"test_number": test_number, "cwe_id": cwe_id, "program": x, "log": log, "result": y_pred, "gt": y})
      self.stat.record(y_pred, y)
      self.stats_per_cwe[cwe_id].record(y_pred, y)

      # Print progress bar
      iterator.set_description(f"{self.stat}")

  def print_stats(self):
    print(f"Overall: {self.stat}")
    print(f"Individual CWE:")
    for (cwe_id, cwe_stats) in self.stats_per_cwe.items():
      print(f"- CWE {cwe_id}: {cwe_stats}")

  def dump_logs(self):
    temp_file_name = f"log_{self.method}_{time.time()}.json"
    json.dump(self.logs, open(f"{self.log_dir}/{temp_file_name}", "w"), indent=2)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--method", type=str, choices=["gpt", "gpt-cot", "neurosym", "random"], default="neurosym")
  parser.add_argument("--percentage", type=float, default=1.0)
  parser.add_argument("--cwe", type=str, nargs="+", default=[])
  parser.add_argument("--filter", type=str, nargs="+", default=[])
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--log-dir", type=str, default=os.path.join(os.path.dirname(__file__), "log"))
  args = parser.parse_args()

  # Create log directory
  os.makedirs(args.log_dir, exist_ok=True)
  random.seed(args.seed)

  # Create a dataset
  dataset = OWASPJavaDataset(percentage=args.percentage, cwe=args.cwe, filter=args.filter)

  # Create a method
  if args.method == "gpt": detector = GPTVulDetector()
  elif args.method == "gpt-cot": detector = GPTCotVulDetector()
  elif args.method == "neurosym": detector = NeuroSymVulDetector()
  elif args.method == "random": detector = RandomVulDetector()
  else: raise Exception(f"Unknown detector `{args.method}`")

  # Do the evaluation
  eval_runner = TestRunner(dataset, detector, args.method, args.log_dir)
  eval_runner.run()
  eval_runner.print_stats()
  eval_runner.dump_logs()

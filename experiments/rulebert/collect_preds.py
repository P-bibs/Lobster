import os
import json
import scallopy
from argparse import ArgumentParser
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from convert_scl_file import to_rule

def get_pred(datapoint):
  query = datapoint['hypothesis'].strip()
  facts = [f.strip() for f in datapoint['facts'].split('.')]
  preds = []
  entities = {}

  query_info = query.split('(')
  query_entities = [e.strip() for e in query_info[1][:-1].split(',')]
  preds.append(query_info[0])
  if not query_info[0] in entities:
    entities[query_info[0]] = set()
  entities[query_info[0]].update(query_entities)

  for fact in facts:
    if not '(' in fact:
      continue
    fact_parsed = fact.split('(')
    preds.append(fact_parsed[0])
    if fact_parsed[0] == 'negcombatant':
      print('here')
    if not fact_parsed[0] in entities:
      entities[fact_parsed[0]] = set()
    entities[fact_parsed[0]].update([e.strip() for e in fact_parsed[1][:-1].split(',')])

  normalized_preds = set()
  for pred in preds:
    # Ensure everything is positive
    if 'neg' == pred[:3]:
      pred = pred[3:]
    normalized_preds.add(pred)

  return normalized_preds, entities

def merge_dict_set(x, y):
  for k, v in x.items():
    if k in y.keys():
        y[k].update(v)
    else:
        y[k] = v

def get_all_preds(dataset_path):
  dataset = json.load(open(dataset_path, 'r'))
  all_pred = set()
  all_entities = dict()
  for datapoint in dataset:
    normalized_preds, entities = get_pred(datapoint)
    merge_dict_set(entities, all_entities)
    all_pred.update(normalized_preds)
  return all_pred

if __name__ == "__main__":
  current_dir = os.path.abspath(__file__)
  data_dir = os.path.abspath(os.path.join(current_dir, '../../data/RuleBert'))

  parser = ArgumentParser()
  parser.add_argument("--data-dir", type=str, default=data_dir)
  parser.add_argument("--dataset", type=str, default="valid_chain_rules_train_0")
  parser.add_argument("--provenance", type=str, default="difftopbottomkclauses")
  args = parser.parse_args()
  print(args)

  # Load the dataset
  dataset_dir = os.path.join(args.data_dir, args.dataset)
  train_dataset_path = os.path.join(dataset_dir, "train.json")
  test_dataset_path = os.path.join(dataset_dir, "test.json")
  preds_path = os.path.join(dataset_dir, "preds.json")

  all_preds = set()
  all_preds.update(get_all_preds(train_dataset_path))
  all_preds.update(get_all_preds(test_dataset_path))
  json.dump(list(all_preds), open(preds_path, 'w'))

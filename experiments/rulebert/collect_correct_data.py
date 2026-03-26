import os
import json
import scallopy
from argparse import ArgumentParser
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from convert_scl_file import to_rule

wrong_dp = ['Chain_0_94231583', 'Chain_0_90329873', 'Chain_1_69502031']

class RuleBertDataset:
  def __init__(self, root, dataset, split):
    self.dataset_dir = os.path.join(root, dataset)
    self.file_name = os.path.join(self.dataset_dir, f"{split}.json")
    self.data = []
    for d in json.load(open(self.file_name, 'r')):
      # systematic question generation error
      if d['meta'] == 'switch_fact' and d['hypothesis'][:3] == 'neg':
        continue
      if d['id'] in wrong_dp:
        continue
      self.data.append(d)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    datapoint = self.data[i]
    query = datapoint['hypothesis'].replace('relation', 'relation_').replace('type', 'type_').strip()
    facts = [f.replace('relation', 'relation_').replace('type', 'type_').strip() for f in datapoint['facts'].split('.')]
    input_facts = {}

    query_info = query.split('(')
    query_pred = query_info[0]
    query_atoms = query_info[1][:-1].split(',')
    scl_query_elements = [f'\"{a}\"' for a in query_atoms]
    scl_query = f"{query_pred}({','.join(scl_query_elements)})"

    for fact in facts:
      if not '(' in fact:
        continue
      fact_parsed = fact.split('(')
      pred = fact_parsed[0]
      clause = tuple(fact_parsed[1][:-1].split(','))
      # Ensure everything is positive
      if 'neg' == pred[:3]:
        pred = pred[3:]
        prob = torch.tensor(0.0)
      else:
        prob = torch.tensor(1.0)

      if not pred in input_facts:
        input_facts[pred] = []
      input_facts[pred].append((prob, clause))
    # rules = [r.replace('relation', 'relation_').replace('type', 'type_') for r in datapoint['rule']]
    rules = datapoint['rule']
    # if any of the rules has less than 0.5's support, it will lead to wrong answer
    # All evidences does not support probability (Why??)
    rule_probs = datapoint['rule_support']

    query_result = torch.tensor(1) if datapoint['output'] else torch.tensor(0)

    return input_facts, scl_query, rules, rule_probs, query_result, datapoint

  @staticmethod
  def collate_fn(batch):
    batched_facts = [fact for (fact, _, _, _, _, _) in batch]
    batched_query = [query for (_, query, _, _, _, _) in batch]
    batched_rules = [rules for (_, _, rules, _, _, _) in batch]
    batched_rule_prob = [probs for (_, _, _, probs, _, _) in batch]
    batched_query_results = torch.stack([query_result for (_, _, _, _, query_result, _) in batch])
    batched_dps =  [dp for (_, _, _, _, _, dp) in batch]
    return batched_facts, batched_query, batched_rules, batched_rule_prob, batched_query_results, batched_dps

def rule_bert_loader(root, dataset, batch_size):
  train_dataset = RuleBertDataset(root, dataset, "train")
  train_loader = DataLoader(train_dataset, batch_size, collate_fn=RuleBertDataset.collate_fn, shuffle=True)
  test_dataset = RuleBertDataset(root, dataset, "test")
  test_loader = DataLoader(test_dataset, batch_size, collate_fn=RuleBertDataset.collate_fn, shuffle=True)
  val_dataset = RuleBertDataset(root, dataset, "val")
  val_loader = DataLoader(val_dataset, batch_size, collate_fn=RuleBertDataset.collate_fn, shuffle=True)
  return (train_loader, test_loader, val_loader)

class RuleBertExecutor:
  def __init__(self, provenance, train_top_k, test_top_k, scl_path, debug = False) -> None:
    self.scallop_ctx = scallopy.ScallopContext(provenance=provenance, train_k=train_top_k, test_k=test_top_k)
    self.scallop_ctx.import_file(scl_path)
    self.deterministic_relations = self.scallop_ctx.relations()
    self.scallop_ctx.set_non_probabilistic(self.deterministic_relations)
    self.debug = debug

  def execute(self, batched_facts, batched_query, batched_rules, batched_rule_probs):
    answers = []
    for facts, query, rules, rule_probs in zip(batched_facts, batched_query, batched_rules, batched_rule_probs):
      current_ctx = self.scallop_ctx.clone()
      rule_preds = [f"p_{ct}" for ct in range(len(rules))]

      query_info = query.split('(')
      query_pred = query_info[0]
      query_tp = query_info[1][:-1].split(',')

      if not 'neg' == query_pred[:3]:
        query_pred_pos = query_pred
        query_pred_neg = 'neg' + query_pred
        pos_question = True
      else:
        query_pred_pos = query_pred[3:]
        query_pred_neg = query_pred
        pos_question = False
      pos_query = query_pred_pos + '(' + query_info[1]
      neg_query = query_pred_neg + '(' + query_info[1]

      for rid, (rule_prob, raw_rule, rule_pred) in enumerate(zip(rule_probs, rules, rule_preds)):
        rule = to_rule(raw_rule)
        rule = rule.add_rel_prob(rule_pred)
        scl_rule = rule.to_scl()
        current_ctx.add_relation(rule_pred, ())
        current_ctx.add_rule(scl_rule)
        facts[rule_pred] = [(rule_prob, ())]

      all_entities = set()
      all_entities.update([eval(tp) for tp in query_tp])
      for pred, tps in facts.items():
        for (_, tp) in tps:
          for entity in tp:
            if isinstance(entity, str):
              all_entities.add(entity)
        current_ctx.add_facts(pred, tps)
      current_ctx.add_facts('entity', [(1.0, (e, )) for e in all_entities])

      # Add query
      current_ctx.add_rule(f"answer_pos() = {pos_query}")
      current_ctx.add_rule(f"answer_neg() = {neg_query}")

      current_ctx.run()
      query_pos_answer = list(current_ctx.relation("answer_pos"))
      query_neg_answer = list(current_ctx.relation("answer_neg"))

      if len(query_pos_answer) == 0:
        pos_prob = torch.tensor(0.0)
      else:
        pos_prob = list(current_ctx.relation("answer_pos"))[0][0]

      if len(query_neg_answer) == 0:
        neg_prob = torch.tensor(0.0)
      else:
        neg_prob = list(current_ctx.relation("answer_neg"))[0][0]

      # False, True
      if pos_question:
        answers.append(torch.stack([neg_prob, pos_prob]))
      else:
        answers.append(torch.stack([pos_prob, neg_prob]))

    return torch.stack(answers)

class Trainer:
  def __init__(self, executor, train_loader, test_loader, val_loader):
    self.device = device
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.val_loader = val_loader
    self.min_test_loss = 10000000000.0
    self.max_accu = 0
    self.executor = executor

  def loss(self, y_pred, y):
    (_, dim) = y_pred.shape
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in y])
    return nn.functional.binary_cross_entropy(y_pred, gt)

  def accuracy(self, y_pred, y):
    batch_size = len(y)
    pred = torch.argmax(y_pred, dim=1)
    num_correct = len([() for i, j in zip(pred, y) if i == j])
    return (num_correct, batch_size)

  def get_corrects(self, y_pred, y):
    pred = torch.argmax(y_pred, dim=1)
    correct_idx = [idx for idx, (i, j) in enumerate(zip(pred, y)) if i == j]
    return correct_idx

  def train(self, num_epochs):
    for i in range(1, num_epochs + 1):
      self.train_epoch(i)
      self.test_epoch(i)

  def get_correct(self, save_path, phase):
    total_count = 0
    total_correct = 0
    total_loss = 0
    if phase == "test":
      iterator = tqdm(self.test_loader)
    elif phase == "val":
      iterator = tqdm(self.val_loader)
    else:
      iterator = tqdm(self.train_loader)

    correct_dids = []
    for (i, (batched_facts, batched_query, batched_rules, batched_rule_prob, y, datapoints)) in enumerate(iterator):
      # if not i == 232:
      #   continue
      batch_size = len(y)
      y_pred = self.executor.execute(batched_facts, batched_query, batched_rules, batched_rule_prob)
      correct_train_idx = self.get_corrects(y_pred, y)
      correct_dids += [datapoints[idx]['id'] for idx in correct_train_idx]
      total_count += batch_size
      total_correct = len(correct_dids)
      correct_perc = 100. * total_correct / total_count
      avg_loss = total_loss / (i + 1)

      iterator.set_description(f"collected {total_correct} out of {total_count}")
      json.dump(correct_dids, open(save_path, 'w'))

length0_meta = ['depth_0', 'unsat_fact', 'inv_unsat_fact', 'inv_fact', 'switch_fact', 'switch_depth_0']
def meta_to_depth(meta):
  if meta in length0_meta:
    return 0
  return int(meta[-1])

def collect_correct_data(data_dir, valid_idxes_path, valid_data_dir, datacount, phase):
  data_path = os.path.join(data_dir, phase + '.json')
  valid_data_path = os.path.join(valid_data_dir, phase + '.json')

  dataset = json.load(open(data_path, 'r'))
  idxes = json.load(open(valid_idxes_path, 'r'))
  length_data = {}

  valid_dataset = [data for data in dataset if data['id'] in idxes]
  for data in valid_dataset:
    depth = meta_to_depth(data['meta'])
    if not depth in length_data:
      length_data[depth] = []
    length_data[depth].append(data)

  new_data = []
  for v in length_data.values():
    new_data += v[:datacount]

  json.dump(new_data, open(valid_data_path, 'w'))
  print('here')


if __name__ == "__main__":
  current_dir = os.path.abspath(__file__)
  data_dir = os.path.abspath(os.path.join(current_dir, '../../data/RuleBert'))

  parser = ArgumentParser()
  parser.add_argument("--data-dir", type=str, default=data_dir)
  parser.add_argument("--dataset", type=str, default="chain_rules_train_0")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--provenance", type=str, default="difftopbottomkclauses")
  parser.add_argument("--train-top-k", type=int, default=3)
  parser.add_argument("--test-top-k", type=int, default=3)
  parser.add_argument("--cuda", type=bool, default=True)
  parser.add_argument("--gpu", type=int, default=0)
  args = parser.parse_args()
  print(args)

  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  scl_path = os.path.join(data_dir, 'scl/pred_decs.scl')
  correct_train_path = os.path.join(data_dir, 'correct_train.json')
  correct_test_path = os.path.join(data_dir, 'correct_test.json')
  correct_val_path = os.path.join(data_dir, 'correct_val.json')

  valid_data_dir = os.path.join(data_dir, 'valid_chain_rules_train_0')
  dataset_path = os.path.join(args.data_dir, args.dataset)

  train_data_ct = 10000
  test_data_ct = 250 # per reasoning length
  val_data_ct = 250

  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: raise Exception("No cuda available")
  else: device = torch.device("cpu")

  if args.cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
  else:
    torch.set_default_tensor_type(torch.FloatTensor)

  # Load the dataset
  (train_loader, test_loader, val_loader) = rule_bert_loader(args.data_dir, args.dataset, args.batch_size)
  executer = RuleBertExecutor(args.provenance, args.train_top_k, args.test_top_k, scl_path)

  # Train
  # trainer = Trainer(executer, train_loader, test_loader, val_loader)
  # trainer.get_correct(correct_val_path, 'val')

  # collect_correct_data(dataset_path, correct_train_path, valid_data_dir, train_data_ct, "train")
  # collect_correct_data(dataset_path, correct_test_path, valid_data_dir, test_data_ct, "test")
  collect_correct_data(dataset_path, correct_val_path, valid_data_dir, val_data_ct, "val")

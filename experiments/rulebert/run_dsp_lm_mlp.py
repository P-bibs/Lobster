import os
import json
from py import process
import scallopy
from argparse import ArgumentParser
from tqdm import tqdm
import copy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from convert_scl_file import to_rule

from transformers import RobertaTokenizer, RobertaForQuestionAnswering, AutoConfig, RobertaModel
import torch
from convert_scl_file import Relation, Rule

preds = ["battle", "successor", "militarycommand", "parent", "relation_", "spouse", "child", "combatant", "birthplace", "predecessor", "relative", "residence", "deathplace"]

length0_meta = ['depth_0', 'unsat_fact', 'inv_unsat_fact', 'inv_fact', 'switch_fact', 'switch_depth_0']
def meta_to_depth(meta):
  if meta in length0_meta:
    return 0
  return int(meta[-1])

def process_context(context):
  context = context.strip()
  sentences = context.split('.')
  clean_sentences = []
  for sentence in sentences:
    if 'If' in sentence:
      continue
    if len(sentence) == 0:
      continue
    clean_sentences.append(sentence.strip())
  return clean_sentences

class MLP(nn.Module):
  def __init__(self, in_dim: int, embed_dim: int, out_dim: int, num_layers: int = 0, softmax = False, normalize = False, sigmoid = False):
    super(MLP, self).__init__()
    layers = []
    layers += [nn.Linear(in_dim, embed_dim), nn.ReLU()]
    for _ in range(num_layers):
      layers += [nn.Linear(embed_dim, embed_dim), nn.ReLU()]
    layers += [nn.Linear(embed_dim, out_dim)]
    self.model = nn.Sequential(*layers)
    self.softmax = softmax
    self.normalize = normalize
    self.sigmoid = sigmoid

  def forward(self, x):
    x = self.model(x)
    if self.softmax: x = nn.functional.softmax(x, dim=1)
    if self.normalize: x = nn.functional.normalize(x)
    if self.sigmoid: x = torch.sigmoid(x)
    return x

class RuleBertDataset:
  def __init__(self, root, dataset, split):
    self.dataset_dir = os.path.join(root, dataset)
    self.file_name = os.path.join(self.dataset_dir, f"{split}.json")
    self.data = json.load(open(self.file_name, 'r'))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    datapoint = self.data[i]
    query = datapoint['hypothesis'].replace('relation', 'relation_').replace('type', 'type_').strip()

    query_info = query.split('(')
    query_pred = query_info[0]
    query_atoms = query_info[1][:-1].split(',')
    scl_query_elements = [f'\"{a}\"' for a in query_atoms]
    scl_query = f"{query_pred}({','.join(scl_query_elements)})"

    context = datapoint['context']
    sentences = process_context(context)

    # rules = [r.replace('relation', 'relation_').replace('type', 'type_') for r in datapoint['rule']]
    rules = datapoint['rule']
    # if any of the rules has less than 0.5's support, it will lead to wrong answer
    # All evidences does not support probability (Why??)
    rule_probs = datapoint['rule_support']
    query_result = torch.tensor(1) if datapoint['output'] else torch.tensor(0)

    return sentences, scl_query, rules, rule_probs, query_result, datapoint

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
  return (train_loader, test_loader)

class RuleBertExecutor(nn.Module):
  def __init__(self, provenance, train_top_k, test_top_k, scl_path, debug = False) -> None:
    super().__init__()
    self.scallop_ctx = scallopy.ScallopContext(provenance=provenance, train_k=train_top_k, test_k=test_top_k)
    self.scallop_ctx.import_file(scl_path)
    self.debug = debug

  def forward(self, batched_facts, batched_sentences, batched_query, batched_rules, batched_rule_probs):
    results = self.execute(batched_facts, batched_query, batched_rules, batched_rule_probs)
    return results

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
        facts[rule_pred] = [(torch.tensor(rule_prob), ())]

      all_entities = set()
      all_entities.update([eval(tp) for tp in query_tp])
      for pred, tps in facts.items():
        for (_, tp) in tps:
          for entity in tp:
            if isinstance(entity, str):
              all_entities.add(entity)
        current_ctx.add_facts(pred, tps)
      current_ctx.add_facts('entity', [(None, (e, )) for e in all_entities])

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

class RuleBertModel(nn.Module):
  def __init__(self, device, num_mlp_layers=2, k=3, debug=False) -> None:
    super(RuleBertModel, self).__init__()
    self.tokenizer = RobertaTokenizer.from_pretrained("deepset/roberta-base-squad2")
    self.roberta_model = RobertaModel.from_pretrained("deepset/roberta-base-squad2")
    self.embed_dim = self.roberta_model.config.hidden_size
    self.relation_extraction = MLP(self.embed_dim, self.embed_dim, len(preds), num_layers=num_mlp_layers, softmax=True)
    self.device = device
    self.k = k

  def forward(self, batched_sentences):

    batched_facts = []

    for sentences in batched_sentences:
      sentence_size = len(sentences)
      facts = {}
      context_tokenized_result = self.tokenizer(sentences, padding=True, return_tensors="pt")
      context_input_ids = context_tokenized_result.input_ids.to(self.device)
      context_attention_mask = context_tokenized_result.attention_mask.to(self.device)
      encoded_contexts = self.roberta_model(context_input_ids, context_attention_mask)
      roberta_embedding = encoded_contexts.last_hidden_state
      context_overall_reps = []
      entity_pairs = []
      for i, s in enumerate(sentences):
        context_overall_rep = torch.mean(roberta_embedding[i, :sum(context_attention_mask[i]), :], dim=0)
        context_overall_reps.append(context_overall_rep)
        sentence_tks = s.split()
        from_entity = sentence_tks[-3]
        to_entity = sentence_tks[-1]
        entity_pairs.append((from_entity, to_entity))

      relation_probs = self.relation_extraction(torch.stack(context_overall_reps))
      for pair, rela_probs, s in zip(entity_pairs, relation_probs, sentences):
        for pred, rela_prob in zip(preds, rela_probs):
          if not pred in facts:
            facts[pred] = []
          if 'not' in s:
            facts[pred].append((torch.tensor(1.0) - rela_prob, pair))
          else:
            facts[pred].append((rela_prob, pair))
      batched_facts.append(facts)
    return batched_facts

class Trainer:
  def __init__(self, executor, train_loader, test_loader, device, learning_rate, rule_bert_k=3, load=False, model_dir=None, model_name=None):
    if not load:
      self.model = RuleBertModel(device, rule_bert_k)
    else:
      model_path = os.path.join(model_dir, model_name + '.best.model')
      self.model = torch.load(open(model_path, 'rb'))
    self.model_dir = model_dir
    self.model_name = model_name
    self.device = device
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.min_test_loss = 10000000000.0
    self.max_accu = 0
    self.executor = executor
    self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

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
    correct_idx = [1 if i == j else 0 for (i, j) in zip(pred, y) ]
    return correct_idx

  def train(self, num_epochs):
    # self.test_epoch(0)
    for i in range(1, num_epochs + 1):
      self.train_epoch(i)
      self.test_epoch(i)

  def train_epoch(self, epoch):
    self.model.train()
    total_count = 0
    total_correct = 0
    total_loss = 0
    iterator = tqdm(self.train_loader)
    for (i, (sentences, batched_query, batched_rules, batched_rule_prob, y, datapoints)) in enumerate(iterator):

      self.optimizer.zero_grad()
      batched_facts = self.model(sentences)
      y_pred = self.executor(batched_facts, sentences, batched_query, batched_rules, batched_rule_prob)
      y_pred = y_pred.clamp(0, 1)
      loss = self.loss(y_pred, y)
      if y_pred.requires_grad:
        loss.backward()
        self.optimizer.step()

      total_loss += loss.item()
      (num_correct, batch_size) = self.accuracy(y_pred, y)
      total_count += batch_size
      total_correct += num_correct
      correct_perc = 100. * total_correct / total_count
      avg_loss = total_loss / (i + 1)

      iterator.set_description(f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Accuracy: {total_correct}/{total_count} ({correct_perc:.2f}%)")

  def test_epoch(self, epoch):
    self.model.eval()
    total_count = 0
    total_correct = 0
    total_loss = 0
    iterator = tqdm(self.test_loader)
    with torch.no_grad():
      for (i, (sentences, batched_query, batched_rules, batched_rule_prob, y, datapoints)) in enumerate(iterator):

        batched_results = self.model(sentences)
        y_pred = self.executor(batched_results, sentences, batched_query, batched_rules, batched_rule_prob)
        loss = self.loss(y_pred, y)

        total_loss += loss.item()
        (num_correct, batch_size) = self.accuracy(y_pred, y)
        if num_correct < 1:
          print ('here')
        total_count += batch_size
        total_correct += num_correct
        correct_perc = 100. * total_correct / total_count
        avg_loss = total_loss / (i + 1)

        iterator.set_description(f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Accuracy: {total_correct}/{total_count} ({correct_perc:.2f}%)")

    # Save model
    if total_correct / total_count > self.max_accu:
      self.max_accu = total_correct / total_count
      torch.save(self.model, os.path.join(self.model_dir, f"{self.model_name}.best.model"))
    torch.save(self.model, os.path.join(self.model_dir, f"{self.model_name}.latest.model"))

  def test_by_depth(self, epoch):
    self.model.eval()
    total_count = 0
    total_correct = 0
    total_loss = 0
    iterator = tqdm(self.test_loader)
    depth_accu = {}
    with torch.no_grad():
      for (i, (sentences, batched_query, batched_rules, batched_rule_prob, y, datapoints)) in enumerate(iterator):
        batched_depth = [meta_to_depth(dp['meta']) for dp in datapoints]
        batched_results = self.model(sentences)
        y_pred = self.executor(batched_results, sentences, batched_query, batched_rules, batched_rule_prob)
        batched_correctness = self.get_corrects(y_pred, y)
        for depth, correct in zip(batched_depth, batched_correctness):
          if not depth in depth_accu:
             depth_accu[depth] = []
          depth_accu[depth].append(correct)

    for depth, accu_ls in depth_accu.items():
      print(f"{depth}: {sum(accu_ls)} out of {len(accu_ls)}({sum(accu_ls)/len(accu_ls)}) correct.")

if __name__ == "__main__":
  current_dir = os.path.abspath(__file__)
  data_dir = os.path.abspath(os.path.join(current_dir, '../../data/RuleBert'))

  parser = ArgumentParser()
  parser.add_argument("--data-dir", type=str, default=data_dir)
  parser.add_argument("--model-name", type=str, default="rule_bert_2")
  parser.add_argument("--dataset", type=str, default="valid_chain_rules_train_0")
  parser.add_argument("--seed", type=int, default=54321)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--provenance", type=str, default="difftopbottomkclauses")
  parser.add_argument("--train-top-k", type=int, default=3)
  parser.add_argument("--test-top-k", type=int, default=3)
  parser.add_argument("--cuda", type=bool, default=True)
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--learning-rate", type=float, default=0.00001)
  parser.add_argument("--rule-bert-k", type=int, default=3 )
  parser.add_argument("--n-epoches", type=int, default=100)
  parser.add_argument("--load", type=bool, default=True)


  args = parser.parse_args()
  print(args)

  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  scl_path = os.path.join(data_dir, 'scl/dsp_lm_mlp.scl')

  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: raise Exception("No cuda available")
  else: device = torch.device("cpu")

  if args.cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
  else:
    torch.set_default_tensor_type(torch.FloatTensor)

  # Setting up data and model directories
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/rulebert"))
  if not os.path.exists(model_dir): os.makedirs(model_dir)

  # Load the dataset
  (train_loader, test_loader) = rule_bert_loader(args.data_dir, args.dataset, args.batch_size)
  executer = RuleBertExecutor(args.provenance, args.train_top_k, args.test_top_k, scl_path)

  # Train
  trainer = Trainer(executer, train_loader, test_loader, device, model_dir=model_dir, model_name=args.model_name, rule_bert_k=args.rule_bert_k, learning_rate=args.learning_rate, load=args.load)
  # trainer.train(args.n_epoches)
  # trainer.test_epoch(0)
  trainer.test_by_depth(0)

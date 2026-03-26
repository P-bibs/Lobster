import os

from argparse import ArgumentParser
from tqdm import tqdm
import csv
import re
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer
import numpy as np

import scallopy

relation_id_map = {
  'daughter': 0,
  'sister': 1,
  'son': 2,
  'aunt': 3,
  'father': 4,
  'husband': 5,
  'granddaughter': 6,
  'brother': 7,
  'nephew': 8,
  'mother': 9,
  'uncle': 10,
  'grandfather': 11,
  'wife': 12,
  'grandmother': 13,
  'niece': 14,
  'grandson': 15,
  'son-in-law': 16,
  'father-in-law': 17,
  'daughter-in-law': 18,
  'mother-in-law': 19,
}

class CLUTRRDataset:
  def __init__(self, root, dataset, split):
    self.dataset_dir = os.path.join(root, f"CLUTRR/{dataset}/")
    self.file_names = [os.path.join(self.dataset_dir, d) for d in os.listdir(self.dataset_dir) if f"_{split}.csv" in d]
    self.data = [row for f in self.file_names for row in list(csv.reader(open(f)))[1:]]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    # Context is a list of sentences
    context = self.data[i][2].strip().lower()

    # Query is of type (sub, obj)
    query_sub_obj = eval(self.data[i][3])
    query = (query_sub_obj[0].lower(), query_sub_obj[1].lower())

    # Answer is one of 20 classes such as daughter, mother, ...
    answer = self.data[i][5]
    return ((context, query), answer)

  @staticmethod
  def collate_fn(batch):
    queries = [query for ((_, query), _) in batch]
    contexts = [context for ((context, _), _) in batch]
    answers = torch.stack([torch.tensor(relation_id_map[answer]) for (_, answer) in batch])
    return ((contexts, queries), answers)


def clutrr_loader(root, dataset, batch_size):
  train_dataset = CLUTRRDataset(root, dataset, "train")
  train_loader = DataLoader(train_dataset, batch_size, collate_fn=CLUTRRDataset.collate_fn, shuffle=True)
  test_dataset = CLUTRRDataset(root, dataset, "test")
  test_loader = DataLoader(test_dataset, batch_size, collate_fn=CLUTRRDataset.collate_fn, shuffle=True)
  return (train_loader, test_loader)


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


class CLUTRRModel(nn.Module):
  def __init__(self, device="cpu", num_mlp_layers=0, debug=False, use_last_hidden_state=False):
    super(CLUTRRModel, self).__init__()

    # Options
    self.device = device
    self.debug = debug
    self.use_last_hidden_state = use_last_hidden_state

    # Roberta as embedding extraction model
    self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", local_files_only=True, add_prefix_space=True)
    self.roberta_model = RobertaModel.from_pretrained("roberta-base")
    self.embed_dim = self.roberta_model.config.hidden_size

    # Entity embedding
    self.relation_extraction = MLP(self.embed_dim * 2, self.embed_dim, len(relation_id_map), num_layers=num_mlp_layers, sigmoid=True)

    # Scallop reasoning context
    self.scallop_ctx = scallopy.ScallopContext("difftopbottomkclauses", k=3)
    self.scallop_ctx.import_file(os.path.abspath(os.path.join(os.path.abspath(__file__), "../scl/clutrr.scl")))
    self.scallop_ctx.set_non_probabilistic(["question"])
    if self.debug:
      self.reason = self.scallop_ctx.forward_function("answer", list(range(len(relation_id_map))), dispatch="single", debug_provenance=True)
    else:
      self.reason = self.scallop_ctx.forward_function("answer", list(range(len(relation_id_map))))

  def forward(self, x):
    (contexts, queries) = x

    # Debug prints
    if self.debug:
      print(contexts)
      print(queries)

    # Preprocess sentences
    clean_contexts = []
    name_token_indices_maps = []
    for (i, context) in enumerate(contexts):
      # First obtain all the names in the context and put them into a set
      names = set(re.findall("\\[(\w+)\\]", context))

      # Then split the context by `[` and `]` so that names are isolated in its own string
      splitted = [u.strip() for t in context.split("[") for u in t.split("]") if u.strip() != ""]

      # Get the ids of the name in the `splitted` array
      is_name_ids = {s: [j for (j, sp) in enumerate(splitted) if sp == s] for s in names}

      # Get the splitted input_ids
      splitted_input_ids_raw = self.tokenizer(splitted).input_ids
      splitted_input_ids = [ids[:-1] if j == 0 else ids[1:] if j == len(splitted_input_ids_raw) - 1 else ids[1:-1] for (j, ids) in enumerate(splitted_input_ids_raw)]
      index_counter = 0
      splitted_input_indices = []
      for (j, l) in enumerate(splitted_input_ids):
        begin_offset = 1 if j == 0 else 0
        end_offset = 1 if j == len(splitted_input_ids) - 1 else 0
        quote_s_offset = 1 if "'s" in splitted[j] and splitted[j].index("'s") == 0 else 0
        splitted_input_indices.append(list(range(index_counter + begin_offset, index_counter + len(l) - end_offset - quote_s_offset)))
        index_counter += len(l) - quote_s_offset

      # Get the token indices for each name
      name_token_indices = {s: [k for phrase_id in is_name_ids[s] for k in splitted_input_indices[phrase_id]] for s in names}

      # Preprocess the context
      clean_contexts.append(context.replace("[", "").replace("]", ""))
      name_token_indices_maps.append(name_token_indices)

    # Use RoBERTa to encode the contexts into overall tensors
    context_tokenized_result = self.tokenizer(clean_contexts, padding=True, return_tensors="pt")
    context_input_ids = context_tokenized_result.input_ids.to(self.device)
    context_attention_mask = context_tokenized_result.attention_mask.to(self.device)
    encoded_contexts = self.roberta_model(context_input_ids, context_attention_mask)

    # Extract features corresponding to the names
    name_feature_maps = []
    for (i, name_token_indices) in enumerate(name_token_indices_maps):
      name_feature_map = {}
      for (name, token_indices) in name_token_indices.items():
        token_features = encoded_contexts.last_hidden_state[i, token_indices, :]
        agg_token_feature = torch.max(token_features, dim=0).values
        name_feature_map[name] = agg_token_feature
      name_feature_maps.append(name_feature_map)

    # Generate the name feature pairs
    splits, name_pairs, name_pairs_features = [], [], []
    for (i, name_feature_map) in enumerate(name_feature_maps):
      names = list(name_feature_map.keys())
      num_pairs = len(name_feature_map) * (len(name_feature_map) - 1)
      splits.append((0, num_pairs) if len(splits) == 0 else (splits[-1][1], splits[-1][1] + num_pairs))
      curr_name_pairs = [(x, y) for x in names for y in names if x != y]
      curr_name_pairs_features = [torch.cat((name_feature_map[x], name_feature_map[y])) for (x, y) in curr_name_pairs]
      name_pairs += curr_name_pairs
      name_pairs_features += curr_name_pairs_features
    name_pairs_features = torch.stack(name_pairs_features)

    # Use MLP to extract relations between names
    name_pair_relations = self.relation_extraction(name_pairs_features)

    # Generate facts for each context
    context_facts, context_disjunctions, question_facts = [], [], []
    for (i, (begin, end)) in enumerate(splits):
      curr_facts = []
      curr_context_disjunctions = []
      for j in range(begin, end):
        (sub, obj) = name_pairs[j]
        curr_facts += [(name_pair_relations[j, k], (k, sub, obj)) for k in range(len(relation_id_map))]
        curr_context_disjunctions.append(list(range((j - begin) * 20, (j - begin + 1) * 20)))
      context_facts.append(curr_facts)
      context_disjunctions.append(curr_context_disjunctions)
      question_facts.append([queries[i]])

    # Run Scallop to reason the result relation
    result = self.reason(context=context_facts, question=question_facts, disjunctions={"context": context_disjunctions})
    result = torch.softmax(result, dim=1)

    # Return the final result
    return result


class Trainer:
  def __init__(self, train_loader, test_loader, device, learning_rate, **args):
    self.device = device
    self.model = CLUTRRModel(device=device, **args).to(device)
    self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader

  def loss(self, y_pred, y):
    (_, dim) = y_pred.shape
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in y])
    return nn.functional.binary_cross_entropy(y_pred, gt)

  def accuracy(self, y_pred, y):
    batch_size = len(y)
    pred = torch.argmax(y_pred, dim=1)
    num_correct = len([() for i, j in zip(pred, y) if i == j])
    return (num_correct, batch_size)

  def train(self, num_epochs):
    for i in range(num_epochs):
      self.train_epoch(i)
      self.test_epoch(i)

  def train_epoch(self, epoch):
    self.model.train()
    total_count = 0
    total_correct = 0
    total_loss = 0
    iterator = tqdm(self.train_loader)
    for (i, (x, y)) in enumerate(iterator):
      self.optimizer.zero_grad()
      y_pred = self.model(x).to("cpu")
      loss = self.loss(y_pred, y)
      total_loss += loss.item()
      loss.backward()
      self.optimizer.step()

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
    with torch.no_grad():
      iterator = tqdm(self.test_loader)
      for (i, (x, y)) in enumerate(iterator):
        y_pred = self.model(x).to("cpu")
        loss = self.loss(y_pred, y)
        total_loss += loss.item()

        (num_correct, batch_size) = self.accuracy(y_pred, y)
        total_count += batch_size
        total_correct += num_correct
        correct_perc = 100. * total_correct / total_count
        avg_loss = total_loss / (i + 1)

        iterator.set_description(f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Accuracy: {total_correct}/{total_count} ({correct_perc:.2f}%)")


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--dataset", type=str, default="data_089907f8")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--learning-rate", type=float, default=0.00001)
  parser.add_argument("--num-mlp-layers", type=int, default=1)
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--use-last-hidden-state", action="store_true")
  parser.add_argument("--cuda", action="store_true")
  parser.add_argument("--gpu", type=int, default=0)
  args = parser.parse_args()

  # Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: raise Exception("No cuda available")
  else: device = torch.device("cpu")

  # Loading dataset
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  (train_loader, test_loader) = clutrr_loader(data_root, args.dataset, args.batch_size)

  # Train
  trainer = Trainer(train_loader, test_loader, device, args.learning_rate, num_mlp_layers=args.num_mlp_layers, debug=args.debug, use_last_hidden_state=args.use_last_hidden_state)
  trainer.train(args.n_epochs)

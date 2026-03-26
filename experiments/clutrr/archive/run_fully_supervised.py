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
    context = [s.strip().lower() for s in self.data[i][2].split(".") if s.strip() != ""]
    proof_state = eval(self.data[i][8])
    return (context, proof_state)

  @staticmethod
  def collate_fn(batch):
    proof_states = [proof_state for (_, proof_state) in batch]
    contexts = [fact for (context, _) in batch for fact in context]
    context_lens = [len(context) for (context, _) in batch]
    context_splits = [(sum(context_lens[:i]), sum(context_lens[:i + 1])) for i in range(len(context_lens))]
    return (contexts, proof_states, context_splits)


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
  def __init__(self, device="cpu", num_mlp_layers=1, debug=False, no_fine_tune_roberta=False, use_softmax=False, provenance="difftopbottomkclauses", top_k=3):
    super(CLUTRRModel, self).__init__()

    # Options
    self.device = device
    self.debug = debug
    self.no_fine_tune_roberta = no_fine_tune_roberta

    # Roberta as embedding extraction model
    self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", local_files_only=True, add_prefix_space=True)
    self.roberta_model = RobertaModel.from_pretrained("roberta-base")
    self.embed_dim = self.roberta_model.config.hidden_size

    # Entity embedding
    self.relation_extraction = MLP(self.embed_dim * 2, self.embed_dim, len(relation_id_map), num_layers=num_mlp_layers, sigmoid=not use_softmax, softmax=use_softmax)

    # Scallop reasoning context
    self.scallop_ctx = scallopy.ScallopContext(provenance=provenance, k=top_k)
    self.scallop_ctx.import_file(os.path.abspath(os.path.join(os.path.abspath(__file__), "../scl/clutrr.scl")))
    self.scallop_ctx.set_non_probabilistic(["question"])
    if self.debug: self.reason = self.scallop_ctx.forward_function("answer", list(range(len(relation_id_map))), dispatch="single", debug_provenance=True)
    else: self.reason = self.scallop_ctx.forward_function("answer", list(range(len(relation_id_map))))

  def _preprocess_contexts(self, contexts, context_splits):
    clean_context_splits = []
    clean_contexts = []
    name_token_indices_maps = []
    for (_, (start, end)) in enumerate(context_splits):
      skip_next = False
      skip_until = 0
      curr_clean_contexts = []
      curr_name_token_indices_maps = []
      for (j, sentence) in zip(range(start, end), contexts[start:end]):
        # It is possible to skip a sentence because the previous one includes the current one.
        if skip_next:
          if j >= skip_until:
            skip_next = False
          continue

        # Get all the names of the current sentence
        names = re.findall("\\[(\w+)\\]", sentence)

        # Check if we need to include the next sentence(s) as well
        num_sentences = 1
        union_sentence = f"{sentence}"
        for k in range(j + 1, end):
          next_sentence = contexts[k]
          next_sentence_names = re.findall("\\[(\w+)\\]", next_sentence)
          if len(names) == 1 or len(next_sentence_names) == 1:
            if len(next_sentence_names) > 0:
              num_sentences += 1
              union_sentence += f". {next_sentence}"
              names += next_sentence_names
            skip_next = True
            if len(next_sentence_names) == 1:
              skip_until = k - 1
            else:
              skip_until = k
          else:
            break

        # Deduplicate the names
        names = set(names)

        # Debug number of sentences
        if self.debug and num_sentences > 1:
          print(f"number of sentences: {num_sentences}, number of names: {len(names)}; {names}")
          print("Sentence:", union_sentence)

        # Then split the context by `[` and `]` so that names are isolated in its own string
        splitted = [u.strip() for t in union_sentence.split("[") for u in t.split("]") if u.strip() != ""]

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

        # Clean up the sentence and add it to the batch
        clean_sentence = union_sentence.replace("[", "").replace("]", "")

        # Preprocess the context
        curr_clean_contexts.append(clean_sentence)
        curr_name_token_indices_maps.append(name_token_indices)

      # Add this batch into the overall list; record the splits
      curr_size = len(curr_clean_contexts)
      clean_context_splits.append((0, curr_size) if len(clean_context_splits) == 0 else (clean_context_splits[-1][1], clean_context_splits[-1][1] + curr_size))
      clean_contexts += curr_clean_contexts
      name_token_indices_maps += curr_name_token_indices_maps

    # Return the preprocessed contexts and splits
    return (clean_contexts, clean_context_splits, name_token_indices_maps)

  def _extract_relations(self, clean_contexts, clean_context_splits, name_token_indices_maps):
    # Use RoBERTa to encode the contexts into overall tensors
    context_tokenized_result = self.tokenizer(clean_contexts, padding=True, return_tensors="pt")
    context_input_ids = context_tokenized_result.input_ids.to(self.device)
    context_attention_mask = context_tokenized_result.attention_mask.to(self.device)
    encoded_contexts = self.roberta_model(context_input_ids, context_attention_mask)

    # Extract features corresponding to the names for each context
    splits, name_pairs, name_pairs_features = [], [], []
    for (begin, end) in clean_context_splits:
      curr_datapoint_name_pairs = []
      curr_datapoint_name_pairs_features = []
      for (j, name_token_indices) in zip(range(begin, end), name_token_indices_maps[begin:end]):
        # Generate the feature_maps
        feature_maps = {}
        for (name, token_indices) in name_token_indices.items():
          token_features = encoded_contexts.last_hidden_state[j, token_indices, :]

          # Detach if we don't want to fine tune roberta
          if self.no_fine_tune_roberta:
            token_features = token_features.detach()

          # Use max pooling to join the features
          agg_token_feature = torch.max(token_features, dim=0).values
          feature_maps[name] = agg_token_feature

        # Generate name pairs
        names = list(name_token_indices.keys())
        curr_sentence_name_pairs = [(m, n) for m in names for n in names if m != n]
        curr_datapoint_name_pairs += curr_sentence_name_pairs
        curr_datapoint_name_pairs_features += [torch.cat((feature_maps[x], feature_maps[y])) for (x, y) in curr_sentence_name_pairs]

      # Generate the pairs for this datapoint
      num_name_pairs = len(curr_datapoint_name_pairs)
      splits.append((0, num_name_pairs) if len(splits) == 0 else (splits[-1][1], splits[-1][1] + num_name_pairs))
      name_pairs += curr_datapoint_name_pairs
      name_pairs_features += curr_datapoint_name_pairs_features

    # Stack all the features into the same big tensor
    name_pairs_features = torch.stack(name_pairs_features)

    # Use MLP to extract relations between names
    name_pair_relations = self.relation_extraction(name_pairs_features)

    # Return the extracted relations and their corresponding symbols
    return (splits, name_pairs, name_pair_relations)

  def forward(self, x):
    (contexts, proof_states, context_splits) = x

    # Go though the preprocessing, RoBERTa model forwarding, and facts extraction steps
    (clean_contexts, clean_context_splits, name_token_indices_maps) = self._preprocess_contexts(contexts, context_splits)
    (splits, name_pairs, name_pair_relations) = self._extract_relations(clean_contexts, clean_context_splits, name_token_indices_maps)

    # To check
    y_pred, y, answers = [], [], []
    for (i, (begin, end)) in enumerate(splits):
      proof_facts = [((sub.lower(), obj.lower()), relation_id_map[rel]) for proof_tree in proof_states[i] for (_, facts) in proof_tree.items() for (sub, rel, obj) in facts]
      name_pair_indices = [name_pairs.index(pair) for (pair, _) in proof_facts if pair in name_pairs]
      y_pred += name_pair_relations[name_pair_indices, :]
      y += [torch.tensor([1.0 if j == answer else 0.0 for j in range(len(relation_id_map))]) for (pair, answer) in proof_facts if pair in name_pairs]
      answers += [answer for (pair, answer) in proof_facts if pair in name_pairs]

    # Compute the loss
    y_pred = torch.stack(y_pred)
    y = torch.stack(y).to(self.device)

    loss = self.loss(y_pred, y)
    (num_correct, batch_size) = self.accuracy(y_pred, answers)

    # Return the results
    return (loss, num_correct, batch_size)

  def loss(self, y_pred, y):
    return nn.functional.binary_cross_entropy(y_pred, y)

  def accuracy(self, y_pred, y):
    batch_size = len(y)
    pred = torch.argmax(y_pred, dim=1)
    num_correct = len([() for i, j in zip(pred, y) if i == j])
    return (num_correct, batch_size)


class Trainer:
  def __init__(self, train_loader, test_loader, device, model_dir, model_name, learning_rate, **args):
    self.device = device
    self.model = CLUTRRModel(device=device, **args).to(device)
    self.model_dir = model_dir
    self.model_name = model_name
    self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.min_test_loss = 10000000000.0

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
    for (i, x) in enumerate(iterator):
      self.optimizer.zero_grad()
      (loss, num_correct, batch_size) = self.model(x)
      loss = loss.to("cpu")
      total_loss += loss.item()
      loss.backward()
      self.optimizer.step()
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
      for (i, x) in enumerate(iterator):
        (loss, num_correct, batch_size) = self.model(x)
        loss = loss.to("cpu")
        total_loss += loss.item()
        total_count += batch_size
        total_correct += num_correct
        correct_perc = 100. * total_correct / total_count
        avg_loss = total_loss / (i + 1)
        iterator.set_description(f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Accuracy: {total_correct}/{total_count} ({correct_perc:.2f}%)")

    # Save model
    if total_loss < self.min_test_loss:
      self.min_test_loss = total_loss
      torch.save(self.model, os.path.join(self.model_dir, f"{self.model_name}.best.model"))
    torch.save(self.model, os.path.join(self.model_dir, f"{self.model_name}.latest.model"))

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--dataset", type=str, default="data_089907f8")
  parser.add_argument("--model-name", type=str, default="clutrr_fully_supervised")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--learning-rate", type=float, default=0.00001)
  parser.add_argument("--num-mlp-layers", type=int, default=1)
  parser.add_argument("--provenance", type=str, default="difftopbottomkclauses")
  parser.add_argument("--top-k", type=int, default=3)
  parser.add_argument("--no-fine-tune-roberta", action="store_true")
  parser.add_argument("--use-softmax", action="store_true")
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

  # Setting up data and model directories
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/clutrr"))
  if not os.path.exists(model_dir): os.makedirs(model_dir)

  # Load the dataset
  (train_loader, test_loader) = clutrr_loader(data_root, args.dataset, args.batch_size)

  # Train
  trainer = Trainer(train_loader, test_loader, device, model_dir, args.model_name, args.learning_rate, num_mlp_layers=args.num_mlp_layers, debug=args.debug, provenance=args.provenance, top_k=args.top_k, use_softmax=args.use_softmax, no_fine_tune_roberta=args.no_fine_tune_roberta)
  trainer.train(args.n_epochs)

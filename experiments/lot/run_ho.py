import os

from argparse import ArgumentParser
from tqdm import tqdm
import jsonlines
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer

import scallopy

class LOTSanityDataset:
  def __init__(self, root, split):
    self.file_name = os.path.join(root, f"LOT/data_hypernyms_hypernyms_explicit_only_short_neg_hypernym_rule_{split}.jsonl")
    self.data = list(jsonlines.open(self.file_name))
    self.indices = list(range(len(self.data)))
    random.shuffle(self.indices)
    if split == "test":
      self.indices = self.indices[:200]

  def __len__(self):
    return len(self.indices)

  def __getitem__(self, i):
    j = self.indices[i]
    phrase = self.data[j]["phrase"].split(".")[0]
    context = [s.strip() for s in self.data[j]["context"].split(".") if s.strip() != ""]
    answer = self.data[j]["answer"]
    return ((phrase, context), answer)

  @staticmethod
  def collate_fn(batch):
    phrases = [phrase for ((phrase, _), _) in batch]
    contexts = [fact for ((_, context), _) in batch for fact in context]
    context_lens = [len(context) for ((_, context), _) in batch]
    context_splits = [(sum(context_lens[:i]), sum(context_lens[:i + 1])) for i in range(len(context_lens))]
    answers = [answer for (_, answer) in batch]
    return ((phrases, contexts, context_splits), answers)


def lot_sanity_loader(root, batch_size):
  train_dataset = LOTSanityDataset(root, "dev")
  train_loader = DataLoader(train_dataset, batch_size, collate_fn=LOTSanityDataset.collate_fn, shuffle=True)
  test_dataset = LOTSanityDataset(root, "test")
  test_loader = DataLoader(test_dataset, batch_size, collate_fn=LOTSanityDataset.collate_fn, shuffle=True)
  return (train_loader, test_loader)


class MLP(nn.Module):
  def __init__(self, in_dim: int, embed_dim: int, out_dim: int, num_layers: int = 0, normalize = False, sigmoid = False):
    super(MLP, self).__init__()
    layers = []
    layers += [nn.Linear(in_dim, embed_dim), nn.ReLU()]
    for _ in range(num_layers):
      layers += [nn.Linear(embed_dim, embed_dim), nn.ReLU()]
    layers += [nn.Linear(embed_dim, out_dim)]
    self.model = nn.Sequential(*layers)
    self.normalize = normalize
    self.sigmoid = sigmoid

  def forward(self, x):
    x = self.model(x)
    if self.normalize: x = nn.functional.normalize(x)
    if self.sigmoid: x = torch.sigmoid(x)
    return x


class LOTModel(nn.Module):
  def __init__(self):
    super(LOTModel, self).__init__()

    # Roberta as embedding extraction model
    self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", local_files_only=True, add_prefix_space=True)
    self.roberta_model = RobertaModel.from_pretrained("roberta-base")
    self.embed_dim = self.roberta_model.config.hidden_size
    for param in self.roberta_model.parameters():
      param.requires_grad = False

    # Entity embedding
    self.entity_synonym = MLP(self.embed_dim * 2, self.embed_dim, 1, sigmoid=True)

    # Inter-relation properties
    self.pairwise_relation_properties = MLP(self.embed_dim * 2, self.embed_dim, 4, sigmoid=True)
    self.triplewise_relation_properties = MLP(self.embed_dim * 3, self.embed_dim, 2, sigmoid=True)

    # Scallop reasoning context
    self.scallop_ctx = scallopy.ScallopContext("difftopbottomkclauses", k=3)
    self.scallop_ctx.import_file(os.path.abspath(os.path.join(os.path.abspath(__file__), "../scl/lot_ho.scl")))
    self.scallop_ctx.set_non_probabilistic(["question", "context"])
    self.reason = self.scallop_ctx.forward_function("answer", [False, True], iter_limit=3)

  def forward(self, x):
    (phrases, contexts, context_splits) = x
    num_total_contexts = len(contexts)
    batch_size = len(context_splits)

    # Find embedding of phrases and contexts
    all_sentences = contexts + phrases
    tokens = self.tokenizer(all_sentences, padding=True, return_tensors="pt")
    tokens_lengths = torch.sum(tokens["attention_mask"], dim=1)
    attention = tokens["attention_mask"].to(torch.float) * 0.1
    attention_length = [int(x.item()) for x in tokens_lengths]
    attention_length_floor = [int(x.item()) for x in torch.floor(tokens_lengths / 3)]
    attention_length_ceil = [int(x.item()) for x in torch.ceil(tokens_lengths / 3)]

    # Use hand-craft attention
    subject_indices = torch.LongTensor([[i, j] for i in range(batch_size) for j in range(attention_length_ceil[i])])
    subject_attention = torch.index_put(torch.clone(attention), tuple(subject_indices.t()), torch.ones(subject_indices.shape[0]))
    relation_indices = torch.LongTensor([[i, j] for i in range(batch_size) for j in range(attention_length_floor[i], attention_length_floor[i] + attention_length_ceil[i])])
    relation_attention = torch.index_put(torch.clone(attention), tuple(relation_indices.t()), torch.ones(relation_indices.shape[0]))
    object_indices = torch.LongTensor([[i, j] for i in range(batch_size) for j in range(attention_length_floor[i] * 2, attention_length[i])])
    object_attention = torch.index_put(torch.clone(attention), tuple(object_indices.t()), torch.ones(object_indices.shape[0]))

    # Get the subject, relation, and object embedding using the hand-crafted attention vectors
    subject_embeddings = self.roberta_model(tokens["input_ids"], subject_attention).pooler_output
    relation_embeddings = self.roberta_model(tokens["input_ids"], relation_attention).pooler_output
    object_embeddings = self.roberta_model(tokens["input_ids"], object_attention).pooler_output

    # Pairwise entity
    entity_pairs = []
    entity_pairs_split = []
    for (i, (start, end)) in enumerate(context_splits):
      r = torch.cat((subject_embeddings[start:end], object_embeddings[start:end], subject_embeddings[num_total_contexts + i].reshape(1, -1), object_embeddings[num_total_contexts + i].reshape(1, -1)))
      pairs = [torch.cat((r[j], r[k])) for j in range(len(r)) for k in range(len(r))]
      entity_pairs_split.append((len(entity_pairs), len(entity_pairs) + len(pairs)))
      entity_pairs += pairs
    e_entity_pairs = torch.stack(entity_pairs)

    # Entity-wise properties
    entity_synonyms = self.entity_synonym(e_entity_pairs).reshape(-1)

    # Pairwise relation
    rela_pairs, rela_triples = [], []
    rela_pairs_split, rela_triples_split = [], []
    for (i, (start, end)) in enumerate(context_splits):
      r = [relation_embeddings[j] for j in range(start, end)] + [relation_embeddings[num_total_contexts + i]]
      pairs = [torch.cat((r[j], r[k])) for j in range(len(r)) for k in range(len(r))]
      triples = [torch.cat((r[j], r[k], r[l])) for j in range(len(r)) for k in range(len(r)) for l in range(len(r))]
      rela_pairs_split.append((len(rela_pairs), len(rela_pairs) + len(pairs)))
      rela_triples_split.append((len(rela_triples), len(rela_triples) + len(triples)))
      rela_pairs += pairs
      rela_triples += triples
    e_rela_pairs = torch.stack(rela_pairs)
    e_rela_triples = torch.stack(rela_triples)

    # Predict inter-relation properties
    pairwise_properties = self.pairwise_relation_properties(e_rela_pairs)
    triplewise_properties = self.triplewise_relation_properties(e_rela_triples)

    # Generate facts for each batch
    question_facts, context_facts = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
    synonym_facts = [[] for _ in range(batch_size)]
    implies_facts, implies_not_facts = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
    inverse_facts, inverse_not_facts = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
    transitive_facts, transitive_not_facts = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
    for (i, (start, end)) in enumerate(context_splits):
      num_context = end - start
      num_relas = num_context + 1
      num_entities = num_relas * 2

      # Context and question
      context_facts[i] = [(j, j, j + num_context, True) for j in range(num_context)]
      question_facts[i] = [(num_context, num_context * 2, num_context * 2 + 1)]

      # Synonym
      entity_pairwise_offset = entity_pairs_split[i][0]
      synonym_facts[i] = [(entity_synonyms[entity_pairwise_offset + j * num_entities + k], (j, k)) for j in range(num_entities) for k in range(num_entities)]

      # Pair-wise
      pairwise_offset = rela_pairs_split[i][0]
      implies_facts[i] = [(pairwise_properties[pairwise_offset + j * num_relas + k, 0], (j, k)) for j in range(num_relas) for k in range(num_relas)]
      implies_not_facts[i] = [(pairwise_properties[pairwise_offset + j * num_relas + k, 1], (j, k)) for j in range(num_relas) for k in range(num_relas)]
      inverse_facts[i] = [(pairwise_properties[pairwise_offset + j * num_relas + k, 2], (j, k)) for j in range(num_relas) for k in range(num_relas)]
      inverse_not_facts[i] = [(pairwise_properties[pairwise_offset + j * num_relas + k, 3], (j, k)) for j in range(num_relas) for k in range(num_relas)]

      # Triple-wise
      triplewise_offset = rela_triples_split[i][0]
      transitive_facts[i] = [(triplewise_properties[triplewise_offset + j * num_relas * num_relas + k * num_relas + l, 0], (j, k, l)) for j in range(num_relas) for k in range(num_relas) for l in range(num_relas)]
      transitive_not_facts[i] = [(triplewise_properties[triplewise_offset + j * num_relas * num_relas + k * num_relas + l, 1], (j, k, l)) for j in range(num_relas) for k in range(num_relas) for l in range(num_relas)]

    # Use scallop to reason
    result = self.reason(
      question=question_facts,
      context=context_facts,
      synonym=synonym_facts,
      implies=implies_facts,
      implies_not=implies_not_facts,
      inverse=inverse_facts,
      inverse_not=inverse_not_facts,
      transitive=transitive_facts,
      transitive_not=transitive_not_facts,
    )
    result = nn.functional.softmax(result, dim=1)

    return result


class Trainer:
  def __init__(self, train_loader, test_loader, learning_rate):
    self.model = LOTModel()
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
      y_pred = self.model(x)
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
        y_pred = self.model(x)
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
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--learning-rate", type=float, default=0.00001)
  args = parser.parse_args()

  # Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Loading dataset
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  (train_loader, test_loader) = lot_sanity_loader(data_root, args.batch_size)

  # Train
  trainer = Trainer(train_loader, test_loader, args.learning_rate)
  trainer.train(args.n_epochs)

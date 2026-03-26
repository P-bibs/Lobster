import os

from argparse import ArgumentParser
from tqdm import tqdm
import jsonlines
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import spacy
from transformers import RobertaModel, RobertaTokenizer

import scallopy

class LOTSanityDataset:
  def __init__(self, root, split):
    self.file_name = os.path.join(root, f"LOT/data_hypernyms_hypernyms_training_mix_short_{split}.jsonl")
    self.data = list(jsonlines.open(self.file_name))
    self.indices = list(range(len(self.data)))
    random.shuffle(self.indices)

  def __len__(self):
    return len(self.indices)

  def __getitem__(self, i):
    j = self.indices[i]
    phrase = self.data[j]["phrase"].split(".")[0].lower()
    context = [s.strip().lower() for s in self.data[j]["context"].split(".") if s.strip() != ""]
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
  train_dataset = LOTSanityDataset(root, "train")
  train_loader = DataLoader(train_dataset, batch_size, collate_fn=LOTSanityDataset.collate_fn, shuffle=True)
  test_dataset = LOTSanityDataset(root, "dev")
  test_loader = DataLoader(test_dataset, batch_size, collate_fn=LOTSanityDataset.collate_fn, shuffle=True)
  return (train_loader, test_loader)


class MLP(nn.Module):
  def __init__(
    self,
    in_dim: int,
    embed_dim: int,
    out_dim: int,
    num_layers: int = 0,
    normalize = False,
    sigmoid = False,
    softmax = False,
  ):
    super(MLP, self).__init__()
    layers = []
    layers += [nn.Linear(in_dim, embed_dim), nn.ReLU()]
    for _ in range(num_layers):
      layers += [nn.Linear(embed_dim, embed_dim), nn.ReLU()]
    layers += [nn.Linear(embed_dim, out_dim)]
    self.model = nn.Sequential(*layers)
    self.normalize = normalize
    self.sigmoid = sigmoid
    self.softmax = softmax

  def forward(self, x):
    x = self.model(x)
    if self.normalize: x = nn.functional.normalize(x)
    if self.sigmoid: x = torch.sigmoid(x)
    if self.softmax: x = torch.softmax(x, dim=1)
    return x


class LOTModel(nn.Module):
  def __init__(self, provenance, k):
    super(LOTModel, self).__init__()

    # Spacy as entity extractor
    self.spacy_nlp = spacy.load("en_core_web_sm")

    # Roberta as embedding extraction model
    self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", local_files_only=True, add_prefix_space=True)
    self.roberta_model = RobertaModel.from_pretrained("roberta-base")
    self.embed_dim = self.roberta_model.config.hidden_size
    for param in self.roberta_model.parameters():
      param.requires_grad = False

    # Inter-relation properties
    self.relation_classifier = MLP(self.embed_dim, self.embed_dim, 10, num_layers=0, softmax=True)

    # Scallop reasoning context
    self.scallop_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    self.scallop_ctx.import_file(os.path.abspath(os.path.join(os.path.abspath(__file__), "../scl/lot_ho_2.scl")))
    self.reason = self.scallop_ctx.forward_function("answer", [False, True])

  def forward(self, x):
    (phrases, contexts, context_splits) = x
    num_total_contexts = len(contexts)
    batch_size = len(context_splits)

    # Find phrases and contexts
    all_sentences = contexts + phrases

    # Use embedding to extract the 10 possible relations
    tokens = self.tokenizer(all_sentences, padding=True, return_tensors="pt")
    all_sentences_embeddings = self.roberta_model(tokens["input_ids"], tokens["attention_mask"]).pooler_output
    all_sentences_relations = self.relation_classifier(all_sentences_embeddings)

    # Use spacy to extract entities
    all_sentences_spacy_docs = [self.spacy_nlp(s) for s in all_sentences]
    all_sentences_entities = [[chunk.text for chunk in doc.noun_chunks] for doc in all_sentences_spacy_docs]

    # Generate facts
    question_facts = [[] for _ in range(batch_size)]
    context_facts = [[] for _ in range(batch_size)]
    for (i, (start, end)) in enumerate(context_splits):
      context_facts[i] = [(all_sentences_relations[j, k], (k, all_sentences_entities[j][0], all_sentences_entities[j][1], True)) for j in range(start, end) for k in range(10) if len(all_sentences_entities[j]) >= 2]
      question_facts[i] = [(all_sentences_relations[j, k], (k, all_sentences_entities[j][0], all_sentences_entities[j][1])) for j in range(num_total_contexts + i, num_total_contexts + i + 1) for k in range(10) if len(all_sentences_entities[j]) >= 2]

    # print(context_facts[0])
    # print(question_facts[0])

    # Run Scallop
    result = self.reason(question=question_facts, context=context_facts)

    # print(result)

    result = nn.functional.softmax(result, dim=1)

    # Get result
    return result


class Trainer:
  def __init__(self, train_loader, test_loader, learning_rate, provenance, k):
    self.model = LOTModel(provenance, k)
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
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--provenance", type=str, default="difftopbottomkclauses")
  parser.add_argument("--top-k", type=int, default=5)
  args = parser.parse_args()

  # Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Loading dataset
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  (train_loader, test_loader) = lot_sanity_loader(data_root, args.batch_size)

  # Train
  trainer = Trainer(train_loader, test_loader, args.learning_rate, args.provenance, args.top_k)
  trainer.train(args.n_epochs)

from typing import Optional, Callable, Tuple
import os
import random
import itertools

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from pathfinder_net import CNNPathFinder32Net, ViTPathFinderNet
from argparse import ArgumentParser
from tqdm import tqdm

pathfinder_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class PathFinder32Dataset(torch.utils.data.Dataset):
  """
  :param data_root, the root directory of the data folder
  :param data_dir, the directory to the pathfinder dataset under the root folder
  :param difficulty, can be picked from "easy", "normal", "hard", and "all"
  """
  def __init__(
    self,
    data_root: str,
    data_dir: str = "pathfinder32",
    difficulty: str = "all",
    transform: Optional[Callable] = pathfinder_img_transform,
  ):
    # Store
    self.transform = transform

    # Get subdirectories
    if difficulty == "all":
      sub_dirs = ["curv_baseline", "curv_contour_length_9", "curv_contour_length_14"]
    elif difficulty == "easy":
      sub_dirs = ["curv_baseline"]
    elif difficulty == "normal":
      sub_dirs = ["curv_contour_length_9"]
    elif difficulty == "hard":
      sub_dirs = ["curv_contour_length_14"]
    else:
      raise Exception(f"Unrecognized difficulty {difficulty}")

    # Get all image paths and their labels
    self.samples = []
    for sub_dir in sub_dirs:
      metadata_dir = os.path.join(data_root, data_dir, sub_dir, "metadata")
      for sample_group_file in os.listdir(metadata_dir):
        sample_group_dir = os.path.join(metadata_dir, sample_group_file)
        sample_group_file = open(sample_group_dir, "r")
        sample_group_lines = sample_group_file.readlines()[:-1]
        for sample_line in sample_group_lines:
          sample_tokens = sample_line[:-1].split(" ")
          sample_img_path = os.path.join(data_root, data_dir, sub_dir, sample_tokens[0], sample_tokens[1])
          sample_label = int(sample_tokens[3])
          self.samples.append((sample_img_path, sample_label))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    (img_path, label) = self.samples[idx]
    img = Image.open(open(img_path, "rb"))
    if self.transform is not None:
      img = self.transform(img)
    return (img, label)

  @staticmethod
  def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.stack([torch.tensor(item[1]).long() for item in batch])
    return (imgs, labels)


def pathfinder_32_loader(data_root, difficulty, batch_size, train_percentage):
  dataset = PathFinder32Dataset(data_root, difficulty=difficulty)
  num_train = int(len(dataset) * train_percentage)
  num_test = len(dataset) - num_train
  (train_dataset, test_dataset) = torch.utils.data.random_split(dataset, [num_train, num_test])
  train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=PathFinder32Dataset.collate_fn, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=PathFinder32Dataset.collate_fn, batch_size=batch_size, shuffle=True)
  return (train_loader, test_loader)


class Trainer():
  def __init__(self, model, train_loader, test_loader, learning_rate, provenance, k, gpu, save_model=False):
    if gpu >= 0:
      device = torch.device("cuda:%d" % gpu)
    else:
      device = torch.device("cpu")
    self.device = device
    if model == "cnn":
      model_cls = CNNPathFinder32Net
    elif model == "vit":
      model_cls = ViTPathFinderNet
    else:
      raise ValueError("Unknown model %s" % model_cls)
    self.network = model_cls(provenance, k).to(self.device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader

    # Aggregated loss (initialized to be a huge number)
    self.save_model = save_model
    self.min_test_loss = 100000000.0

  def loss(self, output, expected_output):
    return torch.mean(torch.square(output - expected_output))

  def accuracy(self, output, expected_output) -> Tuple[int, int]:
    diff = torch.abs(output - expected_output)
    num_correct = len([() for d in diff if d.item() < 0.4999])
    return (len(output), num_correct)

  def train(self, epoch):
    self.network.train()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    num_items = 0
    total_train_correct = 0
    for (input, expected_output) in iter:
      self.optimizer.zero_grad()
      input = input.to(self.device)
      expected_output = expected_output.to(self.device)
      output = self.network(input)
      loss = self.loss(output, expected_output)
      loss.backward()
      self.optimizer.step()
      batch_size, num_correct = self.accuracy(output, expected_output)
      num_items += batch_size
      total_train_correct += num_correct
      correct_perc = 100. * total_train_correct / num_items
      iter.set_description(f"[Train Epoch {epoch}] Batch Loss: {loss.item():.4f}, Overall Accuracy: {correct_perc:.4f}%")

  def test(self, epoch):
    self.network.eval()
    num_items = 0
    test_loss = 0
    total_correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (i, (input, expected_output)) in enumerate(iter):
        input = input.to(self.device)
        expected_output = expected_output.to(self.device)
        output = self.network(input)
        test_loss += self.loss(output, expected_output).item()
        batch_size, num_correct_in_batch = self.accuracy(output, expected_output)
        num_items += batch_size
        total_correct += num_correct_in_batch
        perc = 100. * total_correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Avg loss: {test_loss / (i + 1):.4f}, Accuracy: {total_correct}/{num_items} ({perc:.2f}%)")

    # Save the model
    if self.save_model and test_loss < self.min_test_loss:
      self.min_test_loss = test_loss
      torch.save(self.network, "../model/pathfinder_32/pathfinder_32_net.pkl")

  def run(self, n_epochs):
    for epoch in range(1, n_epochs + 1):
      self.train(epoch)
      self.test(epoch)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("pathfinder_32")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--gpu", type=int, default=-1)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--train-percentage", type=float, default=0.9)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--difficulty", type=str, default="all")
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "vit"])
  parser.add_argument("--top-k", type=int, default=3)
  args = parser.parse_args()

  # Setup parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Load data
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
  (train_loader, test_loader) = pathfinder_32_loader(data_root, args.difficulty, args.batch_size, args.train_percentage)
  trainer = Trainer(args.model, train_loader, test_loader, args.learning_rate, args.provenance, args.top_k, args.gpu)

  # Run!
  trainer.run(args.n_epochs)

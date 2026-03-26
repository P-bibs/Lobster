from typing import Optional, Callable, Tuple
import os
import random
import itertools
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image

import scallopy

class PathFinder128Dataset(torch.utils.data.Dataset):
  pathfinder_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
      (0.1307,), (0.3081,)
    )
  ])

  """
  :param data_root, the root directory of the data folder
  :param data_dir, the directory to the pathfinder dataset under the root folder
  :param difficulty, can be picked from "easy", "normal", "hard", and "all"
  """
  def __init__(
    self,
    data_root: str,
    data_dir: str = "pathfinder128",
    difficulty: str = "all",
    transform: Optional[Callable] = pathfinder_img_transform,
  ):
    # Store
    self.transform = transform

    # Get subdirectories
    easy, normal, hard = ("curv_baseline", 0), ("curv_contour_length_9", 1), ("curv_contour_length_14", 2)
    if difficulty == "all": sub_dirs = [easy, normal, hard]
    elif difficulty == "easy": sub_dirs = [easy]
    elif difficulty == "normal": sub_dirs = [normal]
    elif difficulty == "hard": sub_dirs = [hard]
    else: raise Exception(f"Unrecognized difficulty {difficulty}")

    # Get all image paths and their labels
    self.samples = []
    for (sub_dir, difficulty_id) in sub_dirs:
      metadata_dir = os.path.join(data_root, data_dir, sub_dir, "metadata")
      for sample_group_file in os.listdir(metadata_dir):
        sample_group_dir = os.path.join(metadata_dir, sample_group_file)
        sample_group_file = open(sample_group_dir, "r")
        sample_group_lines = sample_group_file.readlines()[:-1]
        for sample_line in sample_group_lines:
          sample_tokens = sample_line[:-1].split(" ")
          sample_img_path = os.path.join(data_root, data_dir, sub_dir, sample_tokens[0], sample_tokens[1])
          sample_label = int(sample_tokens[3])
          self.samples.append((sample_img_path, difficulty_id, sample_label))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    (img_path, difficulty_id, label) = self.samples[idx]
    img = Image.open(open(img_path, "rb"))
    if self.transform is not None:
      img = self.transform(img)
    return (img, difficulty_id, label)


def pathfinder_128_loader(data_root, batch_size, train_percentage):
  dataset = PathFinder128Dataset(data_root)
  num_train = int(len(dataset) * train_percentage)
  num_test = len(dataset) - num_train
  (train_dataset, test_dataset) = torch.utils.data.random_split(dataset, [num_train, num_test])
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
  return (train_loader, test_loader)


class PathFinder128Net(nn.Module):
  def __init__(
    self,
    provenance="difftopbottomkclauses",
    k=3,
    use_chain=False,
    num_block_x=6,
    num_block_y=6,
    num_nodes=10,
  ):
    super(PathFinder128Net, self).__init__()

    # Build adjacency graph
    if not use_chain: self.build_grid_adjacency(num_block_x, num_block_y)
    else: self.build_chain_adjacency(num_nodes)

    # CNN
    self.cnn = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=5),
      nn.Conv2d(32, 32, kernel_size=5),
      nn.MaxPool2d(4),
      nn.Conv2d(32, 64, kernel_size=5),
      nn.Conv2d(64, 64, kernel_size=5),
      nn.MaxPool2d(4),
      nn.Flatten(),
    )

    # Fully connected for `is_endpoint`
    self.is_endpoint_fc = nn.Sequential(
      nn.Linear(1600, 256),
      nn.ReLU(),
      nn.Linear(256, self.num_blocks),
      nn.Sigmoid(),
    )

    # Fully connected for `edge`
    self.edge_fc = nn.Sequential(
      nn.Linear(1600, 256),
      nn.ReLU(),
      nn.Linear(256, len(self.adjacency)),
      nn.Sigmoid(),
    )

    # Scallop Context
    self.last_fc = nn.Sequential(
      nn.Linear(self.num_blocks + len(self.adjacency), 1),
      nn.Sigmoid(),
    )

  def block_coord_to_block_id(self, x, y):
    return y * self.num_block_x + x

  def build_grid_adjacency(self, num_block_x, num_block_y):
    # block
    self.num_block_x = num_block_x
    self.num_block_y = num_block_y
    self.num_blocks = num_block_x * num_block_y

    # Adjacency
    self.adjacency = []
    for i, j in itertools.product(range(num_block_x), range(num_block_y)):
      for (dx, dy) in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
        x, y = i + dx, j + dy
        if x >= 0 and x < num_block_x and y >= 0 and y < num_block_y:
          source_id = self.block_coord_to_block_id(i, j)
          target_id = self.block_coord_to_block_id(x, y)
          self.adjacency.append((source_id, target_id))

  def build_chain_adjacency(self, num_nodes):
    self.num_blocks = num_nodes
    self.adjacency = [(i, i + 1) for i in range(num_nodes - 1)] + [(i + 1, i) for i in range(num_nodes - 1)]

  def forward(self, input):
    embedding = self.cnn(input)
    edge = self.edge_fc(embedding)
    is_endpoint = self.is_endpoint_fc(embedding)
    feature = torch.cat((edge, is_endpoint), dim=1)
    result = self.last_fc(feature).reshape(-1)
    return result


class Trainer():
  def __init__(self, train_loader, test_loader, device, model_root, learning_rate, use_chain, provenance, k):
    self.device = device
    self.network = PathFinder128Net(provenance, k, use_chain=use_chain).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.model_root = model_root
    self.min_total_loss = 10000000000.0

  def loss(self, output, expected_output):
    return torch.mean(torch.square(output - expected_output))

  def accuracy(self, output, expected_output) -> Tuple[int, int]:
    diff = torch.abs(output - expected_output)
    num_correct = len([() for d in diff if d.item() < 0.4999])
    return (len(output), num_correct)

  def save_model(self, base_name):
    torch.save(self.network, os.path.join(self.model_root, f"{base_name}.pkl"))

  def train(self, epoch):
    self.network.train()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (input, _, expected_output) in iter:
      self.optimizer.zero_grad()
      output = self.network(input.to(self.device)).to("cpu")
      loss = self.loss(output, expected_output)
      loss.backward()
      self.optimizer.step()
      batch_size, num_correct = self.accuracy(output, expected_output)
      correct_perc = 100. * num_correct / batch_size
      iter.set_description(f"[Train Epoch {epoch}] Batch Loss: {loss.item():.4f}, Batch Accuracy: {correct_perc:.4f}%")

  def test(self, epoch):
    self.network.eval()
    num_items = 0
    test_loss = 0
    total_correct = 0

    # Do evaluate
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (i, (input, _, expected_output)) in enumerate(iter):
        output = self.network(input.to(self.device)).to("cpu")
        test_loss += self.loss(output, expected_output).item()
        avg_loss = test_loss / (i + 1)
        batch_size, num_correct_in_batch = self.accuracy(output, expected_output)
        num_items += batch_size
        total_correct += num_correct_in_batch
        perc = 100. * total_correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Avg loss: {avg_loss:.4f}, Accuracy: {total_correct}/{num_items} ({perc:.2f}%)")

    # Save the best model
    if test_loss < self.min_total_loss:
      self.min_total_loss = test_loss
      self.save_model("pathfinder_128_cnn_baseline.best")

    # Save the latest model
    if epoch % 5 == 0: self.save_model("pathfinder_128_cnn_baseline.latest")

  def run(self, n_epochs):
    for epoch in range(1, n_epochs + 1):
      self.train(epoch)
      self.test(epoch)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("pathfinder_128")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--train-percentage", type=float, default=0.9)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--chain", action="store_true")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=3)
  parser.add_argument("--cuda", action="store_true")
  parser.add_argument("--gpu", type=int, default=0)
  args = parser.parse_args()

  # Setup parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: raise Exception("No cuda available")
  else: device = torch.device("cpu")

  # Prepare directories
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
  if not os.path.exists(data_root): os.makedirs(data_root)
  model_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../model/pathfinder_128"))
  if not os.path.exists(model_root): os.makedirs(model_root)

  # Load data
  (train_loader, test_loader) = pathfinder_128_loader(data_root, args.batch_size, args.train_percentage)

  # Run!
  trainer = Trainer(train_loader, test_loader, device, model_root, args.learning_rate, args.chain, args.provenance, args.top_k)
  trainer.run(args.n_epochs)

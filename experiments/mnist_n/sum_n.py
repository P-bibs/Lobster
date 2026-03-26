import os
import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm

import scallopy
import wandb

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class MNISTSumNDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    n: int,
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
    augmentation: int = 2,
  ):
    self.n = n
    # Contains a MNIST dataset
    self.mnist_dataset = torchvision.datasets.MNIST(
      root,
      train=train,
      transform=transform,
      target_transform=target_transform,
      download=download,
    )
    self.index_map = list(range(len(self.mnist_dataset))) * augmentation
    random.shuffle(self.index_map)

  def __len__(self):
    return int(len(self.mnist_dataset) / self.n)

  def __getitem__(self, idx):
    images = []
    total = 0
    for i in range(self.n):
      (img, digit) = self.mnist_dataset[self.index_map[idx * self.n + i]]
      images.append(img)
      total += digit

    images = torch.stack(images)
    return (images, total)

  @staticmethod
  def collate_fn(batch):
    totals = torch.stack([torch.tensor(total).long() for (_, total) in batch])
    image_stacks = torch.stack([images for (images, _) in batch])

    return (image_stacks, totals)


def mnist_sum_n_loader(n, data_dir, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    MNISTSumNDataset(
      n,
      data_dir,
      train=True,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTSumNDataset.collate_fn,
    batch_size=batch_size_train,
    shuffle=True
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTSumNDataset(
      n,
      data_dir,
      train=False,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTSumNDataset.collate_fn,
    batch_size=batch_size_test,
    shuffle=True
  )

  return train_loader, test_loader


class MNISTNet(nn.Module):
  def __init__(self):
    super(MNISTNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
    self.fc1 = nn.Linear(1024, 1024)
    self.fc2 = nn.Linear(1024, 10)

  def forward(self, x):
    x = F.max_pool2d(self.conv1(x), 2)
    x = F.max_pool2d(self.conv2(x), 2)
    x = x.view(-1, 1024)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p = 0.5, training=self.training)
    x = self.fc2(x)
    return F.softmax(x, dim=1)


class MNISTSumNNet(nn.Module):
  def __init__(self, n, provenance, k):
    super(MNISTSumNNet, self).__init__()

    self.n = n

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()

    # Scallop Context
    self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    for i in range(self.n):
      self.scl_ctx.add_relation(f"digit_{i}", "u32", input_mapping=list(range(10)))

    head_title = f"sum_n"
    head_args = "+".join([f"dig{i}" for i in range(self.n)])
    head = f"{head_title}({head_args})"
    body = ", ".join([f"digit_{i}(dig{i})" for i in range(self.n)])
    rule = f"{head} = {body}"

    self.scl_ctx.add_rule(rule)

    self.sum_n = self.scl_ctx.forward_function("sum_n", [(i,) for i in range(self.n * 9 + 1)],
                                               dispatch='parallel')

  def forward(self, image_stacks):
    distributions = self.mnist_net(image_stacks.reshape(-1, 1, 28, 28)).reshape(-1, self.n, 10)

    out = self.sum_n(**{f"digit_{i}": distributions[:,i] for i in range(self.n)})
    return out


def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
  return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
  return F.nll_loss(output, ground_truth)


class Trainer():
  def __init__(self, n, train_loader, test_loader, learning_rate, loss, k, provenance):
    self.network = MNISTSumNNet(n, provenance, k)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    if loss == "nll":
      self.loss = nll_loss
    elif loss == "bce":
      self.loss = bce_loss
    else:
      raise Exception(f"Unknown loss function `{loss}`")

  def train_epoch(self, epoch):
    self.network.train()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    num_items = len(self.train_loader.dataset)
    correct = 0
    total_loss = 0
    for (data, target) in iter:
      self.optimizer.zero_grad()
      output = self.network(data)
      loss = self.loss(output, target)
      loss.backward()
      self.optimizer.step()

      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
      perc = 100. * correct / num_items

      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(self.train_loader.dataset)
    wandb.log({"train_loss": avg_loss, "train_accuracy": perc})

  def test(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target) in iter:
        output = self.network(data)
        test_loss += self.loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")
    wandb.log({"test_loss": test_loss, "test_accuracy": perc})

  def train(self, n_epochs):
    self.test(0)
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test(epoch)


if __name__ == "__main__":
  print(os.getpid())
  # Argument parser
  parser = ArgumentParser("mnist_sum_n")
  parser.add_argument("-n", type=int, default=10)
  parser.add_argument("--n-epochs", type=int, default=1000)
  parser.add_argument("--batch-size-train", type=int, default=8)
  parser.add_argument("--batch-size-test", type=int, default=8)
  parser.add_argument("--learning-rate", type=float, default=0.001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=1)
  parser.add_argument("--cuda", action="store_true")
  parser.add_argument("--gpu", type=int, default=0)
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size_train = args.batch_size_train
  batch_size_test = args.batch_size_test
  learning_rate = args.learning_rate
  loss_fn = args.loss_fn
  k = args.top_k
  provenance = args.provenance
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  wandb.init(project="mnist-sum-n", config=vars(args))

  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: raise Exception("No cuda available")
  else: device = torch.device("cpu")

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))

  # Dataloaders
  train_loader, test_loader = mnist_sum_n_loader(args.n, data_dir, batch_size_train, batch_size_test)

  # Create trainer and train
  trainer = Trainer(args.n, train_loader, test_loader, learning_rate, loss_fn, k, provenance)
  trainer.train(n_epochs)

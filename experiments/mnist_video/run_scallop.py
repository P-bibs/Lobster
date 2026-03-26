from argparse import ArgumentParser
import json
import os
import random

import cv2
from tqdm import tqdm

import torch
import torchvision


class MNISTVideoDataset(torch.utils.data.Dataset):
  mnist_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])

  def __init__(self, root: str, filename: str, train: bool):
    # Load the metadata
    self.root = root
    self.label = "train" if train else "test"
    self.metadata = json.load(open(os.path.join(root, "data", filename)))

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, i):
    dp = self.metadata[i]

    # Load video
    file_name = os.path.join(self.root, "data", "video", self.label, f"{dp['video_id']}.mp4")
    video = cv2.VideoCapture(file_name)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(num_frames):
      ret, frame = video.read()
      frames.append(self.mnist_img_transform(frame)[0:1, :, :])
    video_tensor = torch.stack(frames)

    # Generate label 1, which is a list of digits occurred in the video
    label1 = []
    curr_image_id = None
    for frame in dp["frames_sg"]:
      if curr_image_id == None or curr_image_id != frame["image_id"]:
        curr_image_id = frame["image_id"]
        label1.append(frame["digit"])

    # Generate label 2, which is the list of digits for each frame
    label2 = torch.stack([torch.tensor([1 if i == frame["digit"] else 0 for i in range(10)]) for frame in dp["frames_sg"]])

    # Generate label 3, which is the list of whether the image is different from the previous image in the video
    label3 = [1]
    for i in range(1, len(dp["frames_sg"])):
      label3.append(1 if dp["frames_sg"][i]["image_id"] != dp["frames_sg"][i - 1]["image_id"] else 0)
    label3 = torch.tensor(label3)

    # Return the video and all the labels
    return (video_tensor, (label1, label2, label3))

  def collate_fn(batch):
    videos = torch.stack([item[0] for item in batch])
    label1 = [item[1][0] for item in batch]
    label2 = torch.stack([item[1][1] for item in batch]).to(dtype=torch.float)
    label3 = torch.stack([item[1][2] for item in batch]).to(dtype=torch.float)
    return (videos, (label1, label2, label3))

  def loaders(root, batch_size):
    train_loader = torch.utils.data.DataLoader(
      MNISTVideoDataset(root, "MNIST_video_train_1000.json", train=True),
      batch_size=batch_size,
      shuffle=True,
      collate_fn=MNISTVideoDataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(
      MNISTVideoDataset(root, "MNIST_video_test_10.json", train=False),
      batch_size=batch_size,
      shuffle=True,
      collate_fn=MNISTVideoDataset.collate_fn)
    return train_loader, test_loader


class View(torch.nn.Module):
  def __init__(self, *shape):
    super().__init__()
    self.shape = shape

  def forward(self, input):
    return input.view(*self.shape)


class MNISTVideoNetwork(torch.nn.Module):
  def __init__(self, embedding_size=32):
    super().__init__()
    self.embedding_size = embedding_size
    self.encoder = torch.nn.Sequential(
      torch.nn.Conv2d(1, 32, kernel_size=5),
      torch.nn.MaxPool2d(2),
      torch.nn.Conv2d(32, 64, kernel_size=5),
      torch.nn.MaxPool2d(2),
      View(-1, 10816),
      torch.nn.Linear(10816, 1024),
      torch.nn.ReLU(),
      torch.nn.Linear(1024, self.embedding_size),
      torch.nn.ReLU())
    self.digit_decoder = torch.nn.Sequential(
      torch.nn.Linear(self.embedding_size, 10),
      torch.nn.Softmax(dim=1))
    self.changes_decoder = torch.nn.Sequential(
      torch.nn.Linear(self.embedding_size * 2, 1),
      torch.nn.Sigmoid())

  def forward(self, video_batch):
    (batch_size, num_frames, a, b, c) = video_batch.shape

    # First encode the video frames
    frame_encodings_batch = self.encoder(video_batch.view(batch_size * num_frames, a, b, c))

    # Predict the digits for each frame
    digits_batch = self.digit_decoder(frame_encodings_batch).view(batch_size, num_frames, -1)

    # Predict the changes for each consecutive pair of frames
    frame_encodings = frame_encodings_batch.view(batch_size, num_frames, self.embedding_size)
    zeros = torch.zeros(batch_size, 1, self.embedding_size)
    frame_encodings_with_prepended_zero = torch.cat([zeros, frame_encodings[:, 0:num_frames - 1, :]], dim=1)
    consecutive_frame_encodings = torch.cat([frame_encodings_with_prepended_zero, frame_encodings], dim=2)
    consecutive_frame_encodings_batch = consecutive_frame_encodings.view(batch_size * num_frames, -1)
    changes_batch = self.changes_decoder(consecutive_frame_encodings_batch).view(batch_size, num_frames)

    # Return
    return digits_batch, changes_batch


class Trainer:
  def __init__(self, train_loader, test_loader, model, learning_rate=1e-3):
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

  def loss(self, digits, changes, labels):
    _, label2, label3 = labels
    loss2 = torch.nn.functional.binary_cross_entropy(digits, label2)
    loss3 = torch.nn.functional.mse_loss(changes, label3)
    return loss2 + loss3

  def accuracy(self, digits, changes, labels):
    num_correct_predictions = 0
    num_total_predictions = 0
    (batch_size, num_frames, _) = digits.shape

    # Iterate through all the datapoints
    for datapoint in range(batch_size):
      for frame in range(num_frames):
        # First check if the digit is predicted correctly
        num_total_predictions += 1
        if digits[datapoint, frame].argmax().item() == labels[1][datapoint, frame].argmax().item():
          num_correct_predictions += 1

        # Then check if the change is predicted correctly
        num_total_predictions += 1
        if (changes[datapoint][frame] > 0.5) == (labels[2][datapoint][frame] > 0.5):
          num_correct_predictions += 1

    return num_correct_predictions, num_total_predictions

  def train_epoch(self, epoch):
    self.model.train()

    # Recording the loss and accuracy
    total_loss = 0
    num_batches = 0
    num_correct = 0
    num_total = 0

    # Iterate over batches
    iterator = tqdm(self.train_loader, total=len(self.train_loader))
    for (video, digits) in iterator:
      self.optimizer.zero_grad()

      # Do the prediction
      digits_pred, changes_pred = self.model(video)

      # Compute the loss
      loss = self.loss(digits_pred, changes_pred, digits)
      total_loss += loss.item()
      num_batches += 1

      # Compute the accuracy
      curr_num_correct, curr_num_total = self.accuracy(digits_pred, changes_pred, digits)
      num_correct += curr_num_correct
      num_total += curr_num_total

      # Backpropagate the loss and update the weights
      loss.backward()
      self.optimizer.step()

      # Show the current loss and accuracy
      avg_loss = total_loss / num_batches
      accuracy = num_correct / num_total
      iterator.set_description(f"[Train Epoch {epoch}] Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

  def test_epoch(self, epoch):
    self.model.eval()

    # Recording the loss and accuracy
    total_loss = 0
    num_batches = 0
    num_total = 0
    num_correct = 0

    # Iterate over batches
    iterator = tqdm(self.test_loader, total=len(self.test_loader))
    with torch.no_grad():
      for (video, digits) in iterator:
        digits_pred, changes_pred = self.model(video)
        loss = self.loss(digits_pred, changes_pred, digits)
        total_loss += loss.item()
        num_batches += 1
        curr_num_correct, curr_num_total = self.accuracy(digits_pred, changes_pred, digits)
        num_correct += curr_num_correct
        num_total += curr_num_total

        # Show the current loss and accuracy
        avg_loss = total_loss / num_batches
        accuracy = num_correct / num_total
        iterator.set_description(f"[Test Epoch {epoch}] Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

  def train(self, num_epochs):
    self.test_epoch(0)
    for i in range(1, num_epochs + 1):
      self.train_epoch(i)
      self.test_epoch(i)


if __name__ == "__main__":
  # Setup the parser
  parser = ArgumentParser()
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--num-epochs", type=int, default=10)
  args = parser.parse_args()

  # Set the seed
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Load data directory and dataset
  data_dir = os.path.join(os.path.dirname(__file__), "../data/MNIST_video")
  train_loader, test_loader = MNISTVideoDataset.loaders(data_dir, 32)

  # Initialize a model
  model = MNISTVideoNetwork()

  # Train the model
  trainer = Trainer(train_loader, test_loader, model, args.learning_rate)
  trainer.train(args.num_epochs)

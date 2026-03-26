import os
import argparse
import pickle
from tqdm import tqdm
import scallopy
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from viz import print_bitmap
import util

VERBOSE = False

class MNISTNet(nn.Module):
  def __init__(self, num_classes):
    super(MNISTNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same', padding_mode="zeros")
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same', padding_mode="zeros")
    self.fc1 = nn.Linear(64, 64)
    self.fc2 = nn.Linear(64, 128)
    self.fc3 = nn.Linear(128, 256)
    self.fc4 = nn.Linear(256, 256)
    self.fc5 = nn.Linear(256, num_classes)

    self.fc1.bias.retain_grad()

  def forward(self, x):
    x = F.max_pool2d(self.conv1(x), 2)
    x = F.max_pool2d(self.conv2(x), 2)
    x = x.view(-1, 64)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p = 0.2, training=self.training)
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = self.fc5(x)
    return F.softmax(x, dim=1)

class RasterNet(nn.Module):
    def __init__(self, grid_size, valid_rects):
        super(RasterNet, self).__init__()

        self.grid_size = grid_size
        self.valid_rects = valid_rects

        self.mnist_net = MNISTNet(len(valid_rects))

        coords = [(x, y) for x in range(grid_size) for y in range(grid_size)]
        scl_file = os.path.abspath(os.path.join(os.path.abspath(__file__), "../raster.scl"))
        self.scl = scallopy.Module(
                file=scl_file,
                non_probabilistic=["coord"],
                provenance="diffsamplekproofs",
                #provenance="diffaddmultprob",
                k=16,
                facts={"coord": coords},
                input_mappings={"rectangle": self.valid_rects},
                output_mappings={"pixel_color": coords},
                dispatch='single'
                )

    def forward(self, x):
        output = self.mnist_net(x)
        output.retain_grad()
        self.last_cnn_output = output

        return self.scl(rectangle=output).view(-1, 1, self.grid_size, self.grid_size)

class SingleShapeRasterDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    length = 100
  ):
    self.data = torch.load(os.path.join(root, "bitmaps.pt"))
    self.rects = torch.load(os.path.join(root, "rects.pt"))
    self.valid_rects = pickle.load(open(os.path.join(root, "valid_rects.pkl"), "rb"))

    self.data = self.data[0][None]
    self.rects = self.rects[0][None]
    self.length = length

  def __len__(self):
      return self.length

  def __getitem__(self, idx):
    return self.data[0], self.rects[0]

  @staticmethod
  def collate_fn(batch):
    data = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return data, targets

class RasterDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    train: bool,
    train_split: float = 0.8,
  ):
    self.data = torch.load(os.path.join(root, "bitmaps.pt"))
    self.rects = torch.load(os.path.join(root, "rects.pt"))
    self.valid_rects = pickle.load(open(os.path.join(root, "valid_rects.pkl"), "rb"))

    # Separate into train or test set
    if train:
        self.data = self.data[:int(train_split * self.data.shape[0])]
        self.rects = self.rects[:int(train_split * self.rects.shape[0])]
    else:
        self.data = self.data[int(train_split * self.data.shape[0]):]
        self.rects = self.rects[int(train_split * self.rects.shape[0]):]

    # TODO: random shuffle data before separating into train/test

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    return self.data[idx], self.rects[idx]

  @staticmethod
  def collate_fn(batch):
    data = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return data, targets

def single_shape_raster_loader(data_dir, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    SingleShapeRasterDataset(data_dir),
    collate_fn=RasterDataset.collate_fn,
    batch_size=batch_size_train,
  )
  test_loader = torch.utils.data.DataLoader(
    SingleShapeRasterDataset(data_dir, length=1),
    collate_fn=RasterDataset.collate_fn,
    batch_size=batch_size_test,
  )
  return train_loader, test_loader

def raster_loader(data_dir, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    RasterDataset(
      data_dir,
      train=True,
    ),
    collate_fn=RasterDataset.collate_fn,
    batch_size=batch_size_train,
  )

  test_loader = torch.utils.data.DataLoader(
    RasterDataset(
      data_dir,
      train=False,
    ),
    collate_fn=RasterDataset.collate_fn,
    batch_size=batch_size_test,
  )

  return train_loader, test_loader


class Trainer():
    def __init__(self, train_loader, test_loader, grid_size, lr):
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.network = RasterNet(grid_size, self.train_loader.dataset.valid_rects)

        def loss(output, target):
            recon_loss = F.mse_loss(output, target)
            return recon_loss

            # recon_loss = F.binary_cross_entropy(output, target)
            # regularization_loss = 0 #torch.abs(torch.norm(self.network.embedding.weight.data, p=1) - 1)
            # beta = 0.1
            # return recon_loss + beta * regularization_loss

        self.loss = loss
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)

    def train_epoch(self, epoch):
        self.network.train()
        iterator = tqdm(self.train_loader, total=len(self.train_loader.dataset))
        #iterator = self.train_loader
        for i, (bitmap, _) in enumerate(iterator):
            if i % 100 == 0:
                print(f"Step: {i}")
            output = self.network(bitmap)
            loss = self.loss(output, bitmap)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            iterator.set_description("Epoch {}, Loss {:.4f}".format(epoch, loss.item()))

    def test_epoch(self, epoch):
        self.network.train()
        # iterater = tqdm(self.test_loader, total=len(self.test_loader))
        iterator = self.test_loader
        for i, (bitmap, _) in enumerate(iterator):
            output = self.network(bitmap)
            loss = self.loss(output, bitmap)
            loss.backward()

            if i == 0 and VERBOSE:
                print(util.logits_to_bitmap(output[0][0], self.network.grid_size))
                lowest_loss = 1000
                best_rect_index = -1
                for i, rect in enumerate(self.train_loader.dataset.valid_rects):
                    logits = util.rectangles_to_logits([rect], 5).view(1, 1, self.network.grid_size, self.network.grid_size)
                    rect_loss = self.loss(logits, bitmap).item()
                    if rect_loss < lowest_loss:
                        lowest_loss = rect_loss
                        best_rect_index = i


                last_output = self.network.last_cnn_output
                top5 = last_output[0].topk(5)
                top5rects = [self.train_loader.dataset.valid_rects[top5[1][i]] for i in range(5)]
                zipped = zip(top5[0], top5rects)
                for j, (prob, rect) in enumerate(zipped):
                    rect_loss = self.loss(util.rectangles_to_logits([rect], self.network.grid_size)
                                          .view(1, 1, self.network.grid_size, self.network.grid_size), bitmap).item()
                    print("{:.5f}::{} -- loss: ".format(prob.item(), rect), rect_loss)
                    if j < 2:
                        logits = torch.zeros(1, len(self.train_loader.dataset.valid_rects))
                        index = self.train_loader.dataset.valid_rects.index(rect)
                        logits[0][index] = 1
                        print(util.logits_to_bitmap(self.network.scl(rectangle=logits), self.network.grid_size))

                g = last_output.grad[0]
                grad_sorted = g.argsort()
                best_rect_grad_rank = -1
                for i in range(g.shape[0]):
                    if grad_sorted[i] == best_rect_index:
                        best_rect_grad_rank = i
                        break

                cnn_output_sorted = last_output[0].argsort(descending=True)
                best_rect_cnn_rank = -1
                for i in range(cnn_output_sorted.shape[0]):
                    if cnn_output_sorted[i] == best_rect_index:
                        best_rect_cnn_rank = i
                        break

                print(f"Best rect: {self.train_loader.dataset.valid_rects[best_rect_index]}")
                print(f"\tloss: {lowest_loss}")
                print(f"\tgrad: {last_output.grad[0,best_rect_index]}")
                print(f"\tgrad rank: {best_rect_grad_rank}")
                print(f"\tProb: {last_output[0][best_rect_index]}")
                print(f"\tRanking: {best_rect_cnn_rank}")
                print(util.logits_to_bitmap(util.rectangles_to_logits([self.train_loader.dataset.valid_rects[best_rect_index]], 5), 5))

                print()
                print(f"Bias: {next(self.network.parameters())[0]}")

                print_bitmap(bitmap[0][0])
                print(f"Loss: {loss}")
            #import code; code.interact(local=locals())
            self.optimizer.zero_grad()
            #iterator.set_description(f"[Epoch {epoch}] Test loss: {test_loss:.4f}")

    def train(self, n_epochs=10000):
        for epoch in range(1, n_epochs + 1):
            # objgraph.show_growth()
            # objgraph.show_most_common_types()
            # if epoch == 3:
            #     objgraph.show_refs(self, filename='refs.png')


            print(f"Train epoch {epoch}:")
            self.train_epoch(epoch)

            print("Test epoch:")
            self.test_epoch(epoch)

        # print("Done training")
        # print("Top 5 largest embedding elements:")
        # print(torch.topk(self.network.embedding.weight.data, 10)[0])

        # print("Top 5 corresponding squares:")
        # [print(self.network.squares[i.item()]) for i in torch.topk(self.network.embedding.weight.data, 10)[1][0]]

        # print("Correct square:", self.correct_square)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a png->svg converter")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--picture-size", type=int, default=5)
    args = parser.parse_args()

    VERBOSE = args.verbose

    batch_size = 1
    lr = 0.00001

    # Dataloaders
    #train_loader, test_loader = raster_loader("data", batch_size, batch_size)
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../data"))
    train_loader, test_loader = single_shape_raster_loader(data_dir, batch_size*5, batch_size)

    trainer = Trainer(train_loader, test_loader, args.picture_size, lr)
    trainer.train()

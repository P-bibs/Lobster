from tkinter.tix import Tree
import torchvision
import torch
from typing import *
import random
import math
import numpy as np
import os
import cv2

mnist_height = 28
mnist_width = 28

class MNISTVideoData():
  def __init__(
    self,
    root: str,
    save_path: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
    num_frames: int = 10,
    output_height: int = 64,
    output_width: int = 64,
    min_image_ct: int = 2,
    max_image_ct: int = 5,
    sample_size: int = 1000,
  ):
    assert num_frames > max_image_ct
    assert output_height >= 28
    assert output_width >= 28

    # Contains a MNIST dataset
    self.mnist_dataset = torchvision.datasets.MNIST(
      root,
      train=train,
      transform=transform,
      target_transform=target_transform,
      download=download,
    )
    self.index_map = list(range(len(self.mnist_dataset)))
    random.shuffle(self.index_map)
    self.num_frames = num_frames
    self.output_height = output_height
    self.output_width = output_width
    self.min_image_ct = min_image_ct
    self.max_image_ct = max_image_ct
    self.sample_size = sample_size
    self.save_path = save_path

  def getdatapoint(self):

    image_number = random.choice(range(self.min_image_ct, self.max_image_ct))
    image_ids = random.sample(range(len(self.mnist_dataset)), k=image_number)
    mnist_image_ls = [self.mnist_dataset[self.index_map[image_id]] for image_id in image_ids]

    frame_split = math.floor(self.num_frames / image_number )
    frame_residule = self.num_frames % image_number
    frame_cts = []

    for img_ct in range(len(image_ids)):
        frame_ct = frame_split
        if img_ct < frame_residule:
            frame_ct += 1
        frame_cts.append(frame_ct)

    video_imgs = []
    ans = []
    for image_id, (img, digit), frame_ct in zip(image_ids, mnist_image_ls, frame_cts):
      video_imgs += (self.get_video_one_img(img, frame_ct))
      ans.append((self.index_map[image_id], digit))

    return video_imgs, ans

  def get_datapoints(self):
    dataset = []
    for _ in range(self.sample_size):
      dp = self.getdatapoint()
      dataset.append(dp)
    return dataset

  def save_video(self, video_id, video_imgs):

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 10
    file_path = os.path.join(self.save_path, f'{video_id}.avi')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (self.output_width, self.output_height), 0)
    for video_img in video_imgs:
      # video_img = video_img.resize(self.output_width, self.output_height)
      video_img = cv2.cvtColor(video_img, cv2.COLOR_GRAY2BGR)
      writer.write(video_img)
    writer.release()

    # for i, video_img in enumerate(video_imgs):
    #   cv2.imwrite(os.path.join(self.save_path, f'{video_id}-{i}.jpg'), video_img)

  def get_video_one_img(self, img, frame_ct):
    video_imgs = []
    valid_height = self.output_height - mnist_height
    valid_width = self.output_width - mnist_width
    start_height, end_height = random.sample(range(valid_height), k=2)
    start_width, end_width = random.sample(range(valid_width), k=2)

    y_move_dir = end_height - start_height > 0
    x_move_dir = end_width - start_width > 0

    delta_y_move = (end_height - start_height) / frame_ct
    delta_y_move = math.floor(delta_y_move) if y_move_dir else math.ceil(delta_y_move)
    delta_x_move = (end_width - start_width) / frame_ct
    delta_x_move = math.floor(delta_x_move) if x_move_dir else math.ceil(delta_x_move)

    moved_y = start_height
    moved_x = start_width

    for _ in range(frame_ct):
      blank_image = np.zeros((self.output_height,self.output_width), np.uint8)
      blank_image[moved_y:moved_y+img.size[0], moved_x:moved_x+img.size[1]] = img
      video_imgs.append(blank_image)
      moved_y += delta_y_move
      moved_x += delta_x_move

    return video_imgs

if __name__ == "__main__":
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../data"))
  video_mnist_dir = os.path.join(data_dir, "MNIST_video")
  train = True
  download = True
  num_frames = 20
  output_height = 64
  output_width = 64
  min_image_ct = 2
  max_image_ct = 5
  sample_size = 1

  data_generator = MNISTVideoData(
    root=data_dir,
    save_path=video_mnist_dir,
    train=train,
    download=download,
    num_frames = num_frames,
    output_height = output_height,
    output_width = output_width,
    min_image_ct = min_image_ct,
    max_image_ct = max_image_ct,
    sample_size = sample_size,
  )

  data = data_generator.get_datapoints()
  data_generator.save_video(0, data[0][0])

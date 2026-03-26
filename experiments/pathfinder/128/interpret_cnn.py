import os
from argparse import ArgumentParser
import itertools

import torch
import matplotlib.pyplot as plt

from run_with_cnn import PathFinder128Dataset, PathFinder128Net

def adjacency_list(num_block_x=6, num_block_y=6):
  block_coord_to_block_id = lambda x, y: y * num_block_x + x
  adjacency = []
  for i, j in itertools.product(range(num_block_x), range(num_block_y)):
    for (dx, dy) in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
      x, y = i + dx, j + dy
      if x >= 0 and x < num_block_x and y >= 0 and y < num_block_y:
        source_id = block_coord_to_block_id(i, j)
        target_id = block_coord_to_block_id(x, y)
        adjacency.append((source_id, target_id))
  return adjacency

def block_id_to_block_coord(id, num_block_x=6, num_block_y=6):
  return (id % num_block_x, int(id / num_block_y))

if __name__ == "__main__":
  # Command line arguments
  parser = ArgumentParser("pathfinder_128.interpret_cnn")
  parser.add_argument("--base-name", type=str, default="pathfinder_128_net.best")
  parser.add_argument("--amount", type=int, default=64)
  parser.add_argument("--cuda", action="store_true")
  parser.add_argument("--gpu", type=int, default=0)
  args = parser.parse_args()

  # Prepare directories
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
  model_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../model/pathfinder_128"))
  plot_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../interpret_plot"))

  # Load dataset
  dataset = PathFinder128Dataset(data_root)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.amount, shuffle=True)

  # Load models
  model: PathFinder128Net = torch.load(os.path.join(model_root, f"{args.base_name}.pkl"), map_location="cpu")

  # Precompute
  adjacency = adjacency_list(model.num_block_x, model.num_block_y)
  difficulties = {0: "Easy", 1: "Normal", 2: "Hard"}

  # Do not need gradient
  with torch.no_grad():
    # Pick the first batch
    for (input, difficulty, expected_output) in dataloader:
      # Size
      batch_size, num_channels, width, height = input.shape

      # Run through the model
      embedding = model.cnn(input)
      edge = model.edge_fc(embedding)
      is_endpoint = model.is_endpoint_fc(embedding)
      result = model.connected(edge=edge, is_endpoint=is_endpoint).view(-1)

      # For each sample, plot
      for i in range(batch_size):
        axs = plt.figure().subplots(1, 2)
        image_ax, feature_ax = axs[0], axs[1]

        # Difficulty string
        difficulty_str = difficulties[int(difficulty[i].item())]

        # First draw source image
        source_image = input[i][0]
        image_ax.imshow(source_image)
        image_ax.set_title(f"[{difficulty_str}] Ground Truth: {expected_output[i]}")

        # Then draw is_endpoint heatmap
        feature_ax.imshow(is_endpoint[i].view(model.num_block_x, model.num_block_y))
        for j, (start_block_id, end_block_id) in enumerate(adjacency):
          (start_x, start_y) = block_id_to_block_coord(start_block_id)
          (end_x, end_y) = block_id_to_block_coord(end_block_id)
          alpha = edge[i][j].item()
          feature_ax.plot([start_x, end_x], [start_y, end_y], "white", linewidth=5, alpha=alpha)
          is_correct = "correct" if round(result[i].item()) == expected_output[i].item() else "incorrect"
          feature_ax.set_title(f"Predicted: {result[i]:.3f}, {is_correct}")

        # Show the plot!
        plt.savefig(os.path.join(plot_root, f"{i}.png"))
        plt.close()

      # Only one batch!
      break

import os
import random
from argparse import ArgumentParser
from tqdm import tqdm

import torch

from run_with_cnn import PathFinder128Net, pathfinder_128_loader

def accuracy(output, expected_output):
  diff = torch.abs(output - expected_output)
  num_correct = len([() for d in diff if d.item() < 0.4999])
  return (len(output), num_correct)

def accuracy_single(output, expected_output):
  diff = torch.abs(output - expected_output)
  num_correct = len([() for d in diff if d.item() < 0.4999])
  return (len(output), num_correct)

def test_difficulty(args, data_root, device, model):
  # Load the dataset of given difficulty
  (_, dataloader) = pathfinder_128_loader(data_root, args.batch_size, args.train_percentage)

  # Stats
  num_items, total_correct = 0, 0
  num_easy_items, num_normal_items, num_hard_items = 0, 0, 0
  easy_correct, normal_correct, hard_correct = 0, 0, 0

  # Do not need gradient
  with torch.no_grad():
    # Pick the first batch
    iter = tqdm(dataloader, total=len(dataloader))
    for (input, difficulty_ids, expected_output) in iter:
      # Run through the model
      output = model(input.to(device)).to("cpu")

      # Compute stats
      batch_size, num_correct_in_batch = accuracy(output, expected_output)
      num_items += batch_size
      total_correct += num_correct_in_batch
      perc = 100. * total_correct / num_items

      # Iterate through to find easy/normal/hard stats separately
      for (y_pred, difficulty_id, y) in zip(output, difficulty_ids, expected_output):
        correct = torch.abs(y_pred - y).item() < 0.4999
        if difficulty_id == 0:
          num_easy_items += 1
          if correct: easy_correct += 1
        elif difficulty_id == 1:
          num_normal_items += 1
          if correct: normal_correct += 1
        else:
          num_hard_items += 1
          if correct: hard_correct += 1

      # Compute percentages
      easy_perc = 100. * easy_correct / num_easy_items if num_easy_items > 0 else 0.0
      normal_perc = 100. * normal_correct / num_normal_items if num_normal_items > 0 else 0.0
      hard_perc = 100. * hard_correct / num_hard_items if num_hard_items > 0 else 0.0

      # Print
      iter.set_description(f"[Test] Accuracy: {total_correct}/{num_items} ({perc:.2f}%), Easy: {easy_correct}/{num_easy_items} ({easy_perc:.2f}%), Normal: {normal_correct}/{num_normal_items} ({normal_perc:.2f}%), Hard: {hard_correct}/{num_hard_items} ({hard_perc:.2f}%)")

if __name__ == "__main__":
  # Command line arguments
  parser = ArgumentParser("pathfinder_128.test_cnn")
  parser.add_argument("--base-name", type=str, default="pathfinder_128_net.best")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--train-percentage", type=float, default=0.9)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=3)
  parser.add_argument("--cuda", action="store_true")
  parser.add_argument("--gpu", type=int, default=0)
  args = parser.parse_args()

  # Setup Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: raise Exception("No cuda available")
  else: device = torch.device("cpu")

  # Prepare directories
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
  model_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../model/pathfinder_128"))

  # Load models
  model: PathFinder128Net = torch.load(os.path.join(model_root, f"{args.base_name}.pkl"), map_location=device)

  # Run easy, normal, hard separately
  test_difficulty(args, data_root, device, model)

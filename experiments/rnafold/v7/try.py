import os
import subprocess

THIS_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_FOLDER = os.path.abspath(os.path.join(__file__, "../../../data/rnafold"))
MODEL_FOLDER = os.path.abspath(os.path.join(__file__, "../../../model/rnafold"))

for num_layers in [4, 8, 16]:
  for embed_size in [32, 64, 128]:
    for num_heads in [2, 4, 8, 16]:
      print("==================================")
      print(f"Num Layer: {num_layers}, embed_size: {embed_size}, num_heads: {num_heads}")
      args = [
        "python",
        f"{THIS_FOLDER}/train_token.py",
        "train",
        "--test-input", f"{DATA_FOLDER}/TestSetA.lst",
        "--train-percentage", "100",
        "--test-percentage", "100",
        "--seed", "0",
        "--loss-func", "bce",
        "--max-length", "20",
        "--embed-size", f"{embed_size}",
        "--num-layers", f"{num_layers}",
        "--num-heads", f"{num_heads}",
        "--lr", "0.0001",
        "--epoch", "1",
        # "--inference-retain-k", "5",
        "--batch-size", "4",
        # "--verbose",
        f"{DATA_FOLDER}/TrainSetA.lst",
      ]
      subprocess.run(args)

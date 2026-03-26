import zipfile
import shutil
import sys
import os
from argparse import ArgumentParser

def load_module(input, temp_dir):
  # Input file name and program name
  file_name = os.path.basename(input)
  prog_name = file_name.split("-")[0]

  # Remove the temporary directory
  shutil.rmtree(temp_dir, True)

  # Get zip file
  with zipfile.ZipFile(input, "r") as whl:
    whl.extractall(temp_dir)

  # Load the program
  sys.path.append(os.path.join(temp_dir, prog_name))
  scallop_module = __import__(prog_name)

  # return
  return scallop_module

def main():
  parser = ArgumentParser()
  parser.add_argument("input", type=str)
  parser.add_argument("--temp-dir", type=str, default="target/.tmpwheel.sclcmpl")
  args = parser.parse_args()

  return load_module(args.input, args.temp_dir)

if __name__ == "__main__":
  scallop_module = main()

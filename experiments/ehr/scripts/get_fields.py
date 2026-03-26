import os
import argparse

import csv


def dataset_path(args):
  if args.dataset_root is None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "ehr"))
  else:
    root = args.dataset_root
  return os.path.join(root, args.dataset)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset-root", type=str)
  parser.add_argument("--dataset", type=str, default="r35_pah.EHR.PRO.csv")
  args = parser.parse_args()

  # Get dataset file path and build the database
  dataset_file_path = dataset_path(args)

  # Load the file into csv
  with open(dataset_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    first_row = next(reader)
    for column_name in first_row:
      print(column_name)
    print(f"# of fields: {len(first_row)}")


if __name__ == "__main__":
  main()

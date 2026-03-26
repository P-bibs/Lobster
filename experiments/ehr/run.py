from collections import namedtuple
import os
import argparse

import scallopy


Atom = namedtuple("Atom", ["feature", "comparator", "constant"])


class Query:
  def __init__(self):
    self.atoms = []

  def __and__(self, other):
    if isinstance(other, Atom):
      self.atoms.append(other)
      return self
    elif isinstance(other, Query):
      self.atoms += other.atoms
      return self
    elif isinstance(other, tuple) and len(other) == 3:
      self.atoms.append(Atom(*other))
      return self
    else:
      raise Exception(f"Unknown query `{other}`")


class EHRDatabase:
  PREDICATES = {
    "age": [
      ("PAT_AGE", float),
    ],
    "band_neutrophils": [
      ("X..BAND.NEUTROPHILS..count", float),
      ("X..BAND.NEUTROPHILS..last", float),
      ("X..BAND.NEUTROPHILS..mean", float),
      ("X..BAND.NEUTROPHILS..max", float),
      ("X..BAND.NEUTROPHILS..mean.1", float),
      ("X..BAND.NEUTROPHILS..std", float),
    ],
    "eosinophils": [
      ("X..EOSINOPHILS..mean", float),
      ("X..EOSINOPHILS..min", float),
      ("X..EOSINOPHILS..std", float),
      ("X..EOSINOPHILS..last", float),
      ("X..EOSINOPHILS..mean.1", float),
    ],
    "lymphocytes_manual": [
      ("X..LYMPHOCYTES.MANUAL..last", float),
      ("X..LYMPHOCYTES.MANUAL..count", float),
    ],
    "lymphocytes": [
      ("X..LYMPHOCYTES..first", float),
      ("X..LYMPHOCYTES..first.1", float),
      ("X..LYMPHOCYTES..count", float),
      ("X..LYMPHOCYTES..last", float),
      ("X..LYMPHOCYTES..max", float),
      ("X..LYMPHOCYTES..mean", float),
      ("X..LYMPHOCYTES..mean.1", float),
      ("X..LYMPHOCYTES..min", float),
    ],
    "monocytes_manual": [
      ("X..MONOCYTES.MANUAL..last", float),
      ("X..MONOCYTES.MANUAL..max", float),
    ],
    "monocytes": [
      ("X..MONOCYTES..count", float),
      ("X..MONOCYTES..last", float),
      ("X..MONOCYTES..max", float),
      ("X..MONOCYTES..mean", float),
      ("X..MONOCYTES..max.1", float),
      ("X..MONOCYTES..mean.1", float),
    ],
    "basophils_manual": [
      ("X..BASOPHILS.MANUAL..mean", float),
      ("X..BASOPHILS.MANUAL..min", float),
    ],
    "myelocytes": [
      ("X..MYELOCYTES..mean", float),
    ],
    "neutrophils": [
      ("X..NEUTROPHILS..last.1", float),
      ("X..NEUTROPHILS..mean", float),
      ("X..NEUTROPHILS..min", float),
    ],
    "albumin": [
      ("ALBUMIN..last", float),
      ("ALBUMIN..mean", float),
      ("ALBUMIN..min", float),
      ("ALBUMIN..std", float),
    ],
  }

  def __init__(self, dataset_file_path: str):
    # Preprocess the field name to predicate and argument id pairs
    self.feature_location = {}
    for (predicate, fields) in self.PREDICATES.items():
      for (i, (field_name, _)) in enumerate(fields):
        self.feature_location[field_name] = (predicate, i)

    # First prepare the CSV file option
    self.csv_file = scallopy.io.CSVFileOptions(dataset_file_path)
    self.keys = ["PAT_ID", "appt_date"]

    # Then build the scallop database
    self.db = scallopy.Context()

    # Add the relations
    for (predicate, fields) in self.PREDICATES.items():
      field_names = [field_name for (field_name, _) in fields]
      field_types = tuple([field_ty for (_, field_ty) in fields])
      all_field_types = (str, scallopy.DateTime) + field_types
      csv_file = self.csv_file.with_fields(self.keys + field_names)
      self.db.add_relation(predicate, all_field_types, load_csv=csv_file)

    # Load the file
    self.db.run()

  def query(self, query):
    temp = self.db.clone()
    temp.add_rule(self.query_to_scallop(query))
    temp.run()
    return list(temp.relation("r"))

  def query_to_scallop(self, query: Query):
    head = "r(pat_id, appt_date)"
    body = " and ".join([self.atom_to_scallop(i, a) for (i, a) in enumerate(query.atoms)])
    return f"{head} = {body}"

  def atom_to_scallop(self, i: int, atom: Atom):
    (pred, jth) = self.feature_location[atom.feature]
    pred_arity = len(self.PREDICATES[pred])
    atom_str = pred + "(pat_id, appt_date, " + ", ".join([f"val_{i}" if j == jth else "_" for j in range(pred_arity)]) + ")"
    comp_str = f"val_{i} {atom.comparator} {atom.constant}"
    return f"{atom_str} and {comp_str}"


def dataset_path(args):
  if args.dataset_root is None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "ehr"))
  else:
    root = args.dataset_root
  return os.path.join(root, args.dataset)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset-root", type=str)
  parser.add_argument("--dataset", type=str, default="v1/r35_pah.EHR.PRO.csv")
  args = parser.parse_args()

  # Get dataset file path and build the database
  dataset_file_path = dataset_path(args)
  database = EHRDatabase(dataset_file_path)
  print("Database loaded")

  # Query 1
  q1 = Query() & ("PAT_AGE", ">", 50)
  print("Added query 1")
  r1 = database.query(q1)
  print("Finished computing query 1")
  print(r1)

  # Query 2
  q2 = Query() & ("PAT_AGE", "<", 50) & ("ALBUMIN..mean", ">", "3")
  print("Added query 2")
  r2 = database.query(q2)
  print("Finished computing query 2")


if __name__ == "__main__":
  main()

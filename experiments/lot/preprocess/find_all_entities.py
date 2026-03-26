import os

from argparse import ArgumentParser

# {
#   "property": {"subject": "fruit", "predicate": "/r/PartOf", "object": "angiosperm", "source": "conceptnet", "validity": "never true"},
#   "rules": ["A chestnut is not a flower.", "A chestnut is a fruit.", "A chestnut does not have a brow.", "A flower is part of angiosperm.", "A fruit is not part of angiosperm.", "A chestnut is a woody plant."],
#   "implicit_rule": {"subject": "chestnut", "predicate": "hypernym", "object": "fruit", "source": "wordnet/conceptnet", "validity": "always true"},
#   "statement": {"subject": "chestnut", "predicate": "/r/PartOf", "object": "angiosperm", "validity": "never true"},
#   "split": "dev",
#   "distractors": {
#     "implicit_rule": [
#       {"subject": "chestnut", "predicate": "hypernym", "object": "flower", "source": "wordnet/conceptnet", "validity": "never true"}
#     ],
#     "property": [
#       {"subject": "flower", "predicate": "/r/PartOf", "object": "angiosperm", "source": "conceptnet", "validity": "always true"}
#     ],
#     "statement": [
#       {"subject": "chestnut", "predicate": "/r/IsA", "object": "woody plant", "validity": "always true"},
#       {"subject": "chestnut", "predicate": "meronym", "object": "brow", "validity": "never true"}
#     ]
#   }
# }

import jsonlines

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--input", type=str, default="LOT/data_hypernyms_hypernyms_explicit_only_short_neg_hypernym_rule_dev.jsonl")
  args = parser.parse_args()

  # Dataset
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
  file_name = os.path.join(data_root, args.input)
  dataset = list(jsonlines.open(file_name))

  # All predicates
  entities = {}
  def process_relation(rel):
    if rel["subject"] in entities:
      entities[rel["subject"]] += 1
    else:
      entities[rel["subject"]] = 1
    if rel["object"] in entities:
      entities[rel["object"]] += 1
    else:
      entities[rel["object"]] = 1

  def process_relations(rels):
    for rel in rels:
      process_relation(rel)

  for line in dataset:
    if "property" in line["metadata"]:
      process_relation(line["metadata"]["property"])
    if "implicit_rule" in line["metadata"]:
      process_relation(line["metadata"]["implicit_rule"])
    if "statement" in line["metadata"]:
      process_relation(line["metadata"]["statement"])
    if "distractors" in line["metadata"]:
      if "implicit_rule" in line["metadata"]["distractors"]:
        process_relations(line["metadata"]["distractors"]["implicit_rule"])
      if "statement" in line["metadata"]["distractors"]:
        process_relations(line["metadata"]["distractors"]["statement"])
      if "property" in line["metadata"]["distractors"]:
        process_relations(line["metadata"]["distractors"]["property"])

  print(len(entities))
  print(entities)

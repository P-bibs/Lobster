import os
import random
import json
from argparse import ArgumentParser


def slice_by_question(args):
  source_json = json.load(open(os.path.join(dataset_dir, "questions", args.source_file)))
  scenes= json.load(open(scene_path, 'r'))['scenes']

  info = source_json["info"]
  questions = source_json["questions"]
  all_image_ids = []

  # Crop data
  if args.shuffle: random.shuffle(questions)
  cropped_qs = []

  for question in questions:
    if len(cropped_qs) >= args.num:
      break
    image_id = question['image_index']
    scene = scenes[image_id]
    objs = scene['objects']

    if len(objs) > args.max_obj:
      continue

    if len(question['program']) >= args.max_clause:
      continue

    if image_id in all_image_ids:
      continue
    else:
      all_image_ids.append(image_id)

    cropped_qs.append(question)

  # Result json
  result_json = {"info": info, "questions": cropped_qs}
  return result_json

def slice_by_image(args):

  source_json = json.load(open(os.path.join(dataset_dir, "questions", args.source_file)))
  scenes= json.load(open(scene_path, 'r'))['scenes']

  info = source_json["info"]
  questions = source_json["questions"]
  all_image_ids = []

  # Crop data
  cropped_qs = []
  for sid, scene in enumerate(scenes):
    objs = scene['objects']
    if len(objs) > args.max_obj:
      continue
    all_image_ids.append(sid)
    if len(all_image_ids) >= args.num:
      break

  for question in questions:
    q_image_id = question['image_index']

    if len(question['program']) >= args.max_clause:
      continue

    if q_image_id in all_image_ids:
      cropped_qs.append(question)

  # Result json
  result_json = {"info": info, "questions": cropped_qs}
  return result_json

if __name__ == "__main__":
  phase = "val"
  num = 1000
  max_clause = 100
  max_obj = 100

  # Argument parser
  parser = ArgumentParser("clevr_dataslice")
  parser.add_argument("--source-file", type=str, default=f"CLEVR_{phase}_questions.full.json")
  parser.add_argument("--output-file", type=str, default=f"CLEVR_{phase}_questions_obj_{max_obj}_clause_{max_clause}_image_split.cropped_{num}.json")
  parser.add_argument("--scene-file", type=str, default=f"scenes/CLEVR_{phase}_scenes.json")

  parser.add_argument("--shuffle", action="store_true", default=True)
  parser.add_argument("--num", type=int, default=num)
  parser.add_argument("--seed", type=int, default=1234)

  parser.add_argument("--max_obj", type=int, default=max_obj)
  parser.add_argument("--max_clause", type=int, default=max_clause)

  args = parser.parse_args()

  # Parameters
  random.seed(args.seed)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  dataset_dir = os.path.join(data_dir, "CLEVR")
  scene_path = os.path.join(dataset_dir, args.scene_file)

  # Load data
  result_json = slice_by_image(args)

  # Dump data
  json.dump(result_json, open(os.path.join(dataset_dir, "questions", args.output_file), "w"))

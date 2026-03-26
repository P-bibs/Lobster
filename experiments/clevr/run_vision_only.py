import os
import json
import pickle
import random
from argparse import ArgumentParser
from matplotlib import image

import torch, torchvision
import cv2
from PIL import Image
import os, json, random
import numpy as np
import scallopy
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import math

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, set_global_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_backbone
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import ImageList
from detectron2.modeling import GeneralizedRCNN
from detectron2.structures.boxes import Boxes
import detectron2.data.transforms as T

def check_valid(y_pred_values, y_values):
  y = torch.stack([torch.tensor([1.0 if u[0] == v else 0.0 for u in y_pred_values]) for v in y_values])
  assert torch.sum(y) == y.shape[0]

def scene_num_correct(y_pred_probs, y_values, question_tp):
  if question_tp == 'relate':
    y_discrete = (y_pred_probs>0.5).float()
    correct_num = sum([1 if res[0] == 1 and res[1] == 1 else 0 for res in (y_discrete == y_values).float()])
  else:
    indices = torch.argmax(y_pred_probs, dim=1).detach()
    y_indices = torch.argmax(y_values, dim=1).detach()
    correct_num = sum(indices == y_indices)
  return correct_num

all_shapes = ["cube", "cylinder", "sphere"]
all_colors = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
all_sizes = ["large", "small"]
all_mats = ["metal", "rubber"]
all_relas = ["left", "behind"]

all_attributes = {
  'shape': all_shapes,
  'color': all_colors,
  'size': all_sizes,
  'relate': all_relas,
  'material': all_mats,
}

all_bools = ["true", "false", "True", "False"]
all_nums = [str(i) for i in range(10)]
all_answers = all_shapes + all_colors + all_sizes + all_mats + all_relas + all_bools + all_nums

def scene_to_answer(scene):
  answer = {}
  num_objs = len(scene['objects'])
  indexes = list(range(num_objs))

  left_list = scene['relationships']['left']
  behind_list = scene['relationships']['behind']
  two_object_index = [(int(i), int(j)) for (i, j) in torch.cartesian_prod(torch.tensor(indexes), torch.tensor(indexes)) if not i == j]

  left_answer = {}
  for from_id, to_ids in enumerate(left_list):
    left_answer[from_id] = {}
    for obj_id in indexes:
      if obj_id in to_ids:
        left_answer[from_id][obj_id] = 1.0
      else:
        left_answer[from_id][obj_id] = 0.0

  behind_answer = {}
  for from_id, to_ids in enumerate(behind_list):
    behind_answer[from_id] = {}
    for obj_id in indexes:
      if obj_id in to_ids:
        behind_answer[from_id][obj_id] = 1.0
      else:
        behind_answer[from_id][obj_id] = 0.0

  answer['relate'] = torch.tensor([[left_answer[fid][tid], behind_answer[fid][tid]] for (fid, tid) in two_object_index])

  for qt, attrs in all_attributes.items():
    if qt == 'relate':
      continue

    a = []
    for obj in scene['objects']:
      index = attrs.index(obj[qt])
      res = [0 for _ in range(len(attrs))]
      res[index] = 1
      a.append(res)
    answer[qt] = torch.tensor(a, dtype=torch.float)

  return answer

def extract_bounding_boxes(scene):
  objs = scene['objects']
  rotation = scene['directions']['right']
  bboxes = []

  xmin = []
  ymin = []
  xmax = []
  ymax = []

  for i, obj in enumerate(objs):
    [x, y, z] = obj['pixel_coords']

    [x1, y1, z1] = obj['3d_coords']

    cos_theta, sin_theta, _ = rotation

    x1 = x1 * cos_theta + y1 * sin_theta
    y1 = x1 * -sin_theta + y1 * cos_theta


    height_d = 6.9 * z1 * (15 - y1) / 2.0
    height_u = height_d
    width_l = height_d
    width_r = height_d

    if obj['shape'] == 'cylinder':
      d = 9.4 + y1
      h = 6.4
      s = z1

      height_u *= (s*(h/d + 1)) / ((s*(h/d + 1)) - (s*(h-s)/d))
      height_d = height_u * (h-s+d)/ (h + s + d)

      height_u *= 1.1
      width_l *= 13/(10 + y1)
      width_r = width_l

    if obj['shape'] == 'cube':
      height_u *= 1.45 * 10 / (10 + y1)
      height_d = height_u
      width_l = height_u
      width_r = height_u

    if obj['shape'] == 'sphere':
      height_d = height_d * 1.15
      height_u = height_d
      width_l = height_d
      width_r = height_d

    ymin = y - height_d
    ymax = y + height_u
    xmin = x - width_l
    xmax = x + width_r
    bboxes.append([xmin, ymin, xmax, ymax])

  return Boxes(torch.tensor(bboxes))

def recover_box(pred_boxes, height, width):
  new_pred_boxes = []
  for pred_box in pred_boxes:
    x1 = pred_box[0] * width
    y1 = pred_box[1] * height
    x2 = pred_box[2] * width
    y2 = pred_box[3] * height
    new_pred_boxes.append([x1, y1, x2, y2])
  return torch.tensor(new_pred_boxes)

def save_image(image, boxes, pred_shape_probs, save_path):
  v = Visualizer(image[:, :, ::-1], scale=1.2)
  for box_id, box in enumerate(boxes):
    prob = pred_shape_probs[box_id]
    v.draw_box(box)
    v.draw_text(str(prob), tuple(box[:2].numpy()))
  v = v.get_output()
  bb_img = v.get_image()[:, :, ::-1]
  cv2.imwrite(save_path, bb_img)

def normalize_box(pred_boxes, height, width):
    new_pred_boxes = []
    for pred_box in pred_boxes:
      x1 = pred_box[0] / width
      y1 = pred_box[1] / height
      x2 = pred_box[2] / width
      y2 = pred_box[3] / height
      new_pred_boxes.append([x1, y1, x2, y2])
    return torch.tensor(new_pred_boxes)

def normalize_image(image):
  mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.224]).reshape(3, 1, 1)
  image = image.permute(2, 0, 1)

  image = (image / 255.0 - mean) / std
  return image

class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x

class CLEVRProgram:
  def __init__(self, program):
    # Boolean expressions
    self.greater_than_expr = []
    self.less_than_expr = []
    self.equal_expr = []
    self.exists_expr = []
    self.equal_color_expr = []
    self.equal_material_expr = []
    self.equal_shape_expr = []
    self.equal_size_expr = []

    # Integer expressions
    self.count_expr = []

    # Object expressions
    self.scene_expr = []
    self.unique_expr = []
    self.filter_color_expr = []
    self.filter_material_expr = []
    self.filter_shape_expr = []
    self.filter_size_expr = []
    self.same_color_expr = []
    self.same_material_expr = []
    self.same_shape_expr = []
    self.same_size_expr = []
    self.relate_expr = []
    self.and_expr = []
    self.or_expr = []

    # Query expressions
    self.query_color_expr = []
    self.query_material_expr = []
    self.query_shape_expr = []
    self.query_size_expr = []

    # Start processing program
    for (expr_id, expr) in enumerate(program):
      self._add_expr(expr_id, expr)

    # Last expression is root expression
    if expr_id == len(program) - 1:
      self.root_expr = [(expr_id,)]
      # self.result_type = self._expression_result_type(expr)

  def facts(self):
    return {x: y for (x, y) in self.__dict__.items() if x != "result_type"}

  def _add_expr(self, expr_id, expr):
    fn = expr["function"]
    if fn == "greater_than": self.greater_than_expr.append((expr_id, expr["inputs"][0], expr["inputs"][1]))
    elif fn == "less_than": self.less_than_expr.append((expr_id, expr["inputs"][0], expr["inputs"][1]))
    elif fn == "equal_integer": self.equal_expr.append((expr_id, expr["inputs"][0], expr["inputs"][1]))
    elif fn == "equal_color": self.equal_color_expr.append((expr_id, expr["inputs"][0], expr["inputs"][1]))
    elif fn == "equal_material": self.equal_material_expr.append((expr_id, expr["inputs"][0], expr["inputs"][1]))
    elif fn == "equal_shape": self.equal_shape_expr.append((expr_id, expr["inputs"][0], expr["inputs"][1]))
    elif fn == "equal_size": self.equal_size_expr.append((expr_id, expr["inputs"][0], expr["inputs"][1]))
    elif fn == "exist": self.exists_expr.append((expr_id, expr["inputs"][0]))
    elif fn == "count": self.count_expr.append((expr_id, expr["inputs"][0]))
    elif fn == "scene": self.scene_expr.append((expr_id,))
    elif fn == "unique": self.unique_expr.append((expr_id, expr["inputs"][0]))
    elif fn == "filter_color": self.filter_color_expr.append((expr_id, expr["inputs"][0], expr["value_inputs"][0]))
    elif fn == "filter_material": self.filter_material_expr.append((expr_id, expr["inputs"][0], expr["value_inputs"][0]))
    elif fn == "filter_shape": self.filter_shape_expr.append((expr_id, expr["inputs"][0], expr["value_inputs"][0]))
    elif fn == "filter_size": self.filter_size_expr.append((expr_id, expr["inputs"][0], expr["value_inputs"][0]))
    elif fn == "intersect": self.and_expr.append((expr_id, expr["inputs"][0], expr["inputs"][1]))
    elif fn == "union": self.or_expr.append((expr_id, expr["inputs"][0], expr["inputs"][1]))
    elif fn == "same_color": self.same_color_expr.append((expr_id, expr["inputs"][0]))
    elif fn == "same_material": self.same_material_expr.append((expr_id, expr["inputs"][0]))
    elif fn == "same_shape": self.same_shape_expr.append((expr_id, expr["inputs"][0]))
    elif fn == "same_size": self.same_size_expr.append((expr_id, expr["inputs"][0]))
    elif fn == "relate": self.relate_expr.append((expr_id, expr["inputs"][0], expr["value_inputs"][0]))
    elif fn == "query_color": self.query_color_expr.append((expr_id, expr["inputs"][0]))
    elif fn == "query_material": self.query_material_expr.append((expr_id, expr["inputs"][0]))
    elif fn == "query_shape": self.query_shape_expr.append((expr_id, expr["inputs"][0]))
    elif fn == "query_size": self.query_size_expr.append((expr_id, expr["inputs"][0]))
    else: raise Exception(f"Not implemented: {fn}")

  # def _expression_result_type(self, expr):
  #   fn = expr["function"]
  #   if fn == "greater_than" or fn == "less_than" or fn == "equal_integer" or fn == "exist" or \
  #      fn == "equal_color" or fn == "equal_material" or fn == "equal_shape" or fn == "equal_size":
  #     return "yn_result"
  #   elif fn == "count":
  #     return "num_result"
  #   elif fn == "query_size" or fn == "query_shape" or fn == "query_material" or fn == "query_color":
  #     return "query_result"
  #   else:
  #     raise Exception(f"Unknown result type for expression {expr}")

  def __repr__(self):
    return repr(self.__dict__)



class CLEVRVisionOnlyDataset(torch.utils.data.Dataset):
  def __init__(self, root: str, question_path: str, train: bool = True, device: str = "cpu"):
    self.name = "train" if train else "val"
    self.device = device

    self.scenes = json.load(open(os.path.join(dataset_dir, f"scenes/CLEVR_{self.name}_scenes.json")))['scenes']
    self.questions = json.load(open(os.path.join(root, question_path)))["questions"]

    print(question_path)
    self.image_paths = {question['image_index']: os.path.join(root, f"images/{self.name}/{question['image_filename']}") for question in self.questions}

  def __len__(self):
    return len(self.questions)

  def __getitem__(self, i):
    question = self.questions[i]

    # Generate the program
    clevr_program = CLEVRProgram(question["program"])

    # Get the answer
    answer = self._process_answer(question["answer"])

    scene = self.scenes[question['image_index']]
    image_path = self.image_paths[question['image_index']]
    image = torch.tensor(cv2.imread(image_path), dtype=torch.float)
    image = normalize_image(image)
    # image = image.permute(2, 0, 1)

    height = image.shape[1]
    width = image.shape[2]
    orig_boxes = extract_bounding_boxes(scene)
    pred_boxes = normalize_box(orig_boxes, height, width)

    # answer = {'size': [], 'shape': [], 'rela': [], 'material': [], 'color': []}
    # for obj in scene['objects']:
    #   answer['size'].append(all_sizes.index(obj['size']))
    #   answer['shape'].append(all_shapes.index(obj['shape']))
    #   answer['material'].append(all_mats.index(obj['material']))
    #   answer['color'].append(all_colors.index(obj['color']))
    scene_info = scene_to_answer(scene)

    # Return all of the information
    return (question['image_index'], clevr_program, image, orig_boxes.to(self.device), pred_boxes.to(self.device), answer, scene_info)

  @staticmethod
  def collate_fn(batch):
    batched_programs = []
    batched_rela_objs = []
    batched_images = []
    batched_answer = []
    batched_pred_bboxes = []
    batched_orig_bboxes = []
    batched_rela_objs = []
    batched_scenes = {}

    for _, question, image, orig_boxes, pred_boxes, answer, scene_info in batch:
      batched_programs.append(question)
      batched_pred_bboxes.append(pred_boxes)
      batched_orig_bboxes.append(orig_boxes)
      batched_answer.append(answer)
      batched_images.append(image)
      indexes = torch.tensor(range(pred_boxes.shape[0]))
      two_object_index = torch.tensor([(i, j) for (i, j) in torch.cartesian_prod(indexes, indexes) if not i == j])
      batched_rela_objs.append(two_object_index)

      for qt, info in scene_info.items():
        if not qt in batched_scenes:
          batched_scenes[qt] = []
        batched_scenes[qt].append(info)

    for qt, info in batched_scenes.items():
      batched_scenes[qt] = torch.cat(info)

    batch_split = []
    current_idx = 0
    for vec in batched_pred_bboxes:
      current_idx += vec.shape[0]
      batch_split.append(current_idx)

    return (batched_programs, batched_images, batched_orig_bboxes, batched_pred_bboxes, batched_rela_objs, batch_split), batched_answer, batched_scenes

  def _process_answer(self, answer):
    if answer == "yes": return True
    elif answer == "no": return False
    elif type(answer) == int: return answer
    elif answer.isdigit(): return int(answer)
    else: return answer


def clevr_vision_only_loader(root: str, train_question_path: str, test_question_path: str, batch_size: int, device: str):
  train_loader = torch.utils.data.DataLoader(CLEVRVisionOnlyDataset(root, train_question_path, True, device), collate_fn=CLEVRVisionOnlyDataset.collate_fn, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(CLEVRVisionOnlyDataset(root, test_question_path, False, device), collate_fn=CLEVRVisionOnlyDataset.collate_fn, batch_size=batch_size, shuffle=True)
  return (train_loader, test_loader)


class MLPClassifier(nn.Module):
  def __init__(self, input_dim, latent_dim, output_dim, n_layers, dropout_rate):
    super(MLPClassifier, self).__init__()

    layers = []
    layers.append(nn.Linear(input_dim, latent_dim))
    layers.append(nn.ReLU())
    layers.append(nn.BatchNorm1d(latent_dim))
    layers.append(nn.Dropout(dropout_rate))
    for _ in range(n_layers - 1):
      layers.append(nn.Linear(latent_dim, latent_dim))
      layers.append(nn.ReLU())
      layers.append(nn.BatchNorm1d(latent_dim))
      layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(latent_dim, output_dim))

    self.net = nn.Sequential(*layers)

  def forward(self, x):
    logits = self.net(x)
    return logits

class SceneGraphModel(nn.Module):
  def __init__(self, device):
    super(SceneGraphModel, self).__init__()
    self.device = device

    dim = 7
    hidden_dim = 24

    self.context_feature_extract = ConvInputModel()
    self.pooler = ROIPooler((7, 7), scales=[1/16], sampling_ratio=1, pooler_type="ROIAlignV2" )

    self.shape_clf = MLPClassifier(input_dim=dim * dim * hidden_dim + 4, output_dim=3, latent_dim=args.latent_dim, n_layers=args.model_layer, dropout_rate=0.2)
    self.color_clf = MLPClassifier(input_dim=dim * dim * hidden_dim + 4, output_dim=8, latent_dim=args.latent_dim, n_layers=args.model_layer, dropout_rate=0.2)
    self.size_clf = MLPClassifier(input_dim=dim * dim * hidden_dim + 4, output_dim=2, latent_dim=args.latent_dim, n_layers=args.model_layer, dropout_rate=0.2)
    self.mat_clf = MLPClassifier(input_dim=dim * dim * hidden_dim + 4, output_dim=2, latent_dim=args.latent_dim, n_layers=args.model_layer, dropout_rate=0.2) #
    self.rela_clf = MLPClassifier(input_dim=(dim * dim * hidden_dim + 4) * 2, output_dim=2, latent_dim=args.latent_dim, n_layers=args.rela_model_layer, dropout_rate=0.2) #layer 3

    # A dictionary indexing these classifiers
    self.models = {"shape": self.shape_clf, "color": self.color_clf, "size": self.size_clf, "mat": self.mat_clf, "relate": self.rela_clf}

  def batch_predict(self, relation, features, batch_split):

    model = self.models[relation]
    logits = model(features)
    current_split = 0
    probs = []
    for split in batch_split:
      current_logits = logits[current_split:split]
      if relation == 'relate':
        current_probs = F.sigmoid(current_logits)
      else:
        current_probs = F.softmax(current_logits, dim=1)
      probs.append(current_probs)
      current_split = split
    return torch.cat(probs).reshape(features.shape[0], -1)

  def forward(self, features, orig_bboxes, pred_bboxes, batched_rela_objs, batch_split):
    splits = [(0, r) if i == 0 else (batch_split[i - 1], r) for (i, r) in enumerate(batch_split)]
    objs = [[(o,) for o in range(end - begin)] for (begin, end) in splits]

    object_features = []
    features = torch.stack(features)
    # bboxes = torch.stack(bboxes)
    # bbox_size = torch.stack([torch.stack([torch.abs(x2-x1), torch.abs(y2-y1)]) for x1, y1, x2, y2 in bboxes])
    image_features = self.context_feature_extract(features)
    object_features = self.pooler([image_features], orig_bboxes)
    object_features = object_features.reshape(object_features.shape[0], -1)

    pred_bboxes = torch.cat(pred_bboxes)
    batched_features = torch.cat((object_features, pred_bboxes), dim=1)
    # batched_flattened_features = torch.cat((features.reshape(bboxes.shape[0], -1), bboxes, bbox_size), dim=1)

    current_split = 0
    batched_rela_vecs = []
    batch_rela_split = []
    for split_ct, split in enumerate(batch_split):
      two_object_index = batched_rela_objs[split_ct]
      two_object_indices = two_object_index.reshape(-1) + current_split
      rela_vecs = torch.index_select(batched_features, 0, two_object_indices).reshape(len(two_object_index), -1)
      batched_rela_vecs.append(rela_vecs)
      current_split = split

    current_idx = 0
    for vec in batched_rela_vecs:
      current_idx += vec.shape[0]
      batch_rela_split.append(current_idx)

    batched_rela_vecs = torch.cat(batched_rela_vecs)
    batched_rela_objs = torch.cat(batched_rela_objs)

    shape_prob = self.batch_predict("shape", batched_features, batch_split)
    color_prob = self.batch_predict("color", batched_features, batch_split)
    mat_prob = self.batch_predict("mat", batched_features, batch_split)
    size_prob = self.batch_predict("size", batched_features, batch_split)
    rela_prob = self.batch_predict("relate", batched_rela_vecs, batch_rela_split)

    return shape_prob, color_prob, mat_prob, size_prob, rela_prob, objs, batched_rela_objs, batch_rela_split,


class CLEVRVisionOnlyNet(nn.Module):
  program_relations = [
    "equal_material_expr", "same_size_expr", "exists_expr", "root_expr",
    "count_expr", "and_expr", "or_expr", "query_color_expr",
    "relate_expr", "equal_shape_expr", "less_than_expr", "equal_expr", "equal_color_expr",
    "greater_than_expr", "query_size_expr", "query_shape_expr", "unique_expr", "filter_color_expr",
    "equal_size_expr", "scene_expr", "filter_shape_expr", "same_color_expr",
    "same_material_expr", "filter_material_expr", "same_shape_expr", "query_material_expr",
    "filter_size_expr", "obj"
  ]
  scene_graph_relations = [
    "shape", "color", "material", "size", "relate"
  ]
  relations = program_relations + scene_graph_relations

  def __init__(self, device, k):
    super(CLEVRVisionOnlyNet, self).__init__()

    # Setup scene graph model
    self.sg_model = SceneGraphModel(device)

    # Setup scallopy context
    self.ctx = scallopy.ScallopContext("difftopbottomkclauses", k=k)
    # self.ctx = scallopy.ScallopContext("diffminmaxprob")
    self.ctx.import_file(os.path.abspath(os.path.join(os.path.abspath(__file__), "../scl/clevr_eval_vision_only.scl")))
    self.ctx.set_non_probabilistic(self.program_relations)

    # Setup scallopy forward function
    # self.reason = self.ctx.forward_function(dispatch='single', debug_provenance=True)
    # self.reason = self.ctx.forward_function()
    # self.reason = self.ctx.forward_function(output="result")
    self.reason = self.ctx.forward_function(output="result", output_mapping=all_answers)


  def prob_mat_to_clauses(self, batch_prob, batch_split, all_elements):
    splits = [(0, r) if i == 0 else (batch_split[i - 1], r) for (i, r) in enumerate(batch_split)]
    batched_facts = [[(batch_prob[i, j], (i - begin, e)) for i in range(begin, end) for (j, e) in enumerate(all_elements)] for (begin, end) in splits]
    return batched_facts

  def prob_mat_to_rela_clauses(self, batch_prob, batched_rela_objs, batch_split, all_elements):
    splits = [(0, r) if i == 0 else (batch_split[i - 1], r) for (i, r) in enumerate(batch_split)]
    batched_facts = [[(batch_prob[i, j], (e, batched_rela_objs[i, 0], batched_rela_objs[i, 1])) for i in range(begin, end) for (j, e) in enumerate(all_elements)] for (begin, end) in splits]
    return batched_facts

  # Fake a sg output
  def forward(self, x, ys):
    (programs, images, orig_boxes, pred_boxes, rela_objs, batch_split) = x
    output_relations = ["result" for p in programs]
    shape_prob, color_prob, mat_prob, size_prob, rela_prob, objs, batched_rela_objs, batch_rela_split, = self.sg_model(images, orig_boxes, pred_boxes, rela_objs, batch_split)
    shapes = self.prob_mat_to_clauses(shape_prob, batch_split, all_shapes)
    colors = self.prob_mat_to_clauses(color_prob, batch_split, all_colors)
    mats = self.prob_mat_to_clauses(mat_prob, batch_split, all_mats)
    sizes = self.prob_mat_to_clauses(size_prob, batch_split, all_sizes)
    relas = self.prob_mat_to_rela_clauses(rela_prob, batched_rela_objs, batch_rela_split, all_relas)
    scene_graphs = [{"obj": objs, "shape": shapes, "color": colors, "material": mats, "size": sizes, "relate": relas} for objs, shapes, colors, mats, sizes, relas in zip(objs, shapes, colors, mats, sizes, relas)]

    list_facts = [{**program.facts(), **scene_graph} for (program, scene_graph) in zip(programs, scene_graphs)]
    facts = {k: [fs[k] for fs in list_facts] for k in self.relations}

    disjunctions = {}
    for sg_key, batched_sg_facts in facts.items():
      if sg_key in ['shape', 'color', 'material', 'size']:
        if not sg_key in disjunctions:
          disjunctions[sg_key] = []

        for sg_facts in batched_sg_facts:
          disj_group = {}
          for fid, (prob, sg_fact) in enumerate(sg_facts):
            if not sg_key == 'relate':
                oid, _ = sg_fact
                if not oid in disj_group:
                    disj_group[oid] = []
                disj_group[oid].append(fid)
          disjunctions[sg_key].append(list(disj_group.values()))

    # y_pred_values, y_pred_probs = self.reason(output_relations=output_relations, **facts, disjunctions=disjunctions)
    # y_pred_values, y_pred_probs = self.reason(output_relations=output_relations, **facts)

    y_pred_probs = self.reason(**facts)
    y_pred_values = all_answers

    missing_pred = []
    for y in ys:
      if not str(y) in y_pred_values:
        missing_pred.append(y)
    if not len(missing_pred) == 0:
      to_extend = torch.zeros(y_pred_probs.shape[0], len(missing_pred))
      y_pred_probs = torch.cat((y_pred_probs, to_extend), dim=1)
      y_pred_values = y_pred_values + missing_pred

    y_pred_probs [y_pred_probs == 0] = float('-inf')
    softmaxed_probs = F.softmax(y_pred_probs, dim=1)
    softmaxed_probs = torch.nan_to_num(softmaxed_probs, 0)
    
    return y_pred_values, softmaxed_probs


class Trainer():
  def __init__(self, train_loader, test_loader, device, model_root, learning_rate, k, phase = "train"):
    self.device = device
    if not model_root == None and os.path.exists(model_root + '.best.pt'):
      self.network = torch.load(open(model_root + '.latest.pt', 'rb'))
    else:
      self.network = CLEVRVisionOnlyNet(device, k=k).to(device)

    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.loss_fn = F.binary_cross_entropy
    self.model_root = model_root
    self.min_test_loss = 100000000.0
    self.phase = phase

  def _loss_fn(self, y_pred_values, y_pred_probs, y_values):
    # y = torch.stack([torch.tensor([1.0 if u[0] == v else 0.0 for u in y_pred_values]) for v in y_values]).to(self.device)
    # print(y_pred_probs)
    
    y_batched = []
    for v in y_values:
      y = []
      recorded = False
      for u in y_pred_values:
        if type(u) == tuple:
          assert len(u) == 1
          u = u[0]
          
        if str(u) == str(v):
          y.append(torch.tensor([1.0]))
          if recorded: 
            print('here')
          recorded = True
        else: 
          y.append(torch.tensor([0.0]))
          
      y_batched.append(torch.stack(y))

    y = torch.stack(y_batched).to(self.device)
    y = y.reshape(y_pred_probs.shape[0], y_pred_probs.shape[1])
    
    assert torch.sum(y) == y.shape[0]
    return self.loss_fn(y_pred_probs, y)

  def _num_correct(self, y_pred_values, y_pred_probs, y_values):
    indices = torch.argmax(y_pred_probs, dim=1).to("cpu")
    predicted = [y_pred_values[i] for i in indices]
    return sum([1 if x == y else 0 for (x, y) in zip(predicted, y_values)])

  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    train_loss = 0
    total_correct = 0
    iterator = tqdm(self.train_loader, total=len(self.train_loader))
    for (i, (x, y, _)) in enumerate(iterator):
      batch_size = len(y)

      # Do the prediction and obtain the loss/accuracy
      (y_pred_values, y_pred_probs) = self.network(x, y)
      loss = self._loss_fn(y_pred_values, y_pred_probs, y)
      num_correct = self._num_correct(y_pred_values, y_pred_probs, y)

      # Compute loss
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      # Stats
      train_loss += loss.detach()
      num_items += batch_size
      total_correct += num_correct
      perc = 100. * total_correct / num_items
      avg_loss = train_loss / (i + 1)

      # Prints
      iterator.set_description(f"[Train Epoch {epoch}] Avg loss: {avg_loss:.4f}, Accuracy: {total_correct}/{num_items} ({perc:.2f}%)")
    torch.save(self.network, self.model_root + '.latest.pt')

  def test_epoch(self, epoch, save_model=True):
    self.network.eval()
    num_items = 0
    test_loss = 0
    total_correct = 0
    with torch.no_grad():
      iterator = tqdm(self.test_loader, total=len(self.test_loader))
      for i, (x, y, _) in enumerate(iterator):
        batch_size = len(y)

        # Do the prediction and obtain the loss/accuracy
        (y_pred_values, y_pred_probs) = self.network(x, y)
        loss = self._loss_fn(y_pred_values, y_pred_probs, y)
        num_correct = self._num_correct(y_pred_values, y_pred_probs, y)

        # Stats
        test_loss += loss.detach()
        num_items += batch_size
        total_correct += num_correct
        perc = 100. * total_correct / num_items
        avg_loss = test_loss / (i + 1)

        # Prints
        iterator.set_description(f"[Test Epoch {epoch}] Avg loss: {avg_loss:.4f}, Accuracy: {total_correct}/{num_items} ({perc:.2f}%)")

    # Save model
    if test_loss < self.min_test_loss:
      self.min_test_loss = test_loss
      if save_model:
        torch.save(self.network, self.model_root + '.best.pt')

  def test_scene(self, epoch,):
    self.network.eval()
    num_items = {}
    test_loss = {}
    total_correct = {}
    with torch.no_grad():
      iterator = tqdm(self.test_loader, total=len(self.test_loader))
      for i, (x, _, s) in enumerate(iterator):

        (programs, features, orig_bboxes, pred_bboxes, batched_rela_objs, batch_split) = x
        shape_prob, color_prob, mat_prob, size_prob, rela_prob, _, _, _ = self.network.sg_model(features, orig_bboxes, pred_bboxes, batched_rela_objs, batch_split)
        pred_scene_graphs = {'shape': shape_prob, 'color':color_prob, 'material':mat_prob, 'size':size_prob, 'relate':rela_prob}
        batch_size = {attr: len(info) for attr, info in pred_scene_graphs.items()}

        loss = {}
        num_correct = {}
        perc = {}
        avg_loss = {}

        for attribute in pred_scene_graphs.keys():
          loss[attribute] = self.loss_fn(pred_scene_graphs[attribute], s[attribute])
          num_correct[attribute] = scene_num_correct(pred_scene_graphs[attribute], s[attribute], attribute)

          if not attribute in test_loss:
            test_loss[attribute] = 0
            num_items[attribute] = 0
            total_correct[attribute] = 0

          test_loss[attribute] += loss[attribute].detach()
          num_items[attribute] += batch_size[attribute]
          total_correct[attribute] += num_correct[attribute] if type(num_correct[attribute]) == int else num_correct[attribute].detach()
          perc[attribute] = 100. * total_correct[attribute] / num_items[attribute]
          avg_loss[attribute] = test_loss[attribute] / (i + 1)

        # Prints
        iterator.set_description(f"[Test Epoch {epoch}] avg loss: {avg_loss}, Accuracy: ({perc})")
      print(f'finish test scene: {perc}')

  def train(self, n_epochs):
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)


if __name__ == "__main__":
  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  dataset_dir = os.path.join(data_dir, "CLEVR")
  model_dir = os.path.join(dataset_dir, "models")

  # Argument parser
  # Argument parser
  parser = ArgumentParser("clevr_vision_only")
  parser.add_argument("--phase", type=str, default='train')
  parser.add_argument("--n-epochs", type=int, default=1000)

  # setup question path
  parser.add_argument("--train_question_type", type=str, default="orig_clevr")
  parser.add_argument("--test_question_type", type=str, default="scene")
  parser.add_argument("--train_num", type=int, default=10000)
  parser.add_argument("--val_num", type=int, default=1000)
  parser.add_argument("--train_max_obj", type=int, default=5)
  parser.add_argument("--train_max_clause", type=int, default=10)
  parser.add_argument("--test_max_obj", type=int, default=100)
  parser.add_argument("--test_max_clause", type=int, default=100)
  parser.add_argument("--attr_question_count", type=int, default=5)
  parser.add_argument("--rela_question_count", type=int, default=5)

  # Training hyperparameters
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--latent-dim", type=float, default=1024)
  parser.add_argument("--model_layer", type=int, default=2)
  parser.add_argument("--rela_model_layer", type=int, default=3)
  parser.add_argument("--max_obj", type=float, default=5)
  parser.add_argument("--max_clause", type=float, default=5)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopbottomkclauses")
  parser.add_argument("--train-top-k", type=int, default=3)
  parser.add_argument("--test-top-k", type=int, default=1)
  parser.add_argument("--model_path", type=str, default="")
  parser.add_argument("--dataset_dir", type=str, default=dataset_dir)
  parser.add_argument("--use-cuda", action="store_true")
  parser.add_argument("--gpu", type=int, default=0)
  args = parser.parse_args()

  # Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  device = f"cuda:{args.gpu}" if args.use_cuda else "cpu"
  model_name = f"vision_only_model_{args.learning_rate}_topbotk_{args.train_top_k}_obj_{args.max_obj}_clause_{args.max_clause}_train_{args.train_num}_model_{args.model_layer}_rmodel_{args.rela_model_layer}_latent_{args.latent_dim}_{args.train_question_type}"
  args.model_path = os.path.join(model_dir, model_name)

  if args.train_question_type == "count_only":
    train_question_path =  f"questions/CLEVR_train_all_obj_{args.train_max_obj}_clause_{args.train_max_clause}_aqct_{args.attr_question_count}_uniq_rqct_{args.rela_question_count}_image_split.cropped_{args.train_num}.json"
  elif args.train_question_type == "orig_clevr":
    train_question_path = f"questions/CLEVR_train_questions_obj_{args.train_max_obj}_clause_{args.train_max_clause}_image_split.cropped_{args.train_num}.json"
  else:
    print("Warning: wrong train question type!")
    exit(1)

  if args.test_question_type == "count_only":
    test_question_path =  f"questions/CLEVR_val_all_obj_{args.test_max_obj}_clause_{args.test_max_clause}_aqct_{args.attr_question_count}_uniq_rqct_{args.rela_question_count}_image_split.cropped_{args.val_num}.json"
  elif args.test_question_type == "orig_clevr" or args.test_question_type == "scene":
    test_question_path = f"questions/CLEVR_val_questions_obj_{args.test_max_obj}_clause_{args.test_max_clause}_image_split.cropped_{args.val_num}.json"
  else:
    print("Warning: wrong test question type!")
    exit(1)

  # Dataloaders
  train_loader, test_loader = clevr_vision_only_loader(args.dataset_dir, train_question_path, test_question_path, args.batch_size, device)

  if args.phase == 'train':
    trainer = Trainer(train_loader, test_loader, device, args.model_path, args.learning_rate, args.train_top_k)
    trainer.train(args.n_epochs)
  else:
    trainer = Trainer(train_loader, test_loader, device, args.model_path, args.learning_rate, args.test_top_k)
    if args.test_question_type == "scene":
      trainer.test_scene(0)
    else:
      trainer.test_epoch(0, save_model=False)

  print('end')

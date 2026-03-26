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
      self.result_type = self._expression_result_type(expr)

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

  def _expression_result_type(self, expr):
    fn = expr["function"]
    if fn == "greater_than" or fn == "less_than" or fn == "equal_integer" or fn == "exist" or \
       fn == "equal_color" or fn == "equal_material" or fn == "equal_shape" or fn == "equal_size":
      return "yn_result"
    elif fn == "count":
      return "num_result"
    elif fn == "query_size" or fn == "query_shape" or fn == "query_material" or fn == "query_color":
      return "query_result"
    else:
      raise Exception(f"Unknown result type for expression {expr}")

  def __repr__(self):
    return repr(self.__dict__)


class CLEVRVisionOnlyDataset(torch.utils.data.Dataset):
  def __init__(self, root: str, postfix: str = "", train: bool = True, feature_dim: int = 1024, num=100, device: str = "cpu"):
    self.name = "train" if train else "val"
    self.postfix = postfix
    self.device = device
    # self.feature_dim = 12544
    max_obj = 5
    self.feature_dim = feature_dim
    # Load json dataset
    self.feature_path = os.path.join(root, f"feature_vecs/{self.name}/features.cropped_{num}_dim_{self.feature_dim}_bw_True.pkl")
    self.questions = json.load(open(os.path.join(root, f"questions/CLEVR_{self.name}_agg_questions_obj_{max_obj}.cropped_{num}.json")))["questions"]
    self.image_info = pickle.load(open(self.feature_path, 'rb'))
    self.image_paths = {question['image_index']: os.path.join(root, f"images/{self.name}/{question['image_filename']}") for question in self.questions}

  def __len__(self):
    return len(self.questions)

  def __getitem__(self, i):
    question = self.questions[i]

    # Generate the program
    clevr_program = CLEVRProgram(question["program"])

    # Load the image object features
    # image_path = self.image_paths[i]
    # image = cv2.imread(image_path)
    # pred_boxes, feature_vecs = self._preprocess_image(image)
    pred_boxes, feature_vecs = self.image_info[question['image_index']]

    # Get the answer
    answer = self._process_answer(question["answer"])

    # Return all of the information
    return (question['image_index'], clevr_program, pred_boxes.to(self.device), feature_vecs.to(self.device), answer)

  @staticmethod
  def collate_fn(batch):
    batched_programs = []
    batched_feature_vecs = []
    batched_size_vecs = []
    batched_rela_vecs = []
    batched_rela_objs = []
    batch_image_idx = []

    batched_answer = []
    for image_idx, clevr_program, pred_boxes, feature_vecs, answer in batch:
      batch_image_idx.append(image_idx)
      batched_programs.append(clevr_program)
      size_vecs = torch.cat((feature_vecs, pred_boxes), dim=1)
      batched_size_vecs.append(size_vecs)
      batched_answer.append(answer)

    batch_split = []
    current_idx = 0
    for vec in batched_size_vecs:
      current_idx += vec.shape[0]
      batch_split.append(current_idx)

    batched_size_vecs = torch.cat(batched_size_vecs)
    return (batch_image_idx, batched_programs, batched_size_vecs, batch_split), batched_answer

  def _process_answer(self, answer):
    if answer == "yes": return True
    elif answer == "no": return False
    else: return answer


def clevr_vision_only_loader(root: str, postfix: str, batch_size: int, feature_dim, train_num, test_num, device: str):
  train_loader = torch.utils.data.DataLoader(CLEVRVisionOnlyDataset(root, postfix, True, feature_dim, train_num, device), collate_fn=CLEVRVisionOnlyDataset.collate_fn, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(CLEVRVisionOnlyDataset(root, postfix, False, feature_dim, test_num, device), collate_fn=CLEVRVisionOnlyDataset.collate_fn, batch_size=batch_size, shuffle=True)
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
  def __init__(self, feat_dim, device):
    super(SceneGraphModel, self).__init__()
    self.feat_dim = feat_dim
    self.device = device

    # Classifiers
    self.shape_clf = MLPClassifier(input_dim=self.feat_dim + 4, output_dim=2, latent_dim=args.latent_dim, n_layers=args.model_layer, dropout_rate=0.5)

    # A dictionary indexing these classifiers
    self.models = {"shape": self.shape_clf}

  def batch_predict(self, relation, inputs, batch_split):
    model = self.models[relation]
    logits = model(inputs)
    current_split = 0
    probs = []
    for split in batch_split:
      current_logits = logits[current_split:split]
      current_probs = F.softmax(current_logits, dim=1)
      probs.append(current_probs)
      current_split = split
    return torch.cat(probs).reshape(inputs.shape[0], -1)

  def prob_mat_to_clauses(self, batch_prob, batch_split, all_elements):
    splits = [(0, r) if i == 0 else (batch_split[i - 1], r) for (i, r) in enumerate(batch_split)]
    batched_facts = [[(batch_prob[i, j], (i - begin, e)) for i in range(begin, end) for (j, e) in enumerate(all_elements)] for (begin, end) in splits]
    return batched_facts

  def prob_mat_to_rela_clauses(self, batch_prob, batched_rela_objs, batch_split, all_elements):
    splits = [(0, r) if i == 0 else (batch_split[i - 1], r) for (i, r) in enumerate(batch_split)]
    batched_facts = [[(batch_prob[i, j], (e, batched_rela_objs[i, 0], batched_rela_objs[i, 1])) for i in range(begin, end) for (j, e) in enumerate(all_elements)] for (begin, end) in splits]
    return batched_facts

  def forward(self, batched_size_vecs, batch_split):
    splits = [(0, r) if i == 0 else (batch_split[i - 1], r) for (i, r) in enumerate(batch_split)]
    objs = [[(o,) for o in range(end - begin)] for (begin, end) in splits]

    shape_prob = self.batch_predict("shape", batched_size_vecs, batch_split)
    shapes = self.prob_mat_to_clauses(shape_prob, batch_split, ['cylinder'])
    return [{"obj": objs, "shape": shapes} for objs, shapes in zip(objs, shapes)]


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

  def __init__(self, device, feature_dim, k=3):
    super(CLEVRVisionOnlyNet, self).__init__()

    # Setup scene graph model
    self.sg_model = SceneGraphModel(feature_dim, device)

    # Setup scallopy context
    self.ctx = scallopy.ScallopContext("difftopbottomkclauses", k=k)
    # self.ctx = scallopy.ScallopContext("diffminmaxprob")
    self.ctx.import_file(os.path.abspath(os.path.join(os.path.abspath(__file__), "../scl/clevr_eval_vision_only.scl")))
    self.ctx.set_non_probabilistic(self.program_relations)

    # Setup scallopy forward function
    self.reason = self.ctx.forward_function()

  def forward(self, x):
    (_, programs, size_vecs, split, ) = x
    output_relations = [p.result_type for p in programs]
    scene_graphs = self.sg_model(size_vecs, split)
    list_facts = [{**program.facts(), **scene_graph} for (program, scene_graph) in zip(programs, scene_graphs)]
    facts = {k: [fs[k] if k in fs else [] for fs in list_facts ] for k in self.relations}
    results = self.reason(output_relations=output_relations, **facts)
    return results


class Trainer():
  def __init__(self, train_loader, test_loader, device, model_root, learning_rate, k, feature_dim, phase = "train"):
    self.device = device
    if not model_root == None and os.path.exists(model_root + '.best.pt'):
      self.network = torch.load(open(model_root + '.best.pt', 'rb'))
    else:
      self.network = CLEVRVisionOnlyNet(device, feature_dim=feature_dim, k=k).to(device)

    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.loss_fn = F.binary_cross_entropy
    self.model_root = model_root
    self.min_test_loss = 100000000.0
    self.phase = phase

  def _loss_fn(self, y_pred_values, y_pred_probs, y_values):
    y = torch.stack([torch.tensor([1.0 if str(u[0]) == str(v) else 0.0 for u in y_pred_values]) for v in y_values]).to(self.device)
    return self.loss_fn(y_pred_probs, y)

  def _num_correct(self, y_pred_values, y_pred_probs, y_values):
    indices = torch.argmax(y_pred_probs, dim=1).to("cpu")
    predicted = [y_pred_values[i][0] for i in indices]
    return sum([1 if str(x) == str(y) else 0 for (x, y) in zip(predicted, y_values)])

  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    train_loss = 0
    total_correct = 0
    iterator = tqdm(self.train_loader, total=len(self.train_loader))
    for (i, (x, y)) in enumerate(iterator):
      batch_size = len(y)

      # Do the prediction and obtain the loss/accuracy
      (y_pred_values, y_pred_probs) = self.network(x)
      loss = self._loss_fn(y_pred_values, y_pred_probs, y)
      num_correct = self._num_correct(y_pred_values, y_pred_probs, y)

      # Compute loss
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      # Stats
      train_loss += loss
      num_items += batch_size
      total_correct += num_correct
      perc = 100. * total_correct / num_items
      avg_loss = train_loss / (i + 1)

      # Prints
      iterator.set_description(f"[Train Epoch {epoch}] Avg loss: {avg_loss:.4f}, Accuracy: {total_correct}/{num_items} ({perc:.2f}%)")
    torch.save(self.network, self.model_root + '.latest.pt')

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = 0
    test_loss = 0
    total_correct = 0
    with torch.no_grad():
      iterator = tqdm(self.test_loader, total=len(self.test_loader))
      for i, (x, y) in enumerate(iterator):
        batch_size = len(y)

        # Do the prediction and obtain the loss/accuracy
        (y_pred_values, y_pred_probs) = self.network(x)
        loss = self._loss_fn(y_pred_values, y_pred_probs, y)
        num_correct = self._num_correct(y_pred_values, y_pred_probs, y)

        # Stats
        test_loss += loss
        num_items += batch_size
        total_correct += num_correct
        perc = 100. * total_correct / num_items
        avg_loss = test_loss / (i + 1)

        # Prints
        iterator.set_description(f"[Test Epoch {epoch}] Avg loss: {avg_loss:.4f}, Accuracy: {total_correct}/{num_items} ({perc:.2f}%)")

    # Save model
    if test_loss < self.min_test_loss and self.phase == "train":
      self.min_test_loss = test_loss
      torch.save(self.network, self.model_root + '.best.pt')

  def predict(self, save_image_dir):
    self.network.eval()
    iterator = tqdm(self.test_loader, total=len(self.test_loader))
    image_paths = self.test_loader.dataset.image_paths

    for i, (x, y) in enumerate(iterator):

      # Do the prediction and obtain the loss/accuracy
      (image_idxes, programs, size_vecs, batch_split, ) = x
      batched_image_paths = [image_paths[image_idx] for image_idx in image_idxes]
      save_image_paths = [os.path.join(save_image_dir, str(image_idx) + '.png') for image_idx in image_idxes]
      images = [cv2.imread(image_path) for image_path in batched_image_paths]

      # recover batched boxes
      height = images[0].shape[0]
      width = images[0].shape[1]
      norm_bboxes = size_vecs[:, -4:]
      batched_bboxes = recover_box(norm_bboxes, height, width)
      bboxes = []
      current_split = 0
      for split in batch_split:
        current_bboxes = batched_bboxes[current_split:split]
        bboxes.append(current_bboxes)
        current_split = split

      scene_graphs = self.network.sg_model(size_vecs, batch_split)
      im_shapes = [scene_graph['shape'] for scene_graph in scene_graphs]
      pred_shape_probs = {}

      for image, im_shape, bbox, save_im_path in zip(images, im_shapes, bboxes, save_image_paths):

        pred_shape_probs = {}
        for obj_shape in im_shape:
          obj_id = obj_shape[1][0]
          shape_prob = obj_shape[0]
          pred_shape_probs[obj_id] = "%.2f" % shape_prob.item()

        save_image(image, bbox, pred_shape_probs, save_im_path)

    print("here")

  def train(self, n_epochs):
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)

  def test(self):
    self.test_epoch(0)



if __name__ == "__main__":
  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  dataset_dir = os.path.join(data_dir, "CLEVR")
  model_dir = os.path.join(dataset_dir, "models")
  save_image_dir = "/home/jianih/research/scallop-v2/experiments/data/CLEVR/diagnose_images"

  # Argument parser
  parser = ArgumentParser("clevr_vision_only")
  parser.add_argument("--n-epochs", type=int, default=1000)
  parser.add_argument("--train_num", type=int, default=10000)
  parser.add_argument("--val_num", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--feature-dim", type=float, default=12544)
  parser.add_argument("--latent-dim", type=float, default=1024)
  parser.add_argument("--model_layer", type=int, default=2)
  parser.add_argument("--is_bw", type=float, default=False)
  parser.add_argument("--max_obj", type=float, default=5)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=15)
  parser.add_argument("--dataset-postfix", type=str, default="")
  parser.add_argument("--model_path", type=str, default="")
  parser.add_argument("--dataset_dir", type=str, default=dataset_dir)
  parser.add_argument("--use-cuda", action="store_true")
  parser.add_argument("--gpu", type=int, default=0)
  args = parser.parse_args()

  # Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  device = f"cuda:{args.gpu}" if args.use_cuda else "cpu"
  model_name = f"cylinder_only_model_{args.learning_rate}_topbotk_{args.top_k}_dim_{args.feature_dim}_bw_{args.is_bw}_obj_{args.max_obj}_train_{args.train_num}_val_{args.val_num}_model_{args.model_layer}_latent_{args.latent_dim}"
  args.model_path = os.path.join(model_dir, model_name)

  # Dataloaders
  train_loader, test_loader = clevr_vision_only_loader(args.dataset_dir, args.dataset_postfix, args.batch_size,  args.feature_dim, args.train_num, args.val_num, device)
  trainer = Trainer(train_loader, test_loader, device, args.model_path, args.learning_rate, args.top_k, args.feature_dim)
  # trainer.train(args.n_epochs)
  # trainer.test()
  trainer.predict(save_image_dir)

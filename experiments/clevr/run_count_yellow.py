import os
import json
import torch
import cv2
import random
from argparse import ArgumentParser

def normalize_image(image):
  mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.224]).reshape(3, 1, 1)
  image = image.permute(2, 0, 1)

  image = (image / 255.0 - mean) / std
  return image

def normalize_box(pred_boxes, height, width):
    new_pred_boxes = []
    for pred_box in pred_boxes:
      x1 = pred_box[0] / width
      y1 = pred_box[1] / height
      x2 = pred_box[2] / width
      y2 = pred_box[3] / height
      new_pred_boxes.append([x1, y1, x2, y2])
    return torch.tensor(new_pred_boxes)

class CLEVRVisionOnlyDataset(torch.utils.data.Dataset):
  def __init__(self, root: str, question_path: str, train: bool = True, device: str = "cpu"):
    self.name = "train" if train else "val"
    self.device = device
    self.questions = json.load(open(os.path.join(root, question_path)))["questions"]
    self.image_paths = {question['image_index']: os.path.join(root, f"images/{self.name}/{question['image_filename']}") for question in self.questions}

  def __len__(self):
    return len(self.questions)

  def __getitem__(self, i):
    question = self.questions[i]
    clevr_program = question["program"]
    answer = self._process_answer(question["answer"])

    image_path = self.image_paths[question['image_index']]
    image = torch.tensor(cv2.imread(image_path), dtype=torch.float)
    image = normalize_image(image)

    height = image.shape[1]
    width = image.shape[2]
    orig_boxes = torch.tensor(question['bounding_boxes']).to(self.device)
    pred_boxes = torch.tensor(normalize_box(orig_boxes, height, width)).to(self.device)

    # Return all of the information
    return (clevr_program, image, orig_boxes, pred_boxes, answer)

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

    for question, image, orig_boxes, pred_boxes, answer in batch:
      batched_programs.append(question)
      batched_pred_bboxes.append(pred_boxes)
      batched_orig_bboxes.append(orig_boxes)
      batched_answer.append(answer)
      batched_images.append(image)
      indexes = torch.tensor(range(pred_boxes.shape[0]))
      two_object_index = torch.tensor([(i, j) for (i, j) in torch.cartesian_prod(indexes, indexes) if not i == j])
      batched_rela_objs.append(two_object_index)

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


if __name__ == "__main__":
  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  dataset_dir = os.path.join(data_dir, "CLEVR/ss_lab")
  model_dir = os.path.join(dataset_dir, "models")

  # Argument parser
  # Argument parser
  parser = ArgumentParser("clevr_vision_only")
  parser.add_argument("--phase", type=str, default='train')
  parser.add_argument("--n-epochs", type=int, default=20)

  # setup question path
  parser.add_argument("--train_question_type", type=str, default="count_only")
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
  model_name = f"count_yellow_model"
  args.model_path = os.path.join(model_dir, model_name)

  train_question_path =  f"questions/count_yellow_train.json"
  test_question_path =  f"questions/count_yellow_val.json"

  # Dataloaders
  train_loader, test_loader = clevr_vision_only_loader(args.dataset_dir, train_question_path, test_question_path, args.batch_size, device)

  for batch in train_loader:
    print('here')
    break

#   if args.phase == 'train':
#     trainer = Trainer(train_loader, test_loader, device, args.model_path, args.learning_rate, args.train_top_k)
#     trainer.train(args.n_epochs)
#   else:
#     trainer = Trainer(train_loader, test_loader, device, args.model_path, args.learning_rate, args.test_top_k)
#     if args.test_question_type == "scene":
#       trainer.test_scene(0)
#     else:
#       trainer.test_epoch(0, save_model=False)

  print('end')

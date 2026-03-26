import os
import json
from argparse import ArgumentParser

import torch
import scallopy
from tqdm import tqdm

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


class CLEVRScene:
  def __init__(self, scene):
    # objects
    self.obj = []

    # Basic information
    self.color = []
    self.material = []
    self.shape = []
    self.size = []

    # Relations
    self.relate = []

    # Load objects
    for (i, obj) in enumerate(scene["objects"]):
      # Objects
      self.obj.append((i,))

      # Attributes
      self.color.append((i, obj["color"]))
      self.material.append((i, obj["material"]))
      self.shape.append((i, obj["shape"]))
      self.size.append((i, obj["size"]))

    # Load relations
    for relation in scene["relationships"]:
      for (i, related_objs) in enumerate(scene["relationships"][relation]):
        for j in related_objs:
          self.relate.append((relation, i, j))

  def __repr__(self):
    return repr(self.__dict__)


class CLEVRSanityCheckDataset(torch.utils.data.Dataset):
  def __init__(self, root: str, postfix: str = "", train: bool = True):
    self.name = "train" if train else "val"
    self.postfix = postfix
    self.scenes = json.load(open(os.path.join(root, f"scenes/CLEVR_{self.name}_scenes.json")))
    self.questions = json.load(open(os.path.join(root, f"questions/CLEVR_{self.name}_questions{self.postfix}.json")))

  def __len__(self):
    return len(self.questions["questions"])

  def __getitem__(self, i):
    question = self.questions["questions"][i]
    img_id = question["image_index"]
    clevr_program = CLEVRProgram(question["program"])
    scene = CLEVRScene(self.scenes["scenes"][img_id])
    nl_question = question["question"]
    answer = self._process_answer(question["answer"])
    return (clevr_program, scene, nl_question, answer)

  def _process_answer(self, answer):
    if answer == "yes": return True
    elif answer == "no": return False
    else: return answer


class CLEVRScallop:
  def __init__(self):
    self.ctx = scallopy.ScallopContext()
    self.ctx.import_file(os.path.abspath(os.path.join(os.path.abspath(__file__), "../scl/clevr_eval.scl")))

  def check_datapoint(self, i, datapoint):
    (program, scene, nl_question, answer) = datapoint

    # Clone the context
    temp_ctx = self.ctx.clone()

    # Add the program as facts
    for (relation, facts) in program.__dict__.items():
      if relation == "result_type": continue
      temp_ctx.add_facts(relation, facts)

    # Add the scene as facts
    for (relation, facts) in scene.__dict__.items():
      temp_ctx.add_facts(relation, facts)

    # Run the context
    temp_ctx.run()

    # Get the output
    result = list(temp_ctx.relation(program.result_type))[0][0]
    if str(result) != str(answer):
      print(f"{i}. {nl_question}")
      print(f"Expected {answer}, got {result}")
      print(program)
      print(scene)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("clevr_sanity_check")
  parser.add_argument("--dataset-postfix", type=str, default="")
  args = parser.parse_args()

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  dataset_dir = os.path.join(data_dir, "CLEVR")

  # Load dataset
  dataset = CLEVRSanityCheckDataset(dataset_dir, args.dataset_postfix)

  # Load scallop program
  runner = CLEVRScallop()

  # For each datapoint
  for i in tqdm(range(len(dataset))):
    runner.check_datapoint(i, dataset[i])

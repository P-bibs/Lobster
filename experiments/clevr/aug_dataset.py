import os
import json
from argparse import ArgumentParser

import torch
import scallopy
from tqdm import tqdm
import numpy as np
import copy

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
    program = [{'inputs': [], 'function': 'scene', 'value_inputs': []}, {'inputs': [0], 'function': 'filter_color', 'value_inputs': ['blue']}, {'inputs': [1], 'function': 'count', 'value_inputs': []}]
    clevr_program = CLEVRProgram(program)
    scene = CLEVRScene(self.scenes["scenes"][img_id])
    nl_question = "How many blue objects are there in the scene?"
    return (clevr_program, scene, nl_question)

class CLEVRScallop:
  def __init__(self):
    self.ctx = scallopy.ScallopContext()
    self.ctx.import_file(os.path.abspath(os.path.join(os.path.abspath(__file__), "../scl/clevr_eval.scl")))

  def run_scallop(self, program, scene):
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
    return result

def get_attr_programs(question_types):
  programs = []
  nl_questions = []

  all_shapes = ["cube", "cylinder", "sphere"]
  all_colors = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
  all_sizes = ["large", "small"]
  all_mats = ["metal", "rubber"]

  all_prog_types = {
    "shape": all_shapes,
    "color": all_colors,
    "size": all_sizes,
    "material": all_mats,
  }

  if "all" in question_types:
    question_types = ["shape", "color", "size", "material"]

  prog_types = {}
  for qt in question_types:
    prog_types[qt] = all_prog_types[qt]

  for prog_type, prog_type_ls in prog_types.items():
    for value in prog_type_ls:
      if prog_type == 'rela':
        program = [{'inputs': [], 'function': 'scene', 'value_inputs': []}, {'inputs': [0], 'function': f'relate', 'value_inputs': [value]}, {'inputs': [1], 'function': 'count', 'value_inputs': []}]
        nl_question = f"How many {value} are there in the scene?"
      else:
        program = [{'inputs': [], 'function': 'scene', 'value_inputs': []}, {'inputs': [0], 'function': f'filter_{prog_type}', 'value_inputs': [value]}, {'inputs': [1], 'function': 'count', 'value_inputs': []}]
        nl_question = f"How many {value} are there in the scene?"
      programs.append(program)
      nl_questions.append(nl_question)

  return programs, nl_questions

def get_rela_programs(programs, nl_questions):
  rela_programs = []
  rela_nl_questions = []

  all_relas = ["left", "behind"]

  for value in all_relas:
    for program, nl_question in zip(programs, nl_questions):
      program = program[:-1] + [{'inputs': [1], 'function': f'relate', 'value_inputs': [value]}, {'inputs': [2], 'function': 'count', 'value_inputs': []}]
      nl_question = nl_question + f", and object count to its {value}?"
      rela_programs.append(program)
      rela_nl_questions.append(nl_question)
  return rela_programs, rela_nl_questions

class ResultHolder():

  def __init__(self):
    self.distr = {}
    self.answers = {}

  def insert(self, result, pid):
    if not result in self.distr:
      self.distr[result] = []
    self.distr[result].append(pid)
    self.answers[pid] = result

  def get_answer(self, pid):
    return self.answers[pid]

  def get_count(self):
    return sum([len(pids) for pids in self.distr.values()])

  def get_count_res(self, res):
    if not res in self.distr:
      return 0
    return len(self.distr[res])

  def get_distr(self):
    distr = {res: len(pids) for res, pids in self.distr.items()}
    return distr

  def get_n(self, n):
    # ensure uniform distribution
    # min_ct = min([len(pids) for pids in self.distr.values()])
    all_pids = [pid for pids in self.distr.values() for pid in pids]
    n_pids = np.random.choice(all_pids, n, replace=False)
    return n_pids

  def pop_n_res(self, n, res):
    if not res in self.distr:
      return []
    result = self.distr[res][:n]
    self.distr[res] = self.distr[res][n:]
    return result

  def get_res_count(self):
    return len(self.distr.keys())

class ProgramGenerator():

  def __init__(self, question_types, attr_question_count, rela_question_count, runner):
    self.question_types = question_types
    self.attr_question_count = attr_question_count
    self.rela_question_count = rela_question_count
    self.runner = runner

  def get_progs_result(self, pid, program, clevr_scene, output_distr):
    clevr_program = CLEVRProgram(program)
    new_res = runner.run_scallop(clevr_program, clevr_scene)
    output_distr.insert(new_res, pid)
    return output_distr

  def get_programs(self, scene):
    programs, nl_questions = get_attr_programs(self.question_types)
    clevr_scene = CLEVRScene(scene)
    output_distr = ResultHolder()

    for pid, program in enumerate(programs):
      output_distr = self.get_progs_result(pid, program, clevr_scene, output_distr)
      if output_distr.get_count_res(1) > self.attr_question_count / output_distr.get_res_count() + self.rela_question_count:
        if len(output_distr.answers) > self.attr_question_count + self.rela_question_count:
          break

    # ensure unique existing object for relation
    rela_seed_pids = output_distr.pop_n_res(self.rela_question_count, 1)

    rela_seed_programs, rela_seed_nl_questions = [], []
    for pid in rela_seed_pids:
      rela_seed_programs.append(programs[pid])
      rela_seed_nl_questions.append(nl_questions[pid])
    rela_programs, rela_nl_questions = get_rela_programs(rela_seed_programs, rela_seed_nl_questions)
    selected_rela_pids = np.random.choice(list(range(len(rela_programs))), min(len(rela_programs), self.rela_question_count), replace=False)
    rela_output_distr = ResultHolder()

    for pid in selected_rela_pids:
      rela_program = rela_programs[pid]
      rela_output_distr = self.get_progs_result(pid, rela_program, clevr_scene, rela_output_distr)

    # obtain programs for attributes
    attr_pids = output_distr.get_n(self.attr_question_count)
    attr_qa_info = [(programs[pid], nl_questions[pid], output_distr.get_answer(pid)) for pid in attr_pids]
    rela_qa_info = [(rela_programs[pid], rela_nl_questions[pid], rela_output_distr.get_answer(pid)) for pid in selected_rela_pids]
    qa_info = attr_qa_info + rela_qa_info

    return qa_info

if __name__ == "__main__":

  # Argument parser
  parser = ArgumentParser("clevr_sanity_check")
  parser.add_argument("--dataset-postfix", type=str, default="")
  args = parser.parse_args()
  np.random.seed(1234)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  dataset_dir = os.path.join(data_dir, "CLEVR")
  new_dataset_dir = os.path.join(data_dir, "CLEVR_AGG")

  # Load dataset
  phase = "train"
  crop_num = 100
  max_obj = 5
  max_clause = 10
  question_type = "all"
  attr_question_count = 5
  rela_question_count = 5


  scenes = json.load(open(os.path.join(dataset_dir, f"scenes/CLEVR_{phase}_scenes.json")))
  question_info = json.load(open(os.path.join(dataset_dir, f"questions/CLEVR_{phase}_questions_obj_{max_obj}_clause_{max_clause}_image_split.cropped_{crop_num}.json")))
  questions = question_info['questions']
  all_images = []
  question_per_image = []

  info = question_info['info']
  agg_questions_path = os.path.join(dataset_dir, f"questions/CLEVR_{phase}_{question_type}_obj_{max_obj}_clause_{max_clause}_aqct_{attr_question_count}_uniq_rqct_{rela_question_count}_image_split.cropped_{crop_num}.json")

  # Load scallop program
  runner = CLEVRScallop()
  new_questions = []
  output_distr = {}

  # get_all_possible_progs
  program_gen = ProgramGenerator([question_type], attr_question_count, rela_question_count, runner)

  # For each datapoint select one shape
  for question in questions:
    img_id = question["image_index"]
    if img_id in all_images:
      continue
    else:
      all_images.append(img_id)
      question_per_image.append(question)

  for question in question_per_image:
    img_id = question["image_index"]
    scene = scenes["scenes"][img_id]
    qa_info = program_gen.get_programs(scene)

    for program, nl_question, answer in qa_info:

      new_question = question
      new_question['program'] = program
      new_question['answer'] = answer
      new_question['question'] = nl_question
      new_questions.append(copy.deepcopy(new_question))

  agg_questions = {}
  agg_questions['info'] = info
  agg_questions['questions'] = new_questions
  print(f"question number: {len(new_questions)}")
  json.dump(agg_questions, open(agg_questions_path, 'w'))
  print("end")

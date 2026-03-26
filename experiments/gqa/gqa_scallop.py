import pickle
from tqdm import tqdm
from dataset import get_dataset, Program, GQA_PATH

import scallopy
import scallopy_ext

SCALLOP_FILE = "gqa.scl"
PROVENANCE = "topkproofs"

class Args:
  def __init__(self):
    self.cuda = True
    self.gpu = 0

ARG_ORDER = {
  "COUNT": ("box",),
  "CROP": ("image", "box"),
  "CROP_ABOVE": ("image", "box"),
  "CROP_BELOW": ("image", "box"),
  "CROP_LEFTOF": ("image", "box"),
  "CROP_RIGHTOF": ("image", "box"),
  "EVAL": ("expr",),
  "LOC": ("image", "object"),
  "RESULT": ("var",),
  "VQA": ("image", "question"),
}

def test_one(context: scallopy.ScallopContext, id: str, testcase: dict):
  image_path = GQA_PATH + "images/" + testcase["imageId"] + ".jpg"
  step_facts = [(0, "IMAGE", f'IMAGE("{image_path}")')]

  for i, var_dict in enumerate(testcase["prog"].progs):
    function = var_dict["step_name"]
    arg_str_list = []
    for arg in ARG_ORDER[function]:
      if arg == "expr":
        arg_str_list.append(var_dict["args"][arg])
      elif arg == "object" or arg == "question":
        arg_str_list.append(var_dict["args"][arg].replace("'", '"'))
      else:
        arg_str_list.append('"' + var_dict["args"][arg] + '"')
    expr_str = function + "(" + ",".join(arg_str_list) + ")"
    step_facts.append((i + 1, var_dict["output_var"], expr_str))
  
  ctx = context.clone()
  ctx.add_facts("step", step_facts)
  ctx.run()
  debug = {
    "var_value_bbox": list(ctx.relation("var_value_bbox")),
    "var_value_string": list(ctx.relation("var_value_string")),
    "var_value_int": list(ctx.relation("var_value_int")),
    "final_result": list(ctx.relation("final_result"))
  }
  result = list(ctx.relation("final_result"))

  if result:
    result.sort(key=lambda x: x[0], reverse=True)
    return [tup[1][0] for tup in result[:5]], debug
  else:
    return ["NO RESULT"], debug

def test_no_crop():
  context = scallopy.ScallopContext(provenance=PROVENANCE)
  scallopy_ext.config.configure(Args())
  scallopy_ext.extlib.load_extlib(context)
  context.import_file(SCALLOP_FILE)
  context.set_iter_limit(100)
  context.set_non_probabilistic("step")

  data = get_dataset("mini_question_no_crop.pkl")
  items = tqdm(list(data.items()))
  results = {}
  debug_results = {}
  correct = {1: 0, 3: 0, 5: 0}
  match_substring = lambda s1, s2: s1 in s2 or s2 in s1
  total = 0
  for id, testcase in items:
    if "CROP" not in testcase["prog"].prog_str:
      ground_truth = testcase["answer"]
      predictions, debug = test_one(context, id, testcase)
      results[id] = predictions + [ground_truth]
      debug_results[id] = debug
      
      for k in (1, 3, 5):
        if any(match_substring(ground_truth, pred) for pred in predictions[:k]):
          correct[k] += 1
      total += 1
  
  with open("no_crop_results.txt", "w") as f:
    f.write("id,results (may be 1 to 3),answer\n")
    for id, result in results.items():
      f.write(f"{id},{','.join(result)}\n")

  with open("no_crop_debug.pkl", "wb") as f:
    pickle.dump(debug_results, f, protocol=pickle.HIGHEST_PROTOCOL)
  
  print(results)
  for k in (1, 3, 5):
    print(f"recall@{k}: {correct[k]}/{total} = {round(correct[k]/total, 4)}")

def test_with_crop():
  context = scallopy.ScallopContext(provenance="unit")
  scallopy_ext.config.configure(Args())
  scallopy_ext.extlib.load_extlib(context)
  context.import_file(SCALLOP_FILE)

  data = get_dataset("mini_question_with_crop.pkl")
  results = {}
  for id, testcase in data.items():
    results[id] = test_one(context, testcase)
  
  print(results)

if __name__ == "__main__":
  test_no_crop()

  # data = get_dataset("mini_question_no_crop.pkl")

  # for d in list(data.values())[:100]:
  #   p = d["prog"].prog_str
  #   if "CROP" in p:
  #     print(d["question"], d["answer"])
  #     print(p, end="\n\n")

  # context = scallopy.ScallopContext(provenance=PROVENANCE)
  # scallopy_ext.config.configure(Args())
  # scallopy_ext.extlib.load_extlib(context)
  # context.import_file(SCALLOP_FILE)
  # context.set_iter_limit(100)
  # context.set_non_probabilistic("step")

  # id = "00464293"
  # testcase = data[id]
  # print(testcase)
  # res = test_one(context, id, testcase)
  # print(res)

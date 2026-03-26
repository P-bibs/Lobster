import pickle
from tqdm import tqdm
from dataset import get_dataset, Program, GQA_PATH

import scallopy
import scallopy_ext

SCALLOP_FILE = "vilt.scl"
PROVENANCE = "unit"

class Args:
  def __init__(self):
    self.cuda = True
    self.gpu = 0


def test_one(context, testcase):
  image_path = GQA_PATH + "images/" + testcase["imageId"] + ".jpg"
  ctx = context.clone()
  ctx.add_facts("image_path", [(image_path,)])
  ctx.add_facts("question", [(testcase["question"],)])
  # print(testcase["question"], testcase["answer"])
  ctx.run()
  
  result = list(ctx.relation("result"))
  return [tup[0] for tup in result]
  
def test_no_crop():
  context = scallopy.ScallopContext(provenance=PROVENANCE)
  scallopy_ext.config.configure(Args())
  scallopy_ext.extlib.load_extlib(context)
  context.import_file(SCALLOP_FILE)
  context.set_iter_limit(100)

  data = get_dataset("mini_question_no_crop.pkl")
  items = tqdm(list(data.items()))
  results = {}
  correct = {1: 0, 3: 0, 5: 0}
  match_substring = lambda s1, s2: s1 in s2 or s2 in s1
  total = 0
  for id, testcase in items:
    if "CROP" not in testcase["prog"].prog_str:
      predictions = test_one(context, testcase)
      ground_truth = testcase["answer"]
      results[id] = predictions + [ground_truth]
      
      for k in (1, 3, 5):
        if any(match_substring(ground_truth, pred) for pred in predictions[:k]):
          correct[k] += 1
      total += 1
  
  with open("no_crop_baseline.txt", "w") as f:
    f.write("id,res1,res2,res3,res4,res5,answer\n")
    for id, result in results.items():
      f.write(f"{id},{','.join(result)}\n")
  
  print(results)
  for k in (1, 3, 5):
    print(f"recall@{k}: {correct[k]}/{total} = {round(correct[k]/total, 4)}")


if __name__ == "__main__":
  test_no_crop()

  # context = scallopy.ScallopContext(provenance=PROVENANCE)
  # scallopy_ext.config.configure(Args())
  # scallopy_ext.extlib.load_extlib(context)
  # context.import_file(SCALLOP_FILE)
  # context.set_iter_limit(100)

  # data = get_dataset("mini_question_no_crop.pkl")
  # res = test_one(context, data["00464293"])
  # print(res)
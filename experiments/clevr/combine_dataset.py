import json
import os
import shutil


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

  return bboxes

if __name__ == "__main__":

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  dataset_dir = os.path.join(data_dir, "CLEVR")
  new_dataset_dir = os.path.join(dataset_dir, "ss_lab")


  # Load dataset
  phase = "train"
  all_images = []
  image_dir = os.path.join(dataset_dir, f'images/{phase}')
  new_image_dir = os.path.join(new_dataset_dir, f'images/{phase}')

  scenes = json.load(open(os.path.join(dataset_dir, f"scenes/CLEVR_{phase}_scenes.json")))['scenes']
  question_info = json.load(open(os.path.join(dataset_dir, f"questions/count_yellow_{phase}.json")))

  new_question_path = os.path.join(new_dataset_dir, f'questions/count_yellow_{phase}.json')

  questions = question_info['questions']
  new_question_info = question_info

  new_questions = []
  for question in questions:
    scene = scenes[question['image_index']]
    bbox = extract_bounding_boxes(scene)
    question.pop('question_family_index')
    question.pop('question_index')
    question['bounding_boxes'] = bbox
    question['scene'] = scene
    new_questions.append(question)
    src_image_path = os.path.join(image_dir, question['image_filename'])
    tgt_image_path = os.path.join(new_image_dir, question['image_filename'])
    shutil.copyfile(src_image_path, tgt_image_path)

  new_question_info['questions'] = questions
  json.dump(new_question_info, open(new_question_path, 'w'))
  print('end')

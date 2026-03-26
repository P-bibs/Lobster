import os
import json
import jsbeautifier
from tqdm import tqdm
from PIL import Image

from word_idx_translator import Idx2Word
from preprocess import Query
from transformer import SimpleTransformer
import scallopy
import scallopy_ext


VQAR_DATA = os.path.abspath(os.path.join(__file__, "../dataset"))
VQAR_SCL = os.path.abspath(os.path.join(__file__, "../scl"))
GQA_INFO = "gqa_info.json"
IMAGE_DIR = "images"

META = json.load(open(os.path.join(VQAR_DATA, GQA_INFO)))
IDX2WORD = Idx2Word(META)
TRANSFORMER = SimpleTransformer()


class Args:
    def __init__(self):
        self.cuda = False
        self.gpu = None


def batch_to_json(batch):
    out = {"datapoints": []}
    for (i, datapoint) in enumerate(batch):
        out["datapoints"].append({
            "id": i,
            "image_id": datapoint["image_id"],
            "object_ids": datapoint["object_ids"],
            "scene_graph": datapoint["scene_graph"],
            "question": datapoint["question"]
        })

    options = jsbeautifier.default_options()
    options.indent_size = 2
    json_object = jsbeautifier.beautify(json.dumps(out.copy()), options)
    with open("dataset.json", "w") as outfile:
        outfile.write(json_object)


def clip_recall(gt_output, ctx_output):
    return int(set(gt_output).issubset(set(ctx_output)))


def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2 = x1 + w1, y1 + h1
    x3, y3, w2, h2 = bbox2
    x4, y4 = x3 + w2, y3 + h2

    i_x1, i_y1, i_x2, i_y2 = max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)
    i_area = max(i_x2 - i_x1, 0) * max(i_y2 - i_y1, 0)
    u_area = w1 * h1 + w2 * h2 - i_area

    return i_area / u_area


def owl_recall(gt_bboxes, ctx_bboxes):
    score = 0
    for (_, gt_bbox) in gt_bboxes:
        score += max([iou(gt_bbox, ctx_bbox) for (_, ctx_bbox) in ctx_bboxes], default=0)
    return score / len(gt_bboxes)


def test_datapoint(datapoint, base_ctx, infer_name=True, infer_attr=True, infer_rela=True, debug=True):
    ctx = base_ctx.clone()

    id = datapoint["id"]
    image_id = datapoint['image_id']
    object_ids = datapoint['object_ids']
    bboxes = datapoint['scene_graph']['bboxes']
    names = datapoint['scene_graph']['names']
    attrs = datapoint['scene_graph']['attributes']
    relas = datapoint['scene_graph']['relations']
    query = Query(datapoint['question']['clauses'])
    gt_output = datapoint['question']['output']

    res = {
        "id": id,
        "image_id": image_id,
        "query": query.to_natural_language(),
        "count": 0,
        "score": 0,
    }
    data = {
        "id": id,
        "query_rules": [],
        "gt_output": gt_output,
        "ctx_output": None,
        "gt_bboxes": [],
        "ctx_bboxes": [],
        "name": None,
        "attr": None,
    }
    
    img_path = os.path.join(VQAR_DATA, IMAGE_DIR, f"{image_id}.jpg")
    img = Image.open(img_path)
    W, H = img.size

    # Add image path to ctx
    ctx.set_non_probabilistic("img_path")
    ctx.add_facts("img_path", [(img_path,)])

    # Set settings
    ctx.set_non_probabilistic(["idx", "infer_name", "infer_attr", "infer_rela", "debug", "output"])
    ctx.add_facts("idx", [(id,)])
    ctx.add_facts("infer_name", [(infer_name,)])
    ctx.add_facts("infer_attr", [(infer_attr,)])
    ctx.add_facts("infer_rela", [(infer_rela,)])
    ctx.add_facts("debug", [(debug,)])
    ctx.add_facts("output", [(oid,) for oid in gt_output])

    # Add query attributes
    query_attrs = set()
    for var in query.vars:
        query_attrs.update(var.attrs)
    ctx.set_non_probabilistic("query_attr")
    ctx.add_facts("query_attr", list((attr,) for attr in query_attrs))

    bbox_facts, name_facts, attr_facts, rela_facts = [], [], [], []
    for oid in object_ids:
        # Add output bbox fact for debugging
        x, y, w, h = bboxes[str(oid)]
        x, y, w, h = int(x * W), int(y * H), int(w * W), int(h * H)
        bbox_facts.append((oid, x, y, w, h))

        # Don't add object name from scene graph by default
        if not infer_name:
            idx = names[str(oid)]
            name_facts.append((1, (IDX2WORD.idx_to_name(idx), oid)))

        # Don't add object attr from scene graph by default
        if not infer_attr:
            for idx in attrs[str(oid)]:
                attr_facts.append((1, (IDX2WORD.idx_to_attr(idx), oid)))

        # Don't add relation from scene graph by default
        if not infer_rela and oid in relas:
            for oid2, idx in relas[str(oid)].items():
                if idx >= 0:
                    rela_facts.append((1, (IDX2WORD.idx_to_rela(idx), oid, int(oid2))))

    # Add facts to ctx
    ctx.set_non_probabilistic("out_bbox")
    ctx.add_facts("out_bbox", bbox_facts)
    ctx.add_facts("name", name_facts)
    ctx.add_facts("attr", attr_facts)
    ctx.add_facts("rela", rela_facts)

    # Add query rules to ctx
    for rule in TRANSFORMER.transform(query).split('\n'):
        if len(rule) > 0:
            data["query_rules"].append(rule)
            ctx.add_rule(rule)

    ctx.run()

    data["ctx_output"] = list(ctx.relation("target"))
    res["count"] = len(data["ctx_output"])
    ctx_output = [oid for (_, (oid, )) in data["ctx_output"]]
    for (_, (oid, x, y, w, h)) in list(ctx.relation("obj_bbox")) + list(ctx.relation("out_bbox")):
        if oid in gt_output:
            data["gt_bboxes"].append((oid, (x, y, w, h)))
        if oid in ctx_output:
            data["ctx_bboxes"].append((oid, (x, y, w, h)))

    #res["score"] = clip_metric(gt_output, ctx_output)
    res["score"] = owl_recall(data["gt_bboxes"], data["ctx_bboxes"])

    data["name"] = list(fact for fact in ctx.relation("name") if fact[1][0] != "thing" and fact[1][0] != "object")
    data["attr"] = list(ctx.relation("attr"))

    return res, data


def test_dataset(dataset_file, scl_file, kb_file):
    out = {
        "score": 0,
        "res": [],
        "data": [],
    }
    dataset = json.load(open(dataset_file))["datapoints"]

    # Try dynamically setting names, attrs, relas classes later
    """
    names, attrs, relas = set(), set(), set()
    for datapoint in dataset:
        sg = datapoint['scene_graph']
        for idx in sg['names'].values():
            names.add(IDX2WORD.idx_to_name(idx))
        for idxs in sg['attributes'].values():
            for idx in idxs:
                attrs.add(IDX2WORD.idx_to_attr(idx))
        for idxs in sg['relations'].values():
            for idx in idxs.values():
                if idx >= 0: relas.add(IDX2WORD.idx_to_rela(idx))
    """

    # Configure Scallopy context
    scallopy_ext.config.configure(Args())
    ctx = scallopy.ScallopContext(provenance="topkproofs")
    scallopy_ext.extlib.load_extlib(ctx)
    ctx.import_file(kb_file)
    ctx.import_file(scl_file)

    pbar = tqdm(dataset[:1])
    for i, datapoint in enumerate(pbar):
        try:
            res, data = test_datapoint(datapoint, ctx)
            out["score"] += res["score"]
            out["res"].append(res)
            out["data"].append(data)
            pbar.set_postfix({"score": round(out["score"] / (i + 1), 2)})
        except Exception as e:
            out["res"].append({
                "id": datapoint["id"],
                "exception": str(e),
                "score": 0
            })
    out["score"] /= len(dataset)

    options = jsbeautifier.default_options()
    options.indent_size = 2
    json_object = jsbeautifier.beautify(json.dumps(out.copy()), options)
    with open("data.json", "w") as outfile:
        outfile.write(json_object)


KB = os.path.join(VQAR_SCL, "knowledge_base.scl")
VQAR_NAME, VQAR_NAME_SCL = os.path.join(VQAR_DATA, "vqar_name.json"), os.path.join(VQAR_SCL, "vqar_name_owl.scl")
VQAR_ATTR, VQAR_ATTR_SCL = os.path.join(VQAR_DATA, "vqar_attr.json"), os.path.join(VQAR_SCL, "vqar_attr_owl.scl")
test_dataset(VQAR_NAME, VQAR_NAME_SCL, KB)
test_dataset(VQAR_ATTR, VQAR_ATTR_SCL, KB)
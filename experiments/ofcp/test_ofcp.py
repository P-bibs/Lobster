import os
import json
import jsbeautifier
from tqdm import tqdm
import scallopy
import scallopy_ext

DATASET = os.path.abspath(os.path.join(__file__, "../dataset"))
SCL = os.path.abspath(os.path.join(__file__, "../ofcp.scl"))


class Args:
    def __init__(self):
        self.cuda = False
        self.gpu = None
        self.num_allowed_openai_request = 1000
        self.openai_gpt_model = "gpt-4"
        self.openai_gpt_temperature = 0


scallopy_ext.config.configure(Args())
ctx = scallopy.ScallopContext(provenance="topkproofs")
scallopy_ext.extlib.load_extlib(ctx)

ctx.import_file(SCL)
ctx.set_non_probabilistic(["img_dir", "img_name"])

out = {"res": []}

for image_name in tqdm(sorted(os.listdir(DATASET))):
    tmp_ctx = ctx.clone()
    tmp_ctx.add_facts("img_dir", [(os.path.join(DATASET, image_name),)])
    tmp_ctx.add_facts("img_name", [(image_name,)])
    tmp_ctx.run()

    out["res"].append(
        {
            "name": image_name,
            "classes": [
                name.strip()
                for name in list(tmp_ctx.relation("names"))[0][1][0].split(";")
            ],
            "bbox": list(tmp_ctx.relation("face_bbox")),
            "identity": list(tmp_ctx.relation("identity")),
        }
    )


options = jsbeautifier.default_options()
options.indent_size = 2
json_object = jsbeautifier.beautify(json.dumps(out.copy()), options)
with open("data.json", "w") as outfile:
    outfile.write(json_object)

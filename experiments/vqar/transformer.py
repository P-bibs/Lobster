
import pickle
import os
import sys
import json
import csv
import time

common_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.insert(0, common_path)

from preprocess import Query, BiOp, Variable
from word_idx_translator import Idx2Word



class Transformer():

    def __init__(self, name):
        self.name = name

    def get_new_var(self, var_ids):
            # global ?
            vid = f"V{len(var_ids)}"
            var_ids.append(vid)
            return vid, var_ids

    def gen_header(self, rela_name, arg_types, output=False):
        # args = []
        # for arg_id, arg_type in enumerate(arg_types):
        #     args.append(f"x{arg_id}: {arg_type}")
        # args = ", ".join(args)

        # dl = f""

        # if output:
        #     header = dl + "\n" + f".output {rela_name}"
        # else:
        #     header = dl
        # return header
        return ""

    def transform(self, query):
        raise NotImplementedError

class SimpleTransformer(Transformer):

    def __init__(self):
        super().__init__("simple")

    def rec_transform(self, query, node, var_ids, visited_node=[], is_root=False, rule_ct=0, decl=True):
        if issubclass(type(node), BiOp):
            v1 = node.v1
            v2 = node.v2
            vid, var_ids = self.get_new_var(var_ids)

            if is_root:
                rule_name = "target"
                output = True
                arg_types = ["ObjectID"]
            else:
                rule_name = f"rule_{rule_ct}"
                output = False
                arg_types = ["ObjectID"]

            if decl:
                header = self.gen_header(rule_name, arg_types, output)

            if node.op_name == "and":
                visited_node, prog1 = self.rec_transform(query, v1, var_ids, visited_node, is_root=False, rule_ct = rule_ct + 1)
                visited_node, prog2 = self.rec_transform(query, v2, var_ids, visited_node, is_root=False, rule_ct = rule_ct + 2)
                cur_rule = f"{rule_name}({vid}) :- rule_{rule_ct + 1}({vid}), rule_{rule_ct + 2}({vid})."
            elif node.op_name == "or":
                visited_node, prog1 = self.rec_transform(query, v1, var_ids, visited_node, is_root=False, rule_ct = rule_ct + 1)
                visited_node, prog2 = self.rec_transform(query, v2, var_ids, visited_node, is_root=False, rule_ct = rule_ct + 1, decl=False)
                cur_rule = f"{rule_name}({vid}) :- rule_{rule_ct + 1}({vid})."
            else:
                raise Exception("Unrecognized op name")

            prog = prog1 + prog2 + header + "\n" + cur_rule
            return visited_node, prog

        elif type(node) is Variable:
            if is_root:
                visited_node, new_rule = self.generate_rule(query, node, "target", decl)
            else:
                visited_node, new_rule = self.generate_rule(query, node, rule_ct, decl)
            return visited_node, new_rule

        else:
            raise Exception("Unknown operation")


    def generate_rule(self, query, var, rule_ct, decl=True):
        visited_vars = []
        target_vars = []
        var_description = []
        var_stack = [var]

        while len(var_stack) > 0:
            var = var_stack.pop()
            if var in visited_vars:
                continue
            var_prog = var.to_datalog()
            if not len(var_prog) == 0:
                var_description.append(', '.join(var_prog))
            target_vars.append(var.var_id)
            visited_vars.append(var)
            neighbors = var.get_neighbor()
            var_stack += neighbors

        arg_types = ["ObjectID"]

        if type(rule_ct) is int:
            rule_name = f"rule_{rule_ct}"
            if decl:
                header = self.gen_header(rule_name, arg_types, False)
            else:
                header = ""
        else:
            rule_name = rule_ct
            if decl:
                if rule_name == "target":
                    header = self.gen_header(rule_name, arg_types, True)
                else:
                    header = self.gen_header(rule_name, arg_types, False)
            else:
                header = ""

        new_rule = header + "\n" + f"{rule_name}({target_vars[0]}) = {', '.join(var_description)}\n"
        return visited_vars, new_rule

    def transform(self, query):
        _, new_rule = self.rec_transform(query, query.root, [], is_root=True, rule_ct=0)
        return new_rule


class DetailTransformer(Transformer):

    def __init__(self):
        super().__init__("detail")

    def get_new_var(self, var_ids):
            # global ?
            vid = f"V{len(var_ids)}"
            nid = f"N{len(var_ids)}"
            var_ids.append(vid)
            return vid, nid, var_ids

    def rec_transform(self, query, node, var_ids, visited_node=[], is_root=False, rule_ct=0, interm_rule_ct=0, decl=True, with_name=False, with_rela=True):
        if issubclass(type(node), BiOp):
            v1 = node.v1
            v2 = node.v2
            vid, nid, var_ids = self.get_new_var(var_ids)

            if is_root:
                rule_name = "target"
            else:
                rule_name = f"rule_{rule_ct}"

            if node.op_name == "and":
                visited_node, prog1, with_name1, interm_rule_ct = self.rec_transform(query, v1, var_ids, visited_node, is_root=False, rule_ct = rule_ct + 1, interm_rule_ct=interm_rule_ct, with_name=with_name)
                visited_node, prog2, with_name2, interm_rule_ct = self.rec_transform(query, v2, var_ids, visited_node, is_root=False, rule_ct = rule_ct + 2, interm_rule_ct=interm_rule_ct, with_name=with_name)

                if with_name1:
                    left = f"rule_{rule_ct + 1}({vid}, {nid})"
                else:
                    left =  f"rule_{rule_ct + 1}({vid})"

                if with_name2:
                    right = f"rule_{rule_ct + 2}({vid}, {nid})"
                else:
                    right = f"rule_{rule_ct + 2}({vid})"

                if (with_name1 or with_name2):
                    arg_types = ["ObjectID", "Word"]
                    cur_rule = f"{rule_name}({vid}, {nid}) :- {left}, {right}."
                    with_name = True
                else:
                    arg_types = ["ObjectID"]
                    cur_rule = f"{rule_name}({vid}) :- {left}, {right}."

            elif node.op_name == "or":
                visited_node, prog1, with_name1, interm_rule_ct = self.rec_transform(query, v1, var_ids, visited_node, is_root=False, rule_ct = rule_ct + 1, interm_rule_ct=interm_rule_ct, with_name=with_name)
                visited_node, prog2, with_name2, interm_rule_ct = self.rec_transform(query, v2, var_ids, visited_node, is_root=False, rule_ct = rule_ct + 1, interm_rule_ct=interm_rule_ct, decl=False, with_name=with_name1)

                if ((not with_name1) and with_name2):
                    visited_node, prog1, with_name1, interm_rule_ct = self.rec_transform(query, v1, var_ids, visited_node, is_root=False, rule_ct = rule_ct + 1, interm_rule_ct=interm_rule_ct, with_name=with_name2)

                if with_name1:
                    arg_types = ["ObjectID", "Word"]
                    cur_rule = f"{rule_name}({vid}, {nid}) :- rule_{rule_ct + 1}({vid}, {nid})."
                    with_name = True
                else:
                    arg_types = ["ObjectID"]
                    cur_rule = f"{rule_name}({vid}) :- rule_{rule_ct + 1}({vid})."

            else:
                raise Exception("Unrecognized op name")

            if decl:
                header = self.gen_header(rule_name, arg_types, True)

            prog = prog1 + prog2 + header + "\n" + cur_rule
            return visited_node, prog, with_name, interm_rule_ct

        elif type(node) is Variable:
            if is_root:
                visited_node, new_rule, with_name, interm_rule_ct = self.generate_rule(query, node, "target", interm_rule_ct, decl, with_name=with_name, with_rela=with_rela)
            else:
                visited_node, new_rule, with_name, interm_rule_ct = self.generate_rule(query, node, rule_ct, interm_rule_ct, decl, with_name=with_name, with_rela=with_rela)
            return visited_node, new_rule, with_name, interm_rule_ct

        else:
            raise Exception("Unknown operation")

    def generate_rule(self, query, var, rule_ct, interm_rule_ct, decl=True, with_name=False, with_rela=True):
        visited_vars = []
        target_vars = []
        var_description = []
        var_stack = [var]
        rule_require_name, rule_nid = var.get_name_id()
        with_name = with_name or rule_require_name

        while len(var_stack) > 0:
            var = var_stack.pop()

            if var in visited_vars:
                continue

            require_name, nid = var.get_name_id()
            var_prog = var.to_datalog(with_name, with_rela)
            if not len(var_prog) == 0:
                var_description.append(', '.join(var_prog))
            target_vars.append(var.var_id)
            if require_name:
                target_vars.append(nid)
            visited_vars.append(var)
            neighbors = var.get_neighbor()
            var_stack += neighbors


        if rule_ct == "target":
            arg_types = ["ObjectID" if (target_var[0] == 'V' or target_var[0] == 'O') else "Word" for target_var in target_vars ]
            header = self.gen_header("target", arg_types, True)
            new_rule = header + "\n" + f"target({', '.join(target_vars)}) :- {', '.join(var_description)}.\n"
            return visited_vars, new_rule, with_name, interm_rule_ct

        else:
            # we need intermediate rules to describe the latent variables
            if (not len(visited_vars) == 1):
                rule_name = f"interm_{interm_rule_ct}"
                arg_types = ["ObjectID" if (target_var[0] == 'V' or target_var[0] == 'O') else "Word" for target_var in target_vars ]
                header = self.gen_header(rule_name, arg_types, True)
                interm_rule = header + "\n" + f"{rule_name}({', '.join(target_vars)}) :- {', '.join(var_description)}.\n"
                interm_rule_ct += 1
            else:
                interm_rule = ""

            # Generate the new rule for rec calculation
            if with_name:
                arg_types = ["ObjectID", "Word"]
                rule_vars = [target_vars[0], rule_nid]
            else:
                arg_types = ["ObjectID"]
                rule_vars = [target_vars[0]]

            rule_name = f"rule_{rule_ct}"
            if decl:
                header = self.gen_header(rule_name, arg_types, False)
            else:
                header = ""

            new_rule = interm_rule + header + "\n" + f"{rule_name}({', '.join(rule_vars)}) :- {', '.join(var_description)}.\n"
            return visited_vars, new_rule, with_name, interm_rule_ct

    def transform(self, query):
        _, new_rule, _, _ = self.rec_transform(query, query.root, [], is_root=True, rule_ct=0, with_name=False, with_rela=True)
        end = "query(target(O))."
        return new_rule + "\n" + end

class SceneTransformer():

    def __init__(self, idx2word):
        self.idx2word = idx2word

    def to_prog(self, scene_graph):
        attrs = []
        relations = []
        names = []

        for obj_id, obj_name in scene_graph['names'].items():
            name = self.idx2word.idx_to_name(obj_name)
            if not type(name) == type(None):
                names.append(f"1.0::name(\"{name}\", \"{obj_id}\").")

        for obj_id, attr_ls in scene_graph['attributes'].items():
            for attr in attr_ls:
                attr = self.idx2word.idx_to_attr(attr)
                if not type(attr) == type(None):
                    attrs.append(f"1.0::attr(\"{attr}\", \"{obj_id}\").")

        for subject_id, object_info in scene_graph['relations'].items():
            for object_id, rel in object_info.items():
            # objs_id = [obj_id.strip() for obj_id in objs_id[1:-1].split(",")]s
            #TODO: double check here
                rela = self.idx2word.idx_to_rela(rel)
                if not type(rela) == type(None):
                    relations.append(f"1.0::rela(\"{rela}\", \"{subject_id}\", \"{object_id}\").")

        context = '\n'.join(attrs) + '\n' + '\n'.join(relations) + '\n' + '\n'.join(names)
        return context

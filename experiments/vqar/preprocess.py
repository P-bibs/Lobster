# Prog = Conj | Logic
# Logic = biOp Prog Prog
# Conj = and Des Conj | Des
# Des = rela Relaname O1 O2 | attr Attr O
import json
import os

var = ["s (", "o ("]

class Variable():
    def __init__(self, id):
        self.var_id = f"O{id}"
        self.name_id = f"N{id}"
        self.name = []
        self.hypernyms = []
        self.attrs = []
        self.kgs = []
        # The relations where this object functions as a subject
        self.sub_relas = []
        self.obj_relas = []

    def has_rela(self):
        if len(self.sub_relas) == 0 and len(self.obj_relas) == 0:
            return False
        return True

    def get_name_id(self):
        # if (not len(self.hypernyms) == 0) and len(self.name) == 0:
        #     return True, self.name_id
        # if (not len(self.kgs) == 0) and len(self.name) == 0:
        #     return True, self.name_id
        return False, self.name_id

    def set_name(self, name):
        if name in self.name:
            return

        self.name.append(name)

    def set_kg(self, kg):
        if kg not in self.kgs:
            self.kgs.append(kg)

    def set_hypernym(self, hypernym):
        if hypernym not in self.hypernyms:
            self.hypernyms.append(hypernym)

    def set_attr(self, attr):
        if attr not in self.attrs:
            self.attrs.append(attr)

    def set_obj_relas(self, obj_rela):
        if obj_rela not in self.obj_relas:
            self.obj_relas.append(obj_rela)

    def set_sub_relas(self, sub_rela):
        if sub_rela not in self.sub_relas:
            self.sub_relas.append(sub_rela)

    def get_neighbor(self):
        neighbors = []
        for rela in self.sub_relas:
            neighbors.append(rela.obj)
        for rela in self.obj_relas:
            neighbors.append(rela.sub)
        return neighbors

    def update(self, other):

        self.hypernyms = list(set(self.name + other.name))
        self.hypernyms = list(set(self.hypernyms + other.hypernyms))
        self.attrs = list(set(self.attrs + other.attrs))
        self.kgs = list(set(self.kgs + other.kgs))

    def to_datalog(self, with_name=False, with_rela=True):

        name_query = []

        if (len(self.name) == 0) and with_name:
            name_query.append(f"name({self.name_id}, {self.var_id})")

        if (not len(self.name) == 0):
            for n in self.name:
                name_query.append(f"name(\"{n}\", {self.var_id})")

        attr_query = [
            f"attr(\"{attr}\", {self.var_id})" for attr in self.attrs]
        hypernym_query = [
            f"name(\"{hypernym}\", {self.var_id})" for hypernym in self.hypernyms]
        kg_query = []

        for kg in self.kgs:
            restriction = list(
                filter(lambda x: not x == 'BLANK' and not x == '', kg))
            assert (len(restriction) == 2)
            rel = restriction[0]
            usage = restriction[1]
            kg_query += [
                f"name({self.name_id}, {self.var_id}), oa_rel(\"{rel}\", {self.name_id}, \"{usage}\")"]

        if with_rela:
            rela_query = [rela.to_datalog() for rela in self.sub_relas]
        else:
            rela_query = []

        program = name_query + attr_query + hypernym_query + kg_query + rela_query
        return program

    def to_rust(self):
        name_constraint = self.name + [n for n in self.hypernyms]
        attr_constraints = self.attrs
        kg_contraints = []
        for kg in self.kgs:
            restriction = list(
                filter(lambda x: not x == 'BLANK' and not x == '', kg))
            assert (len(restriction) == 2)
            rel = restriction[0]
            usage = restriction[1]
            kg_contraints.append(f"{rel} {usage}")
        return (name_constraint, attr_constraints, kg_contraints, int(self.var_id[1:]))

    def to_natural_language(self):
        name_constraints = self.name + [n for n in self.hypernyms]
        attr_constraints = self.attrs
        kg_contraints = []
        kg_descriptions = []
        kg_description = ""
        name_descriptions = ""
        attr_descriptions = ""

        for kg in self.kgs:
            restriction = list(
                filter(lambda x: not x == 'BLANK' and not x == '', kg))
            assert (len(restriction) == 2)
            rel = restriction[0]
            usage = restriction[1]
            kg_contraints.append((rel, usage))

        object_name = f"Object{self.var_id[1:]}"
        if not len(name_constraints) == 0:
            name_descriptions = " and ".join(name_constraints)

        if not len(attr_constraints) == 0:
            attr_descriptions = ", ".join(attr_constraints)

        if not (len(name_constraints) == 0 and len(attr_constraints) == 0):
            description = join_without_empty_string([attr_descriptions] + [name_descriptions], " ")
            scene_descriptions = f"{object_name} is {description}. "

        else:
            scene_descriptions = ""

        for rel, usage in kg_contraints:
            kg_descriptions.append(f"{object_name} {rel} {usage}. ")
        kg_description = "\n".join(kg_descriptions)

        description = join_without_empty_string([scene_descriptions, kg_description], "\n")
        return description

def join_without_empty_string(ls, symbol):
    res = ""
    new_ls = []
    for e in ls:
        if e == "":
            continue
        new_ls.append(e)

    return symbol.join(new_ls)

class Relation():
    def __init__(self, rela_name, sub, obj):
        self.rela_name = rela_name
        self.sub = sub
        self.obj = obj
        self.sub.set_sub_relas(self)
        self.obj.set_obj_relas(self)

    def substitute(self, v1, v2):
        if self.sub == v1:
            self.sub = v2
        if self.obj == v1:
            self.obj = v2

    def to_datalog(self):
        rela_query = f"rela(\"{self.rela_name}\", {self.sub.var_id}, {self.obj.var_id})"
        return rela_query

    def to_rust(self):
        return (self.rela_name, int(self.sub.var_id[1:]), int(self.obj.var_id[1:]))

    def to_natural_language(self):
        object_name1 = f"Object{self.sub.var_id[1:]}"
        object_name2 = f"Object{self.obj.var_id[1:]}"
        description = f"{object_name1} is {self.rela_name} {object_name2}. "
        return description
# This is for binary operations on variables


class BiOp():
    def __init__(self, op_name, var, v1, v2):
        self.op_name = op_name
        self.var = var
        self.v1 = v1
        self.v2 = v2

    def to_rust(self):
        return (self.op_name, int(self.var.var_id[1:]), int(self.v1.var_id[1:]), int(self.v2.var_id[1:]))

    def to_natural_language(self):
        object_name1 = f"Object{self.v1.var_id[1:]}"
        object_name2 = f"Object{self.v2.var_id[1:]}"
        target_object_name = f"Object{self.var.var_id[1:]}"
        description = f"{target_object_name} is {object_name1} {self.op_name} {object_name2}. "
        return description

class Or(BiOp):
    def __init__(self, var, v1, v2):
        super().__init__('or', var, v1, v2)


class And(BiOp):
    def __init__(self, var, v1, v2):
        super().__init__('and', var, v1, v2)


class Query():

    def __init__(self, query, dataset="vqar"):
        self.vars = []
        self.relations = []
        self.operations = []
        self.stack = []
        if dataset == "vqar":
            self.preprocess(query)
        else:
            self.gqa_preprocess(query)

    def get_target(self):
        pass

    def get_new_var(self):
        self.vars.append(Variable(len(self.vars)))

    def to_rust(self):
        # Currently only support last layer
        assert (len(self.operations) <= 1)
        var_constraints = []
        rela_constraints = []
        logic_constraints = None

        # Ignore the auxilliary variables
        for v in self.vars[0: len(self.vars) - len(self.operations)]:
            var_constraints.append(v.to_rust())
        for r in self.relations:
            rela_constraints.append(r.to_rust())
        if len(self.operations) == 1:
            logic_constraints = (self.operations[0].to_rust())
        return var_constraints, rela_constraints, logic_constraints

    def to_natural_language(self):
        assert (len(self.operations) <= 1)
        var_constraints = []
        rela_constraints = []
        logic_constraints = []
        var_description = ""
        rela_description = ""
        logic_description = ""

        # Ignore the auxilliary variables
        for v in self.vars[0: len(self.vars) - len(self.operations)]:
            natural_description = v.to_natural_language()
            if not natural_description == "":
                var_constraints.append(natural_description)

        var_description = "\n".join(var_constraints)

        for r in self.relations:
            natural_description = r.to_natural_language()
            if not natural_description == "":
                rela_constraints.append(natural_description)

        rela_description = "\n".join(rela_constraints)

        if len(self.operations) == 1:
            natural_description = self.operations[0].to_natural_language()
            if not natural_description == "":
                logic_constraints.append(natural_description)

        logic_description = "\n".join(logic_constraints)
        target_description = f"The target object is Object{ self.vars[-1].var_id[1:]}"

        nl_ls = [var_description, rela_description, logic_description, target_description]
        nl = join_without_empty_string(nl_ls, "\n")
        return nl

    def gqa_preprocess(self, query):

        if not len(self.vars) == 0:
            self.stack.append(self.vars[-1])
        self.get_new_var()
        self.root = self.vars[-1]
        
        # for clause in question["program"]:
        for ct, clause in enumerate(query):

            # logic operations
            if clause['function'] == "and":
                if len(self.stack) < 2:
                    continue 

                v2 = self.stack.pop()
                v1 = self.stack.pop()
                self.get_new_var()
                var_id = self.vars[-1]

                self.operations.append(And(var_id, v1, v2))
                self.root = self.vars[-1]

            elif clause['function'] == "or":
                if len(self.stack) < 2:
                    continue 

                v2 = self.stack.pop()
                v1 = self.stack.pop()
                self.get_new_var()
                var_id = self.vars[-1]

                self.operations.append(Or(var_id, v1, v2))
                self.root = self.vars[-1]

            # find operations
            elif clause['function'] == "select":
                self.vars[-1].set_name(remove_number(clause['argument']))

            elif clause['function'] == "relate" or clause['function'] == "verify rel":
                args = clause['argument'].split(",")
                sub_lit = remove_number(args[0])
                rela_name = remove_number(args[1])
                object_lit = remove_number(args[2])

                if "s" == sub_lit or "o" == sub_lit:
                    
                    self.get_new_var()
                    sub = self.vars[-2] # subject 
                    obj = self.vars[-1] # object
                    obj.set_name(object_lit)
                    self.root = sub

                elif "s" == object_lit or "o" == object_lit:

                    self.get_new_var()
                    sub = self.vars[-1] # subject 
                    obj = self.vars[-2] # object
                    sub.set_name(sub_lit)
                    self.root = obj

                relation = Relation(rela_name, sub, obj)
                self.relations.append(relation)

            elif clause['function'] == "verify color":
                self.vars[-1].set_attr(remove_number(clause['argument']))
                if not ct == len(query) - 1:
                    self.stack.append(self.vars[-1])
                    self.get_new_var()
                    self.root = self.vars[-1]

            elif clause['function'] == "verify":
                self.vars[-1].set_attr(remove_number(clause['argument']))
                if not ct == len(query) - 1:
                    self.stack.append(self.vars[-1])
                    self.get_new_var()
                    self.root = self.vars[-1]

            elif clause['function'] == "exist":
                if not ct == len(query) - 1:
                    self.stack.append(self.vars[-1])
                    self.get_new_var()
                    self.root = self.vars[-1]

            else:
                raise Exception(f"Not handled function: {clause['operation']}")

        pass

    def preprocess(self, query):

        # for clause in question["program"]:
        for clause in query:

            if clause['function'] == "Initial":
                if not len(self.vars) == 0:
                    self.stack.append(self.vars[-1])
                self.get_new_var()
                self.root = self.vars[-1]

            # logic operations
            elif clause['function'] == "And":

                v1 = self.vars[-1]
                v2 = self.stack.pop()
                self.get_new_var()
                var_id = self.vars[-1]

                self.operations.append(And(var_id, v1, v2))
                self.root = self.vars[-1]

            elif clause['function'] == "Or":
                v1 = self.vars[-1]
                v2 = self.stack.pop()
                self.get_new_var()
                var_id = self.vars[-1]

                self.operations.append(Or(var_id, v1, v2))
                self.root = self.vars[-1]

            # find operations
            elif clause['function'] == "KG_Find":
                self.vars[-1].set_kg(clause['text_input'])

            elif clause['function'] == "Hypernym_Find":
                self.vars[-1].set_hypernym(clause['text_input'])

            elif clause['function'] == "Find_Name":
                self.vars[-1].set_name(clause['text_input'])

            elif clause['function'] == "Find_Attr":
                self.vars[-1].set_attr(clause['text_input'])

            elif clause['function'] == "Relate_Reverse":
                self.get_new_var()
                self.root = self.vars[-1]
                obj = self.vars[-2]
                sub = self.vars[-1]
                rela_name = clause['text_input']
                relation = Relation(rela_name, sub, obj)
                self.relations.append(relation)

            elif clause['function'] == "Relate":
                self.get_new_var()
                self.root = self.vars[-1]
                sub = self.vars[-2]
                obj = self.vars[-1]
                rela_name = clause['text_input']
                relation = Relation(rela_name, sub, obj)
                self.relations.append(relation)

            else:
                raise Exception(f"Not handled function: {clause['function']}")
    


# Optimizers for optimization
class QueryOptimizer():

    def __init__(self, name):
        self.name = name

    def optimize(self, query):
        raise NotImplementedError

# This only works for one and operation at the end
# This is waited for update
# class AndQueryOptimizer(QueryOptimizer):

#     def __init__(self):
#         super().__init__("AndQueryOptimizer")

#     # For any and operation, this can be rewritten as a single object
#     def optimize(self, query):

#         if len(query.operations) == 0:
#             return query

#         assert(len(query.operations) == 1)

#         operation = query.operations[0]
#         # merge every subtree into one
#         if operation.name == "and":
#             v1 = operation.v1
#             v2 = operation.v2
#             v1.merge(v2)

#             for relation in query.relations:
#                 relation.substitute(v2, v1)

#             query.vars.remove(v2)

#             if query.root == operation:
#                 query.root = v1

#         return query


class HypernymOptimizer(QueryOptimizer):

    def __init__(self):
        super().__init__("HypernymOptimizer")

    def optimize(self, query):

        if (query.name is not None and not query.hypernyms == []):
            query.hypernyms = []

        return query


class KGOptimizer(QueryOptimizer):

    def __init__(self):
        super().__init__("HypernymOptimizer")

    def optimize(self, query):

        if (query.name is not None and not query.kgs == []):
            query.kgs = []

        return query

def remove_number(s):
    new_s = ""
    for i in s:
        if i == "(":
            break 
        else:
            new_s += i
    new_s = new_s.strip()
    return new_s

if __name__ == "__main__":
    question_dir = "/home/jianih/research/object_reference/CRIC_core/data/dataset/gqa/"
    question_path = os.path.join(question_dir, "questions/binary/val_balanced_questions.json")

    with open(question_path, 'r') as question_file:
        datapoints = json.load(question_file)
    
    for k, d in datapoints.items():
        q = d["semantic"]
        for c in q:
            c["function"] = c.pop("operation")

        query = Query(q, dataset="gqa")
        rust = query.to_rust()
    
    print("result")
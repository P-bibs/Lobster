import os
import re
import copy

class Rule():
  def __init__(self, head, body):
    self.head = head
    self.body = body

  def to_scl(self):
    return f"{self.head.to_scl()} = {', '.join([r.to_scl() for r in self.body])}"

  def add_bool(self):
    new_body_rels = []
    for rid, body_rel in enumerate(self.body):
      new_body_rels.append(body_rel.add_bool(rid))
    new_head = Relation(self.head.pred, self.head.tp + [' && '.join([f'B{i}' for i in range(len(self.body))])])
    new_rule = Rule(new_head, new_body_rels)
    return new_rule

  def add_entity(self):
    head_tp = self.head.tp
    new_body_rels = self.body
    for var in head_tp:
      new_body_rels += [Relation("entity", [var])]
      new_rule = Rule(self.head, new_body_rels)
    return new_rule

  def add_rel_prob(self, rule_pred):
    new_body_rels = self.body + [Relation(rule_pred, [])]
    new_rule = Rule(self.head, new_body_rels)
    return new_rule

class Relation():
  def __init__(self, pred, tp):
    assert (len(pred) > 0)
    assert (type(tp) == list)
    self.pred = pred
    self.tp = tp

  def to_scl(self):
    return f"{self.pred}({','.join(self.tp)})"

  def is_neg(self):
    return self.pred[:3] == 'neg'

  def add_bool(self, tid):
    new_rel = Relation(self.pred, self.tp + [f"B{tid}"])
    return new_rel

def get_type_header(all_preds):
    decls = []
    for pred in all_preds:
      decls.append(f"type {pred}(String, String)")
    decls.append("type entity(String)")
    decls.append("type answer()")
    return decls

def get_negative_rules(all_preds):
  all_negative_rules = []
  new_all_preds = copy.deepcopy(all_preds)
  for pred in all_preds:
    if pred[:3] == 'neg':
      continue
    neg_pred = 'neg' + pred

    head = Relation(neg_pred, ['A', 'B'])
    body = [Relation(f"~{pred}", ['A', 'B']), Relation('entity', ['A']), Relation('entity', ['B'])]
    rule1 = Rule(head, body)

    # head = Relation(pred, ['A', 'B'])
    # body = [Relation(f"~{neg_pred}", ['A', 'B']), Relation('entity', ['A']), Relation('entity', ['B'])]
    # rule2 = Rule(head, body)
    if not neg_pred in all_preds:
      new_all_preds.add(neg_pred)
    all_negative_rules += [rule1]
  return new_all_preds, all_negative_rules

def collect_rules(rule_dir):

    all_preds = set()
    rules = os.listdir(rule_dir)
    scl_rules = []

    for rid, raw_rule in enumerate(rules):
      preds, rule = collect_preds(raw_rule)
      all_preds.update(preds)
      scl_rule = to_scl_rule(rule)
      scl_rules.append(scl_rule)

    neg_preds, neg_rules = get_negative_rules(all_preds)
    scl_rules += ['rel ' + r.to_scl() for r in neg_rules]
    all_preds.update(neg_preds)

    return scl_rules, all_preds

def to_scl_rule(rule):
  rule = rule.add_entity()
  scl_rule = rule.to_scl()
  return scl_rule

def collect_preds(dl_rule):
  dl_rule = dl_rule.replace("relation", "relation_").replace("type", "type_")
  rule_info = dl_rule[:-1].split(':-')
  body_facts = rule_info[1].split('),')
  facts = [rule_info[0]] + [f + ')' for f in body_facts[:-1]] + [body_facts[-1]]
  relations = []
  preds = []
  for r in facts:
    r = r.strip()
    rela_info = r.split('(')
    tp = rela_info[1][:-1].split(',')
    pred = rela_info[0]
    preds.append(pred)
    relations.append(Relation(pred, tp))
  rule = Rule(relations[0], relations[1:])
  return set(preds), rule

def to_rule(dl_rule):
  dl_rule = dl_rule.replace("relation", "relation_").replace("type", "type_")[:-1]
  rule_info = dl_rule.split(':-')
  body_facts = rule_info[1].split('),')
  facts = [rule_info[0]] + [f + ')' for f in body_facts[:-1]] + [body_facts[-1]]
  relations = []
  for r in facts:
    r = r.strip()
    rela_info = r.split('(')
    tp = rela_info[1][:-1].split(',')
    pred = rela_info[0]
    relations.append(Relation(pred, tp))
  rule = Rule(relations[0], relations[1:])
  rule = rule.add_entity()
  return rule

if __name__ == "__main__":
    current_dir = os.path.abspath(__file__)
    data_dir = os.path.abspath(os.path.join(current_dir, '../../data/RuleBert'))
    all_preds = ["battle", "successor", "militarycommand", "parent", "relation_", "spouse", "child", "combatant", "birthplace", "predecessor", "relative", "residence", "deathplace"]
    all_preds = set(all_preds)
    neg_preds, neg_rules = get_negative_rules(all_preds)
    all_preds.update(neg_preds)
    decls = get_type_header(all_preds)

    rule_file_name = "scl/rulebert.scl"
    rule_path = os.path.join(data_dir, rule_file_name)
    with open(rule_path, 'w') as rule_file:
        rule_file.write('\n'.join(decls))
        rule_file.write('\n')
        rule_file.write('\n')
        rule_file.write('\n'.join(['rel ' + neg_rule.to_scl() for neg_rule in neg_rules]))

import pickle
import json

import scallopy

gt_events = pickle.load(open("example/ground_truth.pkl", "rb"))[0]

all_specs = json.load(open("example/specs.json"))

ctx = scallopy.Context()
ctx.add_program(open("solve.scl").read())

def loc_to_id(loc: str):
  if loc == "start": return 0
  elif loc == "middle": return 1
  elif loc == "end": return 2

def dur_to_id(dur: str):
  if dur == "short": return 0
  elif dur == "medium": return 1
  elif dur == "long": return 2
  elif dur == "all": return 3

batch_size = gt_events.shape[0]
for i in range(batch_size):
  dp_events = gt_events[i]
  dp_specs = [spec for spec in all_specs if spec[0] == i]
  if len(dp_specs) == 0: continue

  event_facts = [(event_id, bool(state_num), frame_id) for (frame_id, temp_vec) in enumerate(dp_events) for (event_id, state_num) in enumerate(temp_vec)]
  spec_facts = [(i, eid, st, loc_to_id(loc), dur_to_id(dur)) for (i, (eid, st, loc, dur)) in enumerate([(event_id, bool(state_num), loc, dur) for (_, event_id, descs) in dp_specs for (state_num, loc, dur) in descs])]

  temp_ctx = ctx.clone()
  temp_ctx.add_facts("event_at_frame", event_facts)
  temp_ctx.add_facts("spec", spec_facts)

  temp_ctx.run()

  result = len(list(temp_ctx.relation("match"))) > 0
  print(result)

  # if result is False:
  #   print(event_facts)
  #   print(spec_facts)
  #   print()
  #   print(list(temp_ctx.relation("match_spec")))
  #   print()
  #   print([fact for fact in event_facts if fact[0] == 33]) # 33,52,53
  #   print([fact for fact in event_facts if fact[0] == 52]) # 33,52,53
  #   print([fact for fact in event_facts if fact[0] == 53]) # 33,52,53
  #   break

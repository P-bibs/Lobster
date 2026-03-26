import pickle
import io
import tokenize


class Program:
  def __init__(self, prog_str, init_state=None):
    self.prog_str = prog_str
    self.state = init_state if init_state is not None else dict()
    self.instructions = self.prog_str.split('\n')
    self.progs = [parse_step(i) for i in self.instructions]

def parse_step(step_str, partial=False):
  tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
  output_var = tokens[0].string
  step_name = tokens[2].string
  parsed_result = dict(
    output_var=output_var,
    step_name=step_name)
  if partial:
    return parsed_result

  arg_tokens = [token for token in tokens[4:-3] if token.string not in [',','=']]
  num_tokens = len(arg_tokens) // 2
  args = dict()
  for i in range(num_tokens):
    args[arg_tokens[2*i].string] = arg_tokens[2*i+1].string
  parsed_result['args'] = args
  return parsed_result


GQA_PATH = "/workspace/aaai_data/gqa_mini_data/"

def get_dataset(fname):
  with open(GQA_PATH + fname, 'rb') as f:
    return pickle.load(f)

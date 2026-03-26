import json
import csv
import sys
import os
import argparse

parser = argparse.ArgumentParser(description='Convert a json clutrr dataset to csv')
parser.add_argument('input', type=str, help='Input json file')
parser.add_argument('output', type=str, help='Output csv file')

# read a json file and extract certain fields and output them as a csv file
def main():
  args = parser.parse_args()
  input_file = args.input
  output_file = args.output
  with open(input_file) as f:
    input_lines = f.readlines()
  output_lines = []

  # Need to extract ",id,story,query,text_query,target,text_target,clean_story,proof_state,f_comb,task_name,story_edges,edge_types,query_edge,genders,syn_story,node_mapping,task_split"
  for i, line in enumerate(input_lines):
    item = json.loads(line)
    number = str(i)
    id_ = hash(line)
    story = item['text_story']
    query = item['query']
    text_query = item['text_query']
    target = item['target']
    text_target = item['text_target']
    clean_story = item['clean_story']
    proof_state = item['proof_state']
    f_comb = item['f_comb']
    task_name = item['task_name']
    story_edges = item['story_edges']
    edge_types = item['edge_types']
    query_edge = item['query_edge']
    genders = item['gender']
    syn_story = item['syn_story']
    node_mapping = item['node_mapping']
    task_split = item['task_split']

    lines.append(",".join([number, id_, story, query, text_query, target, text_target, clean_story, proof_state, f_comb, task_name, story_edges, edge_types, query_edge, genders, syn_story, node_mapping, task_split]))

  with open(output_file, 'w') as f:
    f.write(",id,story,query,text_query,target,text_target,clean_story,proof_state,f_comb,task_name,story_edges,edge_types,query_edge,genders,syn_story,node_mapping,task_split\n")
    f.write("\n".join(lines))


main()

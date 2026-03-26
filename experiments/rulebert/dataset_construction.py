import os
import json
import random

def read_jsonl_dps(jsonl_path):
    with open(jsonl_path, 'r') as json_file:
        dataset = json_file.read()
        data = [json.loads(jline) for jline in dataset.splitlines()]
    return data

def gen_dataset(train_depths, test_depths, dataset_paths, target_path, test_ct):
    train_dps = []
    test_dps = []
    val_dps = []

    for train_depth in train_depths:
        dataset_path = dataset_paths[train_depth]
        train_path = os.path.join(dataset_path, 'train.jsonl')
        val_path = os.path.join(dataset_path, 'val.jsonl')
        train_dps += read_jsonl_dps(train_path)
        val_dps += read_jsonl_dps(val_path)
    for test_depth in test_depths:
        dataset_path = dataset_paths[test_depth]
        test_path = os.path.join(dataset_path, 'test.jsonl')
        test_dps += random.sample(read_jsonl_dps(test_path), test_ct)

    train_path = os.path.join(target_path, 'train.json')
    val_path = os.path.join(target_path, 'val.json')
    test_path = os.path.join(target_path, 'test.json')
    json.dump(train_dps, open(train_path, 'w'))
    json.dump(val_dps, open(val_path, 'w'))
    json.dump(test_dps, open(test_path, 'w'))

if __name__ == "__main__":
    current_dir = os.path.abspath(__file__)
    data_dir = os.path.abspath(os.path.join(current_dir, '../../data/RuleBert'))
    d0_dataset = os.path.join(data_dir, 'chain_rules/Depth_0')
    d1_dataset = os.path.join(data_dir, 'chain_rules/Depth_1')
    d2_dataset = os.path.join(data_dir, 'chain_rules/Depth_2')
    d3_dataset = os.path.join(data_dir, 'chain_rules/Depth_3')
    d4_dataset = os.path.join(data_dir, 'chain_rules/Depth_4')
    d5_dataset = os.path.join(data_dir, 'chain_rules/Depth_5')
    target_dataset_path = os.path.join(data_dir, 'chain_rules_train_0')

    dataset_paths = {
        0: d0_dataset,
        1: d1_dataset,
        2: d2_dataset,
        3: d3_dataset,
        4: d4_dataset,
        5: d5_dataset,
    }

    train_depths = [0]
    test_depths = [0, 1, 2, 3, 4, 5]
    test_ct = 2000
    gen_dataset(train_depths, test_depths, dataset_paths, target_dataset_path, test_ct)
    print('end')

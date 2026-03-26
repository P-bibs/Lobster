import os
import sys
import re
from itertools import product
from collections import defaultdict

OUT_DIR = "./generated/"

def find_macro_inner(lines, macro):
    for i, line in enumerate(lines):
        if line.strip() == f"#ifdef {macro}":
            for j, line in enumerate(lines[i+1:]):
                if line.strip() == "#endif":
                    return [l.strip() for l in lines[i+1:i+j+1]]
    return []

# For example:
#define TYPE u32 u32
#define TYPE u32 u32 u32
#define TYPE f32 u32 u32
param_re = re.compile(r"#define ([a-zA-Z0-9_]+) (.+)")

def replace_arg(text, arg_name, configuration):
    if arg_name.endswith("_INDEX"):
        real_arg_name = arg_name[:-6]
        arg_value = " ".join([str(i) for i, _ in enumerate(configuration[real_arg_name].split(" "))])
    else:
        arg_value = configuration[arg_name]

    arg_value = arg_value.replace("u32", "ValueU32")
    arg_value = arg_value.replace("f32", "ValueF32")
    arg_value = arg_value.replace(" ", ",")
    return text.replace(arg_name, arg_value)

def process(file_name, lines):
    params_raw = find_macro_inner(lines, "SPECIALIZE_PARAMS")
    body_raw = "\n".join(find_macro_inner(lines, "SPECIALIZE_BODY"))
    if len(params_raw) == 0 or len(body_raw) == 0:
        return []

    body_raw = f"#include \"../provenance.h\"\n" + body_raw

    print("Found params", params_raw)
    print("Found body", body_raw)

    param_map = defaultdict(list)
    for line in params_raw:
        m = param_re.match(line)
        if m:
            param_map[m.group(1)].append(m.group(2))

    param_names = param_map.keys()
    params = { name: [(name, arg) for arg in args] for name, args in param_map.items()}

    configurations = list(product(*params.values()))

    specializations = []
    for i, configuration in enumerate(configurations):
        configuration = dict(configuration)
        specialization = body_raw
        for name in configuration.keys():
            specialization = replace_arg(specialization, f"{name}_INDEX", configuration)
            specialization = replace_arg(specialization, name, configuration)
        specializations.append((f"{file_name}_spec_{i}.cu", specialization))
    return specializations

def main():
    # make output directory
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # clear files
    files = os.listdir(OUT_DIR)
    for file in files:
        os.remove(os.path.join(OUT_DIR, file))

    all_names = []
    files = os.listdir('.')
    for file in files:
        if file[-3:] == ".cu":
            with open(file, 'r') as f:
                lines = f.readlines()
            specializations = process(file, lines)
            if len(specializations) > 0:
                print("Specializations for ", file)
                for spec in specializations:
                    print(spec)
            for (path, body) in specializations:
                with open(os.path.join(OUT_DIR, path), "w") as f:
                    f.write(body)

            all_names.append([s for (s, _) in specializations])

    with open(os.path.join(OUT_DIR, "CMakelists.txt"), "w") as f:
        #f.write("cmake_minimum_required(VERSION 3.8)\n")
        #f.write("project(specializations)\n")
        #f.write("set(CMAKE_CXX_STANDARD 11)\n")
        #f.write("set(CMAKE_CXX_STANDARD_REQUIRED ON)\n")
        #f.write("set(CMAKE_CXX_EXTENSIONS OFF)\n")
        f.write("add_library(specializations OBJECT\n")
        for name in all_names:
            for n in name:
                f.write(f"    {n}\n")
        f.write(")\n")

main()

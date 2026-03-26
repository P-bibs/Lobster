import os
from itertools import groupby
from torch.utils.data import Dataset
from collections import defaultdict
import torch
import math
import scallopy

SCALLOP_FOLDER = os.path.abspath(os.path.join(__file__, "../../scallop"))

TOKENS = [
    "LEFT",
    "RIGHT",
    "UNPAIRED",
    "NULL",
]
TOKEN_ID_TO_NAME_MAP = {i: n for (i, n) in enumerate(TOKENS)}
NULL_TOKEN = TOKENS.index("NULL")

# Symbol(s)	Name
# A	Adenine
# C	Cytosine
# G	Guanine
# U T	Uracil
# Symbol(s)	Name	Matches
# N . X	Any base	A C G U
# V	Not U	A C G
# H	Not G	A C U
# D	Not C	A G U
# B	Not A	C G U
# M	Amino	A C
# R	Purine	A G
# W	Weak	A U
# S	Strong	C G
# Y	Pyrimidine	C U
# K	Keto	G U
BASE_NUCLEOTIDES_MAP = {
    "A": [0],
    "C": [1],
    "G": [2],
    "U": [3],
    "T": [3],
    "N": [0, 1, 2, 3],
    "X": [0, 1, 2, 3],
    "V": [0, 1, 2],
    "H": [0, 1, 3],
    "D": [0, 2, 3],
    "B": [1, 2, 3],
    "M": [0, 1],
    "R": [0, 2],
    "W": [0, 3],
    "S": [1, 2],
    "Y": [1, 3],
    "K": [2, 3],
}

NUCLEOTIDES = defaultdict(lambda: [4], BASE_NUCLEOTIDES_MAP)


class BPseqDataset(Dataset):
    def __init__(self, bpseq_list, max_length=None, percentage=100):
        self.bqseq_folder = os.path.dirname(bpseq_list)

        self.data = []
        with open(bpseq_list) as f:
            for l in f:
                l = l.rstrip('\n').split()
                if len(l) == 1:
                    self.data.append(self.read(l[0]))
                elif len(l) == 2:
                    self.data.append(self.read_pdb(l[0], l[1]))

        if max_length is not None:
            self.data = [d for d in self.data if len(d[1]) <= max_length]

        if percentage < 100:
            amount_to_keep = int(len(self.data) * percentage / 100)
            self.data = self.data[:amount_to_keep]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (filename, seq, labels) = self.data[idx]
        clean_labels = self.clean_labels(labels)
        dot_bracket = self.generate_dot_bracket_from_pair_index_matrix(seq, labels)
        clean_seq = self.clean_dot_bracket(seq)
        ref_tokens = self.compute_tokens(seq, dot_bracket)
        return (filename, clean_seq, labels, dot_bracket, ref_tokens)

    def clean_labels(self, labels):
        for (i, j) in enumerate(labels):
            if abs(i - j) == 1:
                labels[i] = 0
        return labels

    def clean_dot_bracket(self, seq):
        seq = seq.replace("()", "..")
        return seq

    def read(self, filename):
        with open(f"{self.bqseq_folder}/{filename}") as f:
            structure_is_known = True
            p = [0]
            s = ['']
            for l in f:
                if not l.startswith('#'):
                    l = l.rstrip('\n').split()
                    if len(l) == 3:
                        if not structure_is_known:
                            raise('invalid format: {}'.format(filename))
                        idx, c, pair = l
                        pos = 'x.<>|'.find(pair)
                        if pos >= 0:
                            idx, pair = int(idx), -pos
                        else:
                            idx, pair = int(idx), int(pair)
                        s.append(c)
                        p.append(pair)
                    elif len(l) == 4:
                        structure_is_known = False
                        idx, c, nll_unpaired, nll_paired = l
                        s.append(c)
                        nll_unpaired = math.nan if nll_unpaired=='-' else float(nll_unpaired)
                        nll_paired = math.nan if nll_paired=='-' else float(nll_paired)
                        p.append([nll_unpaired, nll_paired])
                    else:
                        raise('invalid format: {}'.format(filename))

        if structure_is_known:
            seq = ''.join(s)
            ref_structure = torch.tensor(p)
            return (filename, seq, ref_structure)
        else:
            seq = ''.join(s)
            p.pop(0)
            return (filename, seq, torch.tensor(p))

    def fasta_iter(self, fasta_name):
        fh = open(fasta_name)
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

        for header in faiter:
            # drop the ">"
            headerStr = header.__next__()[1:].strip()

            # join all sequence lines to one.
            seq = "".join(s.strip() for s in faiter.__next__())

            yield (headerStr, seq)

    def read_pdb(self, seq_filename, label_filename):
        it = self.fasta_iter(seq_filename)
        h, seq = next(it)

        p = []
        with open(label_filename) as f:
            for l in f:
                l = l.rstrip('\n').split()
                if len(l) == 2 and l[0].isdecimal() and l[1].isdecimal():
                    p.append([int(l[0]), int(l[1])])

        return (h, seq, torch.tensor(p))

    def generate_dot_bracket_from_pair_index_matrix(self, seq, pair_id_matrix):
        ss = ""
        for (i, j) in enumerate(pair_id_matrix):
            if i == 0: continue
            if j > 0: ss += '(' if j > i else ')'
            else: ss += '.'
        return ss

    def compute_tokens(self, seq, dot_bracket):
        tokens = [None] * len(seq)
        for (i, t) in enumerate(dot_bracket):
            if t == '.': tokens[i] = TOKENS.index("UNPAIRED")
            elif t == '(': tokens[i] = TOKENS.index("LEFT")
            elif t == ')': tokens[i] = TOKENS.index("RIGHT")
        return tokens

    @staticmethod
    def collate_fn(datapoints):
        batch_size = len(datapoints)
        lens = [len(dp[1]) for dp in datapoints]
        max_len = max(lens)

        # 1. File names
        fnames = [dp[0] for dp in datapoints]

        # 2. Inputs
        inputs = [dp[1].ljust(max_len, '0') for dp in datapoints]

        # 3. Labels (BP Seq)
        padded_size = max_len + 1
        labels = torch.zeros((batch_size, padded_size), dtype=torch.int)
        for (i, dp) in enumerate(datapoints):
            dp_size = dp[2].shape[0]
            labels[i, 0:dp_size] = dp[2]

        # 4. Dot-Bracket
        dot_brackets = [dp[3] for dp in datapoints]

        # 5. Tokens
        tokens = torch.zeros((batch_size, max_len, len(TOKENS)))
        for (i, dp) in enumerate(datapoints):
            for (j, token) in enumerate(dp[4]):
                tokens[i, j, token] = 1
            for j in range(lens[i], max_len):
                tokens[i, j, NULL_TOKEN] = 1

        return (fnames, inputs, labels, dot_brackets, tokens)

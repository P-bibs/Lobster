import os
from itertools import groupby
from torch.utils.data import Dataset
import torch
import math

DATA_FOLDER = os.path.abspath(os.path.join(__file__, "../../../data/rnafold"))

class BPseqDataset(Dataset):
    def __init__(self, bpseq_list, percentage=100):
        self.data = []
        with open(bpseq_list) as f:
            for l in f:
                l = l.rstrip('\n').split()
                if len(l)==1:
                    self.data.append(self.read(l[0]))
                elif len(l)==2:
                    self.data.append(self.read_pdb(l[0], l[1]))

        if percentage < 100:
            amount_to_keep = int(len(self.data) * percentage / 100)
            self.data = self.data[:amount_to_keep]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def read(self, filename):
        with open(f"{DATA_FOLDER}/{filename}") as f:
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
            return (filename, seq, torch.tensor(p))
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

    @staticmethod
    def collate_fn(datapoints):
        batch_size = len(datapoints)
        lens = [len(dp[1]) for dp in datapoints]
        padded_size = max(lens) + 1
        fnames = [dp[0] for dp in datapoints]
        inputs = [dp[1].ljust(max(lens), '0') for dp in datapoints]
        labels = torch.zeros((batch_size, padded_size), dtype=torch.int)
        for (i, dp) in enumerate(datapoints):
            dp_size = dp[2].shape[0]
            labels[i, 0:dp_size] = dp[2]
        return (fnames, inputs, labels)

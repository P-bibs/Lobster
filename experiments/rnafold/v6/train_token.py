import os
import sys
import random
import time
from pathlib import Path
import argparse
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import scallopy

from utils.dataset import BPseqDataset
from utils.compbpseq import compare_bpseq, accuracy

THIS_FOLDER = os.path.abspath(os.path.join(__file__, "../"))
DATA_FOLDER = os.path.abspath(os.path.join(__file__, "../../../data/rnafold"))
MODEL_FOLDER = os.path.abspath(os.path.join(__file__, "../../../model/rnafold"))

TOKENS = [
    "HELIX_STACK_LEFT",
    "HELIX_STACK_RIGHT",
    "HAIRPIN_LEFT",
    "HAIRPIN_RIGHT",
    "HAIRPIN_UNPAIRED",
    "INTERN_LOOP_LEFT",
    "INTERN_LOOP_RIGHT",
    "INTERN_LOOP_UNPAIRED",
    "MULTI_LOOP_LEFT",
    "MULTI_LOOP_RIGHT",
    "MULTI_LOOP_UNPAIRED",
    "DANGLE",
]
TOKEN_ID_TO_NAME_MAP = {i: n for (i, n) in enumerate(TOKENS)}

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, latent_dim, max_len=500):
        super(SinusoidalPositionalEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / latent_dim))
        pe = torch.zeros(max_len, latent_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class RNASeqTransformer(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers, dropout=0.1, max_len=1000):
        super(RNASeqTransformer, self).__init__()

        # Embedding dimension for each token (A, C, G, U, padding, unknown)
        self.embedding_dim = 6
        self.latent_dim = latent_dim

        # Linear layer to project one-hot encoded vectors to latent space (B x N x latent_dim)
        self.input_projection = nn.Linear(self.embedding_dim, latent_dim)

        # Positional encoding (sinusoidal)
        self.positional_encoding = SinusoidalPositionalEncoding(latent_dim, max_len)

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Output projection (decoder)
        # Layer Norm after encoding
        self.layer_norm = nn.LayerNorm(latent_dim)

        # Decoder (MLP for output prediction)
        self.fc_out = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, len(TOKENS))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, sequences, src_key_padding_mask=None):
        """
        sequences: List of RNA sequences (B x N strings)
        returns: Decoded sequences in the output space (B x N x O)
        """
        B, N = len(sequences), len(sequences[0])  # Batch size B and sequence length N

        # Step 1: One-hot encode the input RNA sequences (B x N x 5)
        rna_map = defaultdict(lambda: 5, {'0': 0, 'A': 1, 'a': 1, 'C': 2, 'c': 2, 'G': 3, 'g': 3, 'U': 4, 'u': 4})
        one_hot = torch.zeros(B, N, self.embedding_dim)

        for i, seq in enumerate(sequences):
            for j, char in enumerate(seq):
                one_hot[i, j, rna_map[char]] = 1

        # Step 2: Project the one-hot encoded tensor to latent_dim (B x N x latent_dim)
        one_hot = self.input_projection(one_hot)

        # Step 3: Add positional encoding to the one-hot tensor (B x N x latent_dim)
        one_hot = self.positional_encoding(one_hot)

        # Step 4: Use the transformer to encode the sequence (B x N x latent_dim)
        encoded = self.transformer_encoder(one_hot, src_key_padding_mask=src_key_padding_mask)

        # Step 5: Layer normalization after the transformer
        encoded = self.layer_norm(encoded)

        # Step 6: Decode to output dimension using MLP (B x N x O)
        output = self.fc_out(self.dropout(encoded))
        return torch.softmax(output, dim=-1)


class Train:
    def __init__(self):
        self.step = 0

        self.extract_token = scallopy.Context()
        self.extract_token.import_file(f"{THIS_FOLDER}/scallop/extract_token.scl")

        self.infer_structure = scallopy.Context(provenance="topkproofs", k=1)
        self.infer_structure.import_file(f"{THIS_FOLDER}/scallop/infer_structure.scl")
        self.infer_structure.set_non_probabilistic(["length"])

        self.train_loader = None
        self.test_loader = None

    def train(self, epoch):
        self.model.train()
        n_dataset = len(self.train_loader.dataset)
        loss_total, token_acc_total, num = 0, 0, 0
        running_loss, n_running_loss = 0, 0
        sen_total, ppv_total, fval_total, mcc_total = 0, 0, 0, 0
        # start = time.time()
        gt_token_stats, pred_token_stats = defaultdict(lambda: 0), defaultdict(lambda: 0)
        with tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            pbar.set_description(f"[Train Epoch {epoch}]")
            for batch_id, (fnames, seqs, pairs) in enumerate(self.train_loader):
                if self.verbose:
                    print()
                    print("Step: {}, {}".format(self.step, fnames))
                    self.step += 1
                n_batch = len(seqs)
                self.optimizer.zero_grad()
                y = self.model(seqs)
                self.populate_token_stats(y, seqs, pred_token_stats)

                # adj_mat = self.parse_prob_tokens_to_structure(y, seqs, retain_k=self.inference_retain_k)
                # bps = self.generate_bpseq_from_adj_mat(adj_mat)

                loss, token_acc = self.loss(y, seqs, pairs, fname=fnames, token_stats=gt_token_stats)
                loss_total += loss.item()
                token_acc_total += token_acc
                num += n_batch
                if loss.item() > 0.:
                    loss.backward()
                    if self.verbose:
                        for n, p in self.model.named_parameters():
                            print(n, torch.min(p).item(), torch.max(p).item(), torch.min(p.grad).item(), torch.max(p.grad).item())
                    self.optimizer.step()

                # for header, ref, bp in zip(fnames, pairs, bps):
                #     x = compare_bpseq(ref, bp)
                #     [sen, ppv, fval, mcc] = list(accuracy(*x))
                #     sen_total += sen
                #     ppv_total += ppv
                #     fval_total += fval
                #     mcc_total += mcc

                denom = batch_id + 1
                pbar.set_postfix(
                    train_loss='{:.2e}'.format(loss_total / denom),
                    token_acc='{:.2f}'.format(token_acc_total / denom),
                    # sen=f'{sen_total / num:.2f}',
                    # ppv=f'{ppv_total / num:.2f}',
                    # fval=f'{fval_total / num:.2f}',
                )
                pbar.update(n_batch)

                running_loss += loss.item()
                n_running_loss += n_batch
                if n_running_loss >= 100 or num >= n_dataset:
                    running_loss /= n_running_loss
                    if self.writer is not None:
                        self.writer.add_scalar("train/loss", running_loss, (epoch-1) * n_dataset + num)
                    running_loss, n_running_loss = 0, 0

            if self.verbose:
                print("GT:", dict(gt_token_stats))
                print("Pred:", dict(pred_token_stats))

            # elapsed_time = time.time() - start
            # pbar.set_description(f"[Train Epoch {epoch}] Time: {elapsed_time:.3f}s")

    def test(self, epoch):
        self.model.eval()
        n_dataset = len(self.test_loader.dataset)
        loss_total, token_acc_total, num = 0, 0, 0
        sen_total, ppv_total, fval_total, mcc_total = 0, 0, 0, 0
        gt_token_stats, pred_token_stats = defaultdict(lambda: 0), defaultdict(lambda: 0)
        # start = time.time()
        with torch.no_grad(), tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            pbar.set_description(f"[Test Epoch {epoch}]")
            for batch_id, (fnames, seqs, refs) in enumerate(self.test_loader):
                n_batch = len(seqs)
                y = self.model(seqs)
                adj_mat = self.parse_prob_tokens_to_structure(y, seqs, retain_k=self.inference_retain_k)
                bps = self.generate_bpseq_from_adj_mat(adj_mat)
                loss, token_acc = self.loss(y, seqs, refs, fname=fnames, token_stats=gt_token_stats)
                self.populate_token_stats(y, seqs, pred_token_stats)
                loss_total += loss.item()
                token_acc_total += token_acc
                denom = batch_id + 1
                for header, ref, bp in zip(fnames, refs, bps):
                    x = compare_bpseq(ref, bp)
                    [sen, ppv, fval, mcc] = list(accuracy(*x))
                    sen_total += sen
                    ppv_total += ppv
                    fval_total += fval
                    mcc_total += mcc
                num += n_batch
                pbar.set_postfix(
                    test_loss='{:.3e}'.format(loss_total / denom),
                    token_acc='{:.2f}'.format(token_acc_total / denom),
                    sen=f"{sen_total / num:.3f}",
                    ppv=f"{ppv_total / num:.3f}",
                    fval=f"{fval_total / num:.3f}",
                )
                pbar.update(n_batch)

            if self.verbose:
                print("GT:", dict(gt_token_stats))
                print("Pred:", dict(pred_token_stats))

            if self.writer is not None:
                self.writer.add_scalar("test/loss", loss_total / num, epoch * n_dataset)
            # elapsed_time = time.time() - start
            # pbar.set_description(f"[Test Epoch {epoch}]\tLoss: {loss_total / num:.6f}\tTime: {elapsed_time:.3f}s")

    def to_scallop_facts(self, y, seqs, dp_idx):
        s = ""
        for (i, row) in enumerate(y[dp_idx]):
            s += "rel token = {"
            for (j, p) in enumerate(row):
                if j > 0:
                    s += "; "
                s += f"{float(p):.3f}::({i}, {TOKEN_ID_TO_NAME_MAP[j]})"
            s += "}\n"
        return s

    def save_checkpoint(self, outdir, epoch):
        filename = os.path.join(outdir, 'epoch-{}'.format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filename)

    def resume_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return epoch

    def build_optimizer(self, optimizer, model, lr, l2_weight):
        if optimizer == 'Adam':
            return optim.Adam(model.parameters(), lr=lr, amsgrad=False, weight_decay=l2_weight)
        elif optimizer =='AdamW':
            return optim.AdamW(model.parameters(), lr=lr, amsgrad=False, weight_decay=l2_weight)
        elif optimizer == 'RMSprop':
            return optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2_weight)
        elif optimizer == 'SGD':
            return optim.SGD(model.parameters(), nesterov=True, lr=lr, momentum=0.9, weight_decay=l2_weight)
        elif optimizer == 'ASGD':
            return optim.ASGD(model.parameters(), lr=lr, weight_decay=l2_weight)
        else:
            raise('not implemented')

    def loss(self, y, seqs, pairs, dot_brackets=None, fname=None, token_stats=None):
        gt_tokens = self.compute_tokens(y, seqs, pairs)
        if token_stats is not None:
            self.populate_token_stats(gt_tokens, seqs, stats=token_stats)

        # Compute loss
        l = self.loss_fn(y, gt_tokens)

        # Get
        gt_token_ids = torch.argmax(gt_tokens, dim=2).view(-1)
        pred_token_ids = torch.argmax(y, dim=2).view(-1)
        percentage = float(torch.sum(gt_token_ids == pred_token_ids)) / len(gt_token_ids)

        return l, percentage

    def compute_tokens(self, y, seqs, ref_structures):
        (B, N, _) = y.shape
        gt_tokens = torch.zeros((B, N, len(TOKENS)))
        for (db_idx, (seq, ref_structure)) in enumerate(zip(seqs, ref_structures)):
            dot_bracket = self.generate_dot_bracket_from_pair_index_matrix(seq, ref_structure)
            temp_ctx = self.extract_token.clone()
            temp_ctx.add_facts("ss", [(dot_bracket,)])
            temp_ctx.run()
            for (i, t) in temp_ctx.relation("token"):
                gt_tokens[db_idx, i, t] = 1
        return gt_tokens

    def populate_token_stats(self, tokens_tensor, seqs, stats = defaultdict(lambda: 0)):
        for (tokens, seq) in zip(tokens_tensor, seqs):
            for i in range(len(seq)):
                token_name = TOKENS[torch.argmax(tokens[i])]
                stats[token_name] += 1
        return stats

    def generate_dot_bracket_from_pair_index_matrix(self, seq, pair_id_matrix):
        N = len(seq)
        ss = ""
        for (i, j) in enumerate(pair_id_matrix):
            if i == 0: continue
            if j > 0: ss += '(' if j > i else ')'
            else: ss += '.'
        return ss

    def parse_prob_tokens_to_structure(self, y, seqs, retain_k: Optional[int] = None, always_put_dangle: bool = False):
        (B, N, F) = y.shape
        DANGLE = TOKENS.index("DANGLE")
        pred_structure = torch.zeros((B, N + 1, N + 1))
        for (dp_idx, prob_tokens) in enumerate(y):
            # Prepare the input
            if retain_k is not None:
                token_facts = []
                token_disjunctions = [[] for i in range(N)]
                for i in range(N):
                    sampled_top_k = torch.topk(prob_tokens[i], retain_k, dim=0)
                    values, indices = list(sampled_top_k.values), list(sampled_top_k.indices)
                    if always_put_dangle and DANGLE not in indices:
                        indices.append(DANGLE)
                        values.append(prob_tokens[i, DANGLE])
                    for (p, j) in zip(values, indices):
                        fact_id = len(token_facts)
                        token_facts.append((p, (i, j)))
                        token_disjunctions[i].append(fact_id)
            else:
                token_facts = [(prob_tokens[i, j], (i, j)) for i in range(N) for j in range(F)]
                token_disjunctions = [[i * F + j for j in range(F)] for i in range(N)]

            # Execute the scallop program
            length = [(len(seqs[dp_idx]),)]
            temp_ctx = self.infer_structure.clone()
            temp_ctx.add_facts("token", token_facts, disjunctions=token_disjunctions)
            temp_ctx.add_facts("length", length)
            temp_ctx.run()

            # Vectorize the output into
            pred_structure[dp_idx, 0, 0] = 1
            for (p, (i, j)) in temp_ctx.relation("pair"):
                pred_structure[dp_idx, i + 1, j + 1] += p
                pred_structure[dp_idx, j + 1, i + 1] += p
            # for (p, (i, j)) in temp_ctx.relation("pair_residual"):
            #     pred_structure[dp_idx, i + 1, j + 1] += 0.1 * p
            #     pred_structure[dp_idx, j + 1, i + 1] += 0.1 * p
            for (p, (i,)) in temp_ctx.relation("unpaired"):
                pred_structure[dp_idx, i + 1, 0] += p
            # for (p, (i,)) in temp_ctx.relation("unpaired_residual"):
            #     pred_structure[dp_idx, i + 1, 0] += 0.1 * p
            for i in range(len(seqs[dp_idx]), N):
                pred_structure[dp_idx, i + 1, 0] = 1
        return pred_structure

    def generate_bpseq_from_adj_mat(self, adj_mat):
        return adj_mat.argmax(dim=2)

    def build_loss_function(self, loss_func, model, args):
        return torch.nn.BCELoss()

    def build_model(self, args):
        config = {
            'latent_dim' : args.embed_size,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'dropout': args.dropout_rate,
        }

        model = RNASeqTransformer(**config)

        return model, config

    def save_config(self, file, config):
        with open(file, 'w') as f:
            for k, v in config.items():
                k = '--' + k.replace('_', '-')
                if type(v) is bool: # pylint: disable=unidiomatic-typecheck
                    if v:
                        f.write('{}\n'.format(k))
                elif isinstance(v, list) or isinstance(v, tuple):
                    for vv in v:
                        f.write('{}\n{}\n'.format(k, vv))
                else:
                    f.write('{}\n{}\n'.format(k, v))

    def run(self, args, conf=None):
        self.disable_progress_bar = args.disable_progress_bar
        self.verbose = args.verbose
        self.writer = None
        if args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)

        train_dataset = BPseqDataset(args.input, percentage=args.train_percentage, max_length=args.max_length)
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=BPseqDataset.collate_fn)
        if args.test_input is not None:
            test_dataset = BPseqDataset(args.test_input, percentage=args.test_percentage, max_length=args.max_length)
            self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=BPseqDataset.collate_fn)

        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.model, config = self.build_model(args)
        config.update({ 'param': args.param })
        self.inference_retain_k = args.inference_retain_k

        if args.init_param != '':
            init_param = Path(args.init_param)
            if not init_param.exists() and conf is not None:
                init_param = Path(conf) / init_param
            p = torch.load(init_param)
            if isinstance(p, dict) and 'model_state_dict' in p:
                p = p['model_state_dict']
            self.model.load_state_dict(p)

        if args.gpu >= 0:
            self.model.to(torch.device("cuda", args.gpu))
        self.optimizer = self.build_optimizer(args.optimizer, self.model, args.lr, args.l2_weight)
        self.loss_fn = self.build_loss_function(args.loss_func, self.model, args)

        checkpoint_epoch = 0
        if args.resume is not None:
            checkpoint_epoch = self.resume_checkpoint(args.resume)

        if self.test_loader is not None:
            self.test(0)
        for epoch in range(checkpoint_epoch+1, args.epochs+1):
            self.train(epoch)
            if self.test_loader is not None:
                self.test(epoch)
            if args.log_dir is not None:
                self.save_checkpoint(args.log_dir, epoch)

        if args.param is not None:
            model_save_dir = f"{MODEL_FOLDER}/v6/{args.param}"
            model_save_folder = os.path.dirname(model_save_dir)
            os.makedirs(model_save_folder, exist_ok=True)
            torch.save(self.model.state_dict(), model_save_dir)
        if args.save_config is not None:
            self.save_config(args.save_config, config)

        return self.model


    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser('train', help='training')
        # input
        subparser.add_argument('input', type=str, help='Training data of the list of BPSEQ-formatted files')
        subparser.add_argument('--test-input', type=str, help='Test data of the list of BPSEQ-formatted files')
        subparser.add_argument('--gpu', type=int, default=-1, help='use GPU with the specified ID (default: -1 = CPU)')
        subparser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
        subparser.add_argument('--param', type=str, default='param.pth', help='output file name of trained parameters')
        subparser.add_argument('--init-param', type=str, default='', help='the file name of the initial parameters')
        subparser.add_argument('--train-percentage', type=float, default=100)
        subparser.add_argument('--test-percentage', type=float, default=100)
        subparser.add_argument('--max-length', type=int)
        subparser.add_argument('--batch-size', type=int, default=8)

        gparser = subparser.add_argument_group("Training environment")
        gparser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
        gparser.add_argument('--log-dir', type=str, default=None, help='Directory for storing logs')
        gparser.add_argument('--resume', type=str, default=None, help='Checkpoint file for resume')
        gparser.add_argument('--save-config', type=str, default=None, help='save model configurations')
        gparser.add_argument('--disable-progress-bar', action='store_true', help='disable the progress bar in training')
        gparser.add_argument('--verbose', action='store_true', help='enable verbose outputs for debugging')

        gparser = subparser.add_argument_group("Inference")
        gparser.add_argument('--inference-retain-k', type=int)

        gparser = subparser.add_argument_group("Optimizer setting")
        gparser.add_argument('--optimizer', choices=('Adam', 'AdamW', 'RMSprop', 'SGD', 'ASGD'), default='Adam')
        gparser.add_argument('--l1-weight', type=float, default=0., help='the weight for L1 regularization (default: 0)')
        gparser.add_argument('--l2-weight', type=float, default=0., help='the weight for L2 regularization (default: 0)')
        gparser.add_argument('--embedreg-lambda', type=float, default=0., help='regularization strength for embedding layer (default: 0)')
        gparser.add_argument('--score-loss-weight', type=float, default=1., help='the weight for score loss for hinge_mix loss (default: 1)')
        gparser.add_argument('--lr', type=float, default=0.0001, help='the learning rate for optimizer (default: 0.0001)')
        gparser.add_argument('--loss-func', choices=('hinge', 'hinge_mix'), default='hinge', help="loss fuction ('hinge', 'hinge_mix') ")

        gparser = subparser.add_argument_group("Network setting")
        gparser.add_argument('--embed-size', type=int, default=64, help='the dimension of embedding')
        gparser.add_argument('--num-layers', type=int, default=4, help='the number of the transformer layers')
        gparser.add_argument('--num-heads', type=int, default=4, help='the number of the transformer heads')
        gparser.add_argument('--dropout-rate', type=float, default=0.1, help='dropout rate of the Transformer units (default: 0.1)')

        subparser.set_defaults(func = lambda args, conf: Train().run(args, conf))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNA Fold with End-to-End Neural Model")
    subparser = parser.add_subparsers(title='Sub-commands')
    parser.set_defaults(func = lambda args, conf: parser.print_help())
    Train.add_args(subparser)

    DEFAULT_ARGS = [
        "train",
        "--test-input", f"{DATA_FOLDER}/TestSetA.lst",
        "--train-percentage", "100",
        "--test-percentage", "100",
        "--max-length", "100",
        "--embed-size", "64",
        "--num-layers", "4",
        "--num-heads", "32",
        "--lr", "0.0001",
        "--epoch", "1000",
        "--inference-retain-k", "0",
        "--batch-size", "4",
        f"{DATA_FOLDER}/TrainSetA.lst",
    ]
    args = parser.parse_args(DEFAULT_ARGS)

    # args = parser.parse_args()

    conf = list(filter(lambda x: x[0]=='@', sys.argv))
    conf = None if len(conf) == 0 else conf[-1][1:]

    args.func(args, conf)

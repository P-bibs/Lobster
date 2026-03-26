import os
import sys
import random
import time
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BPseqDataset
from layers import NeuralNet
from compbpseq import compare_bpseq, accuracy


DATA_FOLDER = os.path.abspath(os.path.join(__file__, "../../../data/rnafold"))


class EndToEndNeural(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = NeuralNet(**kwargs,
            n_out_paired_layers=2,
            n_out_unpaired_layers=0,
            exclude_diag=True)

    def forward(self, seqs, pairs):
        score_paired, _ = self.net(seqs)
        score_paired_softmax = torch.softmax(score_paired, dim=3)
        bond = score_paired_softmax[:, :, :, 1]
        return bond


class Train:
    def __init__(self):
        self.step = 0
        self.train_loader = None
        self.test_loader = None

    def train(self, epoch):
        self.model.train()
        n_dataset = len(self.train_loader.dataset)
        loss_total, num = 0, 0
        running_loss, n_running_loss = 0, 0
        sen_total, ppv_total, fval_total, mcc_total = 0, 0, 0, 0
        start = time.time()
        with tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            for fnames, seqs, pairs in self.train_loader:
                if self.verbose:
                    print()
                    print("Step: {}, {}".format(self.step, fnames))
                    self.step += 1
                n_batch = len(seqs)
                self.optimizer.zero_grad()
                y = self.model(seqs, pairs)
                bps = self.generate_pair_matrix_from_adjacency_matrix(y)
                loss = self.loss(y, seqs, pairs, fname=fnames)
                loss_total += loss.item()
                num += n_batch
                if loss.item() > 0.:
                    loss.backward()
                    if self.verbose:
                        for n, p in self.model.named_parameters():
                            print(n, torch.min(p).item(), torch.max(p).item(), torch.min(p.grad).item(), torch.max(p.grad).item())
                    self.optimizer.step()

                for header, ref, bp in zip(fnames, pairs, bps):
                    x = compare_bpseq(ref, bp)
                    [sen, ppv, fval, mcc] = list(accuracy(*x))
                    sen_total += sen
                    ppv_total += ppv
                    fval_total += fval
                    mcc_total += mcc

                pbar.set_postfix(
                    train_loss=f'{loss_total/num:.3e}',
                    sen=f'{sen_total/num:.3f}',
                    ppv=f'{ppv_total/num:.3f}',
                    fval=f'{fval_total/num:.3f}',
                )
                pbar.update(n_batch)

                running_loss += loss.item()
                n_running_loss += n_batch
                if n_running_loss >= 100 or num >= n_dataset:
                    running_loss /= n_running_loss
                    if self.writer is not None:
                        self.writer.add_scalar("train/loss", running_loss, (epoch-1) * n_dataset + num)
                    running_loss, n_running_loss = 0, 0
        elapsed_time = time.time() - start
        if self.verbose:
            print()
        print('Train Epoch: {}\tLoss: {:.6f}\tTime: {:.3f}s'.format(epoch, loss_total / num, elapsed_time))

    def test(self, epoch):
        self.model.eval()
        n_dataset = len(self.test_loader.dataset)
        loss_total, num = 0, 0
        sen_total, ppv_total, fval_total, mcc_total = 0, 0, 0, 0
        start = time.time()
        with torch.no_grad(), tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            for fnames, seqs, refs in self.test_loader:
                n_batch = len(seqs)
                y = self.model(seqs, refs)
                bps = self.generate_pair_matrix_from_adjacency_matrix(y)
                loss = self.loss(y, seqs, refs, fname=fnames)
                loss_total += loss.item()
                for header, ref, bp in zip(fnames, refs, bps):
                    x = compare_bpseq(ref, bp)
                    [sen, ppv, fval, mcc] = list(accuracy(*x))
                    sen_total += sen
                    ppv_total += ppv
                    fval_total += fval
                    mcc_total += mcc
                num += n_batch
                pbar.set_postfix(
                    test_loss='{:.3e}'.format(loss_total / num),
                    sen=f"{sen_total / num:.3f}",
                    ppv=f"{ppv_total / num:.3f}",
                    fval=f"{fval_total / num:.3f}",
                )
                pbar.update(n_batch)

        elapsed_time = time.time() - start
        if self.writer is not None:
            self.writer.add_scalar("test/loss", loss_total / num, epoch * n_dataset)
        print('Test Epoch: {}\tLoss: {:.6f}\tTime: {:.3f}s'.format(epoch, loss_total / num, elapsed_time))

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

    def build_model(self, args):
        config = {
            'max_helix_length': args.max_helix_length,
            'embed_size' : args.embed_size,
            'n_linear_layers': args.n_linear_layers,
            'num_filters': args.num_filters if args.num_filters is not None else (96,),
            'filter_size': args.filter_size if args.filter_size is not None else (5,),
            'pool_size': args.pool_size if args.pool_size is not None else (1,),
            'dilation': args.dilation,
            'num_lstm_layers': args.num_lstm_layers,
            'num_lstm_units': args.num_lstm_units,
            'num_transformer_layers': args.num_transformer_layers,
            'num_transformer_hidden_units': args.num_transformer_hidden_units,
            'num_transformer_att': args.num_transformer_att,
            'num_hidden_units': args.num_hidden_units if args.num_hidden_units is not None else (32,),
            'num_paired_filters': args.num_paired_filters,
            'paired_filter_size': args.paired_filter_size,
            'dropout_rate': args.dropout_rate,
            'fc_dropout_rate': args.fc_dropout_rate,
            'num_att': args.num_att,
            'pair_join': args.pair_join,
            'no_split_lr': args.no_split_lr,
        }

        model = EndToEndNeural(**config)

        return model, config


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

    def loss(self, y, seqs, pairs, fname):
        (B, N, N) = y.shape
        gt = torch.zeros((B, N, N))
        for (dp_index, dp) in enumerate(pairs):
            for (i, j) in enumerate(dp):
                gt[dp_index, i, j] = 1
                if j != 0:
                    gt[dp_index, j, i] = 1
        l = self.loss_fn(y.reshape(B * N, N), gt.reshape(B * N, N))
        return l

    def generate_pair_matrix_from_adjacency_matrix(self, adj_mat: torch.Tensor):
        return adj_mat.argmax(dim=2)

    def build_loss_function(self, loss_func, model, args):
        return torch.nn.BCELoss()

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

        train_dataset = BPseqDataset(args.input, args.train_percentage)
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=BPseqDataset.collate_fn)
        if args.test_input is not None:
            test_dataset = BPseqDataset(args.test_input, args.test_percentage)
            self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=BPseqDataset.collate_fn)

        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.model, config = self.build_model(args)
        config.update({ 'param': args.param })

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
            torch.save(self.model.state_dict(), args.param)
        if args.save_config is not None:
            self.save_config(args.save_config, config)

        return self.model


    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser('train', help='training')
        # input
        subparser.add_argument('input', type=str,
                            help='Training data of the list of BPSEQ-formatted files')
        subparser.add_argument('--test-input', type=str,
                            help='Test data of the list of BPSEQ-formatted files')
        subparser.add_argument('--gpu', type=int, default=-1,
                            help='use GPU with the specified ID (default: -1 = CPU)')
        subparser.add_argument('--seed', type=int, default=0, metavar='S',
                            help='random seed (default: 0)')
        subparser.add_argument('--param', type=str, default='param.pth',
                            help='output file name of trained parameters')
        subparser.add_argument('--init-param', type=str, default='',
                            help='the file name of the initial parameters')
        subparser.add_argument('--train-percentage', type=float, default=100)
        subparser.add_argument('--test-percentage', type=float, default=100)
        subparser.add_argument('--batch-size', type=int, default=1)

        gparser = subparser.add_argument_group("Training environment")
        subparser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
        subparser.add_argument('--log-dir', type=str, default=None,
                            help='Directory for storing logs')
        subparser.add_argument('--resume', type=str, default=None,
                            help='Checkpoint file for resume')
        subparser.add_argument('--save-config', type=str, default=None,
                            help='save model configurations')
        subparser.add_argument('--disable-progress-bar', action='store_true',
                            help='disable the progress bar in training')
        subparser.add_argument('--verbose', action='store_true',
                            help='enable verbose outputs for debugging')

        gparser = subparser.add_argument_group("Optimizer setting")
        gparser.add_argument('--optimizer', choices=('Adam', 'AdamW', 'RMSprop', 'SGD', 'ASGD'), default='AdamW')
        gparser.add_argument('--l1-weight', type=float, default=0.,
                            help='the weight for L1 regularization (default: 0)')
        gparser.add_argument('--l2-weight', type=float, default=0.,
                            help='the weight for L2 regularization (default: 0)')
        gparser.add_argument('--embedreg-lambda', type=float, default=0.,
                            help='regularization strength for embedding layer (default: 0)')
        gparser.add_argument('--score-loss-weight', type=float, default=1.,
                            help='the weight for score loss for hinge_mix loss (default: 1)')
        gparser.add_argument('--lr', type=float, default=0.0001,
                            help='the learning rate for optimizer (default: 0.0001)')
        gparser.add_argument('--loss-func', choices=('hinge', 'hinge_mix'), default='hinge',
                            help="loss fuction ('hinge', 'hinge_mix') ")
        gparser.add_argument('--loss-pos-paired', type=float, default=0.5,
                            help='the penalty for positive base-pairs for loss augmentation (default: 0.5)')
        gparser.add_argument('--loss-neg-paired', type=float, default=0.005,
                            help='the penalty for negative base-pairs for loss augmentation (default: 0.005)')
        gparser.add_argument('--loss-pos-unpaired', type=float, default=0,
                            help='the penalty for positive unpaired bases for loss augmentation (default: 0)')
        gparser.add_argument('--loss-neg-unpaired', type=float, default=0,
                            help='the penalty for negative unpaired bases for loss augmentation (default: 0)')

        gparser = subparser.add_argument_group("Network setting")
        gparser.add_argument('--max-helix-length', type=int, default=30,
                        help='the maximum length of helices (default: 30)')
        gparser.add_argument('--embed-size', type=int, default=0,
                        help='the dimention of embedding (default: 0 == onehot)')
        gparser.add_argument('--n-linear-layers', type=int, default=1,
                              help = 'the number of linear layers after hyena embedding')
        gparser.add_argument('--num-filters', type=int, action='append',
                        help='the number of CNN filters (default: 96)')
        gparser.add_argument('--filter-size', type=int, action='append',
                        help='the length of each filter of CNN (default: 5)')
        gparser.add_argument('--pool-size', type=int, action='append',
                        help='the width of the max-pooling layer of CNN (default: 1)')
        gparser.add_argument('--dilation', type=int, default=0,
                        help='Use the dilated convolution (default: 0)')
        gparser.add_argument('--num-lstm-layers', type=int, default=0,
                        help='the number of the LSTM hidden layers (default: 0)')
        gparser.add_argument('--num-lstm-units', type=int, default=0,
                        help='the number of the LSTM hidden units (default: 0)')
        gparser.add_argument('--num-transformer-layers', type=int, default=0,
                        help='the number of the transformer layers (default: 0)')
        gparser.add_argument('--num-transformer-hidden-units', type=int, default=2048,
                        help='the number of the hidden units of each transformer layer (default: 2048)')
        gparser.add_argument('--num-transformer-att', type=int, default=8,
                        help='the number of the attention heads of each transformer layer (default: 8)')
        gparser.add_argument('--num-paired-filters', type=int, action='append', default=[],
                        help='the number of CNN filters (default: 96)')
        gparser.add_argument('--paired-filter-size', type=int, action='append', default=[],
                        help='the length of each filter of CNN (default: 5)')
        gparser.add_argument('--num-hidden-units', type=int, action='append',
                        help='the number of the hidden units of full connected layers (default: 32)')
        gparser.add_argument('--dropout-rate', type=float, default=0.3,
                        help='dropout rate of the CNN and LSTM units (default: 0.0)')
        gparser.add_argument('--fc-dropout-rate', type=float, default=0.3,
                        help='dropout rate of the hidden units (default: 0.0)')
        gparser.add_argument('--num-att', type=int, default=4,
                        help='the number of the heads of attention (default: 0)')
        gparser.add_argument('--pair-join', choices=('cat', 'add', 'mul', 'bilinear'), default='cat',
                            help="how pairs of vectors are joined ('cat', 'add', 'mul', 'bilinear') (default: 'cat')")
        gparser.add_argument('--no-split-lr', default=False, action='store_true')

        subparser.set_defaults(func = lambda args, conf: Train().run(args, conf))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNA Fold with End-to-End Neural Model")
    subparser = parser.add_subparsers(title='Sub-commands')
    parser.set_defaults(func = lambda args, conf: parser.print_help())
    Train.add_args(subparser)

    # DEFAULT_ARGS = [
    #     "train",
    #     "--test-input", f"{DATA_FOLDER}/TestSetB.lst",
    #     "--embed-size", "64",
    #     # "--num-transformer-layers", "4",
    #     "--lr", "0.0001",
    #     "--epoch", "50",
    #     f"{DATA_FOLDER}/TrainSetB.lst"
    # ]
    # args = parser.parse_args(DEFAULT_ARGS)

    args = parser.parse_args()

    conf = list(filter(lambda x: x[0]=='@', sys.argv))
    conf = None if len(conf)==0 else conf[-1][1:]

    args.func(args, conf)

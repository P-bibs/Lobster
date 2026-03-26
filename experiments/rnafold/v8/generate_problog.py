import os
import sys
import random
import time
from pathlib import Path
import argparse
from collections import defaultdict
from typing import Optional
import multimolecule

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import scallopy

from x_transformers import XTransformer, TransformerWrapper, Decoder, Encoder
from multimolecule import RnaTokenizer, RnaErnieModel, RnaFmModel

from utils.dataset import BPseqDataset, TOKENS, TOKEN_ID_TO_NAME_MAP, NUCLEOTIDES
from utils.compbpseq import compare_bpseq, accuracy

THIS_FOLDER = os.path.abspath(os.path.join(__file__, "../"))
DATA_FOLDER = os.path.abspath(os.path.join(__file__, "../../../data/rnafold"))
MODEL_FOLDER = os.path.abspath(os.path.join(__file__, "../../../model/rnafold"))


class RNASeqBiLSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, dropout=0.1, max_len=1000):
        super(RNASeqBiLSTM, self).__init__()
        # Embedding dimension for each token (A, C, G, U, padding, unknown)
        self.embedding_dim = 6
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Linear layer to project one-hot encoded vectors to latent space (B x N x latent_dim)
        self.input_projection = nn.Linear(self.embedding_dim, latent_dim)

        # Positional encoding (sinusoidal)
        self.positional_encoding = SinusoidalPositionalEncoding(latent_dim, max_len)

        # BiLSTM Encoder
        self.bilstm = nn.LSTM(input_size=latent_dim, hidden_size=latent_dim, num_layers=num_layers, dropout=dropout, bidirectional=True, batch_first=True)

        # Output projection (decoder)
        # Layer Norm after encoding
        self.layer_norm = nn.LayerNorm(latent_dim * 2)  # Since BiLSTM is bidirectional, hidden size is doubled

        # Decoder (MLP for output prediction)
        self.fc_out = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, len(TOKENS))
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

        # Step 4: Pass the sequence through the BiLSTM (B x N x 2 * latent_dim)
        lstm_output, _ = self.bilstm(one_hot)

        # Step 5: Layer normalization after the BiLSTM
        lstm_output = self.layer_norm(lstm_output)

        # Step 6: Decode to output dimension using MLP (B x N x O)
        output = self.fc_out(self.dropout(lstm_output))
        return torch.softmax(output, dim=-1)


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


class RNASeqXTransformer(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers, dropout=0.1, max_len=1000):
        super(RNASeqXTransformer, self).__init__()
        self.model = XTransformer(
            dim = latent_dim,
            enc_num_tokens = 6,
            enc_depth = num_layers,
            enc_heads = num_heads,
            enc_max_seq_len = max_len,
            dec_num_tokens = len(TOKENS),
            dec_depth = num_layers,
            dec_heads = num_heads,
            dec_max_seq_len = max_len,
            # tie_token_emb = True  # tie embeddings of encoder and decoder
        )
        # Decoder (MLP for output prediction)
        self.fc_out = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            # nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            # nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(latent_dim // 4, len(TOKENS)),
        )

    def forward(self, sequences, ref_tokens):
        B, N = len(sequences), len(sequences[0])  # Batch size B and sequence length N
        rna_map = defaultdict(lambda: 5, {'0': 0, 'A': 1, 'a': 1, 'C': 2, 'c': 2, 'G': 3, 'g': 3, 'U': 4, 'u': 4})
        nucleotides = torch.zeros(B, N)
        mask = torch.zeros_like(nucleotides).bool()
        for i, seq in enumerate(sequences):
            for j, char in enumerate(seq):
                nucleotides[i, j] = rna_map[char]
                mask[i, j] = char != '0'
        target = torch.argmax(ref_tokens, dim=2)
        encoded = self.model.encoder(nucleotides, mask=mask) # (B, N, len(TOKEN))
        result = self.fc_out(encoded)
        result = torch.softmax(result, dim=2)
        return result


class RNAErnieForNucleotideClassification(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers, dropout=0.1, max_len=1000):
        super(RNAErnieForNucleotideClassification, self).__init__()
        self.tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnaernie')
        self.model = RnaErnieModel.from_pretrained('multimolecule/rnaernie')
        self.fc_out = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, len(TOKENS)))

    def forward(self, sequences, ref_tokens):
        (B, N, _) = ref_tokens.shape
        input = self.tokenizer(sequences, return_tensors='pt')
        output = self.model(**input)
        result = self.fc_out(output.last_hidden_state[:, :N, :])
        result = torch.softmax(result, dim=2)
        return result


class RNAFMForNucleotideClassification(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers, dropout=0.1, max_len=1000):
        super(RNAFMForNucleotideClassification, self).__init__()
        self.tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnafm')
        self.model = RnaFmModel.from_pretrained('multimolecule/rnafm')
        self.fc_out = nn.Sequential(
            nn.Linear(640, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, len(TOKENS)))

    def forward(self, sequences, ref_tokens):
        device = ref_tokens.device
        (B, N, _) = ref_tokens.shape
        input = self.tokenizer(sequences, return_tensors='pt').to(device)
        output = self.model(**input) # B x N x 640
        result = self.fc_out(output.last_hidden_state[:, 1:N+1, :])
        result = torch.softmax(result, dim=2)
        return result

problog_template="""
len(20).
nucleotide(0, a).
nucleotide(1, c).
nucleotide(2, g).
nucleotide(3, a).
nucleotide(4, u).
nucleotide(5, a).
nucleotide(6, g).
nucleotide(7, a).
nucleotide(8, a).
nucleotide(9, a).
nucleotide(10, c).
nucleotide(11, a).
nucleotide(12, u).
nucleotide(13, g).
nucleotide(14, u).
nucleotide(15, a).
nucleotide(16, u).
nucleotide(17, u).
nucleotide(18, g).
nucleotide(19, g).
% nucleotide(20, u).
% nucleotide(21, u).
% nucleotide(22, c).
% nucleotide(23, a).
% nucleotide(24, u).
% nucleotide(25, g).
% nucleotide(26, u).
% nucleotide(27, a).
% nucleotide(28, c).
% nucleotide(29, u).
% nucleotide(30, u).
% nucleotide(31, g).
% nucleotide(32, c).
% nucleotide(33, u).
% nucleotide(34, u).
% nucleotide(35, u).
% nucleotide(36, g).
% nucleotide(37, g).
% nucleotide(38, g).
% nucleotide(39, u).
% nucleotide(40, g).
% nucleotide(41, u).
% nucleotide(42, g).
% nucleotide(43, a).
% nucleotide(44, g).
% nucleotide(45, a).
% nucleotide(46, g).
% nucleotide(47, u).
% nucleotide(48, u).
% nucleotide(49, u).
% nucleotide(50, g).
% nucleotide(51, u).
% nucleotide(52, u).
% nucleotide(53, a).
% nucleotide(54, g).
% nucleotide(55, u).
% nucleotide(56, u).
% nucleotide(57, c).
% nucleotide(58, g).
% nucleotide(59, a).
% nucleotide(60, a).
% nucleotide(61, u).
% nucleotide(62, c).
% nucleotide(63, u).
% nucleotide(64, a).
% nucleotide(65, a).
% nucleotide(66, c).
% nucleotide(67, c).
% nucleotide(68, u).
% nucleotide(69, a).
% nucleotide(70, u).
% nucleotide(71, c).
% nucleotide(72, c).
% nucleotide(73, g).
% nucleotide(74, a).

0.886::token(0,  t_helix_stack_left); 0.004::token(0,  t_helix_stack_right); 0.027::token(0,  t_intern_loop_left); 0.002::token(0,  t_intern_loop_right); 0.051::token(0,  t_intern_loop_unpaired); 0.012::token(0,  t_ext_loop_unpaired).
0.906::token(1,  t_helix_stack_left); 0.003::token(1,  t_helix_stack_right); 0.022::token(1,  t_intern_loop_left); 0.001::token(1,  t_intern_loop_right); 0.044::token(1,  t_intern_loop_unpaired); 0.009::token(1,  t_ext_loop_unpaired).
0.890::token(2,  t_helix_stack_left); 0.003::token(2,  t_helix_stack_right); 0.036::token(2,  t_intern_loop_left); 0.001::token(2,  t_intern_loop_right); 0.044::token(2,  t_intern_loop_unpaired); 0.009::token(2,  t_ext_loop_unpaired).
0.826::token(3,  t_helix_stack_left); 0.003::token(3,  t_helix_stack_right); 0.082::token(3,  t_intern_loop_left); 0.001::token(3,  t_intern_loop_right); 0.064::token(3,  t_intern_loop_unpaired); 0.010::token(3,  t_ext_loop_unpaired).
0.905::token(4,  t_helix_stack_left); 0.002::token(4,  t_helix_stack_right); 0.034::token(4,  t_intern_loop_left); 0.001::token(4,  t_intern_loop_right); 0.043::token(4,  t_intern_loop_unpaired); 0.007::token(4,  t_ext_loop_unpaired).
0.817::token(5,  t_helix_stack_left); 0.003::token(5,  t_helix_stack_right); 0.102::token(5,  t_intern_loop_left); 0.001::token(5,  t_intern_loop_right); 0.059::token(5,  t_intern_loop_unpaired); 0.008::token(5,  t_ext_loop_unpaired).
0.091::token(6,  t_helix_stack_left); 0.007::token(6,  t_helix_stack_right); 0.383::token(6,  t_intern_loop_left); 0.006::token(6,  t_intern_loop_right); 0.486::token(6,  t_intern_loop_unpaired); 0.009::token(6,  t_ext_loop_unpaired).
0.015::token(7,  t_helix_stack_left); 0.003::token(7,  t_helix_stack_right); 0.012::token(7,  t_intern_loop_left); 0.002::token(7,  t_intern_loop_right); 0.948::token(7,  t_intern_loop_unpaired); 0.003::token(7,  t_ext_loop_unpaired).
0.193::token(8,  t_helix_stack_left); 0.003::token(8,  t_helix_stack_right); 0.112::token(8,  t_intern_loop_left); 0.002::token(8,  t_intern_loop_right); 0.667::token(8,  t_intern_loop_unpaired); 0.007::token(8,  t_ext_loop_unpaired).
0.880::token(9,  t_helix_stack_left); 0.002::token(9,  t_helix_stack_right); 0.032::token(9,  t_intern_loop_left); 0.001::token(9,  t_intern_loop_right); 0.064::token(9,  t_intern_loop_unpaired); 0.007::token(9,  t_ext_loop_unpaired).
0.877::token(10, t_helix_stack_left); 0.002::token(10, t_helix_stack_right); 0.040::token(10, t_intern_loop_left); 0.001::token(10, t_intern_loop_right); 0.061::token(10, t_intern_loop_unpaired); 0.007::token(10, t_ext_loop_unpaired).
0.853::token(11, t_helix_stack_left); 0.002::token(11, t_helix_stack_right); 0.067::token(11, t_intern_loop_left); 0.001::token(11, t_intern_loop_right); 0.059::token(11, t_intern_loop_unpaired); 0.008::token(11, t_ext_loop_unpaired).
0.550::token(12, t_helix_stack_left); 0.004::token(12, t_helix_stack_right); 0.334::token(12, t_intern_loop_left); 0.002::token(12, t_intern_loop_right); 0.094::token(12, t_intern_loop_unpaired); 0.008::token(12, t_ext_loop_unpaired).
0.543::token(13, t_helix_stack_left); 0.003::token(13, t_helix_stack_right); 0.192::token(13, t_intern_loop_left); 0.002::token(13, t_intern_loop_right); 0.240::token(13, t_intern_loop_unpaired); 0.007::token(13, t_ext_loop_unpaired).
0.023::token(14, t_helix_stack_left); 0.003::token(14, t_helix_stack_right); 0.013::token(14, t_intern_loop_left); 0.002::token(14, t_intern_loop_right); 0.947::token(14, t_intern_loop_unpaired); 0.003::token(14, t_ext_loop_unpaired).
0.014::token(15, t_helix_stack_left); 0.003::token(15, t_helix_stack_right); 0.010::token(15, t_intern_loop_left); 0.002::token(15, t_intern_loop_right); 0.955::token(15, t_intern_loop_unpaired); 0.003::token(15, t_ext_loop_unpaired).
0.011::token(16, t_helix_stack_left); 0.003::token(16, t_helix_stack_right); 0.004::token(16, t_intern_loop_left); 0.002::token(16, t_intern_loop_right); 0.961::token(16, t_intern_loop_unpaired); 0.002::token(16, t_ext_loop_unpaired).
0.014::token(17, t_helix_stack_left); 0.003::token(17, t_helix_stack_right); 0.003::token(17, t_intern_loop_left); 0.002::token(17, t_intern_loop_right); 0.954::token(17, t_intern_loop_unpaired); 0.003::token(17, t_ext_loop_unpaired).
0.023::token(18, t_helix_stack_left); 0.004::token(18, t_helix_stack_right); 0.007::token(18, t_intern_loop_left); 0.005::token(18, t_intern_loop_right); 0.947::token(18, t_intern_loop_unpaired); 0.003::token(18, t_ext_loop_unpaired).
0.018::token(19, t_helix_stack_left); 0.006::token(19, t_helix_stack_right); 0.023::token(19, t_intern_loop_left); 0.006::token(19, t_intern_loop_right); 0.933::token(19, t_intern_loop_unpaired); 0.003::token(19, t_ext_loop_unpaired).
% 0.006::token(20, t_helix_stack_left); 0.009::token(20, t_helix_stack_right); 0.003::token(20, t_intern_loop_left); 0.007::token(20, t_intern_loop_right); 0.957::token(20, t_intern_loop_unpaired); 0.003::token(20, t_ext_loop_unpaired).
% 0.006::token(21, t_helix_stack_left); 0.017::token(21, t_helix_stack_right); 0.003::token(21, t_intern_loop_left); 0.019::token(21, t_intern_loop_right); 0.934::token(21, t_intern_loop_unpaired); 0.004::token(21, t_ext_loop_unpaired).
% 0.010::token(22, t_helix_stack_left); 0.052::token(22, t_helix_stack_right); 0.006::token(22, t_intern_loop_left); 0.083::token(22, t_intern_loop_right); 0.821::token(22, t_intern_loop_unpaired); 0.009::token(22, t_ext_loop_unpaired).
% 0.008::token(23, t_helix_stack_left); 0.169::token(23, t_helix_stack_right); 0.006::token(23, t_intern_loop_left); 0.203::token(23, t_intern_loop_right); 0.572::token(23, t_intern_loop_unpaired); 0.013::token(23, t_ext_loop_unpaired).
% 0.009::token(24, t_helix_stack_left); 0.611::token(24, t_helix_stack_right); 0.012::token(24, t_intern_loop_left); 0.241::token(24, t_intern_loop_right); 0.084::token(24, t_intern_loop_unpaired); 0.016::token(24, t_ext_loop_unpaired).
% 0.007::token(25, t_helix_stack_left); 0.756::token(25, t_helix_stack_right); 0.009::token(25, t_intern_loop_left); 0.133::token(25, t_intern_loop_right); 0.052::token(25, t_intern_loop_unpaired); 0.016::token(25, t_ext_loop_unpaired).
% 0.004::token(26, t_helix_stack_left); 0.798::token(26, t_helix_stack_right); 0.005::token(26, t_intern_loop_left); 0.111::token(26, t_intern_loop_right); 0.050::token(26, t_intern_loop_unpaired); 0.011::token(26, t_ext_loop_unpaired).
% 0.596::token(27, t_helix_stack_left); 0.008::token(27, t_helix_stack_right); 0.047::token(27, t_intern_loop_left); 0.003::token(27, t_intern_loop_right); 0.317::token(27, t_intern_loop_unpaired); 0.010::token(27, t_ext_loop_unpaired).
% 0.910::token(28, t_helix_stack_left); 0.003::token(28, t_helix_stack_right); 0.024::token(28, t_intern_loop_left); 0.001::token(28, t_intern_loop_right); 0.044::token(28, t_intern_loop_unpaired); 0.007::token(28, t_ext_loop_unpaired).
% 0.898::token(29, t_helix_stack_left); 0.003::token(29, t_helix_stack_right); 0.025::token(29, t_intern_loop_left); 0.001::token(29, t_intern_loop_right); 0.052::token(29, t_intern_loop_unpaired); 0.007::token(29, t_ext_loop_unpaired).
% 0.892::token(30, t_helix_stack_left); 0.003::token(30, t_helix_stack_right); 0.027::token(30, t_intern_loop_left); 0.001::token(30, t_intern_loop_right); 0.052::token(30, t_intern_loop_unpaired); 0.008::token(30, t_ext_loop_unpaired).
% 0.873::token(31, t_helix_stack_left); 0.004::token(31, t_helix_stack_right); 0.048::token(31, t_intern_loop_left); 0.002::token(31, t_intern_loop_right); 0.051::token(31, t_intern_loop_unpaired); 0.009::token(31, t_ext_loop_unpaired).
% 0.381::token(32, t_helix_stack_left); 0.013::token(32, t_helix_stack_right); 0.488::token(32, t_intern_loop_left); 0.007::token(32, t_intern_loop_right); 0.084::token(32, t_intern_loop_unpaired); 0.015::token(32, t_ext_loop_unpaired).
% 0.036::token(33, t_helix_stack_left); 0.026::token(33, t_helix_stack_right); 0.145::token(33, t_intern_loop_left); 0.019::token(33, t_intern_loop_right); 0.753::token(33, t_intern_loop_unpaired); 0.007::token(33, t_ext_loop_unpaired).
% 0.007::token(34, t_helix_stack_left); 0.007::token(34, t_helix_stack_right); 0.003::token(34, t_intern_loop_left); 0.005::token(34, t_intern_loop_right); 0.961::token(34, t_intern_loop_unpaired); 0.003::token(34, t_ext_loop_unpaired).
% 0.008::token(35, t_helix_stack_left); 0.007::token(35, t_helix_stack_right); 0.003::token(35, t_intern_loop_left); 0.006::token(35, t_intern_loop_right); 0.959::token(35, t_intern_loop_unpaired); 0.003::token(35, t_ext_loop_unpaired).
% 0.008::token(36, t_helix_stack_left); 0.014::token(36, t_helix_stack_right); 0.003::token(36, t_intern_loop_left); 0.013::token(36, t_intern_loop_right); 0.942::token(36, t_intern_loop_unpaired); 0.003::token(36, t_ext_loop_unpaired).
% 0.009::token(37, t_helix_stack_left); 0.020::token(37, t_helix_stack_right); 0.002::token(37, t_intern_loop_left); 0.012::token(37, t_intern_loop_right); 0.931::token(37, t_intern_loop_unpaired); 0.004::token(37, t_ext_loop_unpaired).
% 0.011::token(38, t_helix_stack_left); 0.009::token(38, t_helix_stack_right); 0.004::token(38, t_intern_loop_left); 0.007::token(38, t_intern_loop_right); 0.955::token(38, t_intern_loop_unpaired); 0.003::token(38, t_ext_loop_unpaired).
% 0.008::token(39, t_helix_stack_left); 0.014::token(39, t_helix_stack_right); 0.002::token(39, t_intern_loop_left); 0.013::token(39, t_intern_loop_right); 0.941::token(39, t_intern_loop_unpaired); 0.003::token(39, t_ext_loop_unpaired).
% 0.012::token(40, t_helix_stack_left); 0.465::token(40, t_helix_stack_right); 0.018::token(40, t_intern_loop_left); 0.340::token(40, t_intern_loop_right); 0.120::token(40, t_intern_loop_unpaired); 0.017::token(40, t_ext_loop_unpaired).
% 0.003::token(41, t_helix_stack_left); 0.855::token(41, t_helix_stack_right); 0.003::token(41, t_intern_loop_left); 0.072::token(41, t_intern_loop_right); 0.036::token(41, t_intern_loop_unpaired); 0.011::token(41, t_ext_loop_unpaired).
% 0.003::token(42, t_helix_stack_left); 0.872::token(42, t_helix_stack_right); 0.004::token(42, t_intern_loop_left); 0.054::token(42, t_intern_loop_right); 0.031::token(42, t_intern_loop_unpaired); 0.012::token(42, t_ext_loop_unpaired).
% 0.006::token(43, t_helix_stack_left); 0.848::token(43, t_helix_stack_right); 0.007::token(43, t_intern_loop_left); 0.056::token(43, t_intern_loop_right); 0.038::token(43, t_intern_loop_unpaired); 0.017::token(43, t_ext_loop_unpaired).
% 0.014::token(44, t_helix_stack_left); 0.561::token(44, t_helix_stack_right); 0.013::token(44, t_intern_loop_left); 0.126::token(44, t_intern_loop_right); 0.249::token(44, t_intern_loop_unpaired); 0.013::token(44, t_ext_loop_unpaired).
% 0.020::token(45, t_helix_stack_left); 0.012::token(45, t_helix_stack_right); 0.006::token(45, t_intern_loop_left); 0.005::token(45, t_intern_loop_right); 0.937::token(45, t_intern_loop_unpaired); 0.004::token(45, t_ext_loop_unpaired).
% 0.029::token(46, t_helix_stack_left); 0.004::token(46, t_helix_stack_right); 0.009::token(46, t_intern_loop_left); 0.003::token(46, t_intern_loop_right); 0.944::token(46, t_intern_loop_unpaired); 0.002::token(46, t_ext_loop_unpaired).
% 0.010::token(47, t_helix_stack_left); 0.004::token(47, t_helix_stack_right); 0.002::token(47, t_intern_loop_left); 0.003::token(47, t_intern_loop_right); 0.965::token(47, t_intern_loop_unpaired); 0.002::token(47, t_ext_loop_unpaired).
% 0.011::token(48, t_helix_stack_left); 0.003::token(48, t_helix_stack_right); 0.003::token(48, t_intern_loop_left); 0.003::token(48, t_intern_loop_right); 0.966::token(48, t_intern_loop_unpaired); 0.002::token(48, t_ext_loop_unpaired).
% 0.371::token(49, t_helix_stack_left); 0.004::token(49, t_helix_stack_right); 0.018::token(49, t_intern_loop_left); 0.002::token(49, t_intern_loop_right); 0.578::token(49, t_intern_loop_unpaired); 0.005::token(49, t_ext_loop_unpaired).
% 0.890::token(50, t_helix_stack_left); 0.002::token(50, t_helix_stack_right); 0.025::token(50, t_intern_loop_left); 0.001::token(50, t_intern_loop_right); 0.058::token(50, t_intern_loop_unpaired); 0.007::token(50, t_ext_loop_unpaired).
% 0.896::token(51, t_helix_stack_left); 0.003::token(51, t_helix_stack_right); 0.023::token(51, t_intern_loop_left); 0.001::token(51, t_intern_loop_right); 0.054::token(51, t_intern_loop_unpaired); 0.008::token(51, t_ext_loop_unpaired).
% 0.878::token(52, t_helix_stack_left); 0.003::token(52, t_helix_stack_right); 0.032::token(52, t_intern_loop_left); 0.001::token(52, t_intern_loop_right); 0.063::token(52, t_intern_loop_unpaired); 0.008::token(52, t_ext_loop_unpaired).
% 0.852::token(53, t_helix_stack_left); 0.002::token(53, t_helix_stack_right); 0.045::token(53, t_intern_loop_left); 0.001::token(53, t_intern_loop_right); 0.081::token(53, t_intern_loop_unpaired); 0.007::token(53, t_ext_loop_unpaired).
% 0.120::token(54, t_helix_stack_left); 0.010::token(54, t_helix_stack_right); 0.644::token(54, t_intern_loop_left); 0.007::token(54, t_intern_loop_right); 0.200::token(54, t_intern_loop_unpaired); 0.009::token(54, t_ext_loop_unpaired).
% 0.007::token(55, t_helix_stack_left); 0.007::token(55, t_helix_stack_right); 0.003::token(55, t_intern_loop_left); 0.004::token(55, t_intern_loop_right); 0.954::token(55, t_intern_loop_unpaired); 0.003::token(55, t_ext_loop_unpaired).
% 0.008::token(56, t_helix_stack_left); 0.007::token(56, t_helix_stack_right); 0.003::token(56, t_intern_loop_left); 0.004::token(56, t_intern_loop_right); 0.955::token(56, t_intern_loop_unpaired); 0.003::token(56, t_ext_loop_unpaired).
% 0.006::token(57, t_helix_stack_left); 0.021::token(57, t_helix_stack_right); 0.003::token(57, t_intern_loop_left); 0.019::token(57, t_intern_loop_right); 0.926::token(57, t_intern_loop_unpaired); 0.004::token(57, t_ext_loop_unpaired).
% 0.008::token(58, t_helix_stack_left); 0.015::token(58, t_helix_stack_right); 0.003::token(58, t_intern_loop_left); 0.010::token(58, t_intern_loop_right); 0.946::token(58, t_intern_loop_unpaired); 0.003::token(58, t_ext_loop_unpaired).
% 0.008::token(59, t_helix_stack_left); 0.010::token(59, t_helix_stack_right); 0.003::token(59, t_intern_loop_left); 0.006::token(59, t_intern_loop_right); 0.952::token(59, t_intern_loop_unpaired); 0.004::token(59, t_ext_loop_unpaired).
% 0.007::token(60, t_helix_stack_left); 0.006::token(60, t_helix_stack_right); 0.002::token(60, t_intern_loop_left); 0.006::token(60, t_intern_loop_right); 0.959::token(60, t_intern_loop_unpaired); 0.003::token(60, t_ext_loop_unpaired).
% 0.009::token(61, t_helix_stack_left); 0.043::token(61, t_helix_stack_right); 0.005::token(61, t_intern_loop_left); 0.080::token(61, t_intern_loop_right); 0.827::token(61, t_intern_loop_unpaired); 0.008::token(61, t_ext_loop_unpaired).
% 0.006::token(62, t_helix_stack_left); 0.643::token(62, t_helix_stack_right); 0.008::token(62, t_intern_loop_left); 0.221::token(62, t_intern_loop_right); 0.079::token(62, t_intern_loop_unpaired); 0.015::token(62, t_ext_loop_unpaired).
% 0.002::token(63, t_helix_stack_left); 0.871::token(63, t_helix_stack_right); 0.002::token(63, t_intern_loop_left); 0.052::token(63, t_intern_loop_right); 0.034::token(63, t_intern_loop_unpaired); 0.012::token(63, t_ext_loop_unpaired).
% 0.004::token(64, t_helix_stack_left); 0.836::token(64, t_helix_stack_right); 0.003::token(64, t_intern_loop_left); 0.045::token(64, t_intern_loop_right); 0.034::token(64, t_intern_loop_unpaired); 0.027::token(64, t_ext_loop_unpaired).
% 0.002::token(65, t_helix_stack_left); 0.874::token(65, t_helix_stack_right); 0.003::token(65, t_intern_loop_left); 0.052::token(65, t_intern_loop_right); 0.033::token(65, t_intern_loop_unpaired); 0.012::token(65, t_ext_loop_unpaired).
% 0.004::token(66, t_helix_stack_left); 0.781::token(66, t_helix_stack_right); 0.005::token(66, t_intern_loop_left); 0.125::token(66, t_intern_loop_right); 0.052::token(66, t_intern_loop_unpaired); 0.012::token(66, t_ext_loop_unpaired).
% 0.003::token(67, t_helix_stack_left); 0.808::token(67, t_helix_stack_right); 0.004::token(67, t_intern_loop_left); 0.103::token(67, t_intern_loop_right); 0.048::token(67, t_intern_loop_unpaired); 0.012::token(67, t_ext_loop_unpaired).
% 0.003::token(68, t_helix_stack_left); 0.805::token(68, t_helix_stack_right); 0.004::token(68, t_intern_loop_left); 0.039::token(68, t_intern_loop_right); 0.036::token(68, t_intern_loop_unpaired); 0.038::token(68, t_ext_loop_unpaired).
% 0.005::token(69, t_helix_stack_left); 0.709::token(69, t_helix_stack_right); 0.005::token(69, t_intern_loop_left); 0.038::token(69, t_intern_loop_right); 0.040::token(69, t_intern_loop_unpaired); 0.068::token(69, t_ext_loop_unpaired).
% 0.006::token(70, t_helix_stack_left); 0.578::token(70, t_helix_stack_right); 0.005::token(70, t_intern_loop_left); 0.040::token(70, t_intern_loop_right); 0.048::token(70, t_intern_loop_unpaired); 0.110::token(70, t_ext_loop_unpaired).
% 0.006::token(71, t_helix_stack_left); 0.713::token(71, t_helix_stack_right); 0.006::token(71, t_intern_loop_left); 0.049::token(71, t_intern_loop_right); 0.040::token(71, t_intern_loop_unpaired); 0.073::token(71, t_ext_loop_unpaired).
% 0.007::token(72, t_helix_stack_left); 0.678::token(72, t_helix_stack_right); 0.007::token(72, t_intern_loop_left); 0.047::token(72, t_intern_loop_right); 0.040::token(72, t_intern_loop_unpaired); 0.091::token(72, t_ext_loop_unpaired).
% 0.020::token(73, t_helix_stack_left); 0.122::token(73, t_helix_stack_right); 0.013::token(73, t_intern_loop_left); 0.031::token(73, t_intern_loop_right); 0.047::token(73, t_intern_loop_unpaired); 0.283::token(73, t_ext_loop_unpaired).
% 0.028::token(74, t_helix_stack_left); 0.065::token(74, t_helix_stack_right); 0.015::token(74, t_intern_loop_left); 0.024::token(74, t_intern_loop_right); 0.050::token(74, t_intern_loop_unpaired); 0.262::token(74, t_ext_loop_unpaired).

% ====================================================================================================
"""
rules="""
pair_type(a, u).
pair_type(c, g).
pair_type(g, c).
pair_type(g, u).
pair_type(u, a).
pair_type(u, g).

nucleotide_pair(I, J) :- nucleotide(I, X_I), nucleotide(J, X_J), pair_type(X_I, X_J), I < J.

helix_stack(I, J) :- token(I, t_helix_stack_left), substructure(K, L), token(J, t_helix_stack_right), K is I + 1, L is J - 1.

intern_loop_unpaired(I, I) :- token(I, t_intern_loop_unpaired).
intern_loop_unpaired(I, K) :- intern_loop_unpaired(I, J), K is J + 1, token(K, t_intern_loop_unpaired).
intern_loop_part(I, J) :- substructure(I, J).
intern_loop_part(I, J) :- intern_loop_unpaired(I, J).

intern_loop_base(I, J) :- token(I, t_intern_loop_left), K is I + 1, substructure(K, L), M is L + 1, intern_loop_unpaired(M, J).
intern_loop_base(I, J) :- token(I, t_intern_loop_left), K is I + 1, substructure(K, L), M is L + 1, substructure(M, J).
intern_loop_base(I, J) :- token(I, t_intern_loop_left), K is I + 1, intern_loop_unpaired(K, J).
intern_loop_base(I, J) :- intern_loop_base(I, K), L is K + 1, intern_loop_part(L, J).
intern_loop(I, JP1) :- intern_loop_base(I, J), JP1 is J + 1, token(JP1, t_intern_loop_right), nucleotide_pair(I, JP1).

ext_unpaired_region(I, I) :- token(I, t_exp_loop_unpaired).
ext_unpaired_region(I, J) :- ext_unpaired_region(I, JM1), JM1 is J - 1, token(J, t_ext_loop_unpaired).

substructure(I, J) :- helix_stack(I, J).
substructure(I, J) :- intern_loop(I, J).

well_formed_base(I) :- substructure(0, I).
well_formed_base(I) :- ext_unpaired_region(0, I).
well_formed_base(J) :- well_formed_base(I), IP1 is I + 1, substructure(IP1, J).
well_formed_base(J) :- well_formed_base(I), IP1 is I + 1, ext_unpaired_region(IP1, J).
well_formed() :- len(N), well_formed_base(NM1), NM1 is N - 1.

pair(I, J) :- well_formed(), substructure(I, J).
unpaired(I) :- well_formed(), token(I, t_intern_loop_unpaired).
unpaired(I) :- well_formed(), token(I, t_ext_loop_unpaired).

query(pair(_,_)).
query(unpaired(_)).
"""

num_to_struct = {
    0: "t_helix_stack_left",
    1: "t_helix_stack_right",
    2: "t_intern_loop_left",
    3: "t_intern_loop_right",
    4: "t_intern_loop_unpaired",
    5: "t_ext_loop_unpaired"
}


def make_pl(seq, token_facts):
    program = ""
    program += f"len({len(seq)}).\n"
    nucleotides = ""
    for i, c in enumerate(seq):
        nucleotides += f"nucleotide({i}, {c.lower()}).\n"
    program += nucleotides

    last_i = -1
    lines = []
    current_line = []
    for (prob, (i, struct_num)) in token_facts:
        if i != last_i:
            lines.append(";".join(current_line))
            current_line = []
            last_i = i
        current_line.append(f"{prob:.4f}::token({i}, {num_to_struct[struct_num.item()]})")
    
    lines.append(";".join(current_line))
    lines = lines[1:]
    program += ".\n".join(lines)
    program += ".\n\n"

    program += rules
    return program


class Train:
    def __init__(self):
        self.step = 0
        self.PL_NUMBER = 0

        #self.extract_token = scallopy.Context()
        #self.extract_token.import_file(f"{THIS_FOLDER}/scallop/extract_token.scl")

        self.infer_structure = scallopy.Context(provenance="difftopkproofs", k=1)
        self.infer_structure.import_file(f"{THIS_FOLDER}/scallop/infer_structure_optim_1.scl")
        self.infer_structure.set_non_probabilistic(["length", "nucleotide"])

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
            for batch_id, (fnames, seqs, ref_structures, dot_brackets, ref_tokens) in enumerate(self.train_loader):
                n_batch = len(seqs)
                self.optimizer.zero_grad()
                y = self.model(seqs, ref_tokens.to(self.device))

                # # Adjacency matrix
                # adj_mat = self.parse_prob_tokens_to_structure(y, seqs, retain_k=self.inference_retain_k)
                # bps = self.generate_bpseq_from_adj_mat(adj_mat)

                # Loss function
                loss = self.loss(y, ref_tokens.to(self.device), None, ref_structures, seqs)
                loss_total += loss.item()
                if loss.item() > 0.:
                    loss.backward()
                    self.optimizer.step()

                # Compute statistics
                self.populate_token_stats(y, seqs, pred_token_stats)
                self.populate_token_stats(ref_tokens, seqs, gt_token_stats)

                # Compute token accuracy
                token_acc_total += self.compute_token_accuracy(y.to("cpu"), ref_tokens)

                # # Compute structure accuracy (prec/rec/f1)
                # for header, ref, bp in zip(fnames, ref_structures, bps):
                #     x = compare_bpseq(ref, bp)
                #     [sen, ppv, fval, mcc] = list(accuracy(*x))
                #     sen_total += sen
                #     ppv_total += ppv
                #     fval_total += fval
                #     mcc_total += mcc

                denom = batch_id + 1
                num += n_batch
                pbar.set_postfix(
                    loss='{:.2e}'.format(loss_total / denom),
                    token_acc='{:.2f}'.format(token_acc_total / denom),
                    sen=f'{sen_total / num:.2f}',
                    ppv=f'{ppv_total / num:.2f}',
                    fval=f'{fval_total / num:.2f}',
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
            self.print_token_stats("GT", gt_token_stats)
            self.print_token_stats("Pred", pred_token_stats)

        # elapsed_time = time.time() - start
        # pbar.set_description(f"[Train Epoch {epoch}] Time: {elapsed_time:.3f}s")

    def test(self, epoch):
        self.model.eval()
        n_dataset = len(self.test_loader.dataset)
        loss_total, token_acc_total, num = 0, 0, 0
        sen_total, ppv_total, fval_total, mcc_total = 0, 0, 0, 0
        gt_token_stats, pred_token_stats = defaultdict(lambda: 0), defaultdict(lambda: 0)
        # start = time.time()

        #lengths = [len(seqs[0]) for (_, seqs, _, _, _) in self.test_loader]
        #print(lengths)
        #print(sorted(lengths))
        #exit()

        with torch.no_grad(), tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            pbar.set_description(f"[Test Epoch {epoch}]")
            for batch_id, (fnames, seqs, ref_structures, dot_brackets, ref_tokens) in enumerate(self.test_loader):
                n_batch = len(seqs)
                y = self.model(seqs, ref_tokens.to(self.device))
                adj_mat = self.parse_prob_tokens_to_structure(y, seqs, retain_k=self.inference_retain_k)
                if adj_mat is None:
                    continue
                bps = self.generate_bpseq_from_adj_mat(adj_mat)
                loss = self.loss(y, ref_tokens.to(self.device), adj_mat, ref_structures, seqs)
                loss_total += loss.item()
                token_acc_total += self.compute_token_accuracy(y.to("cpu"), ref_tokens)

                # Compute statistics
                self.populate_token_stats(y, seqs, pred_token_stats)
                self.populate_token_stats(ref_tokens, seqs, gt_token_stats)
                for header, ref, bp in zip(fnames, ref_structures, bps):
                    x = compare_bpseq(ref, bp)
                    [sen, ppv, fval, mcc] = list(accuracy(*x))
                    sen_total += sen
                    ppv_total += ppv
                    fval_total += fval
                    mcc_total += mcc

                denom = batch_id + 1
                num += n_batch
                pbar.set_postfix(
                    loss='{:.3e}'.format(loss_total / denom),
                    token_acc='{:.2f}'.format(token_acc_total / denom),
                    sen=f"{sen_total / num:.3f}",
                    ppv=f"{ppv_total / num:.3f}",
                    fval=f"{fval_total / num:.3f}",
                )
                pbar.update(n_batch)

        if self.verbose:
            self.print_token_stats("GT", gt_token_stats)
            self.print_token_stats("Pred", pred_token_stats)

        if self.writer is not None:
            self.writer.add_scalar("test/loss", loss_total / num, epoch * n_dataset)
        # elapsed_time = time.time() - start
        # pbar.set_description(f"[Test Epoch {epoch}]\tLoss: {loss_total / num:.6f}\tTime: {elapsed_time:.3f}s")

    def sanity_check(self):
        n_dataset = len(self.train_loader.dataset)
        loss_total, token_acc_total, num = 0, 0, 0
        sen_total, ppv_total, fval_total, mcc_total = 0, 0, 0, 0
        with torch.no_grad(), tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            pbar.set_description(f"[Sanity Checking]")
            for batch_id, (_, seqs, ref_structures, dot_brackets, ref_tokens) in enumerate(self.train_loader):
                n_batch = len(seqs)
                adj_mat = self.parse_prob_tokens_to_structure(ref_tokens, seqs, retain_k=1)
                bps = self.generate_bpseq_from_adj_mat(adj_mat)
                for dp_idx, (ref, bp) in enumerate(zip(ref_structures, bps)):
                    length = len(dot_brackets[dp_idx])
                    x = compare_bpseq(ref, bp)
                    [sen, ppv, fval, mcc] = list(accuracy(*x))
                    sen_total += sen
                    ppv_total += ppv
                    fval_total += fval
                    mcc_total += mcc

                    acc = len([e for e in ref == bp if e]) / len(bp)
                    if acc < 1:
                        print("Error detected:")
                        print(f" - Sequence:     {seqs[dp_idx]}")
                        print(f" - Dot-Bracket:  {dot_brackets[dp_idx]}")
                        print(f" - Ground Truth: {ref[:length]}")
                        print(f" - Predicted:    {bp[:length]}")
                    # else:
                    #     print(f"Success:")
                    #     print(f" - Sequence:     {seqs[dp_idx]}")
                    #     print(f" - Dot-Bracket:  {dot_brackets[dp_idx]}")
                    #     print(f" - Ground Truth: {ref[:length]}")
                    #     print(f" - Predicted:    {bp[:length]}")
                num += n_batch
                pbar.set_postfix(sen=f"{sen_total / num:.3f}", ppv=f"{ppv_total / num:.3f}", fval=f"{fval_total / num:.3f}")
                pbar.update(n_batch)

    def compute_token_accuracy(self, y, ref_tokens):
        ref_token_ids = torch.argmax(ref_tokens, dim=2).view(-1)
        pred_token_ids = torch.argmax(y, dim=2).view(-1)
        token_acc = float(torch.sum(ref_token_ids == pred_token_ids)) / len(ref_token_ids)
        return token_acc

    def print_token_stats(self, name, token_stats):
        s = "{ "
        for (i, token_type) in enumerate(TOKENS):
            if i > 0: s += ", "
            s += f"{token_type}: {token_stats[token_type]}"
        s += " }"
        print(f"{name}: {s}")

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

    def build_optimizer(self, optimizer, params, lr, l2_weight):
        if optimizer == 'Adam':
            return optim.Adam(params, lr=lr, amsgrad=False, weight_decay=l2_weight)
        elif optimizer =='AdamW':
            return optim.AdamW(params, lr=lr, amsgrad=False, weight_decay=l2_weight)
        elif optimizer == 'RMSprop':
            return optim.RMSprop(params, lr=lr, weight_decay=l2_weight)
        elif optimizer == 'SGD':
            return optim.SGD(params, nesterov=True, lr=lr, momentum=0.9, weight_decay=l2_weight)
        elif optimizer == 'ASGD':
            return optim.ASGD(params, lr=lr, weight_decay=l2_weight)
        else:
            raise('not implemented')

    def loss(self, y, gt_tokens, adj_mat, ref_structures, seqs, dot_brackets=None):
        # Token prediction loss
        l1 = self.loss_fn(y, gt_tokens)

        # # Structure prediction loss
        # (B, N, N) = adj_mat.shape
        # gt_adj_mat = self.generate_adj_matrix_from_bp_seq(ref_structures)
        # l2 = self.loss_fn_2(torch.softmax(adj_mat.reshape(B * N, N), dim=1), gt_adj_mat.reshape(B * N, N))

        # Return the combination of the two losses
        return l1 # + 0.2 * l2

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

    def generate_adj_matrix_from_bp_seq(self, gt_bp_seqs):
        B, Np1 = gt_bp_seqs.shape
        target = torch.zeros((B, Np1, Np1))
        for (dp_idx, row) in enumerate(gt_bp_seqs):
            for (i, j) in enumerate(row):
                target[dp_idx, i, j] = 1
        return target

    def generate_dot_bracket_from_pair_index_matrix(self, seq, pair_id_matrix):
        N = len(seq)
        ss = ""
        for (i, j) in enumerate(pair_id_matrix):
            if i == 0: continue
            if j > 0: ss += '(' if j > i else ')'
            else: ss += '.'
        return ss


    def parse_prob_tokens_to_structure(self, y, seqs, retain_k: Optional[int] = None, sample_with_categorical: bool = False):
        (B, N, F) = y.shape
        pred_structure = torch.zeros((B, N + 1, N + 1))
        for (dp_idx, prob_tokens) in enumerate(y):
            seq = seqs[dp_idx].replace("0", "")
            length = len(seq)
            nucleotide_facts = [(i, j) for (i, n) in enumerate(seq) for j in NUCLEOTIDES[n.upper()]]

            # Prepare the probabilistic input
            if retain_k is not None:
                token_facts = []
                token_disjunctions = [[] for _ in range(length)]
                for i in range(length):
                    if sample_with_categorical:
                        sampled = torch.distributions.Categorical(prob_tokens[i]).sample_n(retain_k)
                        values, indices = list(sampled.values().to_dense()), list(sampled.indices().to_dense())
                    else:
                        sampled = torch.topk(prob_tokens[i, :-1], retain_k, dim=0)
                        values, indices = list(sampled.values), list(sampled.indices)
                    for (p, j) in zip(values, indices):
                        fact_id = len(token_facts)
                        token_facts.append((p, (i, j)))
                        token_disjunctions[i].append(fact_id)
            else:
                token_facts = [(prob_tokens[i, j], (i, j)) for i in range(length) for j in range(F)]
                token_disjunctions = [[i * F + j for j in range(F)] for i in range(length)]

            # Execute the scallop program
            length_facts = [(len(seq),)]

            pl_program = make_pl(seq, token_facts)
            with open(f"experiments/data/rnafold/pl/ssp_{self.PL_NUMBER}.pl", "w") as f:
                f.write(pl_program)
                print("Wrote to ssp_{}.pl".format(self.PL_NUMBER))
            self.PL_NUMBER += 1
            return None

            temp_ctx = self.infer_structure.clone()
            temp_ctx.add_facts("token", token_facts, disjunctions=token_disjunctions)
            temp_ctx.add_facts("length", length_facts)
            temp_ctx.add_facts("nucleotide", nucleotide_facts)
            print("RNA length:", length_facts)
            sys.stdout.flush()
            import os
            os.environ["STRATUM"] = "10,10"
            os.environ["TIME"] = "1"
            temp_ctx.run()
            del os.environ["TIME"]
            del os.environ["STRATUM"]

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

    def build_loss_function(self, loss_func, args):
        if loss_func == "bce":
            self.loss_fn = torch.nn.BCELoss(weight=torch.tensor([0.7, 0.7, 1.0, 1.0, 0.3, 0.3, 0.03]).to(self.device))
            self.loss_fn_2 = torch.nn.BCELoss()
        elif loss_func == "mse":
            self.loss_fn = torch.nn.MSELoss()
        elif loss_func == "bce_with_logits":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        elif loss_func == "ce":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = torch.nn.BCELoss()

    def build_model(self, args):
        if args.model == "transformer":
            config = { 'latent_dim' : args.embed_size, 'num_layers': args.num_layers, 'num_heads': args.num_heads, 'dropout': args.dropout_rate }
            model = RNASeqTransformer(**config)
        elif args.model == "xtransformer":
            config = { 'latent_dim' : args.embed_size, 'num_layers': args.num_layers, 'num_heads': args.num_heads, 'dropout': args.dropout_rate }
            model = RNASeqXTransformer(**config)
        elif args.model == "rnaernie":
            config = { 'latent_dim' : args.embed_size, 'num_layers': args.num_layers, 'num_heads': args.num_heads, 'dropout': args.dropout_rate }
            model = RNAErnieForNucleotideClassification(**config)
        elif args.model == "rnafm":
            config = { 'latent_dim' : args.embed_size, 'num_layers': args.num_layers, 'num_heads': args.num_heads, 'dropout': args.dropout_rate }
            model = RNAFMForNucleotideClassification(**config)
        elif args.model == "bilstm":
            config = { 'latent_dim' : args.embed_size, 'num_layers': args.num_layers, 'dropout': args.dropout_rate }
            model = RNASeqBiLSTM(**config)
        else:
            raise Exception(f"Unknown model {args.model}")
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

        train_max_length = args.train_max_length if args.train_max_length else args.max_length
        if args.train_percentage is None or args.train_percentage > 0:
            train_dataset = BPseqDataset(args.input, percentage=args.train_percentage, max_length=train_max_length)
            self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=BPseqDataset.collate_fn)
        if args.test_input is not None and (args.test_percentage is None or args.test_percentage > 0):
            test_max_length = args.test_max_length if args.test_max_length else args.max_length
            test_min_length = args.test_min_length if args.test_min_length else None
            test_dataset = BPseqDataset(args.test_input, percentage=args.test_percentage, max_length=test_max_length, min_length=test_min_length)
            self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=BPseqDataset.collate_fn)

        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.model, config = self.build_model(args)
        config.update({ 'param': args.param })
        self.inference_retain_k = args.inference_retain_k

        if args.load_params is not None:
            if os.path.exists(args.load_params):
                print(f"[Info] Loading from existing model state dict: {args.load_params}...")
                self.model.load_state_dict(torch.load(args.load_params))
                print(f"[Info] Success!")
            else:
                print("[Warning] load parameter .tch file not found; resort to initializing the model from scratch")

        if args.init_param != '':
            init_param = Path(args.init_param)
            if not init_param.exists() and conf is not None:
                init_param = Path(conf) / init_param
            p = torch.load(init_param)
            if isinstance(p, dict) and 'model_state_dict' in p:
                p = p['model_state_dict']
            self.model.load_state_dict(p)

        self.device = torch.device("cpu")
        if args.gpu >= 0:
            self.device = torch.device("cuda", args.gpu)

        self.model.to(self.device)
        self.optimizer = self.build_optimizer(args.optimizer, self.model.parameters(), args.lr, args.l2_weight)
        self.build_loss_function(args.loss_func, args)

        checkpoint_epoch = 0
        if args.resume is not None:
            checkpoint_epoch = self.resume_checkpoint(args.resume)

        if args.sanity_check:
            self.sanity_check()
            return

        # if self.test_loader is not None:
        #     self.test(0)
        for epoch in range(checkpoint_epoch+1, args.epochs+1):
            if self.train_loader is not None:
                self.train(epoch)
            if self.test_loader is not None:
                self.test(epoch)
            if args.log_dir is not None:
                self.save_checkpoint(args.log_dir, epoch)

            if args.param is not None:
                model_save_dir = f"{MODEL_FOLDER}/v8/checkpoints/{args.param}_epoch_{epoch}.tch"
                model_save_folder = os.path.dirname(model_save_dir)
                os.makedirs(model_save_folder, exist_ok=True)
                torch.save(self.model.state_dict(), model_save_dir)

        if args.param is not None:
            model_save_dir = f"{MODEL_FOLDER}/v8/{args.param}.tch"
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
        subparser.add_argument('--load-params', type=str, help='Loading model file (.tch)')
        subparser.add_argument('--param', type=str, default='param.pth', help='output file name of trained parameters')
        subparser.add_argument('--init-param', type=str, default='', help='the file name of the initial parameters')
        subparser.add_argument('--train-percentage', type=float, default=100)
        subparser.add_argument('--test-percentage', type=float, default=100)
        subparser.add_argument('--max-length', type=int)
        subparser.add_argument('--train-max-length', type=int)
        subparser.add_argument('--test-min-length', type=int)
        subparser.add_argument('--test-max-length', type=int)
        subparser.add_argument('--batch-size', type=int, default=8)

        gparser = subparser.add_argument_group("Training environment")
        gparser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
        gparser.add_argument('--log-dir', type=str, default=None, help='Directory for storing logs')
        gparser.add_argument('--resume', type=str, default=None, help='Checkpoint file for resume')
        gparser.add_argument('--save-config', type=str, default=None, help='save model configurations')
        gparser.add_argument('--disable-progress-bar', action='store_true', help='disable the progress bar in training')
        gparser.add_argument('--verbose', action='store_true', help='enable verbose outputs for debugging')

        gparser = subparser.add_argument_group("Inference")
        gparser.add_argument('--sanity-check', action='store_true')
        gparser.add_argument('--inference-retain-k', type=int)

        gparser = subparser.add_argument_group("Optimizer setting")
        gparser.add_argument('--optimizer', choices=('Adam', 'AdamW', 'RMSprop', 'SGD', 'ASGD'), default='Adam')
        gparser.add_argument('--l1-weight', type=float, default=0., help='the weight for L1 regularization (default: 0)')
        gparser.add_argument('--l2-weight', type=float, default=0., help='the weight for L2 regularization (default: 0)')
        gparser.add_argument('--embedreg-lambda', type=float, default=0., help='regularization strength for embedding layer (default: 0)')
        gparser.add_argument('--score-loss-weight', type=float, default=1., help='the weight for score loss for hinge_mix loss (default: 1)')
        gparser.add_argument('--lr', type=float, default=0.0001, help='the learning rate for optimizer (default: 0.0001)')
        gparser.add_argument('--loss-func', choices=('hinge', 'hinge_mix', 'bce', 'bce_with_logits', 'mse', 'ce'), default='bce', help="loss fuction ('hinge', 'hinge_mix') ")

        gparser = subparser.add_argument_group("Network setting")
        gparser.add_argument('--model', choices=('transformer', 'bilstm', 'xtransformer', 'rnafm', 'rnaernie'), default='xtransformer', help="Model architecture")
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
        # "--sanity-check",
        "--train-percentage", "0",
        "--test-percentage", "100",
        "--seed", "0",
        "--loss-func", "bce",
        "--model", "rnafm",
        "--train-max-length", "100",
        "--test-min-length", "0",
        "--test-max-length", "176",
        "--embed-size", "128",
        "--num-layers", "4",
        "--num-heads", "16",
        "--lr", "0.00001",
        "--epoch", "1",
        "--inference-retain-k", "6",
        "--batch-size", "4",
        "--verbose",
        "--load-params", f"{MODEL_FOLDER}/v8/checkpoints/ArchiveII90_TrainSetA_l500_p100_fullytrained.tch",
        "--test-input", f"{DATA_FOLDER}/archiveII_Test.lst", # testing set
        #"--test-input", f"{DATA_FOLDER}/TestSetA.lst", # testing set
        "--param", f"ArchiveII90_TrainSetA_l500_p1_2", # save/load parameter file
        f"{DATA_FOLDER}/archiveII_Train_TrainSetA.lst", # train set
    ]
    args = parser.parse_args(DEFAULT_ARGS)

    # args = parser.parse_args()

    conf = list(filter(lambda x: x[0]=='@', sys.argv))
    conf = None if len(conf) == 0 else conf[-1][1:]

    args.func(args, conf)

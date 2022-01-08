import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import *
from .protein_features import ProteinFeatures


class ProteinEncoder(nn.Module):

    def __init__(self, args):
        super(ProteinEncoder, self).__init__()
        self.features = ProteinFeatures(
                top_k=args.k_neighbors, num_rbf=args.num_rbf,
                features_type='full',
                direction='bidirectional'
        )
        self.node_in, self.edge_in = self.features.feature_dimensions['full']
        self.W_v = nn.Sequential(
                nn.Linear(self.node_in, args.hidden_size, bias=True),
                Normalize(args.hidden_size)
        )
        self.W_e = nn.Sequential(
                nn.Linear(self.edge_in, args.hidden_size, bias=True),
                Normalize(args.hidden_size)
        )
        self.W_s = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList([
                MPNNLayer(args.hidden_size, args.hidden_size * 3, dropout=args.dropout)
                for _ in range(args.encoder_depth)
        ])
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, X, S, mask):
        V, E, E_idx = self.features(X, mask)
        h_v = self.W_v(V)  # [B, N, H]
        h_e = self.W_e(E)  # [B, N, K, H]
        h_s = self.W_s(S)  # [B, N, H]
        nei_s = gather_nodes(h_s, E_idx)  # [B, N, K, H]

        vmask = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        h = h_v
        for layer in self.layers:
            nei_v = gather_nodes(h, E_idx)  # [B, N, K, H]
            nei_h = torch.cat([nei_v, nei_s, h_e], dim=-1)
            h = layer(h, nei_h, mask_attend=vmask)  # [B, N, H]
            h = h * mask.unsqueeze(-1)  # [B, N, H]
        return h


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--k_neighbors', type=int, default=9)
    parser.add_argument('--encoder_depth', type=int, default=4)
    parser.add_argument('--vocab_size', type=int, default=21)
    parser.add_argument('--num_rbf', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()
    encoder = ProteinEncoder(args).cuda()

    # X is the coordinate: size [batch_size, N_residues, 4, 3]
    # X[:, :, 0, :] is N
    # X[:, :, 1, :] is CA (alpha carbon)
    # X[:, :, 2, :] is C
    # X[:, :, 3, :] is O
    X = torch.randn(1, 5, 4, 3).cuda()

    # S is the amino acid type: [batch_size, N_residues]
    # protein "AMG" -> [0, 13, 6]
    S = torch.ones(1, 5).cuda().long()

    # mask is zero if this residue is padding: [batch_size, N_residues]
    # useful for proteins with different batches in a batch
    mask = torch.ones(1, 5).cuda()

    # output should be size [batch_size, N_residues, N_hidden_size]
    h = encoder(X, S, mask)
    print(h.size())


import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import printt
from .cp import Chemprop
from .tagger import ProteinTagger


class BindingAffinityModel(nn.Module):
    """
        Combine molecule and protein binding site
    """
    def __init__(self, args, model_params, index=None):
        """
        """
        super(BindingAffinityModel, self).__init__()

        # unpack model parameters
        self.model_type = args.model_type
        self.d_model = args.d_model
        ## output_size is size of last layer
        self.output_size = model_params.get("vocab_size") or 1

        # build protein-ligand encoder
        ## bert tagger: (batch, len, dim) -> (batch, dim)
        if self.model_type not in ["ligand"]:
            self.sequence_model = ProteinTagger(args, model_params)  # shared by all models
        ## ignore protein entirely
        else:
            self.sequence_model = Chemprop(args)

        # build MLP: protein, ligand representation -> label(s)
        input_size = args.d_model
        ## shared layers
        self.mlp = nn.Sequential(
                nn.Linear(input_size, args.mlp_hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.mlp_hidden_size, args.mlp_hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout)
        )
        ## different top layers for affinity/tagging
        self.classifier = nn.Linear(args.mlp_hidden_size, self.output_size)
        self.tagger = nn.Linear(args.mlp_hidden_size, self.output_size)

    def forward(self, batch,
                compute_affinity=True, compute_tagging=False):
        """
            Forward pass
            @param (bool) seq2seq  True for tagging
        """
        # "ligand" model is just Chemprop
        if self.model_type in ["ligand"]:
            x, _ = self.sequence_model(batch)
        else:
            # fuse molecule + original protein sequence
            ## x_protein (batch, d_model)
            ## x_mol     (batch, latent_dim)
            ## x_attn    (batch, length)
            ## x_protein_raw  (batch, length, d_model)
            ## (list) x_mol_raw  (batch, atoms, latent_dim)
            x_protein, x_protein_raw, x_attn, x_mol, x_mol_raw = self.sequence_model(batch)
            x = x_protein + x_mol  # fuse molecule + protein

        outputs = {}
        # (default) output single prediction per instance
        if compute_affinity:
            x = self.classifier(self.mlp(x))
            x = x.squeeze()
            outputs["affinity"] = x

        # (optional) output per-token prediction
        if compute_tagging:
            x_seq = self.tagger(self.mlp(x_protein_raw))
            x_seq = torch.softmax(x_seq, dim=-1).squeeze()
            outputs["tagging"] = x_seq

        return outputs


import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cp import Chemprop
from .structure import ProteinEncoder


class ProteinTagger(nn.Module):
    """
        Combine molecule and protein binding site
    """
    def __init__(self, args, model_params):
        super(ProteinTagger, self).__init__()
        # molecule encoder (drug graph) -> latent
        self.chemprop = Chemprop(args)
        # protein sequence encoder
        self.model_type = args.model_type
        if self.model_type in ["bert"]:
            self.bert = model_params["bert"]
        elif self.model_type in ["structure"]:
            self.encoder = ProteinEncoder(args)

        # molecule attention (value = protein seq)
        self.attn = nn.MultiheadAttention(args.d_model, args.nhead,
                                          dropout=args.dropout)

        # initialized in big module

    def forward(self, batch):
        """
            Forward pass feature computation
        """
        # 1) encode ligand
        ## (batch, latent), (batch, atoms, latent)
        x_mol, x_mol_raw = self.chemprop(batch)

        # 2) encode protein sequence
        ## (batch, length+1, 768)
        x_protein_raw = batch["seq"].cuda()
        mask = batch["mask"].cuda()
        if self.model_type in ["bert"]:
            ## no_grad because ESM is big = frozen weights
            with torch.no_grad():
                x_protein_raw = self.bert(x_protein_raw, repr_layers=[12])
                x_protein_raw = x_protein_raw["representations"][12]
                ## remove [CLS] from output
                x_protein_raw = x_protein_raw[:,1:]
        elif self.model_type in ["structure"]:
            x_coords = batch["protein_pos"].cuda()
            # mask is reverse of structure encoder
            x_protein_raw = self.encoder(x_coords, x_protein_raw, ~mask.long())

        key   = x_protein_raw.permute(1, 0, 2)  # (length, batch, latent)
        value = key
        query = x_mol.unsqueeze(0)  # (1, batch, latent)
        if self.model_type in ["bert"]:
            mask  = mask[:,1:]  # remove [CLS]
        # x_protein shape (length=1, batch, 768)
        # x_attn shape (batch, target length=1, source length)
        # x_attn is used for tagging
        # NOTE 20210516 15:53 looks ok, attn not saturated
        x_protein, x_attn = self.attn(query, key, value,
                                      key_padding_mask=mask)
        x_protein, x_attn = x_protein.squeeze(), x_attn.squeeze()
        x_protein_raw = x_protein_raw.permute(2, 0, 1) * x_attn  # (latent, batch, length) * (batch, length)
        x_protein_raw = x_protein_raw.permute(1, 2, 0)  # (batch, length, latent)

        return x_protein, x_protein_raw, x_attn, x_mol, x_mol_raw


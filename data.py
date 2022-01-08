import os
import sys
import json
import copy
import math
import random
from functools import reduce
from collections import defaultdict

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from Bio.Data.IUPACData import protein_letters_3to1

from chemprop.data import get_data_from_smiles, MoleculeDataset

from utils import load_json, load_csv, printt


def load_data(args):
    """
        Load and minimally process data
    """

    # 1) load raw affinity data + pdb/sdf files
    data = get_raw_data(args.data_file, args)

    # 2) use protein sequence (or not = use one-hot embedding for protein id)
    ## tokenize for BERT
    if args.model_type in ["bert"]:
        esm_model, alphabet = torch.hub.load("facebookresearch/esm",
                "esm1_t12_85M_UR50S")
        tokenizer = alphabet.get_batch_converter()
        for split, items in data.items():
            _tokenize(items, "seq", tokenizer)
        printt("finished tokenizing all inputs")
    ## tokenize for non-BERT
    elif args.model_type not in ["ligand"]:
        for split, items in data.items():
            tokenizer = _tokenize(items, "seq")
        printt("finished tokenizing all inputs")
    else:
        esm_model = None

    # 3) split train into separate folds
    data["train"] = _crossval_split(data["train"], args)

    # 4) add additional parameters
    data_params = []
    if args.pocket_file is not None:
        data_tagger = load_json(args.pocket_file)
        data_tagger = {item["id"]: _crop(item["pocket"], args.max_protein_length) for item in data_tagger}
        data_params.append(data_tagger)
    model_params = {}
    if args.model_type in ["bert"]:
        model_params["bert"] = esm_model
    # >>>
    model_params["vocab_size"] = 1

    return data, data_params, model_params


def get_data(data, data_params, fold_num, args, build_dict=True):
    """
        Convert raw data into DataLoaders for training.
        If we use cross-validation, combine non-validation (and non-test)
        folds into training set.
    """
    splits = { "train": [] }
    # split into train/val/test
    folds, val_data, test_data = data["train"], data["val"], data["test"]
    ## if val split is pre-specified, do not take fold
    if len(val_data) == 0:
        val_fold = fold_num
        splits["val"] = folds[val_fold]
    else:
        val_fold = -1  # index should never equal -1 so all folds go to train
        splits["val"] = val_data
    ## if test split is pre-specified, do not take fold
    # 20210310 14:43 verified that test split is fixed and deterministic
    if len(test_data) > 0:
        test_fold = val_fold  # must specify here to allocate train folds
        splits["test"] = test_data
    ## otherwise both val/test labels depend on fold_num
    else:
        test_fold = (fold_num+1) % args.num_folds
        splits["test"] = folds[test_fold]
    ## add remaining to train
    for idx in range(args.num_folds):
        if idx in [val_fold, test_fold]:
            continue
        splits["train"].extend(folds[idx])
    # convert to Dataset object
    ## BERT representations are precomputed here
    for split, items in splits.items():
        splits[split] = AffinityDataset(items, args, data_params)
    # convert to DataLoader
    data_loaders = _get_loader(splits, args)
    return data_loaders

# -------- DATA UTILS --------

def _crossval_split(data, args):
    """
        Split into train/val/test folds
    """
    # randomize first
    random.shuffle(data)
    # split into folds
    folds = [[] for _ in range(args.num_folds)]
    fold_ratio = 1. / args.num_folds
    for fold_num in range(args.num_folds):
        start = int(fold_num * fold_ratio * len(data))
        end = start + int(len(data) * fold_ratio)
        folds[fold_num].extend(data[start:end])
    return folds


def _get_loader(splits, args):
    """
        Convert lists into DataLoader
    """
    # convert to DataLoader
    loaders = {}
    for split, data in splits.items():
        shuffle = split == "train"  # do not shuffle val/test
        # account for test-only datasets
        if len(data) == 0:
            loaders[split] = []
            continue
        collate_fn = lambda batch: _collate_fn(batch, args)
        loaders[split] = DataLoader(data,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    collate_fn=collate_fn,
                                    drop_last=False,
                                    shuffle=shuffle)
    return loaders

# -------- DATA LOADING --------

def get_raw_data(fp, args):
    """
        Load raw data from paths specified in fp
        @return split -> list(dict)
    """
    # raw file has one entry per line
    data_to_load = load_csv(fp)

    # separate out train/test
    data = {
        "train": [],
        "val":   [],
        "test":  []
    }

    # if test split is pre-specified, split into train/test
    # otherwise, allocate all data to train for cross-validation
    for item in data_to_load:
        if "split" in item:
            # >>> debug mode
            #if len(data[item["split"]]) > 400:
            #    continue
            # <<<
            data[item["split"]].append(item)
        else:
            data["train"] = raw_data

    # load all proteins
    aa_code = { k.upper():v for k,v in protein_letters_3to1.items() }
    pdb_data = {}  # path -> structure
    for split, items in data.items():
        for item in tqdm(items, leave=False):
            protein_path = item.get("protein_path")
            pdb_id = item["id"]
            if protein_path is not None:
                if protein_path in pdb_data:
                    continue
                seq, pos = _load_pdb(os.path.join(args.pdb_path, protein_path))
                # errors in PDB (lack carboxyl C atom)
                if seq is None:
                    continue
                seq = _crop("".join(aa_code[aa] for aa in seq),
                        args.max_protein_length)
                pos = _crop(pos, args.max_protein_length)
                pdb_data[pdb_id] = seq, pos

    # load all ligands and labels
    count = 0
    new_data = {split: [] for split in data}
    ## labels included in raw data
    for split, items in data.items():
        for item in items:
            if item.get("ligand_path") is not None:
                ligand = _load_sdf(os.path.join(args.pdb_path, item["ligand_path"]))
            else:
                ligand = item["mol"]
            # skip invalid molecules
            if ligand is None:
                continue
            new_item = item.copy()  # only strings
            new_item["label"] = float(item["label"])
            # Mol object
            new_item["ligand"] = ligand
            # list( 3 letter code ), list(xyz) ->
            # str( 1 letter code ), list(cropped xyz)
            pdb_id = item["id"]
            if item.get("protein_path") is not None:
                if pdb_id not in pdb_data:
                    continue
                new_item["seq"], new_item["protein_pos"] = pdb_data[pdb_id]
            else:
                new_item["seq"] = item["seq"]
            # unique identifier
            if "id" not in new_item:
                new_item["id"] = str(count)
            new_data[split].append(new_item)
            count += 1
    printt(f"Loaded {count} items from {len(data_to_load)} entries.")
    return new_data


def _load_pdb(fp):
    """
        Parse PDB file. See:
        - https://www.wwpdb.org/documentation/file-format-content/
        @return (sequence, position)
    """
    seq, pos = [], []
    atoms_to_keep = ["N", "CA", "C", "O"]
    cur_xyz = {k:[np.nan]*3 for k in atoms_to_keep}
    # >>>
    fp = fp.replace("protein", "pocket")
    # <<<
    with open(fp) as f:
        for line in f:
            # ignore header
            if line[:4] != 'ATOM':
                continue
            # these atoms always appear in this order, in the file
            atom = line[12:16].strip()
            if atom not in atoms_to_keep:
                continue
            # add current residue
            aa = line[17:20].strip()
            if atom == atoms_to_keep[0]:
                seq.append(aa)
                # add previous residue's positions
                if len(seq) > 1:
                    pos.append(np.stack([cur_xyz[a] for a in atoms_to_keep]))
                    cur_xyz = {k:[np.nan]*3 for k in atoms_to_keep}
            # get current residue + position
            xyz = [float(line[30:38]),
                   float(line[38:46]),
                   float(line[46:54])]
            cur_xyz[atom] = np.array(xyz)
        # last residue's positions
        pos.append(np.stack([cur_xyz[a] for a in atoms_to_keep]))
    # reshape into (length, 4=#key atoms, 3=xyz)
    pos = torch.tensor(pos).float()
    if torch.any(torch.isnan(pos)):
        return None, None
    assert pos.shape[0] == len(seq)
    return seq, pos


def _load_sdf(fp):
    mols = Chem.SDMolSupplier(fp)
    if len(mols) == 0:
        return
    assert len(mols) == 1  # should have only 1 ligand
    return mols[0]


def _tokenize(data, key, tokenizer=None, new_key=None,
              single_token=False, add_pad=False):
    """
        Tokenize every item[key] in data[split].
        Modifies item[key] and copies original value to item[key_raw].

        @param (dict) data  split -> list[item]
        @param (str) key  item[key] is iterable
        @param (bool) single_token True to treat entire string as one token
    """
    # figure out old/new keys for before/after tokenizing
    if new_key is not None:
        raw_key = key
    else:
        new_key = key
        raw_key = f"{key}_raw"
    # if tokenizer is not provided, create index
    if tokenizer is None:
        # item[key] is iterable by default, unless single_token=True
        all_values = [item[key] for item in data]
        if not single_token:
            all_values = reduce(lambda x,y: set(x).union(set(y)), all_values)
        tokenizer = _build_index(all_values, use_values=False, add_pad=add_pad)
        f_token = lambda seq: [tokenizer[x] for x in seq]
    else:
        # ESM format [2] gives batch_tokens + adds [CLS]
        f_token = lambda seq: tokenizer([("", seq)])[2][0]
    # tokenize items and modify data in place
    cache = {}  # seq -> tokenized
    for item in data:
        raw_item = item[key]
        item[raw_key] = raw_item
        if raw_item in cache:
            item[new_key] = cache[raw_item]
            continue
        if single_token:
            raw_item = [raw_item]
        # if too long, crop BEFORE tokenizing so [CLS] are preserved
        item[new_key] = f_token(raw_item)
        cache[raw_item] = item[new_key]
    return tokenizer


def _crop(item, max_len):
    if len(item) > max_len:
        padding = math.ceil((len(item) - max_len) / 2)
        item = item[padding:-padding]
    return item


def _build_index(data, use_values=True, add_pad=True, pad="[PAD]"):
    """
        Build map item -> int.

        @param (dict) data key -> [val1, val2, ...]
        @param (bool) use_values True if data.keys() does NOT
            include all potential elements in data.values()
        @param (bool) use_values True if data.keys() is different space
            from data.values()
        @param (bool) add_pad True if 1-index so 0 is padding token.
        @return (dict)
    """
    index = {}
    for item in sorted(data):
        index[item] = len(index)
    if use_values:
        index_range = {}
        if add_pad:
            index_range[pad] = 0  # add padding BEFORE other values
        all_values = reduce(lambda x,y: set(x).union(set(y)), data.values())
        for item in sorted(all_values):
            index_range[item] = len(index_range)
        return index, index_range
    return index


def _collate_fn(batch, args):
    """
        Convert dicts to proper tensors + select indices
        (for sparser computation)
    """
    # collated batch
    new_batch = {}
    # binding label, original identifiers
    for key in ["label"]:
        new_batch[key] = torch.tensor([item[key] for item in batch])
    for key in ["id"]:
        new_batch[key] = [item[key] for item in batch]
    # protein sequence
    if "seq" in batch[0] and args.model_type not in ["ligand"]:
        # x_len is POST-TOKENIZED = it contains [CLS]
        x, x_len = _pad_to_max_length([item["seq"] for item in batch])
        new_batch["seq"] = x
        new_batch["protein_len"] = x_len - 1  # true length does NOT include [CLS]
        # compute mask
        max_len = new_batch["seq"].shape[1]
        # NOTE according to documentation, this mask should be
        # 1 = attend, 0 = ignore as implemented
        # CORRECT for nn.MHA
        mask = torch.arange(max(x_len))[None, :] > x_len[:, None]
        new_batch["mask"] = mask
    # protein 3d structure
    if args.model_type in ["structure"]:
        protein_pos = [item["protein_pos"] for item in batch]
        new_batch["protein_pos"], _ = _pad_to_max_length(protein_pos)
        #ligand_pos = [torch.tensor(item["raw_ligand"].GetConformer().GetPositions()).float() for item in batch]
        #new_batch["ligand_pos"], _ = _pad_to_max_length(ligand_pos)
        # compute Euclidean distances
        '''
        dist = []
        for idx in range(len(protein_pos)):
            dist.append(torch.cdist(protein_pos[idx], ligand_pos[idx], p=2))
        new_batch["dist"] = dist
        '''
    if "pocket" in batch[0]:
        pockets = [item["pocket"] for item in batch]
        x, _ = _pad_to_max_length(pockets)
        new_batch["pocket"] = x
    # ligand MoleculeDatapoint objects
    if "ligand" in batch[0]:
        # these smiles are NOT strings; they are MoleculeDatapoint objects
        smiles = [item["ligand"] for item in batch]
        smiles = MoleculeDataset(smiles)
        new_batch["ligand"] = smiles.batch_graph()
    return new_batch


def _pad_to_max_length(items):
    # collate = pad and convert to tensor
    # batch_size, max_len
    if type(items[0]) is not torch.Tensor:
        idx_raw = [torch.tensor(x) for x in items]
    else:
        idx_raw = items
    lengths = torch.tensor([len(x) for x in items])
    # ESM padding 1
    idxs = nn.utils.rnn.pad_sequence(idx_raw,
                                     batch_first=True,
                                     padding_value=1)
    return idxs, lengths


class AffinityDataset(Dataset):
    """
        Protein-ligand binding dataset
    """

    def __init__(self, dataset, args,
                 data_params):
        """
            @param (list(dict)) dataset
        """
        super(AffinityDataset, self).__init__()

        # load chemical structure
        self.dataset = sorted(dataset, key=lambda x:x["id"])

        if len(data_params) > 0:
            self.protein_to_pocket = data_params[0]
            self.dataset = [x for x in self.dataset if x["id"] in
                            self.protein_to_pocket]
        else:
            self.protein_to_pocket = None

        all_smiles = [item["ligand"] for item in dataset]
        ## convert to MoleculeDataset
        self.mol_dataset = get_data_from_smiles([[smiles] for smiles in all_smiles])
        ## map from id -> index in MoleculeDataset
        self.smiles_idx = {item["id"]: i for i, item in enumerate(self.dataset)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
            @return a single example from the dataset
        """
        item = self.dataset[idx].copy()

        # must retrieve SMILES index before converting mol
        item["raw_ligand"] = item["ligand"]
        # item["ligand"] = self.mol_dataset[self.smiles_idx[item["id"]]]
        item["ligand"] = self.mol_dataset[self.smiles_idx[item["id"]]]

        # >>>
        if self.protein_to_pocket is not None:
            item["pocket"] = self.protein_to_pocket[item["id"]]
        # <<<

        return item


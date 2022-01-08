import os
import sys
import csv
import json
import glob
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn.metrics as metrics

# -------- general

def load_json(fp):
    data = []
    with open(fp) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_csv(fp):
    data = []
    with open(fp) as f:
        reader = csv.DictReader(f)
        for line in reader:
            data.append(line)
    return data


def log(item, fp):
    # pre-process item
    item_new = {}
    for key, val in item.items():
        if type(val) is list:
            key_std = f"{key}_std"
            item_new[key] = float(np.mean(val))
            item_new[key_std] = float(np.std(val))
        else:
            item_new[key] = val
    # initialization: write keys
    if not os.path.exists(fp):
        with open(fp, "w+") as f:
            f.write("")
    # append values
    with open(fp, "a") as f:
        json.dump(item_new, f)
        f.write(os.linesep)


def get_timestamp():
    return datetime.now().strftime('%H:%M:%S')


def printt(*args, **kwargs):
    print(get_timestamp(), *args, **kwargs)


def print_res(scores):
    """
        @param (dict) scores key -> score(s)
    """
    for key, val in scores.items():
        if type(val) is list:
            print_str = f"{np.mean(val):.3f} +/- {np.std(val):.3f}"
        else:
            print_str = f"{val:.3f}"
        print(f"{key}\t{print_str}")


def get_model_path(fold_dir):
    # load last model saved (we only save if improvement in validation performance)
    # convoluted code says "sort by epoch, then batch"
    models = sorted(glob.glob(f"{fold_dir}/*.pth"),
                    key=lambda s:(int(s.split("/")[-1].split("_")[3]),
                                  int(s.split("/")[-1].split("_")[2])))
    if len(models) == 0:
        print(f"no models found at {fold_dir}")
        return
    checkpoint = models[-1]
    return checkpoint

# -------- metrics

def compute_metrics(true, pred):
    """
        these lists are JAGGED IFF sequence=True
        @param pred (n, sequence, 1) preds for prob(in) where in = 1
        @param true (n, sequence, 1) targets, binary vector
    """
    # metrics depend on task
    as_sequence = type(true[0]) is list
    if as_sequence:
        f_metrics = {
            "roc_auc": _compute_roc_auc,
            "prc_auc": _compute_prc_auc
        }
    else:
        f_metrics = {
            "mse": _compute_mse,
            "rmse": _compute_rmse
        }
    scores = defaultdict(list)
    for key, f in f_metrics.items():
        if as_sequence:
            for t,p in zip(true, pred):
                scores[key].append(f(t, p))
            scores[key] = np.mean(scores[key])
        else:
            scores[key] = f(true, pred)
    return scores


def _compute_roc_auc(true, pred):
    try:
        return metrics.roc_auc_score(true, pred)
    except:
        # single target value
        return 0.5


def _compute_prc_auc(true, pred):
    if np.sum(true) == 0:
        return 0.5
    precision, recall, _ = metrics.precision_recall_curve(true, pred)
    prc_auc = metrics.auc(recall, precision)
    return prc_auc


def _compute_mse(true, pred):
    # technically order doesn't matter but "input" then "target"
    true, pred = torch.tensor(true), torch.tensor(pred)
    return F.mse_loss(pred, true).item()


def _compute_rmse(true, pred):
    # technically order doesn't matter but "input" then "target"
    true, pred = torch.tensor(true), torch.tensor(pred)
    return torch.sqrt(F.mse_loss(pred, true)).item()


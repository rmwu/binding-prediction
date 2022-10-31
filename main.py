import os
import sys
import csv
import json
import random
from typing import Tuple, Literal, Optional
from collections import defaultdict

import numpy as np
import torch
from tensorboardX import SummaryWriter

from tap import Tap

from data import load_data, get_data
from model import load_model, to_cuda
from train import train, evaluate
from utils import printt, print_res, load_json, log

from chemprop.args import TrainArgs as ChempropArgs


class TrainArgs(ChempropArgs): #Tap):
    """
        Extends Tap from Kyle.
    """
    # configuration
    config_file: str = None  # override arguments (e.g. first time training)
    args_file: str = "args.json"  # save arguments (e.g. reloading trained model)
    log_file: str = "results.json"  # save results  (e.g. metrics, losses, etc.)

    # args for chemprop to run
    dataset_type: str = "classification"
    ffn_hidden_size: int = 300  # chemprop default is 300

    # ======== data ========
    # directories for your data
    data_path: str = None  # root to data directory
    save_path: str = None  # root to model checkpoints

    # directories for required files
    ckpt_path: str = "/data/scratch/rmwu/hub"  # torch hub pretrained models
    pdb_path:  str = "/data/rsg/chemistry/rmwu/data/raw/pdb"  # root to PDB files

    # filenames
    ## labels file + paths to pdb files
    data_file: str = None
    ## paths to sdf ligand files
    mol_file: str = None
    ## pdb, mol id -> label
    label_file: str = None

    ## (optional) tagging for protein pocket
    pocket_file: str = None

    # data loading
    max_protein_length: int = 1000  # center crop to maximum length
    num_workers: int = 10  # DataLoader workers
    batch_size: int = 50
    ligand_type: Literal["2d", "3d"] = "2d"  # load from SMILES string or SDF file

    # logging
    run_name: str = None  # (optional) tensorboard folder, aka "comment" field
    log_frequency: int = 10  # log to tensorboard every [n] batches

    # ====== training ======
    mode: Literal["train", "test"] = "train"
    metric: Literal["mse", "roc_auc", "prc_auc"] = "mse"  # report scores per fold/epoch

    num_folds: int = 5  # number of different seeds / cross-validation folds
    epochs: int = 60  # max epochs to run
    patience: int = 5  # lack of validation improvement = end
    warmup_epochs: int = 0  # train non-affinity model first

    gpu: int = 0
    seed: int = 0

    save_pred: bool = False  # save predictions on test set
    no_tqdm: bool = False  # set to True if running dispatcher (hyperparameters)

    # ======== model =======
    # (optional) load checkpoint for entire model
    checkpoint_path: str = None

    # protein encoder model
    ## "ligand"  ignores protein, use ligand only
    model_type: Literal["ligand", "bert", "structure"] = "ligand"
    ## representation size for sequence encoder (i.e. size of BERT)
    d_model: int = 768

    # protein-ligand attention:
    ## MHA
    nhead: int = 16  # must be factor of d_model

    ## Structure encoder (Wengong)
    k_neighbors: int = 9
    encoder_depth: int = 4
    vocab_size: int = 21
    num_rbf: int = 16

    # GCN
    dist_threshold: float = 10  # threshold in angstroms for constructing graph
    mpnn_rounds: int = 0  # number of times to refine distances

    # MLP: (mol, protein) -> label
    dropout: float = 0.2
    mlp_hidden_size: int = 1000

    # Chemprop
    ## mlp (must match protein encoder for attention)
    hidden_size: int = 768
    ## number of message-passing steps
    depth: int = 3  # default

    # ==== optimization ====
    # loss terms
    binding_loss_weight: float = 1
    tagging_loss_weight: float = 0

    # optimization
    lr: float = 1e-4
    weight_decay: float = 1e-5

    def process_args(self):
        # used for dispatcher (bash script auto-formats)
        ## process run_name
        if self.run_name is None:
            self.run_name = self.save_path.split("/")[-1]

        # default checkpoint_path to save_path if mode == "test"
        if self.mode == "test" and self.checkpoint_path is None:
            self.checkpoint_path = self.save_path

        # load configuration = override specified values
        ## first load config_file
        if self.config_file is not None:
            with open(self.config_file) as f:
                config = json.load(f)
            for k,v in config.items():
                self.__dict__[k] = v
        ## next load all saved parameters
        if self.checkpoint_path is not None:
            if not os.path.exists(self.checkpoint_path):
                printt("invalid checkpoint_path", self.checkpoint_path)
            if os.path.exists(self.args_file):
                with open(self.args_file) as f:
                    config = json.load(f)
            for k,v in config.items():
                if k not in ["checkpoint_path", "gpu"]:
                    self.__dict__[k] = v

        # prepend data root
        self.data_file  = os.path.join(self.data_path, self.data_file)
        if self.pocket_file is not None:
            self.pocket_file = os.path.join(self.data_path, self.pocket_file)
        if self.mol_file is not None:
            self.mol_file = os.path.join(self.data_path, self.mol_file)
        if self.label_file is not None:
            self.label_file = os.path.join(self.data_path, self.label_file)
        # prepend output root
        self.args_file = os.path.join(self.save_path, self.args_file)
        self.log_file  = os.path.join(self.save_path, self.log_file)


def parse_args():
    return TrainArgs().parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    torch.hub.set_dir(args.ckpt_path)

    # prepare logger
    writer = SummaryWriter(comment=args.run_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # save args (do not pickle object for readability)
    args.save(args.args_file, with_reproducibility=False)

    # load raw data
    data_folds, data_params, model_params = load_data(args)
    printt("finished loading raw data")

    # save scores for each fold
    test_scores = defaultdict(list)
    # (optional) save predictions for each fold
    if args.save_pred or args.mode == "test":
        all_preds = {}
    for fold in range(args.num_folds):
        set_seed(args.seed)
        # make save folder
        fold_dir = os.path.join(args.save_path, f"fold_{fold}")
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        printt("fold {} seed {}\nsaved to {}".format(fold, args.seed, fold_dir))
        # load and convert data to DataLoaders
        loaders = get_data(data_folds, data_params, fold, args)
        printt("finished creating data splits")
        # get model and load checkpoint, if relevant
        model = load_model(args, model_params, fold)
        printt("finished loading model")

        # run training loop
        if args.mode != "test":
            train_loader = loaders["train"]
            val_loader = loaders["val"]
            best_score, best_epoch, best_path = train(
                    train_loader, val_loader,
                    model, writer, fold_dir, args)
            printt("finished training best epoch {} loss {:.3f}".format(
                    best_epoch, best_score))

            # 20210310 13:42 verified that model really is loading from scratch
            with torch.no_grad():
                model = load_model(args, model_params, fold)
                model.load_state_dict(torch.load(best_path,
                    map_location="cpu")["model"])
                printt(f"loaded model from {best_path}")
                # move to cuda if applicable
                if args.gpu >= 0:
                    model = to_cuda(model, args)

        # run testing loop
        if args.save_pred or args.mode == "test":
            # TODO should test mode eval on all splits?
            test_score, preds = evaluate(loaders["test"], model, writer, args,
                                            return_output=True)
            test_score = evaluate(loaders["test"], model, writer, args)
            # for fixed values, assert that returned values same as saved
            for key in ["mol", "protein", "label"]:
                if key in all_preds:
                    pass
                else:
                    all_preds[key] = preds[key]
            all_preds[fold] = preds["output"]
        else:
            test_score = evaluate(loaders["test"], model, writer, args)
        # print and save
        for key, val in test_score.items():
            test_scores[key].append(val)
        printt("fold {}".format(fold))
        print_res(test_score)
        test_score["fold"] = fold
        # set next seed
        args.seed += 1
        # end of fold ========

    # save predictions
    if args.save_pred or args.mode == "test":
        # gather and average predictions from each fold
        all_folds = [all_preds[fold] for fold in range(args.num_folds)]
        avg_preds = torch.tensor(all_folds).mean(dim=0).tolist()
        all_preds["mean"] = avg_preds
        # zip into rows and save
        save_fp = os.path.join(args.save_path, "pred.csv")
        with open(save_fp, "w+") as f:
            writer = csv.writer(f)
            writer.writerow(all_preds.keys())
            for row in zip(*all_preds.values()):
                writer.writerow(row)

    printt(f"{args.num_folds} folds average")
    print_res(test_scores)
    log(test_scores, args.log_file)
    # end of all folds ========


if __name__ == "__main__":
    main()


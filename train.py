"""
    Training script
"""

import os
import sys
from itertools import zip_longest
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from chemprop.train import evaluate_predictions

from utils import printt, print_res, get_model_path
from utils import compute_metrics, log


def train(train_loader, val_loader, model,
          writer, fold_dir, args):

    # optimizer
    optimizer = Adam(model.parameters(),
                     lr=args.lr,
                     weight_decay=args.weight_decay)
    ## load optimizer state
    if args.checkpoint_path is not None:
        checkpoint = get_model_path(fold_dir)
        if checkpoint is not None:
            start_epoch = int(checkpoint.split("/")[-1].split("_")[3])
            with torch.no_grad():
                optimizer.load_state_dict(torch.load(checkpoint,
                    map_location="cpu")["optimizer"])
            printt("Finished loading optimizer")
        else:
            start_epoch = 0
    else:
        start_epoch = 0

    # loss functions
    f_loss = {
        #"affinity": nn.MSELoss(),
        "affinity": nn.BCELoss(),
        "tagging": nn.BCELoss()
    }

    # validation
    best_loss = float("inf")
    best_epoch = start_epoch
    # >>> TODO checkpoint better name
    best_path = os.path.join(fold_dir, "model_0.pth")

    # logging
    log_path = os.path.join(fold_dir, "log.json")

    num_batches = start_epoch * (len(train_loader) // args.batch_size)
    iterator = range(start_epoch, start_epoch+args.epochs)
    if not args.no_tqdm:
        iterator = tqdm(iterator,
                        initial=start_epoch,
                        desc="train epoch", ncols=50)
    for epoch in iterator:

        # start epoch!
        iterator = enumerate(train_loader)
        if not args.no_tqdm:
            iterator = tqdm(iterator,
                            total=len(train_loader),
                            desc="train batch",
                            leave=False, ncols=50)
        for batch_num, batch in iterator:
            # reset optimizer gradients
            model.train()
            optimizer.zero_grad()

            # forward pass
            losses = compute_losses(model, batch, args, f_loss)
            loss = losses["loss"]

            # backpropagate
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            # write to tensorboard
            num_batches += 1
            loss = loss.cpu().item()  # CPU for logging
            if num_batches % args.log_frequency == 0:
                for loss_type, loss in losses.items():
                    log_key = f"train_{loss_type}"
                    writer.add_scalar(log_key, loss, num_batches)

            # end of batch ========

        # evaluate (end of epoch)
        avg_train_loss, avg_train_score = train_epoch_end(num_batches,
                train_loader, model,
                log_path, fold_dir, writer, args,
                max_batch=5, split="train")  # adjust based on batch size
        val_loss, avg_val_score = train_epoch_end(num_batches,
                val_loader, model, log_path, fold_dir, writer, args)
        printt('\nepoch', epoch)
        printt('train loss', avg_train_loss, args.metric, avg_train_score)
        printt('val loss', val_loss, args.metric, avg_val_score)

        # save model
        path_suffix = f"{num_batches}_{epoch}_{avg_val_score:.3f}_{val_loss:.3f}.pth"
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            # save model ONLY IF best
            best_path = os.path.join(fold_dir, f"model_best_{path_suffix}")
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, best_path)
            printt(f"\nsaved model to {best_path}")

        # check if out of patience
        if epoch - best_epoch >= args.patience:
            break

        # end of epoch ========
    # end of all epochs ========

    return best_loss, best_epoch, best_path


def train_epoch_end(num_batches, val_loader,
                    model, log_path, fold_dir, writer, args,
                    max_batch=None, split="val"):
    """
        Evaluate at end of training epoch and write to log file
    """
    log_item = { "batch": num_batches }
    val_loss, avg_val_score = float("inf"), 0
    if len(val_loader) == 0:
        return val_loss, avg_val_score

    # run inference
    val_scores = evaluate(val_loader, model, writer, args,
                          max_batch=max_batch)
    avg_val_score = np.nanmean(val_scores[args.metric])
    for key, value in val_scores.items():
        log_item[f"{split}_{key}"] = value

    # tensorboard
    for key, value in log_item.items():
        if key not in ["batch"]:
            writer.add_scalar(key, value, num_batches)
    # append to JSON log
    log(log_item, log_path)
    # early stopping based on loss only
    val_loss = val_scores["loss"]
    return val_loss, avg_val_score


def evaluate(val_loader, model, writer, args,
             eval_per_task=False, return_output=False, max_batch=None):
    """
        @param (bool) eval_per_task  sort results by each task, i.e. evaluate
                                     separately per protein
        @param (bool) return_output  in addition to scores, return raw
                                     predictions + labels
        @param (int) max_batch       number of batches to sample
    """
    f_loss = {
        #"affinity": nn.MSELoss(),
        "affinity": nn.BCELoss(),
        "tagging": nn.BCELoss()
    }

    with torch.no_grad():
        model.eval()
        all_output, all_target = defaultdict(list), defaultdict(list)
        all_losses = defaultdict(list)
        # (optional) save predictions
        if return_output or eval_per_task:
            ids = []
        # loop through all batches
        iterator = enumerate(val_loader)
        if not args.no_tqdm:
            iterator = tqdm(iterator,
                            total=len(val_loader),
                            desc="evaluation",
                            leave=False, ncols=50)
        for batch_num, batch in iterator:
            # model predictions
            losses, outputs, targets = compute_losses(model, batch, args, f_loss,
                                                      save_outputs=True)
            losses["loss"] = losses["loss"].cpu().item()  # CPU for logging
            for loss_type, loss in losses.items():
                all_losses[loss_type].append(loss)
            for loss_type, out in outputs.items():
                all_output[loss_type].extend(out)
            for loss_type, tar in targets.items():
                all_target[loss_type].extend(tar)

            # currently unused
            if return_output or eval_per_task:
                ids.extend(batch["id"])

            if max_batch is not None and batch_num >= max_batch:
                break

    # evaluate metrics
    scores = {}
    for key in all_target:
        scores_key = compute_metrics(all_target[key], all_output[key])
        for k,v in scores_key.items():
            scores[f"{key}_{k}"] = v
            if k == args.metric:
                scores[k] = v
    # average loss over batches
    for loss_type, losses in all_losses.items():
        scores[loss_type] = torch.mean(torch.tensor(losses)).item()
    # (optional) gather all predictions
    if return_output or eval_per_task:
        assert len(ids) == len(all_target)
        output = {
            "id": ids,
            "label": all_target,
            "output": all_output
        }
        # (optional) additionally evaluate per task
        if eval_per_task:
            pass
    if return_output:
        return scores, output
    return scores


def compute_losses(model, batch, args, f_loss, save_outputs=False):
    """
        Compute all losses
    """
    loss = 0
    losses = {}
    if save_outputs:
        outputs, targets = {}, {}

    # flags based on loss weight
    compute_tagging  = args.tagging_loss_weight > 0
    compute_affinity = args.binding_loss_weight > 0

    outputs = model(batch,
                    compute_affinity=compute_affinity,
                    compute_tagging=compute_tagging)

    # compute loss
    if compute_affinity:
        binding_loss = compute_loss(batch, outputs["affinity"], f_loss["affinity"])
        loss = loss + args.binding_loss_weight * binding_loss
        losses["binding_loss"] = binding_loss.cpu().item()
        if save_outputs:
            outputs["binding"] = outputs["affinity"].tolist()
            targets["binding"] = batch["label"].tolist()

    if compute_tagging:
        tagging_loss = compute_loss(batch, outputs["tagging"], f_loss["tagging"],
                                    key="tagging_label", len_key="protein_len",
                                    as_sequence=True, per_instance=True)
        loss = loss + args.tagging_loss_weight * tagging_loss
        losses["tagging_loss"] = tagging_loss.cpu().item()
        if save_outputs:
            outputs["tagging"] = outputs["tagging"].tolist()
            targets["tagging"] = batch["tagging_label"].tolist()

    losses["loss"] = loss

    if save_outputs:
        return losses, outputs, targets
    return losses


def compute_loss(batch, output, f_loss,
                 key="label", len_key="protein_len",
                 as_sequence=False, per_instance=False, as_int=False):
    """
        Compute prediction loss. Assumption: per-instance prediction.
    """
    # move both to GPU
    if not per_instance:
        output = output.squeeze()
        target = batch[key].cuda().float()
        loss = f_loss(output, target)
    else:
        losses = []
        if as_sequence:
            for idx in range(len(output)):
                out, tar = output[idx], batch[key][idx]
                max_len = min(batch[len_key][idx], len(out), len(tar))
                out = out[:max_len]
                tar = tar[:max_len].cuda()
                if as_int:
                    tar = tar.long()
                losses.append(f_loss(out, tar))
        else:
            pass
        loss = torch.stack(losses).mean()
    return loss


import os
import sys
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import printt, get_model_path
from .binding import BindingAffinityModel


def load_model(args, model_params, fold):
    """
        Model factory
    """
    # load model with specified arguments
    kwargs = {}
    model = BindingAffinityModel(args, model_params, **kwargs)
    printt("loaded model with kwargs:", " ".join(kwargs.keys()))
    # initialize weights
    _init(model)

    # (optional) load checkpoint if provided
    if args.checkpoint_path is not None:
        fold_dir = os.path.join(args.checkpoint_path, f"fold_{fold}")
        checkpoint = get_model_path(fold_dir)
        if checkpoint is not None:
            # extract current model
            state_dict = model.state_dict()
            # load onto CPU, transfer to proper GPU
            pretrain_dict = torch.load(checkpoint, map_location="cpu")["model"]
            pretrain_dict = {k:v for k,v in pretrain_dict.items() if ("mlp" not
                             in k )} #or "mlp2" in k)}
            # update current model
            state_dict.update(pretrain_dict)
            model.load_state_dict(state_dict)
            printt("loaded checkpoint from", checkpoint)
        else:
            printt("no checkpoint found")

        # >>> todo
        #print('freezing...')
        #for param in model.sequence_model.parameters():
        #    param.requires_grad = False
        #for param in model.mlp2.parameters():
        #    param.requires_grad = False
        # <<<

    # move to cuda if applicable
    if args.gpu >= 0:
        model = to_cuda(model, args)
        ## NOTE cannot get DataParallel to work with TAPE
        ## fails for Chemprop too...
        #model = nn.DataParallel(model)
    return model


def to_cuda(model, args):
    """
        move model to cuda
    """
    # specify number in case test =/= train GPU
    model = model.cuda(args.gpu)
    return model


def _init(model):
    """
        Wrapper around Xavier normal initialization
    """
    for name, param in model.named_parameters():
        # NOTE must name parameter "bert"
        if "bert" in name:
            continue
        # bias terms
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        # weight terms
        else:
            nn.init.xavier_normal_(param)


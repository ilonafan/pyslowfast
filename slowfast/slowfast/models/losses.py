#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from pytorchvideo.losses.soft_target_cross_entropy import (
    SoftTargetCrossEntropyLoss,
)


class ContrastiveLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, dummy_labels=None):
        targets = torch.zeros(inputs.shape[0], dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss(reduction=self.reduction).cuda()(
            inputs, targets
        )
        return loss


class MultipleMSELoss(nn.Module):
    """
    Compute multiple mse losses and return their average.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(MultipleMSELoss, self).__init__()
        self.mse_func = nn.MSELoss(reduction=reduction)

    def forward(self, x, y):
        loss_sum = 0.0
        multi_loss = []
        for xt, yt in zip(x, y):
            if isinstance(yt, (tuple,)):
                if len(yt) == 2:
                    yt, wt = yt
                    lt = "mse"
                elif len(yt) == 3:
                    yt, wt, lt = yt
                else:
                    raise NotImplementedError
            else:
                wt, lt = 1.0, "mse"
            if lt == "mse":
                loss = self.mse_func(xt, yt)
            else:
                raise NotImplementedError
            loss_sum += loss * wt
            multi_loss.append(loss)
        return loss_sum, multi_loss


def entropy(p):
    return Categorical(probs=p).entropy()

def entropy_loss(logits, reduction='mean'):

    losses = entropy(F.softmax(logits, dim=1))  # (N)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')

def cross_entropy(logits, labels, reduction='mean'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def kl_div(p, q, base=2):
    # p, q is in shape (batch_size, n_classes)
    if base == 2:
        return (p * p.log2() - p * q.log2()).sum(dim=1)
    else:
        return (p * p.log() - p * q.log()).sum(dim=1)


def symmetric_kl_div(p, q, base=2):
    return kl_div(p, q, base) + kl_div(q, p, base)


def js_div(p, q, base=2):
    # Jensen-Shannon divergence, value is in (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, base) + 0.5 * kl_div(q, m, base)

def get_aux_loss_func(cfg):
    if cfg.MODEL.LOSS_FUNC_AUX == 's-mae':
        aux_loss_func = F.smooth_l1_loss
    elif cfg.MODEL.LOSS_FUNC_AUX == 'mae':
        aux_loss_func = F.l1_loss
    elif cfg.MODEL.LOSS_FUNC_AUX == 'mse':
        aux_loss_func = F.mse_loss
    else:
        raise AssertionError(f'{cfg.MODEL.LOSS_FUNC_AUX} loss is not supported for auxiliary loss yet.')
    return aux_loss_func

# class ELRLoss(nn.Module):
#     'Compute early learning regularization loss'
#     def __init__(self, num_examp, num_classes=60, lam=3, beta=0.7):
#         """
#         Args:
#         `num_examp`: Total number of training examples.
#         `num_classes`: Number of classes in the classification problem.
#         `lam`: Regularization strength; must be a positive float, controling the strength of the ELR.
#         `beta`: Temporal ensembling momentum for target estimation.
#         """

#         super(ELRLoss, self).__init__()
#         self.num_classes = num_classes
#         self.USE_CUDA = torch.cuda.is_available()
#         self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp, self.num_classes)
#         self.beta = beta
#         self.lam = lam
        

def elr_loss(cfg, index, output, label, train_meter):
    """
    Args:
    `index`: Training sample index, due to training set shuffling, index is used to track training examples in different iterations.
    `output`: Model's logits, same as PyTorch provided loss functions.
    'label`: Labels, same as PyTorch provided loss functions.
    """

    y_pred = F.softmax(output, dim=1)
    y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
    y_pred_ = y_pred.data.detach()

    train_meter.target[index] = cfg.MODEL.BETA * train_meter.target[index] + (1 - cfg.MODEL.BETA) * ((y_pred_)/(y_pred_).sum(dim=1, keepdim=True))
    ce_loss = F.cross_entropy(output, label)
    elr_reg = ((1 - (train_meter.target[index] * y_pred).sum(dim=1)).log()).mean()
    final_loss = ce_loss + cfg.MODEL.LAM * elr_reg
    return final_loss

def elr_plus_loss(cfg, output, label, mixed_target):
    y_pred = F.softmax(output,dim=1)
    y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)

    # if self.num_classes == 100:
    #     y_labeled = y_labeled*self.q
    #     y_labeled = y_labeled/(y_labeled).sum(dim=1,keepdim=True)

    # ce_loss = torch.mean(-torch.sum(y_labeled * F.log_softmax(output, dim=1), dim = -1))
    ce_loss = F.cross_entropy(output, label)
    reg = ((1 - (mixed_target * y_pred).sum(dim=1)).log()).mean()
    final_loss = ce_loss + cfg.ELR_PLUS.LAM * reg
    return final_loss

def update_target(cfg, train_meter, output, index, mix_index, mixup_l = 1):
    y_pred_ = F.softmax(output, dim=1)
    train_meter.target[index] = cfg.ELR_PLUS.BETA * train_meter.target[index] + (1 - cfg.ELR_PLUS.BETA) *  (y_pred_)/(y_pred_).sum(dim=1, keepdim=True)
    mixed_target = mixup_l * train_meter.target[index] + (1 - mixup_l) * train_meter.target[index][mix_index]
    return mixed_target


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": partial(
        SoftTargetCrossEntropyLoss, normalize_targets=False
    ),
    "contrastive_loss": ContrastiveLoss,
    "elr_loss": nn.CrossEntropyLoss,
    "elr_plus_loss": nn.CrossEntropyLoss,
    "cdr": partial(
        SoftTargetCrossEntropyLoss, normalize_targets=False
    ),
    "mse": nn.MSELoss,
    "multi_mse": MultipleMSELoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]

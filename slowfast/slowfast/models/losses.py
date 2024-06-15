#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Function

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

class ELRLoss(nn.Module):
    'Compute early learning regularization loss'
    def __init__(self, cfg, num_examp):
        """
        Args:
        `num_examp`: Total number of training examples.
        `num_classes`: Number of classes in the classification problem.
        `lam`: Regularization strength; must be a positive float, controling the strength of the ELR.
        `beta`: Temporal ensembling momentum for target estimation.
        """

        super(ELRLoss, self).__init__()
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.USE_CUDA = torch.cuda.is_available()
        if cfg.MODEL.MODEL_NAME == 'MViT':
            self.ce_func = SoftTargetCrossEntropyLoss(normalize_targets=False) # Only for MVITv2
        else:
            self.ce_func = nn.CrossEntropyLoss(reduction='mean')
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp, self.num_classes)
        self.beta = cfg.ELR.BETA
        self.lam = cfg.ELR.LAM
        
    def forward(self, index, output, label):
        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()

        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        # ce_loss = F.cross_entropy(output, label)
        ce_loss = self.ce_func(output, label)
        elr_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + self.lam * elr_reg
        return final_loss

class IDFDLoss(nn.Module):
    def __init__(self, tau2):
        super().__init__()
        self.tau2 = tau2

    def forward(self, x, ff, y):
        L_id = F.cross_entropy(x, y)
        
        norm_ff = ff / (ff**2).sum(0, keepdim=True).sqrt()
        coef_mat = torch.mm(norm_ff.t(), norm_ff)
        coef_mat.div_(self.tau2)
        a = torch.arange(coef_mat.size(0), device=coef_mat.device)
        L_fd = F.cross_entropy(coef_mat, a)
        return L_id, L_fd
    
class NonParametricClassifierOP(Function):
    @staticmethod
    def forward(ctx, x, y, memory, params):

        tau = params[0].item()
        out = x.mm(memory.t())
        out.div_(tau)
        ctx.save_for_backward(x, memory, y, params)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, memory, y, params = ctx.saved_tensors
        tau = params[0]
        momentum = params[1]

        grad_output.div_(tau)

        grad_input = grad_output.mm(memory)
        grad_input.resize_as_(x)

        weight_pos = memory.index_select(0, y.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(x.mul(1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return grad_input, None, None, None, None


class NonParametricClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, tau=1.0, momentum=0.5):
        super(NonParametricClassifier, self).__init__()
        self.register_buffer('params', torch.tensor([tau, momentum]))
        stdv = 1. / np.sqrt(input_dim / 3.)
        self.register_buffer(
            'memory',
            torch.rand(output_dim, input_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        out = NonParametricClassifierOP.apply(x, y, self.memory, self.params)
        return out


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": partial(
        SoftTargetCrossEntropyLoss, normalize_targets=False
    ),
    "contrastive_loss": ContrastiveLoss,
    "elr_loss": ELRLoss,
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

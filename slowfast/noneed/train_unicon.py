#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import math
import numpy as np
import os
import pickle
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from datetime import datetime

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.utils.env import pathmgr
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.models.contrastive import (
    contrastive_forward,
    contrastive_parameter_surgery,
)
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule
from slowfast.models.losses import entropy, cross_entropy, entropy_loss, symmetric_kl_div, js_div, get_aux_loss_func, update_target
from slowfast.models.utils import mixup_data, update_ema_variables
from slowfast.models.losses import SupConLoss

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader,
    train_loader2,
    model1,
    model2,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model1.train()
    model2.eval()
    inputs_x1, inputs_x2, inputs_x3, inputs_x4 = [None] * 2, [None] * 2, [None] * 2, [None] * 2
    inputs_u1, inputs_u2, inputs_u3, inputs_u4 = [None] * 2, [None] * 2, [None] * 2, [None] * 2
    w_x = []
    unlabeled_train_iter = None
    train_meter.iter_tic()
    data_size = len(train_loader) 

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

    if cfg.MODEL.FROZEN_BN:
        misc.frozen_bn_stats(model1)
        misc.frozen_bn_stats(model2)
        
    # Explicitly declare reduction to mean.
    # if cfg.MODEL.LOSS_FUNC == 'elr_loss' or cfg.MODEL.LOSS_FUNC == 'elr_loss_plus':
    #     loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(num_examp=len(train_loader.dataset), num_classes=cfg.MODEL.NUM_CLASSES, lam=cfg.MODEL.LAM, beta=cfg.MODEL.BETA)
    # else:
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
    
    if cur_epoch >= cfg.SOLVER.WARMUP_EPOCHS:
        unlabeled_train_iter = iter(train_loader2) 
        
    for cur_iter, (inputs, labels, index, time, meta) in enumerate(
        train_loader
    ):  
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            # inputs[0] format is (B, C, T, H, W) 
            if isinstance(inputs, (list,)):   
                for i in range(len(inputs)):
                    if isinstance(inputs[i], (list,)):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if not isinstance(labels, list):
                labels = labels.cuda(non_blocking=True)
                index = index.cuda(non_blocking=True)
                time = time.cuda(non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        batch_size = (
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        )
    
        # Update the learning rate.
        epoch_exact = cur_epoch + float(cur_iter) / data_size
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples
        # else:
        #     # Transform label to one-hot
        #     labels = torch.zeros(batch_size, cfg.MODEL.NUM_CLASSES).cuda().scatter_(1, labels.long().view(-1,1), 1)  
        
        if cur_epoch >= cfg.SOLVER.WARMUP_EPOCHS:
            # print("len of inputs is: ", len(inputs))
            inputs_x1 = [inputs[0][:, 0:3, :, :, :]]
            inputs_x2 = [inputs[0][:, 3:6, :, :, :]]
            inputs_x3 = [inputs[0][:, 6:9, :, :, :]]
            inputs_x4 = [inputs[0][:, 9:12, :, :, :]]
            prob = train_loader.dataset.probability
            w_x = [prob[i] for i in index]
            print("w_x is: ", w_x)
            
            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, cfg.MODEL.NUM_CLASSES).cuda().scatter_(1, labels.long().view(-1,1), 1)     
            # print("labels_x is: ", labels_x)
            # label_ind = torch.argmax(labels, dim=1)
            # labels_x = F.one_hot(label_ind, num_classes=cfg.MODEL.NUM_CLASSES) 
            # w_x = w_x.view(-1,1).type(torch.FloatTensor)
            w_x = torch.FloatTensor(w_x).unsqueeze(-1)
            # print("shape of w_x: ", w_x.shape)
            
            try:
                inputs_u, _, _, _, _= unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(train_loader2)
                inputs_u, _, _, _, _ = unlabeled_train_iter.next()
            
            if cfg.NUM_GPUS:
                w_x = w_x.cuda(non_blocking=True)
                # inputs[0] format is (B, C, T, H, W) 
                if isinstance(inputs_u, (list,)):   
                    for i in range(len(inputs_u)):
                        if isinstance(inputs_u[i], (list,)):
                            for j in range(len(inputs_u[i])):
                                inputs_u[i][j] = inputs_u[i][j].cuda(non_blocking=True)
                        else:
                            inputs_u[i] = inputs_u[i].cuda(non_blocking=True)
                else:
                    inputs_u = inputs_u.cuda(non_blocking=True)
                # if not isinstance(labels_u, list):
                #     index_u = index_u.cuda(non_blocking=True)
                #     time_u = time_u.cuda(non_blocking=True)
                # for key, val in meta_u.items():
                #     if isinstance(val, (list,)):
                #         for i in range(len(val)):
                #             val[i] = val[i].cuda(non_blocking=True)
                #     else:
                #         meta_u[key] = val.cuda(non_blocking=True)
            inputs_u1 = [inputs_u[0][:, 0:3, :, :, :]]
            inputs_u2 = [inputs_u[0][:, 3:6, :, :, :]]
            inputs_u3 = [inputs_u[0][:, 6:9, :, :, :]]
            inputs_u4 = [inputs_u[0][:, 9:12, :, :, :]]
            
        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):

            # Explicitly declare reduction to mean.
            perform_backward = True
            optimizer.zero_grad()

            if cur_epoch < cfg.SOLVER.WARMUP_EPOCHS:
                preds = model1(inputs)
                loss = loss_fun(preds, labels)
            else:
                with torch.no_grad():
                    # # Label co-guessing of unlabeled samples
                    outputs_u11 = model1(inputs_u1)
                    outputs_u12 = model1(inputs_u2)
                    outputs_u21 = model2(inputs_u1)
                    outputs_u22 = model2(inputs_u2)            
                    
                    ## Pseudo-label
                    # pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
                    pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + outputs_u21 + outputs_u22) / 4 
                    T = 0.5
                    ptu = pu**(1/T)    ## Temparature Sharpening
                    
                    targets_u = ptu / ptu.sum(dim=1, keepdim=True)
                    targets_u = targets_u.detach()                  

                    ## Label refinement
                    outputs_x1  = model1(inputs_x1)
                    outputs_x2 = model2(inputs_x2)            
                    
                    # px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                    px = (torch.softmax(outputs_x1, dim=1) + outputs_x2) / 2

                    w_x = w_x.tile((1, cfg.MODEL.NUM_CLASSES))
                    px = w_x * labels_x + (1-w_x) * px         
                    ptx = px**(1/T)    ## Temparature sharpening 
                    
                    targets_x = ptx / ptx.sum(dim=1, keepdim=True)           
                    targets_x = targets_x.detach()
                
                ## Unsupervised Contrastive Loss
                f1 = model1(inputs_u3)
                f2 = model1(inputs_u4)
                f1 = F.normalize(f1, dim=1)
                f2 = F.normalize(f2, dim=1)
                features = torch.cat([f1.unsqueeze(-1), f2.unsqueeze(-1)], dim=-1)
                contrastive_criterion = SupConLoss()
                loss_simCLR = contrastive_criterion(features)
                
                # MixMatch
                alpha = 0.5
                l = np.random.beta(alpha, alpha)        
                l = max(l, 1-l)
                # all_inputs  = torch.cat((inputs_x3[0], inputs_x4[0], inputs_u3[0], inputs_u4[0]), dim=0)
                all_inputs = torch.cat((inputs_x3[0], inputs_x4[0]), dim=0)
                # all_targets = torch.cat((targets_x, targets_x, targets_u, targets_u), dim=0)
                all_targets = torch.cat((targets_x, targets_x), dim=0)

                idx = torch.randperm(all_inputs.size(0))

                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]
                
                ## Mixup
                mixed_input  = l * input_a  + (1 - l) * input_b        
                mixed_target = l * target_a + (1 - l) * target_b
                
                mixed_input_x = mixed_input[:batch_size*2]
                mixed_input_u = mixed_input[batch_size*2:]

                # logits_x = model1([mixed_input_x])
                logits_x = model1([all_inputs])  # no mixup
                # logits_u = model1([mixed_input_u])
                
                # logits = torch.cat((logits_x, logits_u), dim=0)
                # logits_x = logits[:batch_size*2]
                # logits_u = logits[batch_size*2:]        
                
                ## Combined Loss
                # criterion  = SemiLoss()
                # Lx, Lu, lamb = criterion(cfg, logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], cur_epoch, cfg.SOLVER.WARMUP_EPOCHS)
                # Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_target[:batch_size*2], dim=1))
                Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * all_targets, dim=1)) # no mixup
                
                ## Regularization
                prior = torch.ones(cfg.MODEL.NUM_CLASSES)/ cfg.MODEL.NUM_CLASSES
                prior = prior.cuda()        
                # pred_mean = torch.softmax(logits, dim=1).mean(0)
                pred_mean = torch.softmax(logits_x, dim=1).mean(0)
                penalty = torch.sum(prior*torch.log(prior/pred_mean))

                ## Total Loss
                lambda_c = 0.025
                # loss = Lx + lamb * Lu + lambda_c*loss_simCLR + penalty
                if math.isnan(loss_simCLR):
                    logger.info("ERROR: Got NaN simCLR loss at Epoch {}: Iteration {}".format(cur_epoch, cur_iter))
                    loss = Lx + penalty
                elif math.isnan(penalty):
                    logger.info("ERROR: Got NaN penalty at Epoch {}: Iteration {}".format(cur_epoch, cur_iter))
                    loss = Lx + lambda_c*loss_simCLR
                elif math.isnan(Lx):
                    logger.info("ERROR: Got NaN Lx loss at Epoch {}: Iteration {}".format(cur_epoch, cur_iter))
                    loss = penalty + lambda_c*loss_simCLR
                else:
                    loss = Lx + penalty + lambda_c*loss_simCLR
                
                preds = logits_x
                # _, top_max_k_inds = torch.topk(mixed_target[:batch_size*2], 1, dim=1, largest=True, sorted=True)
                _, top_max_k_inds = torch.topk(all_targets, 1, dim=1, largest=True, sorted=True)
                labels = top_max_k_inds

        loss_extra = None
        if isinstance(loss, (list, tuple)):
            loss, loss_extra = loss

        # check Nan Loss.
        # misc.check_nan_losses(loss)
        
        if math.isnan(loss):
            # logger.info("ERROR: Got NaN losses {}".format(datetime.now()))
            logger.info("ERROR: Got NaN losses at Epoch {}: Iteration {}".format(cur_epoch, cur_iter))
            train_meter.iter_toc()  # do measure allreduce for this meter
            torch.cuda.synchronize()
            train_meter.iter_tic()
            continue
        
        if perform_backward:
            scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        
        if cfg.MODEL.LOSS_FUNC == 'cdr':
            num_gradual = round((cfg.SOLVER.MAX_EPOCH + 1) * 0.1)
            clip_narry = np.linspace(0.5, 1, num=num_gradual)
            clip_narry = clip_narry[::-1]
            if cur_epoch < num_gradual:
                clip = clip_narry[cur_epoch]
            else:
                clip = 1 - 0.5
        
            to_concat_g = []
            to_concat_v = []
            for param in model1.parameters():
                # print('The dim of param is: ', param.dim())
                # if param.dim() in [2, 4]:
                to_concat_g.append(param.grad.data.view(-1))
                to_concat_v.append(param.data.view(-1))
            
            all_g = torch.cat(to_concat_g)
            all_v = torch.cat(to_concat_v)
            metric = torch.abs(all_g * all_v)
            num_params = all_v.size(0)
            nz = int(clip * num_params)
            top_values, _ = torch.topk(metric, nz)
            thresh = top_values[-1]

            for param in model1.parameters():
                # if param.dim() in [2, 4]:
                mask = (torch.abs(param.data * param.grad.data) >= thresh).type(torch.cuda.FloatTensor)
                mask = mask * clip
                param.grad.data = mask * param.grad.data
        
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            grad_norm = torch.nn.utils.clip_grad_value_(
                model1.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model1.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        else:
            grad_norm = optim.get_grad_norm_(model1.parameters())
        # Update the parameters. (defaults to True)
        model1, update_param = contrastive_parameter_surgery(
            model1, cfg, epoch_exact, cur_iter
        )
        if update_param:
            scaler.step(optimizer)
        scaler.update()

        if cfg.MIXUP.ENABLE:
            _top_max_k_vals, top_max_k_inds = torch.topk(
                labels, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
            preds = preds.detach()
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, grad_norm = du.all_reduce([loss, grad_norm])
                loss, grad_norm = (
                    loss.item(),
                    grad_norm.item(),
                )
            elif cfg.MASK.ENABLE:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, grad_norm = du.all_reduce([loss, grad_norm])
                    if loss_extra:
                        loss_extra = du.all_reduce(loss_extra)
                loss, grad_norm, top1_err, top5_err = (
                    loss.item(),
                    grad_norm.item(),
                    0.0,
                    0.0,
                )
                if loss_extra:
                    loss_extra = [one_loss.item() for one_loss in loss_extra]
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, grad_norm, top1_err, top5_err = du.all_reduce(
                        [loss.detach(), grad_norm, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, grad_norm, top1_err, top5_err = (
                    loss.item(),
                    grad_norm.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                grad_norm,
                batch_size
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                loss_extra,
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )
        train_meter.iter_toc()  # do measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter, cfg.OUTPUT_DIR)  #Fix: Add cfg.OUTPUT_DIR
        torch.cuda.synchronize()
        train_meter.iter_tic()
    
    del inputs
    del inputs_u

    # in case of fragmented memory
    torch.cuda.empty_cache()
    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch, cfg.OUTPUT_DIR)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(
    val_loader, model1, model2, val_meter, cur_epoch, cfg, writer
):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model1.eval()
    model2.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, index, time, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
            index = index.cuda()
            time = time.cuda()
        batch_size = (
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        )
        val_meter.data_toc()

        # if cfg.DETECTION.ENABLE:
        #     # Compute the predictions.
        #     preds = model(inputs, meta["boxes"])
        #     ori_boxes = meta["ori_boxes"]
        #     metadata = meta["metadata"]

        #     if cfg.NUM_GPUS:
        #         preds = preds.cpu()
        #         ori_boxes = ori_boxes.cpu()
        #         metadata = metadata.cpu()

        #     if cfg.NUM_GPUS > 1:
        #         preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
        #         ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
        #         metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

        #     val_meter.iter_toc()
        #     # Update and log stats.
        #     val_meter.update_stats(preds, ori_boxes, metadata)

        # else:
            # if cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            #     if not cfg.CONTRASTIVE.KNN_ON:
            #         return
            #     train_labels = (
            #         model.module.train_labels
            #         if hasattr(model, "module")
            #         else model.train_labels
            #     )
            #     yd, yi = model(inputs, index, time)
            #     K = yi.shape[1]
            #     C = (
            #         cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM
            #     )  # eg 400 for Kinetics400
            #     candidates = train_labels.view(1, -1).expand(batch_size, -1)
            #     retrieval = torch.gather(candidates, 1, yi)
            #     retrieval_one_hot = torch.zeros((batch_size * K, C)).cuda()
            #     retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            #     yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
            #     probs = torch.mul(
            #         retrieval_one_hot.view(batch_size, -1, C),
            #         yd_transform.view(batch_size, -1, 1),
            #     )
            #     preds = torch.sum(probs, 1)
            # elif cfg.MODEL.MODEL_NAME == 'MVIT_PNP':
            #     preds = model(inputs)['logits']    
            # else:
        preds = model1(inputs) + model2(inputs)

        if cfg.DATA.MULTI_LABEL:
            if cfg.NUM_GPUS > 1:
                preds, labels = du.all_gather([preds, labels])
        else:
            if cfg.DATA.IN22k_VAL_IN1K != "":
                preds = preds[:, :1000]
            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

            # Combine the errors across the GPUs.
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])

            # Copy the errors from GPU to CPU (sync point).
            top1_err, top5_err = top1_err.item(), top5_err.item()

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_err,
                top5_err,
                batch_size
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                    global_step=len(val_loader) * cur_epoch + cur_iter,
                )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter, cfg.OUTPUT_DIR)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )
            
            # Write val predictions into file      
            if cur_epoch == cfg.SOLVER.MAX_EPOCH:
                save_path = os.path.join(cfg.OUTPUT_DIR, "val_pred.dat")
                if os.path.exists(save_path):
                    save_path = os.path.join(cfg.OUTPUT_DIR, "val_pred_2.dat")

                if du.is_root_proc():
                    with pathmgr.open(save_path, "wb") as f:
                        pickle.dump([all_preds, all_labels], f)

                logger.info(
                    "Successfully saved val prediction results to {}".format(save_path)
                )

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)

def get_smoothed_label_distribution(labels, num_class, epsilon):
    smoothed_label = torch.full(size=(labels.size(0), num_class), fill_value=epsilon / (num_class - 1))
    addier = torch.mul(labels.cpu(), 1 - epsilon)
    smoothed_label = torch.add(smoothed_label, addier)
    # smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1 - epsilon)
    return smoothed_label.to(labels.device) 

def calculate_elr_plus_loss(cfg, model, inputs, labels, index, train_meter):
    mixed_inputs, mixed_labels, mixup_l, mixup_index = mixup_data(inputs, labels, cfg.ELR_PLUS.ALPHA)   
    
    preds_original = train_meter.model_ema([inputs])
    preds_original = preds_original.data.detach()
    
    mixed_target = update_target(cfg, train_meter, preds_original, index, mixup_index, mixup_l)
    preds = model([mixed_inputs])
    loss = elr_plus_loss(cfg, preds, mixed_labels, mixed_target)
    return loss, preds


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        flops, params = misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model1 = build_model(cfg)
    model2 = build_model(cfg)
    
    # Additional logger info for train head only
    logger.info(f"Train head only: {cfg.TRAIN.TRAIN_HEAD_ONLY}")
    
    # if cfg.TRAIN.TRAIN_HEAD_ONLY:
    #     _freeze_except_head(cfg, model)
    #     logger.info("Freeze model except the head.")
    
    logger.info("Two models are built!")
    flops, params = 0.0, 0.0
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        flops, params = misc.log_model_info(model1, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer1 = optim.construct_optimizer(model1, cfg)
    optimizer2 = optim.construct_optimizer(model2, cfg)
    # Create a GradScaler for mixed precision training
    scaler1 = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    scaler2 = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    logger.info("Optimizers and scalers are built!")

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task=cfg.TASK)
        last_checkpoint_2 = cu.get_last_checkpoint_unicon(cfg.OUTPUT_DIR, task=cfg.TASK)
        if last_checkpoint is not None:
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model1,
                cfg.NUM_GPUS > 1,
                optimizer1,
                scaler1 if cfg.TRAIN.MIXED_PRECISION else None,
            )
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint_2,
                model2,
                cfg.NUM_GPUS > 1,
                optimizer2,
                scaler2 if cfg.TRAIN.MIXED_PRECISION else None,
            )
            start_epoch = checkpoint_epoch + 1
        # elif "ssl_eval" in cfg.TASK:
        #     last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task="ssl")
        #     checkpoint_epoch = cu.load_checkpoint(
        #         last_checkpoint,
        #         model,
        #         cfg.NUM_GPUS > 1,
        #         optimizer,
        #         scaler if cfg.TRAIN.MIXED_PRECISION else None,
        #         epoch_reset=True,
        #         clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
        #     )
        #     start_epoch = checkpoint_epoch + 1
        else:
            start_epoch = 0
    
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model1,
            cfg.NUM_GPUS > 1,
            optimizer1,
            scaler1 if cfg.TRAIN.MIXED_PRECISION else None,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
        )
        cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model2,
            cfg.NUM_GPUS > 1,
            optimizer2,
            scaler2 if cfg.TRAIN.MIXED_PRECISION else None,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    # Create the video train and val loaders.    
    eval_train_loader = loader.construct_loader(cfg, "eval_train")
    val_loader = loader.construct_loader(cfg, "val")
    warmup_loader = loader.construct_loader(cfg, "warmup")
    # precise_bn_loader = (
    #     loader.construct_loader(cfg, "train", is_precise_bn=True)
    #     if cfg.BN.USE_PRECISE_STATS
    #     else None
    # )

    # if (
    #     cfg.TASK == "ssl"
    #     and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
    #     and cfg.CONTRASTIVE.KNN_ON
    # ):
    #     if hasattr(model, "module"):
    #         model.module.init_knn_labels(train_loader)
    #     else:
    #         model.init_knn_labels(train_loader)

    # Create meters.
    # if cfg.DETECTION.ENABLE:
    #     train_meter = AVAMeter(len(train_loader), cfg, mode="train")
    #     val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    # elif cfg.MODEL.LOSS_FUNC == 'elr_loss':
    #     train_meter = EarlyLearningMeter(len(train_loader), cfg, len(train_loader.dataset))
    #     val_meter = ValMeter(len(val_loader), cfg)
    # elif cfg.MODEL.LOSS_FUNC == 'elr_plus_loss':
    #     model_ema = build_model(cfg)
    #     for param in model_ema.parameters():
    #         param.data = torch.zeros_like(param.data)
    #         param.requires_grad = False
    #     logger.info(f"Model_ema is built!")
        
    #     train_meter = EarlyLearningPlusMeter(len(train_loader), cfg, len(train_loader.dataset), model_ema)
    #     val_meter = ValMeter(len(val_loader), cfg)
    # else:
    
    val_meter = ValMeter(len(val_loader), cfg)
    warmup_meter1 = TrainMeter(len(warmup_loader), cfg)
    warmup_meter2 = TrainMeter(len(warmup_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        # if cur_epoch > 0 and cfg.DATA.LOADER_CHUNK_SIZE > 0:
        #     num_chunks = math.ceil(
        #         cfg.DATA.LOADER_CHUNK_OVERALL_SIZE / cfg.DATA.LOADER_CHUNK_SIZE
        #     )
        #     skip_rows = (cur_epoch) % num_chunks * cfg.DATA.LOADER_CHUNK_SIZE
        #     logger.info(
        #         f"=================+++ num_chunks {num_chunks} skip_rows {skip_rows}"
        #     )
        #     cfg.DATA.SKIP_ROWS = skip_rows
        #     logger.info(f"|===========| skip_rows {skip_rows}")
        #     train_loader = loader.construct_loader(cfg, "train")
        #     loader.shuffle_dataset(train_loader, cur_epoch)

        # if cfg.MULTIGRID.LONG_CYCLE:
        #     cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
        #     if changed:
        #         (
        #             model,
        #             optimizer,
        #             train_loader,
        #             val_loader,
        #             precise_bn_loader,
        #             train_meter,
        #             val_meter,
        #         ) = build_trainer(cfg)

        #         # Load checkpoint.
        #         if cu.has_checkpoint(cfg.OUTPUT_DIR):
        #             last_checkpoint = cu.get_last_checkpoint(
        #                 cfg.OUTPUT_DIR, task=cfg.TASK
        #             )
        #             assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
        #         else:
        #             last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
        #         logger.info("Load from {}".format(last_checkpoint))
        #         cu.load_checkpoint(
        #             last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
        #         )

        # loader.shuffle_dataset(train_loader, cur_epoch)
        # if hasattr(train_loader.dataset, "_set_epoch_num"):
        #     train_loader.dataset._set_epoch_num(cur_epoch)
        
        
        # Train for one epoch.
        epoch_timer.epoch_tic()
        if cur_epoch < cfg.SOLVER.WARMUP_EPOCHS:
            
            # Shuffle the dataset.
            loader.shuffle_dataset(warmup_loader, cur_epoch)
            if hasattr(warmup_loader.dataset, "_set_epoch_num"):
                warmup_loader.dataset._set_epoch_num(cur_epoch)
                
            train_epoch(
                warmup_loader,
                None,
                model1,
                model2,
                optimizer1,
                scaler1,
                warmup_meter1,
                cur_epoch,
                cfg,
                writer,
            )
            
            train_epoch(
                warmup_loader,
                None,
                model2,
                model1,
                optimizer2,
                scaler2,
                warmup_meter2,
                cur_epoch,
                cfg,
                writer,
            )
            epoch_timer.epoch_toc()
            logger.info(
                f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
                f"from {start_epoch} to {cur_epoch} take "
                f"{epoch_timer.avg_epoch_time():.2f}s in average and "
                f"{epoch_timer.median_epoch_time():.2f}s in median."
            )
            logger.info(
                f"For epoch {cur_epoch}, each iteraction takes "
                f"{epoch_timer.last_epoch_time()/len(warmup_loader):.2f}s in average. "
                f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
                f"{epoch_timer.avg_epoch_time()/len(warmup_loader):.2f}s in average."
            )
        else:
            prob = Calculate_JSD(cfg, model1, model2, eval_train_loader)           
            threshold = torch.mean(prob)
            d_mu = 0.7
            tau = 5
            if threshold.item() >= d_mu:
                threshold = threshold - (threshold-torch.min(prob))/tau
            SR = torch.sum(prob<threshold).item() / eval_train_loader.dataset.num_videos  
            
            print("SR is: ", SR)
            
            labeled_train_loader = loader.construct_loader(cfg, "train", sample_ratio=SR, prob=prob, noise="labeled")  
            unlabeled_train_loader = loader.construct_loader(cfg, "train", sample_ratio=SR, prob=prob, noise="unlabeled")        
            
            train_meter1 = TrainMeter(len(labeled_train_loader), cfg) 
            train_meter2 = TrainMeter(len(labeled_train_loader), cfg)  
            
            logger.info('Train Model1!')
            train_epoch(
                labeled_train_loader,
                unlabeled_train_loader,
                model1,
                model2,
                optimizer1,
                scaler1,
                train_meter1,
                cur_epoch,
                cfg,
                writer,
            )
            logger.info('Train Model2!')
            train_epoch(
                labeled_train_loader,
                unlabeled_train_loader,
                model2,
                model1,
                optimizer2,
                scaler2,
                train_meter2,
                cur_epoch,
                cfg,
                writer,
            )
            epoch_timer.epoch_toc()
            logger.info(
                f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
                f"from {start_epoch} to {cur_epoch} take "
                f"{epoch_timer.avg_epoch_time():.2f}s in average and "
                f"{epoch_timer.median_epoch_time():.2f}s in median."
            )
            logger.info(
                f"For epoch {cur_epoch}, each iteraction takes "
                f"{epoch_timer.last_epoch_time()/len(labeled_train_loader):.2f}s in average. "
                f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
                f"{epoch_timer.avg_epoch_time()/len(labeled_train_loader):.2f}s in average."
            )

        is_checkp_epoch = (
            cu.is_checkpoint_epoch(
                cfg,
                cur_epoch,
                None if multigrid is None else multigrid.schedule,
            )
            or cur_epoch == cfg.SOLVER.MAX_EPOCH - 1
        )
        is_eval_epoch = (
            misc.is_eval_epoch(
                cfg,
                cur_epoch,
                None if multigrid is None else multigrid.schedule,
            )
            and not cfg.MASK.ENABLE
        )

        # Compute precise BN stats.
        # if (
        #     (is_checkp_epoch or is_eval_epoch)
        #     and cfg.BN.USE_PRECISE_STATS
        #     and len(get_bn_modules(model)) > 0
        # ):
        #     calculate_and_update_precise_bn(
        #         precise_bn_loader,
        #         model,
        #         min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
        #         cfg.NUM_GPUS > 0,
        #     )
        # _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model1,
                optimizer1,
                cur_epoch,
                cfg,
                scaler1 if cfg.TRAIN.MIXED_PRECISION else None,
            )
            
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model2,
                optimizer2,
                cur_epoch,
                cfg,
                scaler2 if cfg.TRAIN.MIXED_PRECISION else None,
            )
            
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(
                val_loader,
                model1,
                model2,
                val_meter,
                cur_epoch,
                cfg,
                writer,
            )

    if start_epoch == cfg.SOLVER.MAX_EPOCH and not cfg.MASK.ENABLE: # final checkpoint load
        eval_epoch(val_loader, model1, model2, val_meter, start_epoch, cfg, writer)
        
    if writer is not None:
        writer.close()
    result_string = (
        "_p{:.2f}_f{:.2f} _t{:.2f}_m{:.2f} _a{:.2f} Top5 Acc: {:.2f} MEM: {:.2f} f: {:.4f}"
        "".format(
            params / 1e6,
            flops,
            epoch_timer.median_epoch_time() / 60.0
            if len(epoch_timer.epoch_times)
            else 0.0,
            misc.gpu_mem_usage(),
            100 - val_meter.min_top1_err,
            100 - val_meter.min_top5_err,
            misc.gpu_mem_usage(),
            flops,
        )
    )
    logger.info("training done: {}".format(result_string))
    return result_string


# KL divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## Jensen-Shannon Divergence 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

## Calculate JSD
def Calculate_JSD(cfg, model1, model2, eval_train_loader):  
    JS_dist = Jensen_Shannon()
    JSD = torch.zeros(eval_train_loader.dataset.num_videos).cuda()    

    for batch_idx, (inputs, labels, index, time, meta) in enumerate(eval_train_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
            index = index.cuda()
            time = time.cuda()
        batch_size = (
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        )

        # model1.train()
        # model2.train()
        
        ## Get outputs of both network
        # with torch.no_grad():
        #     out1 = nn.Softmax(dim=1).cuda()(model1(inputs))     
        #     out2 = nn.Softmax(dim=1).cuda()(model2(inputs))

        ## Get the Prediction
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            out1 = model1(inputs)
            out2 = model2(inputs)
        
        out = (out1 + out2)/2     

        ## Divergence calculator to record the diff. between ground truth and output prob. dist.  
        dist = JS_dist(out, F.one_hot(labels, num_classes = cfg.MODEL.NUM_CLASSES))
        # print("Shape of dist, :", dist.shape)
        # print("Device of dist, :", dist.device)
        # print("Shape of index, :", index.shape)
        # print("Device of index, :", index.device)
        JSD[index] = dist 
        # JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist
        # batch_idx < len(eval_train_loader)-1:
        #     JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist
        # else: 
        #     JSD[int(batch_idx*batch_size):] = dist if
    return JSD

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    lambda_u = 30
    return lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, cfg, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, linear_rampup(epoch, warm_up)

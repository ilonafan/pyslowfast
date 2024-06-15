#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import math
import numpy as np
import os
import pickle
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
import torch.nn.functional as F
import faiss
from pathlib import Path
import pandas as pd

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
from slowfast.models.losses import RankLoss

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    v_centers,
    v_clusters,
    t_centers,
    clean_idx,
    hard_labels,
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
    model.train()
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
        misc.frozen_bn_stats(model)
        
    # Explicitly declare reduction to mean.
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

    # print("The type of loss_fun is: ", loss_fun.__class__.__name__)
    
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

        batch_size = int((
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        ) / 2)
        
        data = [inputs[0][1::2, :, :, :, :]]
        data_aug = [inputs[0][0::2, :, :, :, :]]
        # print("data shape: {}, data_aug shape: {}".format(data[0].shape, data_aug[0].shape))
        
        # Update the learning rate.
        epoch_exact = cur_epoch + float(cur_iter) / data_size
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples
        
        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):

            # Explicitly declare reduction to mean.
            perform_backward = True
            optimizer.zero_grad()
            
            # Hyperparameter
            temperature = 0.3 
            alpha = 8 # Select the value as Mixup paper suggested
            
            if cur_epoch < cfg.SOLVER.WARMUP_EPOCHS:
                preds, feat, error_recon = model(data)  
                # print("preds shape: {}, labels shape: {}".format(preds.shape, labels.shape))
                
                shuffle_idx = torch.randperm(batch_size)
                mapping = {k:v for (v,k) in enumerate(shuffle_idx)}
                reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])     
                _, feat_aug, error_recon_aug = model([data_aug[0][shuffle_idx]])  
                feat_aug = feat_aug[reverse_idx]
                
                # Reconstruction loss
                loss_recon = error_recon.mean() 
                loss_recon_aug = error_recon_aug.mean()    
                
                # Instance contrastive loss
                sim_clean = torch.mm(feat, feat.t())
                mask = (torch.ones_like(sim_clean) - torch.eye(batch_size, device=sim_clean.device)).bool()
                sim_clean = sim_clean.masked_select(mask).view(batch_size, -1)

                sim_aug = torch.mm(feat, feat_aug.t())
                sim_aug = sim_aug.masked_select(mask).view(batch_size, -1)   
                
                logits_pos = torch.bmm(feat.view(batch_size,1,-1),feat_aug.view(batch_size,-1,1)).squeeze(-1)
                logits_neg = torch.cat([sim_clean,sim_aug],dim=1)

                logits = torch.cat([logits_pos,logits_neg],dim=1)
                instance_labels = torch.zeros(batch_size).long().cuda()
                loss_cc = loss_fun(logits/temperature, instance_labels)    
                
                # Cross-modal mixup rank loss
                L = np.random.beta(alpha, alpha)   
                L = max(L, 1-L)  
                one_hot_labels = torch.zeros(batch_size, cfg.MODEL.NUM_CLASSES).cuda().scatter_(1, labels.view(-1,1), 1) 
                
                all_inputs = torch.cat((data[0], data_aug[0]),dim=0)
                idx = torch.randperm(batch_size*2) 
                all_labels = torch.cat((one_hot_labels,one_hot_labels),dim=0)      
                
                input_mix = L * all_inputs + (1 - L) * all_inputs[idx]  
                labels_mix = L * all_labels + (1 - L) * all_labels[idx]
                
                preds_mix, feat_mix, _ = model([input_mix])  
                loss_ce_mix = loss_fun(preds_mix, labels_mix)  # Mixup cross entropy loss
                
                # Compute video centers of each mixup input in the batch
                v_centers = v_centers.cuda()
                
                epsilon = torch.tensor(1e-10).type(torch.FloatTensor)
                v_centers_ = torch.sum(v_centers ** 2, dim=1, keepdim=True)
                feat_mix_ = torch.sum(feat_mix ** 2, dim=1, keepdim=True)

                distance_ = torch.sqrt(torch.max(v_centers_ - 2 * torch.mm(v_centers, feat_mix.T) + feat_mix_.T, epsilon))
                assignment = torch.argmin(distance_, dim=0) # (B) 
                v_centers_b = v_centers[assignment]
                
                # Compute distances between v_centers_b and t_vecs
                v_centers_b_ = torch.sum(v_centers_b ** 2, dim=1, keepdim=True)
                t_centers_ = torch.sum(t_centers ** 2, dim=1, keepdim=True)

                # print(v_centers_b.shape)
                # print(t_centers.shape)
                vt_distance = torch.sqrt(torch.max(v_centers_b_ - 2 * torch.mm(v_centers_b, t_centers.T) + t_centers_.T, epsilon))
                # vt_distance = torch.min(vt_distance_, dim=0)
                # vt_distance = vt_distance.values
                # print(vt_distance.shape)
                
                criterion = RankLoss()
                loss_cm_mix = L * criterion(vt_distance, all_labels) + (1 - L) * criterion(vt_distance, all_labels[idx])
                
                # Joint loss
                loss = loss_ce_mix + loss_cm_mix + loss_cc + loss_recon + loss_recon_aug
                print("The cross modal rank loss is: ", loss_cm_mix)
                
                # For train meter statistics
                # preds = preds_mix
                # labels = labels_mix
                
            else:
                target = hard_labels[index].cuda(non_blocking=True) 
                clean_idx_batch = clean_idx[index] 
                if True not in clean_idx_batch.tolist():
                    logger.info("Warning: No clean idx at Epoch {}: Iteration {}".format(cur_epoch, cur_iter))
                    train_meter.iter_toc()  # do measure allreduce for this meter
                    torch.cuda.synchronize()
                    train_meter.iter_tic()
                    continue
                
                # Cross entropy loss
                output, feat, error_recon = model(data)
                loss_ce = loss_fun(output[clean_idx_batch], target[clean_idx_batch])   
                
                # Instance contrastive loss
                shuffle_idx = torch.randperm(batch_size)
                mapping = {k:v for (v,k) in enumerate(shuffle_idx)}
                reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])     
                _, feat_aug, error_recon_aug = model([data_aug[0][shuffle_idx]])   
                feat_aug = feat_aug[reverse_idx]
                
                # Recontruction loss
                loss_recon = error_recon.mean()
                loss_recon_aug = error_recon_aug.mean()
                
                sim_clean = torch.mm(feat, feat.t())
                mask = (torch.ones_like(sim_clean) - torch.eye(batch_size, device=sim_clean.device)).bool()
                sim_clean = sim_clean.masked_select(mask).view(batch_size, -1)

                sim_aug = torch.mm(feat, feat_aug.t())
                sim_aug = sim_aug.masked_select(mask).view(batch_size, -1)   

                logits_pos = torch.bmm(feat.view(batch_size,1,-1),feat_aug.view(batch_size,-1,1)).squeeze(-1)
                logits_neg = torch.cat([sim_clean,sim_aug],dim=1)

                logits = torch.cat([logits_pos,logits_neg],dim=1)
                instance_labels = torch.zeros(batch_size).long().cuda()
                loss_cc = loss_fun(logits/temperature, instance_labels)                

                # L = np.random.beta(alpha, alpha)    
                # L = max(L, 1-L)
                one_hot_target = torch.zeros(batch_size, cfg.MODEL.NUM_CLASSES).cuda().scatter_(1, target.view(-1,1), 1)  
                # one_hot_target = one_hot_target[clean_idx_batch]      
        
                # all_inputs = torch.cat((data[0][clean_idx_batch], data_aug[0][clean_idx_batch]),dim=0)
                # idx = torch.randperm(clean_idx_batch.sum()*2) 
                # all_labels = torch.cat((one_hot_target, one_hot_target),dim=0)

                # input_mix = L * all_inputs + (1 - L) * all_inputs[idx]  
                # labels_mix = L * all_labels + (1 - L) * all_labels[idx]

                # _, feat_mix, _ = model([input_mix])  

                # logits_proto = torch.mm(feat_mix, prototypes.t())/ temperature      
                # loss_proto = -torch.mean(torch.sum(F.log_softmax(logits_proto, dim=1) * labels_mix, dim=1))    
                
                
                # Cross-modal rank loss   
                # Get video centers of each input in the batch
                v_centers = v_centers.cuda()
                epsilon = torch.tensor(1e-10).type(torch.FloatTensor)
                v_centers_b = v_centers[v_clusters[index][clean_idx_batch]]
                
                # Compute distances between v_centers_b and t_vecs
                v_centers_b_ = torch.sum(v_centers_b ** 2, dim=1, keepdim=True)
                t_centers_ = torch.sum(t_centers ** 2, dim=1, keepdim=True)

                vt_distance = torch.sqrt(torch.max(v_centers_b_ - 2 * torch.mm(v_centers_b, t_centers.T) + t_centers_.T, epsilon))
                # vt_distance = torch.min(vt_distance_, dim=0)
                # vt_distance = vt_distance.values
                
                criterion = RankLoss()
                loss_cm = criterion(vt_distance, one_hot_target[clean_idx_batch])
                
                print("The cross modal rank loss is: ", loss_cm)
                loss = loss_ce + loss_cm + loss_cc + loss_recon + loss_recon_aug
                
                # For train meter statistics
                preds = output[clean_idx_batch]
                labels = target[clean_idx_batch]
                
                ############### 
                # preds = model(inputs)
                # loss = loss_fun(preds, labels)

        loss_extra = None
        if isinstance(loss, (list, tuple)):
            loss, loss_extra = loss

        # check Nan Loss.
        misc.check_nan_losses(loss)
        if perform_backward:
            scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            grad_norm = torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        else:
            grad_norm = optim.get_grad_norm_(model.parameters())
        # Update the parameters. (defaults to True)
        model, update_param = contrastive_parameter_surgery(
            model, cfg, epoch_exact, cur_iter
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

        top1_err, top5_err = None, None
        
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

    # in case of fragmented memory
    torch.cuda.empty_cache()
    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch, cfg.OUTPUT_DIR)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(
    val_loader, model, val_meter, cur_epoch, cfg, train_loader, writer
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
    model.eval()
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

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            if cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                if not cfg.CONTRASTIVE.KNN_ON:
                    return
                train_labels = (
                    model.module.train_labels
                    if hasattr(model, "module")
                    else model.train_labels
                )
                yd, yi = model(inputs, index, time)
                K = yi.shape[1]
                C = (
                    cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM
                )  # eg 400 for Kinetics400
                candidates = train_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)
                retrieval_one_hot = torch.zeros((batch_size * K, C)).cuda()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
                probs = torch.mul(
                    retrieval_one_hot.view(batch_size, -1, C),
                    yd_transform.view(batch_size, -1, 1),
                )
                preds = torch.sum(probs, 1)
            elif cfg.MODEL.MODEL_NAME == 'MVIT_PNP':
                preds = model(inputs)['logits']    
            elif cfg.TRAIN.DATASET == 'rrl':
                preds, _, _ = model(inputs)
            else:
                preds = model(inputs)

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
            if cur_epoch == cfg.SOLVER.MAX_EPOCH - 1:
                save_path = os.path.join(cfg.OUTPUT_DIR, "val_pred.dat")

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
    model = build_model(cfg)
    
    # Additional logger info for train head only
    logger.info(f"Train head only: {cfg.TRAIN.TRAIN_HEAD_ONLY}")
    
    # if cfg.TRAIN.TRAIN_HEAD_ONLY:
    #     _freeze_except_head(cfg, model)
    #     logger.info("Freeze model except the head.")
    
    logger.info("Model is built!")
    flops, params = 0.0, 0.0
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        flops, params = misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    logger.info("Optimizer and scaler is built!")

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task=cfg.TASK)
        if last_checkpoint is not None:
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
            start_epoch = checkpoint_epoch + 1
        elif "ssl_eval" in cfg.TASK:
            last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task="ssl")
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
                epoch_reset=True,
                clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            )
            start_epoch = checkpoint_epoch + 1
        else:
            start_epoch = 0
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
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
    train_loader = loader.construct_loader(cfg, "train")
    eval_loader = loader.construct_loader(cfg, "eval")
    val_loader = loader.construct_loader(cfg, "val")
    # precise_bn_loader = (
    #     loader.construct_loader(cfg, "train", is_precise_bn=True)
    #     if cfg.BN.USE_PRECISE_STATS
    #     else None
    # )
    precise_bn_loader = (
        loader.construct_loader(cfg, "eval", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    if (
        cfg.TASK == "ssl"
        and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        and cfg.CONTRASTIVE.KNN_ON
    ):
        if hasattr(model, "module"):
            model.module.init_knn_labels(train_loader)
        else:
            model.init_knn_labels(train_loader)

    # Create loss function for ELR method explicitly.
    if cfg.MODEL.LOSS_FUNC == 'elr_loss':
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(cfg, num_examp=train_loader.dataset.num_videos)
    else:
        loss_fun = None
    
    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Declare some global variables
    
    
    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        if cur_epoch > 0 and cfg.DATA.LOADER_CHUNK_SIZE > 0:
            num_chunks = math.ceil(
                cfg.DATA.LOADER_CHUNK_OVERALL_SIZE / cfg.DATA.LOADER_CHUNK_SIZE
            )
            skip_rows = (cur_epoch) % num_chunks * cfg.DATA.LOADER_CHUNK_SIZE
            logger.info(
                f"=================+++ num_chunks {num_chunks} skip_rows {skip_rows}"
            )
            cfg.DATA.SKIP_ROWS = skip_rows
            logger.info(f"|===========| skip_rows {skip_rows}")
            train_loader = loader.construct_loader(cfg, "train")
            loader.shuffle_dataset(train_loader, cur_epoch)

        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(
                        cfg.OUTPUT_DIR, task=cfg.TASK
                    )
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        if hasattr(train_loader.dataset, "_set_epoch_num"):
            train_loader.dataset._set_epoch_num(cur_epoch)
        
        # Cluster embeddings and compute distances between video centers and text centers
        features, labels, probs = compute_features(cfg, eval_loader, model)  
        hard_labels = probs.clone()
        clean_idx = labels.clone()
        
        v_centers, v_clusters = compute_clusterings(cfg, features)
        t_centers = get_text_embeddings(cfg)
        
        # v_centers_ = torch.sum(v_centers ** 2, dim=1, keepdim=True)
        # t_centers_ = torch.sum(t_centers ** 2, dim=1, keepdim=True)

        # epsilon = torch.tensor(1e-10).type(torch.FloatTensor)
        # distance_ = torch.sqrt(torch.max(v_centers_ - 2 * torch.mm(v_centers, t_centers.T) + t_centers_.T, epsilon))
        # distance = torch.min(distance_, dim=1)
        # distance = distance.values
        
        # # mean_th = probs.mean(0)[labels]  
        # std, mean = torch.std_mean(probs, dim=0) 
        # mean_th = std[labels] + mean[labels] 
            
        # if cur_epoch < cfg.SOLVER.WARMUP_EPOCHS:        
        #     for c in range(cfg.MODEL.NUM_CLASSES):   
        #         prototype = features[np.where(labels.numpy()==c)].mean(0)    #compute prototypes as mean embeddings       
        #         prototypes.append(torch.Tensor(prototype))      
            
        if cur_epoch >= cfg.SOLVER.WARMUP_EPOCHS:
            if cur_epoch == cfg.SOLVER.WARMUP_EPOCHS:
                # Initalize the soft label as model's softmax prediction
                gt_score = probs[labels>=0, labels] # (N)
                
                # Compute threshold of each clustering
                num_clusters = cfg.MODEL.NUM_CLASSES
                N = eval_loader.dataset.num_videos
                thresholds = torch.zeros((num_clusters, probs.shape[1]))
                for i in range(num_clusters):
                    soft_labels_cluster = []
                    for j in range(N):
                        if v_clusters[j] == i:
                            soft_labels_cluster.append(train_loader.dataset.soft_labels[j])
                    soft_labels_cluster = torch.stack(soft_labels_cluster, dim=0)
                    std, mean = torch.std_mean(soft_labels_cluster, dim=0)
                    thresholds[i] = std + mean
                
                th = torch.diagonal(thresholds, 0)
                gt_clean = gt_score >= th[labels]
                train_loader.dataset.soft_labels = probs.clone()  
                train_loader.dataset.soft_labels[gt_clean] = torch.zeros(gt_clean.sum(), cfg.MODEL.NUM_CLASSES).scatter_(1, labels[gt_clean].view(-1,1), 1)     
            
            # Generate new soft label 
            clean_idx, hard_labels, pred_score = label_clean(cfg, features, labels, probs, train_loader, v_clusters)
            
            # Write val predictions into file      
            if cur_epoch == cfg.SOLVER.MAX_EPOCH - 1:
                save_path = os.path.join(cfg.OUTPUT_DIR, "pseudo_labels.dat")
                if du.is_root_proc():
                    with pathmgr.open(save_path, "wb") as f:
                        pickle.dump([clean_idx.tolist(), hard_labels.tolist(), pred_score.tolist(), train_loader.dataset.soft_labels.tolist()], f)
                logger.info(
                    "Successfully saved pseudo labels to {}".format(save_path)
                )
            
            features = features[clean_idx]   
            pseudo_labels = hard_labels[clean_idx]   
            
            logger.info("The number of clean labels: {} at Epoch {}".format(pseudo_labels.shape[0], cur_epoch)) 
            
            # for c in range(cfg.MODEL.NUM_CLASSES):   
            #     if c not in pseudo_labels.tolist():
            #         prototype = features_orig[np.where(labels.numpy()==c)].mean(0)
            #         logger.info("Warning: No pseudo labels of class {} at Epoch {}".format(c, cur_epoch))
            #     else:
            #         prototype = features[np.where(pseudo_labels.numpy()==c)].mean(0)    #compute prototypes with pseudo-label     
            #     prototypes.append(torch.Tensor(prototype))            
                
        # # Normalize the prototypes  
        # prototypes = torch.stack(prototypes).cuda()
        # prototypes = F.normalize(prototypes, p=2, dim=1)    
        
        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
            v_centers, #
            v_clusters, #
            t_centers, #
            clean_idx,
            hard_labels,
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
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
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
            (
                misc.is_eval_epoch(
                    cfg,
                    cur_epoch,
                    None if multigrid is None else multigrid.schedule,
                )
                and not cfg.MASK.ENABLE
            )
            or cur_epoch == cfg.SOLVER.MAX_EPOCH - 1
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch and cur_epoch >= 20:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(
                val_loader,
                model,
                val_meter,
                cur_epoch,
                cfg,
                train_loader,
                writer,
            )
    if start_epoch == cfg.SOLVER.MAX_EPOCH and not cfg.MASK.ENABLE: # final checkpoint load
        eval_epoch(val_loader, model, val_meter, start_epoch, cfg, train_loader, writer)
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

def compute_features(cfg, eval_loader, model):
    model.eval()
    N = eval_loader.dataset.num_videos
    
    features = np.zeros((N, 512), dtype='float32')  # feat.shape[1]=128 
    targets = torch.zeros(N, dtype=torch.long)                        
    probs = torch.zeros(N, cfg.MODEL.NUM_CLASSES) 
    
    for i, (inputs, labels, index, _, _) in enumerate(eval_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # labels = labels.cuda()
            # index = index.cuda()
        # batch_size = (
        #     inputs[0][0].size(0)
        #     if isinstance(inputs[0], list)
        #     else inputs[0].size(0)
        # )
        
        with torch.no_grad():
            prob, feat, _ = model(inputs)             
            feat = feat.data.cpu().numpy()
            # prob = F.softmax(output, dim=1)
            prob = prob.data.cpu()
            
            features[index.numpy()] = feat
            targets[index] = labels
            probs[index] = prob
        # if i == 0:
        #     features = np.zeros((N, feat.shape[1]), dtype='float32')
        #     targets = torch.zeros(N, dtype=torch.long)                        
        #     probs = torch.zeros(N, cfg.MODEL.NUM_CLASSSES) 
        # elif i < len(eval_loader) - 1:
        #     features[i * batch_size: (i + 1) * batch_size] = feat
        #     targets[i * batch_size: (i + 1) * batch_size] = labels
        #     probs[i * batch_size: (i + 1) * batch_size] = prob
        # else:
        #     # special treatment for final batch
        #     features[i * batch_size:] = feat
        #     targets[i * batch_size:] = labels
        #     probs[i * batch_size:] = prob
    return features, targets, probs   

def label_clean(cfg, features, labels, probs, train_loader, clusters):     
        N = features.shape[0]   
        temperature = 0.3   # Hyperparameter
        num_clusters = cfg.MODEL.NUM_CLASSES
        thresholds = torch.zeros((num_clusters, probs.shape[1]))
        
        # Compute threshold of each clustering
        for i in range(num_clusters):
            soft_labels_cluster = []
            for j in range(N):
                if clusters[j] == i:
                    soft_labels_cluster.append(train_loader.dataset.soft_labels[j])
            soft_labels_cluster = torch.stack(soft_labels_cluster, dim=0)
            std, mean = torch.std_mean(soft_labels_cluster, dim=0)
            thresholds[i] = std + mean
        
        # Update soft labels
        # score = torch.zeros((N, probs.shape[1]))
        score = train_loader.dataset.soft_labels
        for i in range(num_clusters):
            cluster_idx = (clusters == i).nonzero(as_tuple=True)[0]
            
            if cluster_idx.shape[0] == 1:
                continue
            
            feat = torch.from_numpy(features)
            feat_cluster = torch.index_select(feat, 0, cluster_idx)
            
            epsilon = torch.tensor(1e-10).type(torch.FloatTensor)
            nx = torch.sum(feat_cluster ** 2, dim=1, keepdim=True)
            qq = torch.mm(feat_cluster, feat_cluster.T)
            distance = torch.sqrt(torch.max(nx - 2 * qq + nx.T, epsilon))
            n = cluster_idx.shape[0]
            dis_ = distance.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
            weights = F.softmax(-dis_ / temperature, dim=1) 
            
            for j in range(cluster_idx.shape[0]):
                indices = torch.cat((cluster_idx[0:j], cluster_idx[j+1:]))
                neighbor_labels = train_loader.dataset.soft_labels[indices]
                idx = cluster_idx[j]
                score[idx] = torch.matmul(weights[j], neighbor_labels)
        train_loader.dataset.soft_labels = (score + probs)/2
        
        # Consider the ground-truth label as clean if the soft label outputs a score higher than the threshold
        gt_score = train_loader.dataset.soft_labels[labels>=0,labels]
        gt_clean = gt_score >= torch.diagonal(thresholds, 0)[labels]
        train_loader.dataset.soft_labels[gt_clean] = torch.zeros(gt_clean.sum(), cfg.MODEL.NUM_CLASSES).scatter_(1, labels[gt_clean].view(-1,1), 1)  
        
        # Get the hard pseudo label and the clean subset used to calculate supervised loss
        max_score, hard_labels = torch.max(train_loader.dataset.soft_labels, 1)  
        clean_idx = max_score > torch.max(torch.diagonal(thresholds, 0))
        
        return  clean_idx, hard_labels, gt_score
    
def compute_clusterings(cfg, u_vecs):
    # Hyperparameter
    iters = 10 # 10
    beta = -30
    epsilon = torch.tensor(1e-10).type(torch.FloatTensor)
    
    u_vecs = torch.from_numpy(u_vecs)
    num_cluster = cfg.MODEL.NUM_CLASSES
    # input_num, feature_dim = u_vecs.size()
    input_num = u_vecs.shape[0]
    ini_interval = int(input_num/num_cluster)  

    # Initialize clusterings
    o = torch.unsqueeze(u_vecs[0, :], dim=0)
    count = 1
    while(num_cluster-count > 0):
        current_o = torch.unsqueeze(u_vecs[ini_interval*count, :], dim=0)  #ini_interval*count
        o = torch.cat([o, current_o], dim=0)
        count += 1

    for i in range(iters):
        nx = torch.sum(o**2, dim=1, keepdim=True) # (num_cluster, 1)
        ny = torch.sum(u_vecs**2, dim=1, keepdim=True) # (N, 1)
        qq = nx - 2 * torch.mm(o, u_vecs.T) + ny.T # (num_cluaster, N)
        distances = torch.sqrt(torch.max(qq, epsilon))
        
        assignment = F.softmax(beta*distances, dim=0)   # assignments  [output_num_capsule, input_num_capsule]
        o = torch.mm(assignment, u_vecs)     # cluster centers  [num_cluster, dim_cluster]
        weights = torch.sum(assignment, dim=1, keepdim=True)
        o = o / weights
    
    # Update the final distances and determine the cluesterings
    clusters = torch.argmax(assignment, dim=0)  # (N)
    # centers_ = torch.sum(o ** 2, dim=1, keepdim=True)
    # u_vecs_ = torch.sum(u_vecs ** 2, dim=1, keepdim=True)
    # distance_ = torch.sqrt(torch.max(centers_ - 2 * torch.mm(o, u_vecs.T) + u_vecs_.T, epsilon))
    
    # distance = torch.min(distance_, dim=0)
    # distance = distance.values # (N)
    # clusters = torch.argmin(distance_, dim=0)  # (N)
    
    return  o, clusters  

def get_text_embeddings(cfg):
    name = cfg.DATA.PATH_TO_DATA_DIR
    file_name = Path(name).stem + ".csv"
    path = Path("/cvhci/temp/lfan/label_embed_xclip").joinpath(file_name)
    df = pd.read_csv(path)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu') 
    t_vecs = torch.from_numpy(df.values).float().to(device)

    return t_vecs
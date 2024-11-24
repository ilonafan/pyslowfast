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
from pathlib import Path
import pandas as pd
import csv

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
from slowfast.models.contrastive import contrastive_parameter_surgery
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
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
            
            preds, feat, error_recon = model(data)  
            
            shuffle_idx = torch.randperm(batch_size)
            mapping = {k:v for (v,k) in enumerate(shuffle_idx)}
            reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])     
            _, feat_aug, error_recon_aug = model([data_aug[0][shuffle_idx]])  
            feat_aug_rev = feat_aug[reverse_idx]
            
            # Reconstruction loss
            loss_recon = error_recon.mean() + error_recon_aug.mean()    
            
            # Instance contrastive loss
            sim_clean = torch.mm(feat, feat.t()) 
            mask = (torch.ones_like(sim_clean) - torch.eye(batch_size, device=sim_clean.device)).bool()
            sim_clean = sim_clean.masked_select(mask).view(batch_size, -1)
            
            sim_aug = torch.mm(feat, feat_aug_rev.t())
            sim_aug = sim_aug.masked_select(mask).view(batch_size, -1)   
            
            logits_pos = torch.bmm(feat.view(batch_size,1,-1),feat_aug_rev.view(batch_size,-1,1)).squeeze(-1)
            logits_neg = torch.cat([sim_clean,sim_aug],dim=1)
            
            logits = torch.cat([logits_pos,logits_neg],dim=1)
            instance_labels = torch.zeros(batch_size).long().cuda()
            loss_cc = loss_fun(logits/cfg.TAU1, instance_labels)  
            
            output_aug, feat_aug, _ = model(data_aug) 
            
            loss = loss_cc + loss_recon
            logger.info("Epoch: {} Iter: {} Loss CC, Recon = {:.2f}, {:.2f}".format(cur_epoch, cur_iter, loss_cc, loss_recon))
            
            # Classification loss
            if cur_epoch < cfg.SOLVER.WARMUP_EPOCHS:
                L = np.random.beta(cfg.ALPHA, cfg.ALPHA)   
                L = max(L, 1-L)  
                one_hot_labels = torch.zeros(batch_size, cfg.MODEL.NUM_CLASSES).cuda().scatter_(1, labels.view(-1,1), 1) 
                
                all_inputs = torch.cat((data[0], data_aug[0]),dim=0)
                idx = torch.randperm(batch_size*2) 
                all_labels = torch.cat((one_hot_labels,one_hot_labels),dim=0)      
                
                input_mix = L * all_inputs + (1 - L) * all_inputs[idx]  
                labels_mix = L * all_labels + (1 - L) * all_labels[idx]
                
                preds_mix, _, _ = model([input_mix])  
                loss_ce_mix = loss_fun(preds_mix, labels_mix)  
                loss = loss + loss_ce_mix
                logger.info("Epoch: {} Iter: {} Loss CE Mix = {:.2f}".format(cur_epoch, cur_iter, loss_ce_mix))
            else:
                clean_idx_batch = clean_idx[index] 
                noisy_idx_batch = ~clean_idx[index]
                target = train_loader.dataset.soft_labels[index].cuda(non_blocking=True) 
                labels = hard_labels[index].cuda(non_blocking=True)
                
                if True in clean_idx_batch.tolist():
                    loss_ce = loss_fun(preds[clean_idx_batch], target[clean_idx_batch]) + loss_fun(output_aug[clean_idx_batch], target[clean_idx_batch])
                    loss = loss + loss_ce
                    logger.info("Epoch: {} Iter: {} Loss CE = {:.2f}".format(cur_epoch, cur_iter, loss_ce))
                    
                if True in noisy_idx_batch.tolist():
                    w_u = linear_rampup(cfg, cur_epoch, cfg.LAM_U)
                    probs_u = torch.softmax(output_aug[noisy_idx_batch], dim=1)
                    loss_u = torch.mean((probs_u - target[noisy_idx_batch])**2)
                    loss = loss + w_u * loss_u
                    logger.info("Epoch: {} Iter: {} L2 Loss = {:.4f}".format(cur_epoch, cur_iter, loss_u))
        
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
        
        if cfg.MIXUP.ENABLE and cur_epoch < cfg.SOLVER.WARMUP_EPOCHS:
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
    val_loader, model, val_meter, cur_epoch, cfg, writer, optimizer, scaler
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
            elif cfg.TRAIN.DATASET == 'rrl':
                preds, _, _ = model(inputs)
            else:
                preds = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            elif cfg.MODEL.NUM_CLASSES < 5:
                if cfg.DATA.IN22k_VAL_IN1K != "":
                    preds = preds[:, :1000]
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, 1)
                
                top1_err = 1.0 - num_topks_correct / preds.size(0) * 100.0 
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    top1_err = du.all_reduce(top1_err)

                # Copy the errors from GPU to CPU (sync point).
                top1_err = top1_err.item()
                
                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    # top5_err,
                    batch_size
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )
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
    is_best = val_meter.log_epoch_stats(cur_epoch)
    
    if is_best and cur_epoch != cfg.SOLVER.MAX_EPOCH - 1:
        cu.save_best_checkpoint(
            cfg.OUTPUT_DIR,
            model,
            optimizer,
            cur_epoch,
            cfg,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
        )
    
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
    return is_best


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
    
    is_best = False
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
        
        # Begin the training pipeline of CRL
        features, labels, probs = compute_features(cfg, eval_loader, model)  # normalized low dimensional features, labels, logits
        
        clean_idx = labels.clone()
        hard_labels = labels.clone()
        num_clusters = cfg.MODEL.NUM_CLASSES
        
        # Cluster video low dimensional embeddings
        feat = torch.from_numpy(features)
        if cfg.CLUSTER == 'kmeans':
            kmeans = KMeans(n_clusters=num_clusters, n_init="auto").fit(features) # Initialized with k-means++
            y_pred = kmeans.predict(features)
            v_cluster_ids = torch.from_numpy(y_pred).to(torch.int64)
            v_centers = torch.from_numpy(kmeans.cluster_centers_)
        elif cfg.CLUSTER == 'agglomerative':
            y_pred = AgglomerativeClustering(n_clusters=num_clusters).fit_predict(features)
            v_cluster_ids = torch.from_numpy(y_pred)
            v_centers = torch.zeros(num_clusters, feat.shape[1])
            for c in range(cfg.MODEL.NUM_CLASSES):   
                v_centers[c] = feat[v_cluster_ids == c].mean(0)    #compute v_centers as mean embeddings      
        elif cfg.CLUSTER == 'spectral':
            y_pred = SpectralClustering(n_clusters=num_clusters, assign_labels='discretize', random_state=0).fit_predict(features)
            v_cluster_ids = torch.from_numpy(y_pred).to(torch.int64)
            v_centers = torch.zeros(num_clusters, feat.shape[1])
            for c in range(cfg.MODEL.NUM_CLASSES):   
                v_centers[c] = feat[v_cluster_ids == c].mean(0)    #compute v_centers as mean embeddings   
        # Only for evaluation
        cluster_assignment = evaluate_clustering_results(cfg, y_pred, cur_epoch, writer)
        
        if cur_epoch >= cfg.SOLVER.WARMUP_EPOCHS:
            if cur_epoch == cfg.SOLVER.WARMUP_EPOCHS:
                # Initalize the soft label as model's softmax prediction
                train_loader.dataset.soft_labels = probs.clone()    
            # Generate new soft label 
            clean_idx, hard_labels = label_clean(cfg, feat, labels, probs, train_loader, v_cluster_ids, cluster_assignment, cur_epoch, is_best)
            logger.info("The number of clean labels: {} at Epoch {}".format(clean_idx.sum(), cur_epoch)) 
        
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
        if is_checkp_epoch and cur_epoch >= 25:
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
            is_best = eval_epoch(
                val_loader,
                model,
                val_meter,
                cur_epoch,
                cfg,
                writer,
                optimizer, 
                scaler
            )
    if start_epoch == cfg.SOLVER.MAX_EPOCH and not cfg.MASK.ENABLE: # final checkpoint load
        is_best = eval_epoch(val_loader, model, val_meter, start_epoch, cfg, writer, optimizer, scaler)
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
    
    features = np.zeros((N, 48), dtype='float32')
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
                
        input = [inputs[0][1::2, :, :, :, :]]
        input_aug = [inputs[0][0::2, :, :, :, :]]
        
        with torch.no_grad():
            prob, feat, _ = model(input)             
            feat = feat.data.cpu().numpy()
            prob = prob.data.cpu()
            
            prob_aug, feat_aug, _ = model(input_aug)
            feat_aug = feat_aug.data.cpu().numpy()
            prob_aug = prob_aug.data.cpu()
            
            features[index.numpy()] = (feat + feat_aug) / 2
            targets[index] = labels
            probs[index] = (prob + prob_aug) / 2
    return features, targets, probs   

def label_clean(cfg, features, labels, probs, train_loader, v_cluster_ids, cluster_assignment, cur_epoch, is_best):
    N = features.shape[0]
    num_clusters = cfg.MODEL.NUM_CLASSES
    all_score = torch.zeros(N, cfg.MODEL.NUM_CLASSES) 
    
    # Co-refinement within clustering 
    for i in range(num_clusters):
        features_cluster = features[v_cluster_ids == i]
        
        # Calculate weights based on pairwise euclidean distance within cluster
        eu_dis = pairwise_distance(features_cluster, features_cluster)
        n = features_cluster.shape[0]
        dis = eu_dis.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
        weights = F.softmax(-dis/0.3, dim=1)  # cluster_len x (cluster_len - 1)
        
        soft_labels_cluster = train_loader.dataset.soft_labels[v_cluster_ids == i] # cluster_len x prob.shape[1]
        weights = weights.to(soft_labels_cluster.device)
        score_cluster = torch.zeros((weights.shape[0]), cfg.MODEL.NUM_CLASSES)
        for j in range(weights.shape[0]):
            neighbors = torch.cat((soft_labels_cluster[:j], soft_labels_cluster[j+1:])) # (cluster_len - 1) x prob.shape[1]
            score_cluster[j] = torch.matmul(weights[j], neighbors) # 1 x prob.shape[1]
        all_score[v_cluster_ids == i] = score_cluster 
    refined_scores = cfg.BETA * all_score + (1 - cfg.BETA) * probs
    
    # Calculate JSD
    JS_dist = Jensen_Shannon()
    jsd = JS_dist(refined_scores, F.one_hot(labels, num_classes = cfg.MODEL.NUM_CLASSES)) 
    
    # Fit Gaussian mixture model and divide dateset by first criterion
    prob_jsd, gmm_jsd = gmm_fit_func(jsd.reshape(-1, 1))
    fst_gt_clean = prob_jsd[:, gmm_jsd.means_.argmin()] > prob_jsd[:, gmm_jsd.means_.argmax()]
    
    # Compute weighted refined scores and divide dateset by second criterion
    class_weights = compute_class_weights(cfg, labels, train_loader, cur_epoch)   
    cluster_th = compute_cluster_threshold(cfg, v_cluster_ids, train_loader, class_weights)  
    weighted_scores = class_weights[labels] * refined_scores[labels>=0,labels]
    sec_gt_clean =  weighted_scores >= cluster_th[v_cluster_ids][labels>=0,labels]
        
    # Combine two criteria of sample selection
    gt_clean = torch.logical_or(torch.from_numpy(fst_gt_clean), sec_gt_clean)
    
    # Generate soft pseudo labels for noisy data
    gt_score = refined_scores[labels>=0, labels]
    max_score, max_score_idx = torch.max(refined_scores, 1)  
    train_loader.dataset.soft_labels = torch.zeros(probs.shape[0], cfg.MODEL.NUM_CLASSES)
    train_loader.dataset.soft_labels[labels>=0, labels] = gt_score
    train_loader.dataset.soft_labels[max_score_idx>=0, max_score_idx] = max_score
    
    # Generate hard labels
    train_loader.dataset.soft_labels[gt_clean] = torch.zeros(gt_clean.sum(), cfg.MODEL.NUM_CLASSES).scatter_(1, labels[gt_clean].view(-1,1), 1)
    max_prob, hard_labels = torch.max(train_loader.dataset.soft_labels, 1) 
    clean_idx = max_prob >= 1
    
    return clean_idx, hard_labels

def compute_class_weights(cfg, labels, train_loader, cur_epoch):
    noise_prop = cfg.NOISE_PROP 
    N = train_loader.dataset.num_videos
    eps = 0.05
    class_weights = torch.zeros(cfg.MODEL.NUM_CLASSES)
    label_freq = 0
    
    if cur_epoch  ==  cfg.SOLVER.WARMUP_EPOCHS:
        num_noise = noise_prop * N
        for i in range(cfg.MODEL.NUM_CLASSES):
            label_freq = len([j for j in range(labels.shape[0]) if labels[j] == i])
            class_weights[i] = -(label_freq - num_noise / (cfg.MODEL.NUM_CLASSES - 1)) / label_freq
        class_weights = (class_weights - class_weights.min()) / (class_weights.max() - class_weights.min()) + eps
    elif cur_epoch  >  cfg.SOLVER.WARMUP_EPOCHS:
        max_prob, hard_labels = torch.max(train_loader.dataset.soft_labels, 1) 
        pred_clean_idx = max_prob >= 1
        for i in range(cfg.MODEL.NUM_CLASSES):
            label_freq = len([j for j in range(labels.shape[0]) if labels[j] == i])
            class_weights[i] = -len([j for j in range(labels.shape[0]) if pred_clean_idx[j] and hard_labels[j] == i]) / label_freq
        class_weights = (class_weights - class_weights.min()) / (class_weights.max() - class_weights.min()) + eps
    return class_weights

def compute_cluster_threshold(cfg, v_cluster_ids, train_loader, class_weights):
    num_clusters = cfg.MODEL.NUM_CLASSES
    cw_sum = sum(class_weights) / num_clusters - 0.05
    cluster_th = torch.zeros((num_clusters, train_loader.dataset.soft_labels.shape[1]))
    for i in range(num_clusters):
        cluster_mean = torch.mean(train_loader.dataset.soft_labels[v_cluster_ids == i])
        cluster_th[i] = cw_sum * cluster_mean
    return cluster_th

def pairwise_distance(data1, data2):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu') 
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)
    # N*1*M
    A = data1.unsqueeze(dim=1)
    # 1*N*M
    B = data2.unsqueeze(dim=0)
    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis

def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

class Jensen_Shannon(torch.nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)
    
def gmm_fit_func(input_loss):
    input_loss = np.array(input_loss)
    gmm = GaussianMixture(n_components=2,max_iter=30,tol=1e-2,reg_covar=5e-4) 
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    return prob, gmm

class Clustering_Metrics:
    ari = adjusted_rand_score
    nmi = normalized_mutual_info_score

    @staticmethod
    def acc(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row, col = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(row, col)]) * 1.0 / y_pred.size, col
    
def evaluate_clustering_results(cfg, y_pred, cur_epoch, writer):
    data_dir = cfg.DATA.PATH_TO_DATA_DIR
    if Path(data_dir).parts[-2] == "objectlevel":
        dist_dir = Path(data_dir).parts[-3] + "/" + Path(data_dir).parts[-2] + "/" + Path(data_dir).parts[-1] + "/train.csv"
    else:
        dist_dir = Path(data_dir).parts[-2] + "/" + Path(data_dir).parts[-1] + "/train.csv"
    dir_name = Path("/cvhci/temp/lfan/label/clean_label").joinpath(Path(dist_dir))
    
    clean_labels = []
    with open(dir_name, newline='') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            clean_labels.append(int(row[1]))
    clean_labels = torch.Tensor(clean_labels)
    
    y = np.array(clean_labels)
    acc, col = Clustering_Metrics.acc(y, y_pred)
    cluster_assignment = torch.from_numpy(col)
    nmi, ari = Clustering_Metrics.nmi(y, y_pred), Clustering_Metrics.ari(y, y_pred)
    logger.info("Epoch: {} {} clustering ACC, NMI, ARI = {:.4f}, {:.4f}, {:.4f}".format(cur_epoch + 1, cfg.CLUSTER, acc, nmi, ari))
    
    # write to tensorboard format if available.
    if writer is not None:
        writer.add_scalars(
            {
                "Clustering/ACC": acc,
                "Clustering/NMI": nmi,
                "Clustering/ARI": ari,
            },
            global_step=cur_epoch,
        )
    return cluster_assignment

def linear_rampup(cfg, cur_epoch, lam_u, rampup_length=5):
    cur = np.clip((cur_epoch - cfg.SOLVER.WARMUP_EPOCHS) / rampup_length, 0.0, 1.0)
    return lam_u * float(cur)
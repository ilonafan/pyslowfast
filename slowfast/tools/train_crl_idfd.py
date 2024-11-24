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
from slowfast.models.contrastive import (
    contrastive_forward,
    contrastive_parameter_surgery,
)
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from slowfast.models.head_helper import Normalize
from slowfast.models.losses import NonParametricClassifier, IDFDLoss
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import skfda

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    npc,
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
    norm = Normalize(2)
    loss_idfd = IDFDLoss(tau2=cfg.TAU3)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = norm.to(device)
    loss_idfd = loss_idfd.to(device)
    
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
            
            # IDFD loss
            outputs = npc(feat, index)
            loss_id, loss_fd = loss_idfd(outputs, feat, index)
            loss_clustering = loss_id + loss_fd
            
            shuffle_idx = torch.randperm(batch_size)
            mapping = {k:v for (v,k) in enumerate(shuffle_idx)}
            reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])     
            _, feat_aug, error_recon_aug = model([data_aug[0][shuffle_idx]])  
            feat_aug = feat_aug[reverse_idx]
            
            # Reconstruction loss
            loss_recon = error_recon.mean()  # loss_recon
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
            loss_cc = loss_fun(logits/cfg.TAU1, instance_labels)    # loss_instance  (loss_cc)          
            
            loss = loss_cc + loss_recon + loss_recon_aug + loss_clustering
            
            logger.info("Epoch: {} Loss CC, Recon, Recon_aug, ID, FD = {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(cur_epoch, loss_cc, loss_recon, loss_recon_aug, loss_id, loss_fd))            
                
                
        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {
                    "Clustering/ID loss": loss_id,
                    "Clustering/FD loss": loss_fd,
                },
                global_step=data_size * cur_epoch + cur_iter,
            )
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
    
    # Save the checkpoint if it has better val results 
    if is_best:
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npc = NonParametricClassifier(input_dim=48, output_dim=train_loader.dataset.num_videos, tau=cfg.TAU2, momentum=0.5)
    npc = npc.to(device)
    
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
            npc,
            writer,       
        )
        
        # Only for evaluation after each epoch
        num_clusters = cfg.MODEL.NUM_CLASSES
        if cfg.CLUSTER == 'kmeans':
            trainFeatures = npc.memory
            z = trainFeatures.cpu().numpy()
            kmeans = KMeans(n_clusters=num_clusters, n_init="auto").fit(z)
            y_pred = kmeans.predict(z)
        elif cfg.CLUSTER == 'agglomerative':
            trainFeatures = npc.memory
            z = trainFeatures.cpu().numpy()
            y_pred = AgglomerativeClustering(n_clusters=num_clusters).fit_predict(z)
        elif cfg.CLUSTER == 'spectral':
            trainFeatures = npc.memory
            z = trainFeatures.cpu().numpy()
            y_pred = SpectralClustering(n_clusters=num_clusters, assign_labels='discretize', random_state=0).fit_predict(z)
        elif cfg.CLUSTER == 'fuzzyc':
            trainFeatures = npc.memory
            z = trainFeatures.cpu().numpy()
            fd = skfda.FDataGrid(z)
            y_pred = skfda.ml.clustering.FuzzyCMeans(n_clusters=num_clusters, random_state=0).fit_predict(fd)
        evaluate_clustering_results(cfg, y_pred, cur_epoch, writer)
        
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

        # # Save a checkpoint.
        if is_checkp_epoch and cur_epoch >= 28:
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
                writer,
                optimizer,
                scaler
            )
    if start_epoch == cfg.SOLVER.MAX_EPOCH and not cfg.MASK.ENABLE: # final checkpoint load
        eval_epoch(val_loader, model, val_meter, start_epoch, cfg, writer, optimizer, scaler)
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
        dir_name = Path("/cvhci/temp/lfan/label/clean_label").joinpath(Path(dist_dir))
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


# def label_clean_jsd(cfg, features, labels, probs, train_loader, v_cluster_ids, v_centers, cur_epoch):
#     temperature = 0.3
#     noise_prop = 0.5
#     N = features.shape[0]
#     num_clusters = cfg.MODEL.NUM_CLASSES
#     score = torch.zeros(N, cfg.MODEL.NUM_CLASSES) 
    
#     # Co-refinement within clustering 
#     for i in range(num_clusters):
#         features_cluster = features[v_cluster_ids == i]
        
#         # Calculate pairwise euclidean distance within cluster
#         eu_dis = pairwise_distance(features_cluster, features_cluster)
#         n = features_cluster.shape[0]
#         dis = eu_dis.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
        
#         # Mask neighbours with large distances
#         dis_mean = dis.mean(1)
#         mask = [dis[i, :] > dis_mean[i] for i in range(dis.shape[0])]
#         mask = torch.vstack(mask)
#         weights = F.softmax(-dis/temperature, dim=1)  # cluster_len x (cluster_len - 1)
#         weights[mask] = 0 
        
#         soft_labels_cluster = train_loader.dataset.soft_labels[v_cluster_ids == i] # cluster_len x prob.shape[1]
#         weights = weights.to(soft_labels_cluster.device)
#         score_cluster = torch.zeros((weights.shape[0]), cfg.MODEL.NUM_CLASSES)
#         for j in range(weights.shape[0]):
#             neighbors = torch.cat((soft_labels_cluster[:j], soft_labels_cluster[j+1:])) # (cluster_len - 1) x prob.shape[1]
#             score_cluster[j] = torch.matmul(weights[j], neighbors) # 1 x prob.shape[1]
#         score[v_cluster_ids == i] = score_cluster 
#     soft_labels_cur = cfg.BETA * score + (1 - cfg.BETA) * probs

#     # Calculate JSD
#     JS_dist = Jensen_Shannon()
#     jsd = JS_dist(soft_labels_cur, F.one_hot(labels, num_classes = cfg.MODEL.NUM_CLASSES)) 
    
#     # Fit Gaussian mixture model and divide dateset into clean subset and noisy subset
#     prob_jsd, gmm_jsd = gmm_fit_func(jsd.reshape(-1, 1))
#     # prob_jsd, gmm_jsd = gmm_fit_func(weight_jsd.reshape(-1, 1))
#     fst_gt_clean = prob_jsd[:, gmm_jsd.means_.argmin()] > prob_jsd[:, gmm_jsd.means_.argmax()]
    
#     # Compute class weights
    
#     if cur_epoch  ==  cfg.SOLVER.WARMUP_EPOCHS:
#         num_noise = noise_prop * N
#     elif cur_epoch  >  cfg.SOLVER.WARMUP_EPOCHS:
#         max_prob, hard_labels = torch.max(train_loader.dataset.soft_labels, 1) 
#         num_noise = torch.sum(max_prob < 1).item()  # Use the sample selection results of last epoch as prior knowledge of this epoch
    
#     if cfg.SAMPLE_SELECTION == 'jsd':
#         class_weights = compute_class_weights_jsd(cfg, labels, num_noise)
#         weight_jsd = class_weights[labels] * jsd
#         # Consider samples as clean if jsd is less equal than cluster threshold
#         cluster_th = compute_cluster_threshold_jsd(cfg, jsd, v_cluster_ids, class_weights)
#         sec_gt_clean = weight_jsd <= cluster_th[v_cluster_ids]
#     elif cfg.SAMPLE_SELECTION == 'soft_label':
#         class_weights = compute_class_weights(cfg, labels, num_noise)
#         weight_soft_labels = class_weights[labels] * soft_labels_cur[labels>=0,labels]
#         # Consider samples as clean if jsd is less equal than cluster threshold
#         cluster_th = compute_cluster_threshold(cfg, v_cluster_ids, train_loader, class_weights)
#         sec_gt_clean = weight_soft_labels >= cluster_th[v_cluster_ids][labels>=0,labels]
#         # sec_gt_clean = soft_labels_cur[labels>=0,labels] >= cluster_th[v_cluster_ids][labels>=0,labels]
        
#     # Combine two criterians of sample selections
#     gt_clean = torch.logical_or(torch.from_numpy(fst_gt_clean), sec_gt_clean)
    
#     # Generate soft pseudo labels for noisy data
#     gt_score = soft_labels_cur[labels>=0, labels]
#     max_score, max_score_idx = torch.max(soft_labels_cur, 1)  
    
#     # ct_dist = pairwise_distance(v_centers[v_cluster_ids], train_loader.dataset.t_vecs)
#     # _, min_dis_idx = torch.min(ct_dist, 1)
#     # min_dis_score = soft_labels_cur[min_dis_idx>=0, min_dis_idx]
    
#     # smoothed_prob = (1 - (gt_score + max_score + min_dis_score)) / (cfg.MODEL.NUM_CLASSES - 3)
#     # train_loader.dataset.soft_labels = smoothed_prob[:, None] * torch.ones(probs.shape[0], cfg.MODEL.NUM_CLASSES)
#     train_loader.dataset.soft_labels = torch.zeros(probs.shape[0], cfg.MODEL.NUM_CLASSES)
#     train_loader.dataset.soft_labels[labels>=0, labels] = gt_score
#     train_loader.dataset.soft_labels[max_score_idx>=0, max_score_idx] = max_score
#     # train_loader.dataset.soft_labels[min_dis_idx>=0, min_dis_idx] = min_dis_score
    
#     # Generate hard labels for clean data
#     train_loader.dataset.soft_labels[gt_clean] = torch.zeros(gt_clean.sum(), cfg.MODEL.NUM_CLASSES).scatter_(1, labels[gt_clean].view(-1,1), 1)
#     max_prob, hard_labels = torch.max(train_loader.dataset.soft_labels, 1) 
#     clean_idx = max_prob >= 1
    
#     # Write val predictions into file      
#     if cur_epoch == cfg.SOLVER.MAX_EPOCH - 1:
#         save_path = os.path.join(cfg.OUTPUT_DIR, "pseudo_labels.dat")
#         if du.is_root_proc():
#             with pathmgr.open(save_path, "wb") as f:
#                 pickle.dump([clean_idx.tolist(), 
#                             soft_labels_cur.tolist(), 
#                             hard_labels.tolist(), 
#                             gt_score.tolist(), 
#                             train_loader.dataset.soft_labels.tolist(), 
#                             # train_loader.dataset.t_vecs.tolist(), 
#                             v_cluster_ids.tolist(), 
#                             fst_gt_clean.tolist(), 
#                             sec_gt_clean.tolist(), 
#                             class_weights.tolist()], f)
#         logger.info(
#             "Successfully saved pseudo labels to {}".format(save_path)
#         )
    
#     return clean_idx, hard_labels
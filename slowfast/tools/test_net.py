#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
import csv
import math
from tqdm import tqdm

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, time, meta) in enumerate(
        test_loader
    ):

        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        elif cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            if not cfg.CONTRASTIVE.KNN_ON:
                test_meter.finalize_metrics()
                return test_meter
            # preds = model(inputs, video_idx, time)
            train_labels = (
                model.module.train_labels
                if hasattr(model, "module")
                else model.train_labels
            )
            yd, yi = model(inputs, video_idx, time)
            batchSize = yi.shape[0]
            K = yi.shape[1]
            C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
            candidates = train_labels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot = torch.zeros((batchSize * K, C)).cuda()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
            probs = torch.mul(
                retrieval_one_hot.view(batchSize, -1, C),
                yd_transform.view(batchSize, -1, 1),
            )
            preds = torch.sum(probs, 1)
        elif cfg.TRAIN.DATASET == 'rrl':
            preds, feat_lowD, feat = model(inputs)
        elif cfg.TASK == "TSNE":
            preds, feat = model(inputs)
        else:
            # Perform the forward pass.
            preds = model(inputs)
        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather([preds, labels, video_idx])
        if cfg.NUM_GPUS:
            preds = preds.cpu() 
            labels = labels.cpu()
            video_idx = video_idx.cpu()
        if cfg.NUM_GPUS and (cfg.TRAIN.DATASET == 'rrl' or cfg.TASK == "TSNE"):
            feat = feat.cpu()
        if cfg.NUM_GPUS and cfg.TRAIN.DATASET == 'rrl':
            feat_lowD = feat_lowD.cpu()
        test_meter.iter_toc()
        
        
        # Visualize the resultant embeddings with t-sne 
        if cfg.TASK == 'TSNE' and cfg.TRAIN.DATASET == 'rrl':
            # Update and log embeddings.
            test_meter.update_embeddings_rrl(
                preds.detach(), feat.detach(), feat_lowD.detach(), labels.detach(), video_idx.detach()
            )
        elif cfg.TASK == 'TSNE':
            test_meter.update_embeddings(
                preds.detach(), feat.detach(), labels.detach(), video_idx.detach()
            )
        elif not cfg.VIS_MASK.ENABLE:
            # Update and log stats.
            test_meter.update_stats(
                preds.detach(), labels.detach(), video_idx.detach()
            )
        test_meter.log_iter_stats(cur_iter)
        test_meter.iter_tic()
    
    if cfg.TASK == 'TSNE' and cfg.TRAIN.DATASET == 'rrl':
        all_embeddings = test_meter.embeddings.clone().detach()
        all_embeddings_lowD = test_meter.embeddings_lowD.clone().detach()
        all_labels = test_meter.video_labels
        
        if cfg.NUM_GPUS:
            all_embeddings = all_embeddings.cpu()
            all_embeddings_lowD = all_embeddings_lowD.cpu()
            all_labels = all_labels.cpu()

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, "embeddings.csv")
            save_path_low = os.path.join(cfg.OUTPUT_DIR, "embeddings_lowD.csv")

        if du.is_root_proc():
            res = torch.hstack((all_embeddings, all_labels.unsqueeze(1)))
            res_lowD = torch.hstack((all_embeddings_lowD, all_labels.unsqueeze(1)))
            emb_list = res.tolist()
            emb_lowD_list = res_lowD.tolist()
            
            with open(save_path, 'w', newline='') as file:
                csv_writer = csv.writer(file)
                for emb in emb_list:
                    csv_writer.writerow(emb)
            with open(save_path_low, 'w', newline='') as file:
                csv_writer = csv.writer(file)
                for emb_low in emb_lowD_list:
                    csv_writer.writerow(emb_low)
        logger.info(
            "Successfully saved resultant embeddings to {} and {}".format(save_path, save_path_low)
        )   
    elif cfg.TASK == 'TSNE':
        all_embeddings = test_meter.embeddings.clone().detach()
        all_labels = test_meter.video_labels
        
        if cfg.NUM_GPUS:
            all_embeddings = all_embeddings.cpu()
            all_labels = all_labels.cpu()

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, "embeddings.csv")

        if du.is_root_proc():
            res = torch.hstack((all_embeddings, all_labels.unsqueeze(1)))
            emb_list = res.tolist()

            with open(save_path, 'w', newline='') as file:
                csv_writer = csv.writer(file)
                for emb in emb_list:
                    csv_writer.writerow(emb)
        logger.info(
            "Successfully saved resultant embeddings to {}".format(save_path)
        )   

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE and cfg.TASK != "TSNE":    
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)      
        save_path = os.path.join(cfg.OUTPUT_DIR, "test_pred.dat")

        if du.is_root_proc():
            with pathmgr.open(save_path, "wb") as f:
                pickle.dump([all_preds, all_labels], f)

        logger.info(
            "Successfully saved prediction results to {}".format(save_path)
        )
            
        test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
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

    if len(cfg.TEST.NUM_TEMPORAL_CLIPS) == 0:
        cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]

    test_meters = []
    for num_view in cfg.TEST.NUM_TEMPORAL_CLIPS:

        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view

        # Print config.
        # logger.info("Test with config:")
        # logger.info(cfg)

        # Build the video model and print model statistics.
        model = build_model(cfg)
        flops, params = 0.0, 0.0
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            model.eval()
            flops, params = misc.log_model_info(
                model, cfg, use_train_input=False
            )

        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, use_train_input=False)
        if (
            cfg.TASK == "ssl"
            and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            and cfg.CONTRASTIVE.KNN_ON
        ):
            train_loader = loader.construct_loader(cfg, "train")
            if hasattr(model, "module"):
                model.module.init_knn_labels(train_loader)
            else:
                model.init_knn_labels(train_loader)

        cu.load_test_checkpoint(cfg, model)

        # Create video testing loaders.
        test_loader = loader.construct_loader(cfg, "test")  
        logger.info("Testing model for {} iterations".format(len(test_loader)))

        if cfg.DETECTION.ENABLE:
            assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
            test_meter = AVAMeter(len(test_loader), cfg, mode="test")
        else:
            assert (
                test_loader.dataset.num_videos
                % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
                == 0
            )
            # Create meters for multi-view testing.
            test_meter = TestMeter(
                test_loader.dataset.num_videos
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES
                if not cfg.TASK == "ssl"
                else cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
                len(test_loader),
                cfg.MODEL.MODEL_NAME,
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
            )

        # Set up writer for logging to Tensorboard format.
        if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
            cfg.NUM_GPUS * cfg.NUM_SHARDS
        ):
            writer = tb.TensorboardWriter(cfg)
        else:
            writer = None

        # # Perform multi-view test on the entire dataset.
        test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
        test_meters.append(test_meter)
        if writer is not None:
            writer.close()
        
        if cfg.TASK == "fps":
            evaluate_computational_performance(model, test_loader, cfg)
            break
    return


def init_measurement():
    MEASURE_REPETITION = 300
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    infer_durations = np.zeros((MEASURE_REPETITION,1))
    return starter,ender,infer_durations


def get_measurement_stats(infer_durations):
    duration_mean = np.mean(infer_durations)
    duration_std = np.std(infer_durations)
    return duration_mean, duration_std


def get_num_params(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def evaluate_computational_performance(model, test_loader, cfg):
    WARMUP_REPETITION = 100
    MEASURE_REPETITION = 300
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_properties = torch.cuda.get_device_properties(device)
    device_memory = math.floor(getattr(device_properties, "total_memory") / 1e9) # unit: GB

    starter, ender, infer_durations = init_measurement()

    model = model.to(device)
    num_params = get_num_params(model)

    with torch.no_grad():
        # GPU warm-up
        for _ in tqdm(range(WARMUP_REPETITION), desc="GPU warm-up", total=WARMUP_REPETITION):
            for _, (inputs, labels, video_idx, time, meta) in enumerate(
                test_loader
            ):
                if cfg.NUM_GPUS:
                    # Transfer the data to the current GPU device.
                    if isinstance(inputs, (list,)):
                        for i in range(len(inputs)):
                            inputs[i] = inputs[i].cuda(non_blocking=True)
                    else:
                        inputs = inputs.cuda(non_blocking=True)
                    # Transfer the data to the current GPU device.
                    labels = labels.cuda()
                    video_idx = video_idx.cuda()
                    for key, val in meta.items():
                        if isinstance(val, (list,)):
                            for i in range(len(val)):
                                val[i] = val[i].cuda(non_blocking=True)
                        else:
                            meta[key] = val.cuda(non_blocking=True)
                preds, _, _ = model(inputs)
                break
            
        for rep in tqdm(range(MEASURE_REPETITION), desc="Measuring inference time", total=MEASURE_REPETITION):
            starter.record()
            for _, (inputs, labels, video_idx, time, meta) in enumerate(
                test_loader
            ):
                if cfg.NUM_GPUS:
                    # Transfer the data to the current GPU device.
                    if isinstance(inputs, (list,)):
                        for i in range(len(inputs)):
                            inputs[i] = inputs[i].cuda(non_blocking=True)
                    else:
                        inputs = inputs.cuda(non_blocking=True)
                    # Transfer the data to the current GPU device.
                    labels = labels.cuda()
                    video_idx = video_idx.cuda()
                    for key, val in meta.items():
                        if isinstance(val, (list,)):
                            for i in range(len(val)):
                                val[i] = val[i].cuda(non_blocking=True)
                        else:
                            meta[key] = val.cuda(non_blocking=True)
                preds, _, _ = model(inputs)
                break
            ender.record()
            
            # Wait for GPU sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # time unit is milliseconds
            curr_time = curr_time / 1000 # ms -> s
            infer_durations[rep] = curr_time
    duration_mean, duration_std = get_measurement_stats(infer_durations)
    summary = {
        "Device": f"{torch.cuda.get_device_name(device)} ({device_memory} GB)",
        "#Parameters": f"{num_params/1e6:.2f} M",
        "Inference time": f"Mean: {duration_mean:.3f}s, Std: {duration_std:.3f}s",
        "FPS": f"{cfg.TEST.BATCH_SIZE * cfg.TEST.NUM_SPATIAL_CROPS * cfg.DATA.NUM_FRAMES/ duration_mean:.3f}",
    }
    print(summary)
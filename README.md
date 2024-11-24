<!-- ## Updates

- [2021.09.17] Code for flying guide dog prototype and the Pedestrian and Vehicle Traffic Lights (PVTL) dataset are released. -->

# Exploring Video-Based Driver Activity Recognition under Noisy Labels

PyTorch implementation of paper "**Exploring Video-Based Driver Activity Recognition under Noisy Labels**".


## Overview

- `detectron2_repo/`: Detectron2 repository
- `slowfast/`
    - `config/`: Config yaml files for different experiments
    - `scripts/`: Script ipynb files for dataset preparation and result visualization
    - `slowfast/`: Implementations
        - `config/default.py`: Default config
        - `datasets/`: Implementations of datasets
        - `models/`: Implementations of models
        - `utils/`: Helper functions
  - `tools/`: Scripts for training/validation/testing

## Dependencies

- [Python](https://python.org/), version = 3.10
- [Ubuntu 16.04 Desktop](https://ubuntu.com/download)
- [PyTorch](https://PyTorch.org/), version = 1.12.1
- [CUDA](https://developer.nvidia.com/cuda-downloads), version = 11.6
- [Anaconda3](https://www.anaconda.com/)

## Installation

Please find installation instructions for PyTorch and PySlowFast in [INSTALL.md](INSTALL.md) and follow the instructions in [DATASET.md](slowfast/datasets/DATASET.md) to prepare the datasets in Kinetics format.

## Experiments

We verify the effectiveness of the PNP method and Robust Early-Learning method on simulated noisy datasets. 

In this repository, we provide the subset we used for this project. You should download the NTU60 dataset and create the subset according to the csv files. The dataset should be put into the same folder of labels as the instructions in [DATASET.md](slowfast/datasets/DATASET.md).

To generate noise labels, you can run the [generate_noisy_label.ipynb](slowfast/script/generate_noisy_label.ipynb) in the script folder with any noise proportion.


Here is a training example: 
```bash
python tools/run.py \
  --cfg configs/Kinetics/MViTv2_S_16x4.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
```

To perform test, you can set the TRAIN.ENABLE to False, and do not forget to pass the path to the model you want to test to TEST.CHECKPOINT_FILE_PATH.


## Acknowledgements

Great thanks for these open-source repositories: 

- PySlowFast: [PySlowFast](https://github.com/facebookresearch/SlowFast)


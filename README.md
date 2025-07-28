# HM2-Net

# HM2-Net: Hybrid Mamba Network for Medical Image Segmentation

This repository provides the official implementation of **HM2-Net**, a hybrid U-Net architecture that integrates multi-scale state space modeling with uncertainty-guided attention for robust and efficient medical image segmentation.

## Installation

Install the required Python packages using:
`pip install -r HM2-Net/requirements.txt`


## Inference

Run segmentation on your dataset using:
`python HM2-Net/test_ours.py
--volume_path /path/to/your/volume
--model_dir /path/to/saved/model`
- `--volume_path`: Path to the input volume directory
- `--model_dir`: Path to the pre-trained model directory (e.g., `.pth` file or checkpoint folder).


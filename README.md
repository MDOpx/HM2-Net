# HM2-Net

This repository provides the official implementation of **HM2-Net**, a hybrid U-Net architecture that integrates multi-scale state space modeling with uncertainty-guided attention for robust and efficient medical image segmentation.

## Installation

- Python version: **3.10**
- Install the required Python packages using:
`pip install -r hm2net/requirements.txt`

## 📦 Pretrained Models (Optional for ViT-based backbones)
You can download ViT-based pretrained models from the following link:
`https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz`
Example models:
- R50-ViT-B_16
- ViT-B_16
- ViT-L_16

## Run the code
Run segmentation on your dataset using:
`python hm2net/test.py --dataset {dataset_name} --vit_name R50-ViT-B_16`


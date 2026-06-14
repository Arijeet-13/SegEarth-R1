#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR

# Clean previous builds to avoid stale artifact issues
rm -rf build/ dist/ *.egg-info MultiScaleDeformableAttention.egg-info

# Set CUDA_HOME if not already set (adjust path to match your CUDA 11.6 install)
# export CUDA_HOME=/usr/local/cuda-11.6

# Set target GPU architecture for your specific GPU (uncomment and adjust):
# export TORCH_CUDA_ARCH_LIST="8.6"   # RTX 3090, RTX 3080, RTX A5000
# export TORCH_CUDA_ARCH_LIST="8.0"   # A100
# export TORCH_CUDA_ARCH_LIST="7.5"   # RTX 2080 Ti, T4
# export TORCH_CUDA_ARCH_LIST="7.0"   # V100
# export TORCH_CUDA_ARCH_LIST="6.1"   # GTX 1080 Ti, P40

python setup.py build install

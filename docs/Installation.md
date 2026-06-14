# Installation⚙️

## Requirement:

* Linux with Python ≥ 3.10.
* PyTorch ≥ 2.0 and torchvision that matches the PyTorch installation.

## Set up conda environment:
```bash
conda create -n segearthr1 python=3.10
conda activate segearthr1

git clone https://github.com/earth-insights/SegEarth-R1.git
cd SegEarth-R1

# Install PyTorch 2.0.1 with CUDA 11.7 (fully compatible with CUDA 11.6 drivers)
# PyTorch bundles its own CUDA runtime — you only need an NVIDIA driver >= 515.43
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt
```

## Important: Set CUDA_HOME
Before compiling CUDA kernels, ensure `CUDA_HOME` points to your CUDA 11.6 toolkit:
```bash
export CUDA_HOME=/usr/local/cuda-11.6
# Also ensure it's on PATH:
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Install detectron2:
Build detectron2 from source to ensure it compiles against your CUDA 11.6/11.7 environment:
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

If you encounter compilation errors (e.g., arch mismatch), set the GPU architecture first:
```bash
# Adjust "8.6" to match your GPU:
#   8.6 = RTX 3090 / RTX 3080 / RTX A5000
#   8.0 = A100
#   7.5 = RTX 2080 Ti / T4
#   7.0 = V100
export TORCH_CUDA_ARCH_LIST="8.6"
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Verify the installation:
```bash
python -m detectron2.utils.collect_env
```

## CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile the CUDA kernel for MSDeformAttn:

```bash
cd segearth_r1/model/mask_decoder/Mask2Former_Simplify/modeling/pixel_decoder/ops
sh make.sh
```

**Tip:** If you get a GPU architecture error, edit `make.sh` to uncomment and set the `TORCH_CUDA_ARCH_LIST` line that matches your GPU before running.

Verify the module compiled successfully:
```python
python -c "import MultiScaleDeformableAttention; print('MSDeformAttn loaded successfully!')"
```

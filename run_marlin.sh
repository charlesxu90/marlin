#!/bin/bash

SECONDS=0

echo "Starting MARLIN training..."

# CUDA environment setup
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

# Use GPU 1 (GPU 0 has 33GB used, GPU 1 is mostly free)
export CUDA_VISIBLE_DEVICES=1

# Optional: Limit TensorFlow GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

Rscript MARLIN_training.R

duration=$SECONDS
echo "MARLIN training completed in $(($duration / 60)) minutes and $(($duration % 60)) seconds."
echo "Results are saved in the 'output' directory."
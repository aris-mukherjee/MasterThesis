#!/bin/bash
#SBATCH  --output=no_norm_train_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
#SBATCH  --constraint=geforce_gtx_titan_x

source /itet-stor/arismu/net_scratch/conda/bin/activate pytorch_env
conda activate pytorch_env

CUDA_VISIBLE_DEVICES=0 python -u NO_NORM_train_FETS.py
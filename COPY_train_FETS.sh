#!/bin/bash
#SBATCH  --output=train_log_FETS_TTA/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
#SBATCH  --constraint=geforce_gtx_titan_x

source /itet-stor/arismu/net_scratch/conda/bin/activate pytorch_env
conda activate pytorch_env

CUDA_VISIBLE_DEVICES=0 python -u COPY_train_FETS.py
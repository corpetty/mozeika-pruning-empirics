#!/bin/bash
#SBATCH --job-name=exp36-lenet-finetune
#SBATCH --output=/home/petty/slurm-logs/%j-%x/out.log
#SBATCH --error=/home/petty/slurm-logs/%j-%x/err.log
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --partition=gpu

mkdir -p /home/petty/slurm-logs/${SLURM_JOB_ID}-${SLURM_JOB_NAME}

/home/petty/torch-env/bin/python3 \
    /home/petty/pruning-research/experiments/36_lenet_finetune.py

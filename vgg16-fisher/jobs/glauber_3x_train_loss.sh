#!/bin/bash
#SBATCH --job-name=ai-research/glauber-3x-train-loss
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/petty/slurm-logs/%j-%x/glauber-3x-train-loss.log
#SBATCH --error=/home/petty/slurm-logs/%j-%x/glauber-3x-train-loss.log

mkdir -p /home/petty/slurm-logs/${SLURM_JOB_ID}-${SLURM_JOB_NAME//\//-}
source /home/petty/torch-env/bin/activate
python3 /home/petty/pruning-research/experiments/31_glauber_3x_with_train_loss.py

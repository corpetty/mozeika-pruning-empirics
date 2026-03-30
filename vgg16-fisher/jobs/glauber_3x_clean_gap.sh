#!/bin/bash
#SBATCH --job-name=ai-research/glauber-clean-gap
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/petty/slurm-logs/%j-%x/glauber-clean-gap.log
#SBATCH --error=/home/petty/slurm-logs/%j-%x/glauber-clean-gap.log

mkdir -p "/home/petty/slurm-logs/${SLURM_JOB_ID}-${SLURM_JOB_NAME//\//-}"
source /home/petty/torch-env/bin/activate
python3 /home/petty/pruning-research/experiments/32_glauber_3x_clean_gap.py

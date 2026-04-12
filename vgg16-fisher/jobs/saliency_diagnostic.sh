#!/bin/bash
#SBATCH --job-name=ai-research/saliency-diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/petty/slurm-logs/%j-%x/job.log
#SBATCH --error=/home/petty/slurm-logs/%j-%x/job.log
#SBATCH --open-mode=append

mkdir -p /home/petty/slurm-logs/${SLURM_JOB_ID}-${SLURM_JOB_NAME}
echo "Job ${SLURM_JOB_ID} started at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

source /home/petty/torch-env/bin/activate
cd /home/petty/pruning-research
python3 experiments/34_saliency_diagnostic.py

echo "Job finished at $(date)"

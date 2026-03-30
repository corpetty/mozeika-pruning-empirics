#!/bin/bash
#SBATCH --job-name=kv/exp25-niah-robust
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/home/petty/slurm-logs/exp25-niah-%j.out
#SBATCH --error=/home/petty/slurm-logs/exp25-niah-%j.err

set -e
echo "=== Job $SLURM_JOB_ID started $(date) ==="

sudo systemctl stop ollama || true
sleep 2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

cd /home/petty/pruning-research/kv-subspace
/home/petty/torch-env/bin/python3 experiments/exp25_niah_robust.py

echo "=== Job $SLURM_JOB_ID done $(date) ==="

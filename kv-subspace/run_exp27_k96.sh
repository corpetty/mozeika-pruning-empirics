#!/bin/bash
#SBATCH --job-name=kv/exp27-k96
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=8:00:00
#SBATCH --output=/home/petty/slurm-logs/exp27-k96-%j.out
#SBATCH --error=/home/petty/slurm-logs/exp27-k96-%j.err

set -e
echo "=== Job $SLURM_JOB_ID (k96_4bit) started $(date) ==="

sudo systemctl stop ollama || true
sleep 2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

cd /home/petty/pruning-research/kv-subspace
/home/petty/torch-env/bin/python3 experiments/exp27_downstream_tasks.py --config k96_4bit

echo "=== Job $SLURM_JOB_ID done $(date) ==="

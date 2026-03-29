#!/bin/bash
#SBATCH --job-name=kv/exp18-adaptive-policy
#SBATCH --output=/home/petty/slurm-logs/exp18-%j.out
#SBATCH --error=/home/petty/slurm-logs/exp18-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Exp 18: Adaptive Layer K-Scheduling ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
date

# Stop ollama to free VRAM
sudo systemctl stop ollama || true
sleep 3

cd /home/petty/pruning-research/kv-subspace
/home/petty/torch-env/bin/python3 experiments/exp18_adaptive_policy.py

echo "=== Done ===" && date

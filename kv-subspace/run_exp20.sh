#!/bin/bash
#SBATCH --job-name=kv/exp20
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/petty/slurm-logs/exp20-%j.out
#SBATCH --error=/home/petty/slurm-logs/exp20-%j.err

echo "=== Exp 20: V-specific k Threshold Scan ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
date

sudo systemctl stop ollama || true
sleep 2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

cd /home/petty/pruning-research/kv-subspace
/home/petty/torch-env/bin/python3 experiments/exp20_v_threshold.py

echo "=== Done ==="
date

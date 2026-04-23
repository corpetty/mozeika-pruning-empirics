#!/bin/bash
#SBATCH --job-name=expF1-rotation-ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/petty/slurm-logs/expF1-%j.out
#SBATCH --error=/home/petty/slurm-logs/expF1-%j.err

echo "=== Task F1: Rotation Ablation ==="
echo "Job: $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
date

sudo systemctl stop ollama || true
sleep 2

cd /home/petty/pruning-research/kv-subspace
source /home/petty/torch-env/bin/activate

python3 experiments/exp36_rotation_ablation.py

echo "=== Done ==="
date

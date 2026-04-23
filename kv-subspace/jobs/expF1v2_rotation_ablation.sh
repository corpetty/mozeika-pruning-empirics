#!/bin/bash
#SBATCH --job-name=expF1v2-rotation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/home/petty/slurm-logs/expF1v2-%j.out
#SBATCH --error=/home/petty/slurm-logs/expF1v2-%j.err

echo "=== Task F1 v2: Rotation Ablation (corrected) ==="
echo "Job: $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
date

sudo systemctl stop ollama || true
sleep 2

cd /home/petty/pruning-research/kv-subspace
source /home/petty/torch-env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 experiments/exp37_rotation_ablation_v2.py

echo "=== Done ==="
date

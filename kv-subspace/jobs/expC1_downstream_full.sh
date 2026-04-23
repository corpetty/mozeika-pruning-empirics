#!/bin/bash
#SBATCH --job-name=expC1-downstream
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/home/petty/slurm-logs/expC1-%j.out
#SBATCH --error=/home/petty/slurm-logs/expC1-%j.err

echo "=== Task C1: Full-N Downstream Tasks ==="
echo "Job: $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
date

sudo systemctl stop ollama || true
sleep 2

cd /home/petty/pruning-research/kv-subspace
source /home/petty/torch-env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 experiments/expC1_downstream_full.py

echo "=== Done ==="
date

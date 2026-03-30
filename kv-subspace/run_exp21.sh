#!/bin/bash
#SBATCH --job-name=kv/exp21-llama3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=/home/petty/slurm-logs/exp21-%j.out
#SBATCH --error=/home/petty/slurm-logs/exp21-%j.err

echo "=== Exp 21: Llama-3.1 Architecture Validation ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
date

# Stop Ollama to free VRAM headroom
sudo systemctl stop ollama

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/petty/pruning-research/kv-subspace
/home/petty/torch-env/bin/python3 experiments/exp21_llama3_validation.py

echo "=== Done ==="
date

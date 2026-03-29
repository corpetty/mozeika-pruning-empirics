#!/bin/bash
#SBATCH --job-name=kv/exp13-long-ctx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/petty/slurm-logs/exp13-%j.out
#SBATCH --error=/home/petty/slurm-logs/exp13-%j.err

set -e

echo "=== Exp 13: Long-Context Scaling ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"

# Stop Ollama entirely to free all VRAM
echo "Stopping Ollama..."
sudo systemctl stop ollama || { echo "sudo stop failed, trying ollama stop..."; ollama stop qwen35moe-32k 2>/dev/null || true; }
sleep 5
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# Force GPU0 (GPU1 is in use by another job)
export CUDA_VISIBLE_DEVICES=0

# Memory fragmentation fix
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/petty/pruning-research/kv-subspace

echo "Running experiment 13..."
/home/petty/torch-env/bin/python3 experiments/long_context_scaling.py

echo "Done: $(date)"

# Restart Ollama (model will reload on next request)
echo "Restarting Ollama..."
sudo systemctl start ollama || true

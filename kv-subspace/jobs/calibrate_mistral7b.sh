#!/bin/bash
#SBATCH --job-name=subrotq_calib
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/petty/slurm-logs/subrotq_calib_%j.log
#SBATCH --error=/home/petty/slurm-logs/subrotq_calib_%j.log

# SubRotQ Basis Calibration Job
# Generates PCA basis for K-cache compression using Mistral-7B-v0.3

set -e

echo "=== SubRotQ Basis Calibration ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo

# Stop Ollama to free GPU memory
echo "Stopping Ollama service..."
sudo systemctl stop ollama
sleep 2

# Check GPU
echo "GPU status:"
nvidia-smi --query-gpu=index,name,memory.free,memory.used --format=csv
echo

# Activate venv
cd /home/petty/pruning-research
source venv/bin/activate

# Set CUDA device (use GPU0 for calibration)
export CUDA_VISIBLE_DEVICES=0

# Run calibration
cd kv-subspace
python calibrate_subrotq_basis.py \
    --model mistralai/Mistral-7B-v0.3 \
    --rank 128 \
    --bits 4 \
    --calib-tokens 2048 \
    --dataset wikitext \
    --dataset-split train \
    --output results/subrotq_basis_mistral7b_k128.bin \
    --device cuda

echo
echo "Job complete: $(date)"

# Restart Ollama
echo "Restarting Ollama service..."
sudo systemctl start ollama

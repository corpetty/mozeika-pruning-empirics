#!/bin/bash
#SBATCH --job-name=gemma4_basis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/petty/slurm-logs/gemma4_basis_%j.out
#SBATCH --error=/home/petty/slurm-logs/gemma4_basis_%j.err

# Stop Ollama to free GPU memory
echo "Stopping Ollama..."
sudo systemctl stop ollama
sleep 2

# Activate environment
source /home/petty/pruning-research/kv-subspace/.venv/bin/activate

# Set environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

cd /home/petty/pruning-research/kv-subspace

echo "Starting Gemma4-E4B basis generation..."
echo "Time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo ""

python3 scripts/generate_gemma4_basis.py \
    --model google/gemma-4-E4B-it \
    --output results/gemma4_e4b_pca_basis_k128.npz \
    --k 128 \
    --num-samples 500 \
    --max-tokens 512 \
    --device cuda:0

EXIT_CODE=$?

echo ""
echo "Job finished at $(date)"
echo "Exit code: $EXIT_CODE"

# Restart Ollama
echo "Restarting Ollama..."
sudo systemctl start ollama

exit $EXIT_CODE

#!/bin/bash
#SBATCH --job-name=kvsub/exp17-fixed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/petty/slurm-logs/exp17-fixed-%j.out
#SBATCH --error=/home/petty/slurm-logs/exp17-fixed-%j.err

set -euo pipefail
cd /home/petty/pruning-research/kv-subspace

# Stop Ollama to free VRAM
sudo systemctl stop ollama || true
sleep 3

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Remove old results so experiment re-runs clean
# (keep backup just in case)
if [ -f results/exp17_cross_domain.csv ]; then
    cp results/exp17_cross_domain.csv results/exp17_cross_domain_OLD.csv
    rm results/exp17_cross_domain.csv
fi

echo "=== Exp 17 (fixed eval logic) ==="
/home/petty/torch-env/bin/python3 experiments/exp17_cross_domain.py
echo "=== DONE ==="

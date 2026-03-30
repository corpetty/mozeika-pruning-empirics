#!/bin/bash
#SBATCH --job-name=kvsub/exp19-fixed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/petty/slurm-logs/exp19-fixed-%j.out
#SBATCH --error=/home/petty/slurm-logs/exp19-fixed-%j.err

set -euo pipefail
cd /home/petty/pruning-research/kv-subspace

sudo systemctl stop ollama || true
sleep 3

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Back up old results, re-run clean
if [ -f results/exp19_online_basis.csv ]; then
    cp results/exp19_online_basis.csv results/exp19_online_basis_OLD.csv
    rm results/exp19_online_basis.csv
fi

echo "=== Exp 19 (fixed drift comparison + update instrumentation) ==="
python experiments/exp19_online_basis.py
echo "=== DONE ==="

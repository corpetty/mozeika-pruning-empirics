#!/bin/bash
#SBATCH --job-name=kvsub-exp22
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/petty/slurm-logs/exp22-quantizer-%j.out
#SBATCH --error=/home/petty/slurm-logs/exp22-quantizer-%j.err

set -euo pipefail
cd /home/petty/pruning-research/kv-subspace

sudo systemctl stop ollama || true
sleep 3

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Exp 22: SubRotQ vs PolarQuant quantizer comparison ==="
echo "Configs: k={64,96,112,128} x bits={4,8} x quantizer={subrotq,polarquant}"
/home/petty/torch-env/bin/python3 experiments/exp22_quantizer_comparison.py
echo "=== DONE ==="

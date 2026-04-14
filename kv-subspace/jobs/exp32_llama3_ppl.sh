#!/bin/bash
#SBATCH --job-name=exp32-llama3-ppl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/petty/slurm-logs/exp32-%j.out
#SBATCH --error=/home/petty/slurm-logs/exp32-%j.err

set -e
echo "=== Exp32: Llama-3.1-8B WikiText-2 PPL ==="
echo "Job: $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
date

# Stop Ollama so GPU0 is free (it runs on GPU1 normally, but be safe)
sudo systemctl stop ollama 2>/dev/null || true
sleep 2

cd /home/petty/pruning-research/kv-subspace

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_CACHE=/home/petty/.cache/huggingface/datasets

source .venv/bin/activate
python3 experiments/exp32_llama3_wikitext2_ppl.py

echo "=== Done ===" && date

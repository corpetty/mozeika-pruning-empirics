#!/bin/bash
#SBATCH --job-name=eval_baseline_arc
#SBATCH --output=/home/petty/slurm-logs/%x_%j.out
#SBATCH --error=/home/petty/slurm-logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=1:00:00

# Stop Ollama to free GPU memory
sudo systemctl stop ollama

# Activate environment
cd /home/petty/pruning-research/kv-subspace
source .venv/bin/activate

# Run baseline evaluation
python scripts/eval_lm_harness.py \
    --model google/gemma-4-E4B-it \
    --tasks arc_easy \
    --batch_size 1 \
    --num_fewshot 0 \
    --device cuda:0 \
    --output_path results/

echo "Baseline evaluation complete"

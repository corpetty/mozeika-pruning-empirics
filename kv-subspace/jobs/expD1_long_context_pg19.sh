#!/bin/bash
#SBATCH --job-name=expD1-longctx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/petty/slurm-logs/expD1-%j.out
#SBATCH --error=/home/petty/slurm-logs/expD1-%j.err

echo "=== Experiment D1: Long-Context PPL on PG-19 ==="
echo "Job: $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
date

sudo systemctl stop ollama || true
sleep 2

cd /home/petty/pruning-research/kv-subspace
source /home/petty/torch-env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_CACHE=/home/petty/.cache/huggingface/datasets

python3 experiments/expD1_long_context_pg19.py

echo "=== Done ==="
date

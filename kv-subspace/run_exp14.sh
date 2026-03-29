#!/bin/bash
#SBATCH --job-name=kv/exp14-throughput
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/petty/slurm-logs/exp14-%j.out
#SBATCH --error=/home/petty/slurm-logs/exp14-%j.err

set -e

echo "=== Exp 14: Throughput Benchmark ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"

echo "Stopping Ollama..."
sudo systemctl stop ollama || true
sleep 5
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/petty/pruning-research/kv-subspace
echo "Running experiment 14..."
/home/petty/torch-env/bin/python3 experiments/exp14_throughput.py

echo "Done: $(date)"
echo "Restarting Ollama..."
sudo systemctl start ollama || true

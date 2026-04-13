#!/bin/bash
#SBATCH --job-name=calib_tiny
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/petty/slurm-logs/calib_tinyllama_%j.out

cd /home/petty/pruning-research
source venv/bin/activate

cd kv-subspace
export CUDA_VISIBLE_DEVICES=0

python calibrate_subrotq_basis.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --rank 64 \
  --bits 4 \
  --calib-tokens 2048 \
  --output results/subrotq_basis_tinyllama_k64.bin \
  --device cuda

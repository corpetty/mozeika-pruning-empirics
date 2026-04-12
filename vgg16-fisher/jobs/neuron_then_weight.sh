#!/bin/bash
#SBATCH --job-name=ai-research/neuron-then-weight
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/petty/slurm-logs/%j-%x/job.log
#SBATCH --error=/home/petty/slurm-logs/%j-%x/job.log

mkdir -p /home/petty/slurm-logs/${SLURM_JOB_ID}-${SLURM_JOB_NAME//\//-}

source /home/petty/torch-env/bin/activate
cd /home/petty/pruning-research

echo "Job ${SLURM_JOB_ID} started at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python experiments/33_neuron_then_weight_pruning.py

echo "Job finished at $(date)"

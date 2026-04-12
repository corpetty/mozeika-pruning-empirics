#!/bin/bash
#SBATCH --job-name=ai-research/vgg16-finite-temp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/petty/slurm-logs/%j-%x/job.log
#SBATCH --error=/home/petty/slurm-logs/%j-%x/job.log

mkdir -p /home/petty/slurm-logs/${SLURM_JOB_ID}-${SLURM_JOB_NAME//\//-}

source /home/petty/torch-env/bin/activate
cd /home/petty/pruning-research

echo "Job ${SLURM_JOB_ID} started at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Start from dense (ImageNet pretrained + 1 epoch CIFAR-10 fine-tune), geometric annealing T: 1e-7 → 1e-10 over 80 rounds
# Regrowth enabled with 5% cap per round
python experiments/34_vgg16_finite_temp_annealing.py \
    --T-start 1e-7 \
    --T-end 1e-10 \
    --schedule geometric \
    --max-rounds 80 \
    --target-sparsity 0.99 \
    --data-root /home/petty/.openclaw/workspace-ai-research/data

echo "Job finished at $(date)"

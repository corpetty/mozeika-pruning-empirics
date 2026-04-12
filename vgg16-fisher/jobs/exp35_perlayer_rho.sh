#!/bin/bash
#SBATCH --job-name=ai-research/exp35-perlayer-rho
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/petty/slurm-logs/%j-%x/job.log
#SBATCH --error=/home/petty/slurm-logs/%j-%x/job.log
#SBATCH --open-mode=append

mkdir -p /home/petty/slurm-logs/${SLURM_JOB_ID}-${SLURM_JOB_NAME}
echo "Job ${SLURM_JOB_ID} started at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

source /home/petty/torch-env/bin/activate
cd /home/petty/pruning-research

python3 experiments/35_vgg16_perlayer_rho.py \
    --max-rounds 80 \
    --target-sparsity 0.99 \
    --train-epochs 3 \
    --fisher-batches 5 \
    --seed 0

echo "Job finished at $(date)"

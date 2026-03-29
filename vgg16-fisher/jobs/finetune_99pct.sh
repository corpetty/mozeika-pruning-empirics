#!/bin/bash
#SBATCH --job-name=ai-research/finetune-vgg16-99pct
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/petty/slurm-logs/%j-%x.log
#SBATCH --error=/home/petty/slurm-logs/%j-%x.err
#SBATCH --time=12:00:00

cd /home/petty/pruning-research/vgg16-fisher
source /home/petty/torch-env/bin/activate

python3 finetune_sparse.py \
  --checkpoint vgg16_pruned_and_compressed_v4_99pct.pt \
  --resume vgg16_finetuned_99pct.pt \
  --start-epoch 13 \
  --data-root /home/petty/.openclaw/workspace-ai-research/data \
  --epochs 100 \
  --lr 1e-4 \
  --batch-size 128 \
  --save-every 10 \
  --output vgg16_finetuned_99pct.pt \
  --device cuda

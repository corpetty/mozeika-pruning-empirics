#!/bin/bash
#SBATCH --job-name=ai-research/extract-vgg16-99pct-finetuned
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/petty/slurm-logs/%j-%x.log
#SBATCH --error=/home/petty/slurm-logs/%j-%x.err
#SBATCH --time=2:00:00

cd /home/petty/pruning-research/vgg16-fisher
source /home/petty/torch-env/bin/activate

python3 extract_and_export.py \
  --checkpoint vgg16_finetuned_99pct.pt \
  --output-dir /home/petty/pruning-research/vgg16-fisher \
  --data-root /home/petty/.openclaw/workspace-ai-research/data \
  --input-size 224 \
  --device cpu \
  --verify

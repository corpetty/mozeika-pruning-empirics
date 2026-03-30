#!/bin/bash
#SBATCH --job-name=ai-research/quantize-vgg16-int8
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/petty/slurm-logs/%j-%x.log
#SBATCH --error=/home/petty/slurm-logs/%j-%x.err
#SBATCH --time=1:00:00

cd /home/petty/pruning-research/vgg16-fisher
source /home/petty/torch-env/bin/activate

# Check onnxruntime-tools available (needed for quantize_static)
python3 -c "from onnxruntime.quantization import quantize_static; print('onnxruntime quantization OK')"

python3 quantize_int8.py \
  --onnx vgg16_finetuned_99pct_compact.onnx \
  --output vgg16_finetuned_99pct_compact_int8_static.onnx \
  --data-root /home/petty/.openclaw/workspace-ai-research/data \
  --n-calibration 512 \
  --input-size 224

#!/bin/bash
cd /home/petty/pruning-research/kv-subspace
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/generate_gemma4_basis.py \
    --k 128 \
    --num-samples 200 \
    --max-tokens 512 \
    --skip-v \
    --output results/gemma4_e4b_pca_basis_k128_hetero.npz

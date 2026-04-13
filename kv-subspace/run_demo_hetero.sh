#!/bin/bash
cd /home/petty/pruning-research/kv-subspace
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python demo_subrotq_scaling.py \
    --basis results/gemma4_e4b_pca_basis_k128_hetero.npz \
    --baseline-ctx 4096 \
    --subrotq-ctx 16384

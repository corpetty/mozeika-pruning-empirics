#!/bin/bash
cd /home/petty/pruning-research/kv-subspace
source .venv/bin/activate
python scripts/generate_gemma4_basis.py --num-samples 200 --skip-v

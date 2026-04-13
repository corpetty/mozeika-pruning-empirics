#!/bin/bash
cd /home/petty/pruning-research/kv-subspace
source .venv/bin/activate
python scripts/eval_lm_harness.py --model google/gemma-4-E4B-it --tasks arc_easy --batch_size 1 --num_fewshot 0 2>&1 | head -200

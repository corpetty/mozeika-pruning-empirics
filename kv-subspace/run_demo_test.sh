#!/bin/bash
#SBATCH --job-name=kv/demo-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/home/petty/slurm-logs/demo-test-%j.out
#SBATCH --error=/home/petty/slurm-logs/demo-test-%j.err

echo "=== kvpatch demo end-to-end test ==="
echo "Job: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"
date

sudo systemctl stop ollama

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/petty/pruning-research/kv-subspace

echo "--- Test 1: builtin-long doc, default k=112 ---"
/home/petty/torch-env/bin/python3 demo.py \
    --builtin-long \
    --question "What are the five scientific topics mentioned in this document? List each briefly." \
    --ctx 8192 \
    --k 112 \
    --save-basis /tmp/demo_basis.pkl

echo ""
echo "--- Test 2: same doc, reload saved basis (skip calibration) ---"
/home/petty/torch-env/bin/python3 demo.py \
    --builtin-long \
    --question "What role does entropy play in thermodynamics according to this document?" \
    --ctx 8192 \
    --basis /tmp/demo_basis.pkl

echo ""
echo "--- Test 3: baseline (no patch) for comparison ---"
/home/petty/torch-env/bin/python3 demo.py \
    --builtin-long \
    --question "What are the five scientific topics mentioned in this document? List each briefly." \
    --ctx 8192 \
    --no-patch

echo "=== Done ==="
date

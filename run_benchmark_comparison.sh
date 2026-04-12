#!/usr/bin/env bash
set -e

PYTHON="/home/petty/torch-env/bin/python3"
ROUNDS=5
SEED=42
OUTPUT="/home/petty/.openclaw/workspace/lenet_benchmark_results.txt"

echo "========================================"
echo "LeNet-300-100 Glauber Pruning Benchmark"
echo "========================================"
echo "Comparing 3 implementations:"
echo "  v1: Loop-based parameter updates"
echo "  v2: Vectorized operations"
echo "  v3: Baseline (manual loop)"
echo ""
echo "Config: $ROUNDS rounds, seed=$SEED"
echo "Python: $PYTHON"
echo "========================================"
echo ""

cd /home/petty/pruning-research

$PYTHON benchmark_lenet.py \
  --python "$PYTHON" \
  --repeats 3 \
  --warmup 1 \
  --outfile "$OUTPUT" \
  "lenet_v1_loop.py --rounds $ROUNDS --seed $SEED --save-dir /tmp/v1 --csv-log-name history_v1.csv" \
  "lenet_v2_vectorized.py --rounds $ROUNDS --seed $SEED --save-dir /tmp/v2 --csv-log-name history_v2.csv" \
  "lenet_v3_baseline.py --rounds $ROUNDS --seed $SEED --save-dir /tmp/v3 --csv-log-name history_v3.csv"

echo ""
echo "✅ Benchmark complete!"
echo "Results saved to: $OUTPUT"
echo ""
cat "$OUTPUT"

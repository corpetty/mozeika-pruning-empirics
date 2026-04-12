#!/usr/bin/env bash
set -e

PYTHON="/home/petty/torch-env/bin/python3"
ROUNDS=5
OUTPUT_DIR="/home/petty/.openclaw/workspace/benchmark_results"

mkdir -p "$OUTPUT_DIR"

echo "====================================="
echo "LeNet-300-100 Pruning Benchmark"
echo "====================================="
echo "Testing 3 implementations:"
echo "  v1: Loop-based (baseline)"
echo "  v2: Vectorized operations"
echo "  v3: Original baseline"
echo ""
echo "Config: $ROUNDS rounds, torch-env, GPU"
echo "====================================="
echo ""

# V1: Loop-based
echo "[1/3] Running v1_loop..."
time $PYTHON /home/petty/pruning-research/lenet_v1_loop.py \
  --rounds $ROUNDS \
  --save-dir "$OUTPUT_DIR/v1_loop" \
  --csv-log-name history_v1.csv \
  2>&1 | tee "$OUTPUT_DIR/v1_loop.log"

# V2: Vectorized
echo ""
echo "[2/3] Running v2_vectorized..."
time $PYTHON /home/petty/pruning-research/lenet_v2_vectorized.py \
  --rounds $ROUNDS \
  --save-dir "$OUTPUT_DIR/v2_vectorized" \
  --csv-log-name history_v2.csv \
  2>&1 | tee "$OUTPUT_DIR/v2_vectorized.log"

# V3: Baseline
echo ""
echo "[3/3] Running v3_baseline..."
time $PYTHON /home/petty/pruning-research/lenet_v3_baseline.py \
  --rounds $ROUNDS \
  --save-dir "$OUTPUT_DIR/v3_baseline" \
  --csv-log-name history_v3.csv \
  2>&1 | tee "$OUTPUT_DIR/v3_baseline.log"

echo ""
echo "====================================="
echo "Benchmark complete!"
echo "Results saved in: $OUTPUT_DIR"
echo "====================================="
echo ""
echo "Summary:"
grep "real" "$OUTPUT_DIR"/*.log || true

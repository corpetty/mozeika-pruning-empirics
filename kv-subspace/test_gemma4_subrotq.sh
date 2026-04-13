#!/bin/bash
set -e

echo "=== Testing Gemma4-26B with SubRotQ k=128/4-bit ==="
echo "Using Mistral-7B basis (cross-architecture validated in exp30)"
echo ""

GGUF="/tmp/gemma4-26b.gguf"
BASIS="/home/petty/pruning-research/kv-subspace/results/subrotq_basis_mistral7b_k128.bin"
LLAMA_CLI="/tmp/llama.cpp/build/bin/llama-cli"

# Verify files exist
if [[ ! -f "$GGUF" ]]; then
    echo "ERROR: GGUF not found at $GGUF"
    exit 1
fi

if [[ ! -f "$BASIS" ]]; then
    echo "ERROR: Basis not found at $BASIS"
    exit 1
fi

if [[ ! -f "$LLAMA_CLI" ]]; then
    echo "ERROR: llama-cli not found at $LLAMA_CLI"
    exit 1
fi

echo "✓ GGUF: $(du -h $GGUF | cut -f1)"
echo "✓ Basis: $(du -h $BASIS | cut -f1) (Mistral-7B k=128)"
echo "✓ llama-cli: $LLAMA_CLI"
echo ""

echo "Running SubRotQ test (GPU0 only, 50 tokens)..."
CUDA_VISIBLE_DEVICES=0 "$LLAMA_CLI" \
  -m "$GGUF" \
  -p "Explain quantum entanglement in simple terms:" \
  -n 50 \
  --subrotq \
  --subrotq-rank 128 \
  --subrotq-bits 4 \
  --subrotq-basis "$BASIS" \
  2>&1 | grep -E "(SubRotQ|compress|decompress|Explain|quantum)" | head -40

echo ""
echo "=== Test complete ==="

#!/bin/bash
# Phase 8: Gemma4-26B SubRotQ Test Script
# Run this manually if Claude Code sandbox blocks external file ops

set -e

GGUF_SRC="/usr/share/ollama/.ollama/models/blobs/sha256-7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df"
GGUF_DST="/tmp/gemma4-26b.gguf"
LLAMA_CLI="/tmp/llama.cpp/build/bin/llama-cli"
BASIS="$(pwd)/results/subrotq_basis_mistral7b_k128.bin"

export CUDA_VISIBLE_DEVICES=0

echo "=== Step 1: Extract GGUF from Ollama ==="
if [ ! -f "$GGUF_DST" ]; then
    echo "Copying GGUF (17GB, may take a minute)..."
    cp "$GGUF_SRC" "$GGUF_DST"
    echo "Done. Size: $(du -h "$GGUF_DST" | cut -f1)"
else
    echo "GGUF already at $GGUF_DST ($(du -h "$GGUF_DST" | cut -f1))"
fi

echo ""
echo "=== Step 2: Verify GGUF ==="
file "$GGUF_DST"
echo "First 4 bytes (should be GGUF magic):"
xxd -l 4 "$GGUF_DST"

echo ""
echo "=== Step 3: Check basis file ==="
if [ -f "$BASIS" ]; then
    echo "Basis: $BASIS ($(du -h "$BASIS" | cut -f1))"
else
    echo "ERROR: Basis not found at $BASIS"
    exit 1
fi

echo ""
echo "=== Step 4: Test SubRotQ on Gemma4 ==="
echo "Running llama-cli with SubRotQ k=128/4-bit..."
$LLAMA_CLI \
    -m "$GGUF_DST" \
    -p "Write a detailed explanation of quantum entanglement" \
    -n 100 \
    --subrotq \
    --subrotq-rank 128 \
    --subrotq-bits 4 \
    --subrotq-basis "$BASIS" \
    2>&1 | tee /tmp/gemma4_subrotq_output.log

echo ""
echo "=== Done ==="
echo "Full output saved to /tmp/gemma4_subrotq_output.log"

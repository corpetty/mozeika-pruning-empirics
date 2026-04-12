#!/bin/bash
# Force Ollama to use single GPU (free GPU0 for research)
# Run with: sudo bash force_single_gpu.sh

set -e

echo "=== Forcing Ollama to Single-GPU Mode ==="

# Backup current config
cp /etc/default/ollama /etc/default/ollama.backup.$(date +%Y%m%d-%H%M%S)
echo "✓ Backed up /etc/default/ollama"

# Check if OLLAMA_NUM_GPU already exists
if grep -q "^OLLAMA_NUM_GPU=" /etc/default/ollama; then
    echo "OLLAMA_NUM_GPU already set, updating..."
    sed -i 's/^OLLAMA_NUM_GPU=.*/OLLAMA_NUM_GPU=1/' /etc/default/ollama
else
    echo "Adding OLLAMA_NUM_GPU=1..."
    echo "OLLAMA_NUM_GPU=1" >> /etc/default/ollama
fi

echo ""
echo "Updated /etc/default/ollama:"
echo "---"
cat /etc/default/ollama
echo "---"

echo ""
echo "Restarting Ollama service..."
systemctl restart ollama

echo ""
echo "Waiting for Ollama to start..."
sleep 5

# Check if service is running
if systemctl is-active --quiet ollama; then
    echo "✓ Ollama restarted successfully"
else
    echo "✗ Ollama failed to start"
    echo "Check logs: journalctl -u ollama -n 50"
    exit 1
fi

echo ""
echo "Loading model to verify single-GPU mode..."
curl -s http://localhost:11434/api/generate -d '{
  "model": "gemma4:26b",
  "prompt": "Hello",
  "stream": false,
  "options": {"num_predict": 5}
}' >/dev/null 2>&1 && echo "✓ Model loaded successfully" || echo "✗ Model load failed"

sleep 2

echo ""
echo "GPU Memory Usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
  awk '{printf "  GPU %s: %d / %d MiB (%.1f%%)\n", $1, $2, $3, ($2/$3)*100}'

echo ""
echo "Process allocation:"
nvidia-smi pmon -c 1 | grep ollama || echo "  (no active Ollama process)"

echo ""
echo "=== Summary ==="
echo "Ollama is now restricted to 1 GPU."
echo ""
echo "Expected behavior:"
echo "  - GPU0: ~370 MiB (idle, free for research)"
echo "  - GPU1: ~22-24 GB (Gemma4 26B + 8-bit KV cache)"
echo "  - Max context: ~28K tokens (2× baseline)"
echo ""
echo "If both GPUs still show high usage, check:"
echo "  1. Run 'ollama stop gemma4:26b' to unload model"
echo "  2. Wait 30 seconds for memory to release"
echo "  3. Run 'ollama run gemma4:26b' to reload on single GPU"
echo ""
echo "To revert to multi-GPU, remove 'OLLAMA_NUM_GPU=1' from /etc/default/ollama"
echo "and restart: sudo systemctl restart ollama"

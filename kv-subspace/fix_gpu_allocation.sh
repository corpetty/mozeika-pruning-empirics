#!/bin/bash
# Fix GPU allocation: Hide GPU0 completely from Ollama
# Run with: sudo bash fix_gpu_allocation.sh

set -e

echo "=== Fixing GPU Allocation: Hide GPU0 from Ollama ==="

# Backup current config
cp /etc/default/ollama /etc/default/ollama.backup.$(date +%Y%m%d-%H%M%S)
echo "✓ Backed up /etc/default/ollama"

# Change CUDA_VISIBLE_DEVICES from 0 to 1 (show only GPU1 to Ollama)
sed -i 's/^CUDA_VISIBLE_DEVICES=0/CUDA_VISIBLE_DEVICES=1/' /etc/default/ollama

# Ensure OLLAMA_NUM_GPU=1 is set
if ! grep -q "^OLLAMA_NUM_GPU=1" /etc/default/ollama; then
    echo "OLLAMA_NUM_GPU=1" >> /etc/default/ollama
fi

echo ""
echo "Updated /etc/default/ollama:"
echo "---"
cat /etc/default/ollama
echo "---"

echo ""
echo "Key change: CUDA_VISIBLE_DEVICES=0 → CUDA_VISIBLE_DEVICES=1"
echo "  (Ollama will only see GPU1, GPU0 is completely hidden)"

echo ""
echo "Stopping Ollama and unloading models..."
systemctl stop ollama
sleep 3

# Verify GPUs are idle
echo ""
echo "GPU state before restart:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo ""
echo "Starting Ollama..."
systemctl start ollama
sleep 5

# Check if service is running
if systemctl is-active --quiet ollama; then
    echo "✓ Ollama started successfully"
else
    echo "✗ Ollama failed to start"
    echo "Check logs: journalctl -u ollama -n 50"
    exit 1
fi

echo ""
echo "Loading model on GPU1 only..."
curl -s http://localhost:11434/api/generate -d '{
  "model": "gemma4:26b",
  "prompt": "Test",
  "stream": false,
  "options": {"num_predict": 5}
}' >/dev/null 2>&1 && echo "✓ Model loaded" || echo "✗ Model load failed"

sleep 3

echo ""
echo "Final GPU Memory Usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
  awk '{printf "  GPU %s: %d / %d MiB", $1, $2, $3; if ($2 < 1000) printf " ✓ FREE\n"; else printf " (active)\n"}'

echo ""
echo "Process allocation:"
nvidia-smi pmon -c 1 | grep -E "# gpu|ollama" || echo "  (no Ollama on GPU0)"

echo ""
echo "=== Summary ==="
echo "Ollama now sees ONLY GPU1 (CUDA_VISIBLE_DEVICES=1)"
echo ""
echo "Expected:"
echo "  - GPU0: <1 GB (completely free for research)"
echo "  - GPU1: ~22-24 GB (Gemma4 26B + 8-bit KV cache)"
echo ""
echo "To verify GPU0 is truly free, try running a research job:"
echo "  CUDA_VISIBLE_DEVICES=0 python3 your_script.py"

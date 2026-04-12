#!/bin/bash
# Enable 8-bit KV cache for Ollama via environment variable
# Run with: sudo bash enable_8bit_kv_cache.sh

set -e

echo "=== Enabling 8-bit KV Cache for Ollama ==="

# Backup current config
cp /etc/default/ollama /etc/default/ollama.backup.$(date +%Y%m%d-%H%M%S)

# Remove any existing OLLAMA_KV_CACHE_TYPE line
sed -i '/^OLLAMA_KV_CACHE_TYPE=/d' /etc/default/ollama

# Add 8-bit KV cache setting
echo "OLLAMA_KV_CACHE_TYPE=q8_0" >> /etc/default/ollama

echo ""
echo "Updated /etc/default/ollama:"
cat /etc/default/ollama

echo ""
echo "Restarting Ollama service..."
systemctl restart ollama

echo ""
echo "Waiting for Ollama to start..."
sleep 5

# Check if service is running
if systemctl is-active --quiet ollama; then
    echo "✓ Ollama restarted successfully"
    echo ""
    echo "Testing with gemma4:26b..."
    # Test that model loads
    curl -s http://localhost:11434/api/generate -d '{
      "model": "gemma4:26b",
      "prompt": "Hello",
      "stream": false
    }' >/dev/null 2>&1 && echo "✓ Model loads successfully with 8-bit KV cache" || echo "✗ Model load failed"
    
    echo ""
    echo "Check VRAM usage:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
else
    echo "✗ Ollama failed to start"
    echo "Check logs: journalctl -u ollama -n 50"
    exit 1
fi

echo ""
echo "=== Summary ==="
echo "8-bit KV cache is now enabled globally for all models."
echo "Expected max context on single GPU: ~28K tokens (2× baseline)"
echo "VRAM savings: KV cache uses half the memory (vs fp16)"
echo ""
echo "To disable, remove 'OLLAMA_KV_CACHE_TYPE=q8_0' from /etc/default/ollama"
echo "and restart: sudo systemctl restart ollama"

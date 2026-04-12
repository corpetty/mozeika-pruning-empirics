#!/bin/bash
# Force Ollama to GPU1 only by patching systemd service directly
# Run with: sudo bash force_gpu1_systemd.sh

set -e

echo "=== Forcing Ollama to GPU1 via systemd override ==="

# Create systemd override directory
mkdir -p /etc/systemd/system/ollama.service.d/

# Create override file with CUDA_VISIBLE_DEVICES=1
cat > /etc/systemd/system/ollama.service.d/gpu-override.conf <<'EOF'
[Service]
# Force Ollama to only see GPU1 (hide GPU0 for research)
Environment="CUDA_VISIBLE_DEVICES=1"
EOF

echo "✓ Created systemd override: /etc/systemd/system/ollama.service.d/gpu-override.conf"
cat /etc/systemd/system/ollama.service.d/gpu-override.conf

echo ""
echo "Reloading systemd and restarting Ollama..."
systemctl daemon-reload
systemctl stop ollama
sleep 5

# Verify GPUs are idle after stop
echo ""
echo "GPU state after stop:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo ""
echo "Starting Ollama with GPU1 only..."
systemctl start ollama
sleep 5

if systemctl is-active --quiet ollama; then
    echo "✓ Ollama started"
else
    echo "✗ Ollama failed to start"
    journalctl -u ollama -n 20
    exit 1
fi

echo ""
echo "Loading model..."
curl -s http://localhost:11434/api/generate -d '{
  "model": "gemma4:26b",
  "prompt": "Test",
  "stream": false,
  "options": {"num_predict": 5}
}' >/dev/null 2>&1 && echo "✓ Model loaded" || echo "✗ Load failed"

sleep 3

echo ""
echo "=== Final GPU State ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
  awk '{
    used_pct = ($2/$3)*100
    printf "GPU %s: %d / %d MiB (%.0f%%)", $1, $2, $3, used_pct
    if ($2 < 1000) printf " ✓ FREE FOR RESEARCH\n"
    else printf " (Ollama active)\n"
  }'

echo ""
nvidia-smi pmon -c 1 | grep -E "gpu|ollama"

echo ""
echo "=== Summary ==="
echo "Systemd override forces CUDA_VISIBLE_DEVICES=1 at service level."
echo "Expected: GPU0 <1GB (free), GPU1 ~22-24GB (Gemma4 + 8-bit KV)"
echo ""
echo "To verify GPU0 is free:"
echo "  CUDA_VISIBLE_DEVICES=0 python3 -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "To remove override:"
echo "  sudo rm /etc/systemd/system/ollama.service.d/gpu-override.conf"
echo "  sudo systemctl daemon-reload && sudo systemctl restart ollama"

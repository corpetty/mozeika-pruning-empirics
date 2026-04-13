#!/usr/bin/env bash
# Phase 9: SubRotQ Ollama Backend Injection
# Run this script from your terminal (not sandboxed Claude Code)
# Usage: bash phase9_ollama_inject.sh [step]
#   step 1: Discovery — find Ollama's llama.cpp backend
#   step 2: Build SubRotQ shared library
#   step 3: Inject into Ollama
#   step 4: Configure SubRotQ parameters
#   step 5: Test context scaling

set -euo pipefail
STEP="${1:-discover}"

LLAMA_SRC="/tmp/llama.cpp"
BASIS_FILE="/home/petty/pruning-research/kv-subspace/results/subrotq_basis_mistral7b_k128.bin"
OLLAMA_BIN="$(which ollama)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[Phase9]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err() { echo -e "${RED}[ERROR]${NC} $*"; }

########################################################################
# STEP 1: Discovery — Understand Ollama's llama.cpp integration
########################################################################
discover() {
    log "=== Step 1: Ollama Discovery ==="

    echo ""
    log "Ollama binary: $OLLAMA_BIN"
    file "$OLLAMA_BIN"
    echo ""

    # Check if Ollama uses shared libs
    log "Checking dynamic library dependencies..."
    ldd "$OLLAMA_BIN" 2>/dev/null || echo "Static binary or not ELF"
    echo ""

    # Modern Ollama uses runner executables, not shared libs
    log "Checking for Ollama runner executables..."
    OLLAMA_RUNNERS=""
    for dir in \
        /usr/lib/ollama/runners \
        /usr/local/lib/ollama/runners \
        "$HOME/.ollama/runners" \
        "$(dirname "$OLLAMA_BIN")/../lib/ollama/runners" \
        /usr/share/ollama/lib/ollama/runners; do
        if [ -d "$dir" ]; then
            OLLAMA_RUNNERS="$dir"
            log "Found runners at: $dir"
            ls -la "$dir"/
            echo ""
            # Check each runner for llama symbols
            for runner in "$dir"/*; do
                if [ -d "$runner" ]; then
                    log "Runner: $runner"
                    ls -la "$runner"/
                    # Check for shared libs inside runner dir
                    for f in "$runner"/*.so "$runner"/ollama_llama_server; do
                        if [ -f "$f" ]; then
                            log "  Found: $f ($(du -h "$f" | cut -f1))"
                            ldd "$f" 2>/dev/null | grep -i llam || true
                        fi
                    done
                fi
            done
            break
        fi
    done

    if [ -z "$OLLAMA_RUNNERS" ]; then
        warn "No runners directory found. Checking alternative locations..."
        # Maybe embedded in the binary
        strings "$OLLAMA_BIN" | grep -i "runner\|llama_server\|libllama" | head -20

        # Check /usr/lib/ollama directly
        if [ -d /usr/lib/ollama ]; then
            log "Found /usr/lib/ollama:"
            find /usr/lib/ollama -type f | head -30
        fi
    fi

    echo ""
    log "Checking Ollama service configuration..."
    systemctl cat ollama 2>/dev/null || warn "No systemd service found"
    echo ""

    log "Current Ollama process info..."
    ps aux | grep -i ollama | grep -v grep || warn "Ollama not running"
    echo ""

    log "GPU status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
    echo ""

    log "=== Discovery complete ==="
    echo ""
    echo "Next steps:"
    echo "  1. If Ollama uses runners with .so files → we can replace libllama.so"
    echo "  2. If Ollama uses static ollama_llama_server → we need to rebuild the runner"
    echo "  3. If Ollama is fully static → use LD_PRELOAD or fall back to direct llama-cli"
    echo ""
    echo "Run: bash phase9_ollama_inject.sh build"
}

########################################################################
# STEP 2: Build SubRotQ llama.cpp as shared library
########################################################################
build() {
    log "=== Step 2: Build SubRotQ Shared Library ==="

    if [ ! -d "$LLAMA_SRC" ]; then
        err "llama.cpp not found at $LLAMA_SRC"
        exit 1
    fi

    cd "$LLAMA_SRC"
    log "Building in $LLAMA_SRC/build-shared..."

    mkdir -p build-shared && cd build-shared

    # Build as shared library with CUDA on GPU0
    CUDA_VISIBLE_DEVICES=0 cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128

    CUDA_VISIBLE_DEVICES=0 cmake --build . -j$(nproc)

    echo ""
    log "Build artifacts:"
    find . -name "*.so*" -type f | while read f; do
        echo "  $f ($(du -h "$f" | cut -f1))"
    done
    find . -name "ollama_llama_server" -o -name "llama-server" -type f | while read f; do
        echo "  $f ($(du -h "$f" | cut -f1))"
    done

    # Also build the server executable (Ollama uses this)
    log "Checking for llama-server..."
    if [ -f bin/llama-server ]; then
        log "llama-server built successfully: bin/llama-server"
    else
        warn "llama-server not found in build. Ollama may need this."
    fi

    echo ""
    log "=== Build complete ==="
    echo "Run: bash phase9_ollama_inject.sh inject"
}

########################################################################
# STEP 3: Inject SubRotQ into Ollama
########################################################################
inject() {
    log "=== Step 3: Inject SubRotQ into Ollama ==="

    BUILD_DIR="$LLAMA_SRC/build-shared"
    if [ ! -d "$BUILD_DIR" ]; then
        err "Build directory not found. Run 'build' step first."
        exit 1
    fi

    # Strategy depends on discovery results
    # Try multiple injection strategies

    echo ""
    log "Strategy A: Replace runner shared libraries"
    echo "---"

    # Find Ollama's runner directory
    RUNNER_DIR=""
    for dir in \
        /usr/lib/ollama/runners \
        /usr/local/lib/ollama/runners \
        "$HOME/.ollama/runners" \
        "$(dirname "$OLLAMA_BIN")/../lib/ollama/runners"; do
        if [ -d "$dir" ]; then
            RUNNER_DIR="$dir"
            break
        fi
    done

    if [ -n "$RUNNER_DIR" ]; then
        log "Found runner directory: $RUNNER_DIR"

        # Find CUDA runner (usually named like cuda_v12 or metal)
        CUDA_RUNNER=$(find "$RUNNER_DIR" -maxdepth 1 -type d -name "*cuda*" | head -1)
        if [ -z "$CUDA_RUNNER" ]; then
            CUDA_RUNNER=$(find "$RUNNER_DIR" -maxdepth 1 -type d | head -1)
        fi

        if [ -n "$CUDA_RUNNER" ]; then
            log "Target runner: $CUDA_RUNNER"
            log "Current contents:"
            ls -la "$CUDA_RUNNER"/

            echo ""
            warn "About to backup and replace libraries in $CUDA_RUNNER"
            read -p "Continue? [y/N] " confirm
            if [ "$confirm" != "y" ]; then
                echo "Aborted."
                exit 0
            fi

            # Backup
            BACKUP_DIR="${CUDA_RUNNER}.backup.$(date +%Y%m%d_%H%M%S)"
            sudo cp -a "$CUDA_RUNNER" "$BACKUP_DIR"
            log "Backed up to: $BACKUP_DIR"

            # Replace libraries
            for lib in libllama.so libggml.so libggml-base.so libggml-cuda.so; do
                SRC=$(find "$BUILD_DIR" -name "$lib*" -type f | head -1)
                DST="$CUDA_RUNNER/$lib"
                if [ -f "$SRC" ] && [ -f "$DST" ]; then
                    log "Replacing $DST with $SRC"
                    sudo cp "$SRC" "$DST"
                elif [ -f "$SRC" ]; then
                    log "Adding $lib to runner (not previously present)"
                    sudo cp "$SRC" "$CUDA_RUNNER/"
                fi
            done

            log "Injection complete!"
        fi
    else
        warn "No runner directory found."
        echo ""
        log "Strategy B: LD_PRELOAD injection"
        echo "---"
        echo "If Ollama statically links llama.cpp, we can try LD_PRELOAD."
        echo "This will be configured in step 4 (configure)."
    fi

    echo ""
    log "=== Injection complete ==="
    echo "Run: bash phase9_ollama_inject.sh configure"
}

########################################################################
# STEP 4: Configure SubRotQ parameters
########################################################################
configure() {
    log "=== Step 4: Configure SubRotQ Parameters ==="

    # Create systemd override for Ollama
    OVERRIDE_DIR="/etc/systemd/system/ollama.service.d"
    OVERRIDE_FILE="$OVERRIDE_DIR/subrotq.conf"

    log "Creating systemd override at $OVERRIDE_FILE..."

    sudo mkdir -p "$OVERRIDE_DIR"

    # Check if LD_PRELOAD is needed
    BUILD_DIR="$LLAMA_SRC/build-shared"
    LIBLLAMA=$(find "$BUILD_DIR" -name "libllama.so" -type f | head -1)

    cat <<CONF | sudo tee "$OVERRIDE_FILE"
[Service]
# GPU assignment: Ollama on GPU1
Environment="CUDA_VISIBLE_DEVICES=1"

# SubRotQ configuration
Environment="LLAMA_SUBROTQ_ENABLED=1"
Environment="LLAMA_SUBROTQ_RANK=128"
Environment="LLAMA_SUBROTQ_BITS=4"
Environment="LLAMA_SUBROTQ_BASIS=$BASIS_FILE"

# LD_PRELOAD fallback (uncomment if runner replacement didn't work)
# Environment="LD_PRELOAD=$LIBLLAMA"

# Increase context limit
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
CONF

    log "Override created:"
    cat "$OVERRIDE_FILE"

    echo ""
    log "Reloading systemd..."
    sudo systemctl daemon-reload

    echo ""
    log "=== Configuration complete ==="
    echo ""
    echo "Run: bash phase9_ollama_inject.sh test"
}

########################################################################
# STEP 5: Test context scaling
########################################################################
test_scaling() {
    log "=== Step 5: Test SubRotQ Context Scaling ==="

    warn "This will restart Ollama. Current sessions will be interrupted."
    read -p "Continue? [y/N] " confirm
    if [ "$confirm" != "y" ]; then
        echo "Aborted."
        exit 0
    fi

    log "Restarting Ollama..."
    sudo systemctl restart ollama
    sleep 3

    # Check it started
    if ! systemctl is-active --quiet ollama; then
        err "Ollama failed to start!"
        sudo journalctl -u ollama --no-pager -n 30
        echo ""
        err "Check logs above. To restore:"
        echo "  sudo rm /etc/systemd/system/ollama.service.d/subrotq.conf"
        echo "  sudo systemctl daemon-reload && sudo systemctl restart ollama"
        exit 1
    fi
    log "Ollama is running."

    # Check for SubRotQ in logs
    log "Checking for SubRotQ activation in logs..."
    sudo journalctl -u ollama --no-pager -n 50 | grep -i "subrotq" || warn "No SubRotQ messages in log"
    echo ""

    # Quick sanity test
    log "Quick generation test (default context)..."
    RESULT=$(curl -s http://localhost:11434/api/generate -d '{
        "model": "gemma4:26b",
        "prompt": "What is 2+2? Answer in one word.",
        "stream": false,
        "options": {"num_ctx": 4096}
    }' 2>&1)
    echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('response','ERROR')[:200])" 2>/dev/null || echo "$RESULT" | head -5
    echo ""

    # Context scaling tests
    log "Context scaling tests..."
    echo "ctx_length,vram_before_mb,vram_after_mb,success,time_ms" > /tmp/subrotq_scaling.csv

    for ctx in 8192 16384 32768 49152 65536 81920; do
        log "Testing context length: $ctx..."

        # Record VRAM before
        VRAM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')

        # Generate with specified context
        # Use a simple repeating prompt to fill context
        PROMPT="Repeat after me and continue the pattern: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20. Now count from 1 to 100 slowly:"

        START=$(date +%s%N)
        RESPONSE=$(curl -s --max-time 120 http://localhost:11434/api/generate -d "{
            \"model\": \"gemma4:26b\",
            \"prompt\": \"$PROMPT\",
            \"stream\": false,
            \"options\": {\"num_ctx\": $ctx, \"num_predict\": 50}
        }" 2>&1)
        END=$(date +%s%N)
        ELAPSED=$(( (END - START) / 1000000 ))

        # Record VRAM after
        VRAM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')

        # Check success
        SUCCESS="false"
        RESP_TEXT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('response','')[:100])" 2>/dev/null || echo "")
        if [ -n "$RESP_TEXT" ] && [ "$RESP_TEXT" != "" ]; then
            SUCCESS="true"
        fi

        echo "$ctx,$VRAM_BEFORE,$VRAM_AFTER,$SUCCESS,$ELAPSED" >> /tmp/subrotq_scaling.csv
        log "  ctx=$ctx | VRAM: ${VRAM_BEFORE}→${VRAM_AFTER} MB | success=$SUCCESS | ${ELAPSED}ms"

        if [ "$SUCCESS" = "false" ]; then
            warn "  Failed at ctx=$ctx. Response: $(echo "$RESPONSE" | head -3)"
            # If OOM, stop testing larger contexts
            if echo "$RESPONSE" | grep -qi "out of memory\|OOM\|CUDA error"; then
                err "  OOM detected. Stopping scaling test."
                break
            fi
        fi

        # Cool down
        sleep 2
    done

    echo ""
    log "=== Scaling Results ==="
    column -t -s',' /tmp/subrotq_scaling.csv

    # Copy results
    cp /tmp/subrotq_scaling.csv /home/petty/pruning-research/kv-subspace/results/phase9_scaling.csv
    log "Results saved to results/phase9_scaling.csv"

    echo ""
    log "GPU status:"
    nvidia-smi
}

########################################################################
# Restore: Undo injection
########################################################################
restore() {
    log "=== Restoring Ollama to Original State ==="

    # Find backup
    RUNNER_DIR=""
    for dir in \
        /usr/lib/ollama/runners \
        /usr/local/lib/ollama/runners \
        "$HOME/.ollama/runners"; do
        if [ -d "$dir" ]; then
            RUNNER_DIR="$dir"
            break
        fi
    done

    if [ -n "$RUNNER_DIR" ]; then
        BACKUP=$(ls -d "${RUNNER_DIR}"/*cuda*.backup.* 2>/dev/null | sort -r | head -1)
        if [ -n "$BACKUP" ]; then
            CUDA_RUNNER=$(echo "$BACKUP" | sed 's/.backup.[0-9_]*//')
            log "Restoring $CUDA_RUNNER from $BACKUP"
            sudo rm -rf "$CUDA_RUNNER"
            sudo mv "$BACKUP" "$CUDA_RUNNER"
        fi
    fi

    # Remove override
    if [ -f /etc/systemd/system/ollama.service.d/subrotq.conf ]; then
        log "Removing SubRotQ systemd override..."
        sudo rm /etc/systemd/system/ollama.service.d/subrotq.conf
    fi

    sudo systemctl daemon-reload
    sudo systemctl restart ollama

    log "Ollama restored to original state."
}

########################################################################
# Fallback: Direct llama-cli test (skip Ollama)
########################################################################
fallback() {
    log "=== Fallback: Direct llama-cli Testing ==="
    log "Skipping Ollama injection, testing SubRotQ directly with a GGUF model."
    echo ""

    # Check for any available GGUF files
    log "Looking for GGUF files..."
    find /tmp -name "*.gguf" -type f 2>/dev/null | while read f; do
        echo "  $f ($(du -h "$f" | cut -f1))"
    done
    find /home/petty -name "*.gguf" -type f 2>/dev/null | head -5 | while read f; do
        echo "  $f ($(du -h "$f" | cut -f1))"
    done

    LLAMA_CLI="$LLAMA_SRC/build/bin/llama-cli"
    if [ ! -f "$LLAMA_CLI" ]; then
        err "llama-cli not found at $LLAMA_CLI"
        exit 1
    fi

    echo ""
    log "To download Mistral-7B GGUF for testing:"
    echo "  huggingface-cli download TheBloke/Mistral-7B-v0.1-GGUF mistral-7b-v0.1.Q4_K_M.gguf --local-dir /tmp/"
    echo ""
    log "Then test SubRotQ:"
    echo "  CUDA_VISIBLE_DEVICES=0 $LLAMA_CLI \\"
    echo "    -m /tmp/mistral-7b-v0.1.Q4_K_M.gguf \\"
    echo "    --subrotq --subrotq-rank 128 --subrotq-bits 4 \\"
    echo "    --subrotq-basis $BASIS_FILE \\"
    echo "    -c 32768 -n 100 \\"
    echo "    -p 'Explain quantum entanglement:'"
}

########################################################################
# Main dispatcher
########################################################################
case "${STEP}" in
    discover|1)  discover ;;
    build|2)     build ;;
    inject|3)    inject ;;
    configure|4) configure ;;
    test|5)      test_scaling ;;
    restore)     restore ;;
    fallback)    fallback ;;
    *)
        echo "Usage: $0 {discover|build|inject|configure|test|restore|fallback}"
        echo ""
        echo "Steps (run in order):"
        echo "  discover  — Find Ollama's llama.cpp backend structure"
        echo "  build     — Build SubRotQ llama.cpp as shared library"
        echo "  inject    — Replace Ollama's backend with SubRotQ version"
        echo "  configure — Add SubRotQ env vars to Ollama service"
        echo "  test      — Restart Ollama and test context scaling"
        echo ""
        echo "Utilities:"
        echo "  restore   — Undo injection, restore original Ollama"
        echo "  fallback  — Skip Ollama, test with llama-cli directly"
        ;;
esac

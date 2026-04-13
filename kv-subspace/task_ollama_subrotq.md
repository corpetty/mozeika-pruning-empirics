# Phase 9: SubRotQ Ollama Injection — Execution Guide

## Status: Script Ready (`phase9_ollama_inject.sh`)

## Important Architecture Note

Modern Ollama (v0.3+) does **not** use llama.cpp as a shared library in the way originally assumed. Instead, it bundles **runner executables** — typically found at:
- `/usr/lib/ollama/runners/cuda_v12/` (system install)
- `~/.ollama/runners/` (user install)

Each runner directory contains:
- `ollama_llama_server` — the main runner binary
- `libllama.so`, `libggml*.so` — shared libraries loaded by the runner

This means **library replacement should work** if we find the right runner directory.

## Execution Steps

Run from terminal (not sandboxed Claude Code):

```bash
cd ~/pruning-research/kv-subspace

# Step 1: Discover Ollama's backend structure (READ-ONLY)
bash phase9_ollama_inject.sh discover

# Step 2: Build SubRotQ as shared library (GPU0)
bash phase9_ollama_inject.sh build

# Step 3: Replace Ollama's libraries (REQUIRES SUDO, makes backups)
bash phase9_ollama_inject.sh inject

# Step 4: Configure SubRotQ env vars in systemd
bash phase9_ollama_inject.sh configure

# Step 5: Restart Ollama and test context scaling
bash phase9_ollama_inject.sh test
```

## Recovery

```bash
bash phase9_ollama_inject.sh restore
```

## Fallback (if Ollama injection fails)

```bash
bash phase9_ollama_inject.sh fallback
```

Downloads Mistral-7B GGUF and tests SubRotQ directly with llama-cli.

## Expected Results

| Context | Without SubRotQ | With SubRotQ (k=128/4-bit) |
|---------|----------------|---------------------------|
| 32K     | ~20.6 GB       | ~18 GB (baseline)         |
| 49K     | OOM             | ~20 GB                    |
| 65K     | OOM             | ~22 GB                    |
| 81K     | OOM             | ~24 GB (near limit)       |

## Key Risks

1. **ABI mismatch** — Our llama.cpp fork may have different symbols than Ollama's bundled version. If Ollama crashes on load, rebase SubRotQ patches onto Ollama's exact llama.cpp version.
2. **Gemma4 tensor mismatch** — Already documented (expected 1014 tensors, got 658). The GGUF format issue persists even with injection.
3. **SubRotQ env vars not read** — If llama.cpp only supports CLI flags (not env vars), SubRotQ won't activate. Pre-check:
   ```bash
   grep -r "LLAMA_SUBROTQ" /tmp/llama.cpp/src/ /tmp/llama.cpp/common/
   ```
   If missing, either patch llama.cpp to read env vars or patch Ollama's Go code to pass `--subrotq` flags.

## Constraints
- GPU0 for builds, GPU1 for Ollama — DO NOT CHANGE
- Backup all Ollama files before replacement
- Never stop Ollama without permission

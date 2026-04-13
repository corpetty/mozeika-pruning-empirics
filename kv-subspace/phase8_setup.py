#!/usr/bin/env python3
"""Phase 8: Gemma4-26B GGUF extraction and SubRotQ setup verification."""
import os
import shutil
import struct
import subprocess

GGUF_SRC = "/usr/share/ollama/.ollama/models/blobs/sha256-7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df"
GGUF_DST = "/tmp/gemma4-26b.gguf"
LLAMA_CLI = "/tmp/llama.cpp/build/bin/llama-cli"
BASIS = os.path.join(os.path.dirname(__file__), "results", "subrotq_basis_mistral7b_k128.bin")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Step 1: Copy GGUF
print("=== Step 1: Extract GGUF from Ollama ===")
if not os.path.exists(GGUF_DST):
    src_size = os.path.getsize(GGUF_SRC)
    print(f"Source: {GGUF_SRC}")
    print(f"Size: {src_size / (1024**3):.1f} GB")
    print("Copying (this may take a minute)...")
    shutil.copy2(GGUF_SRC, GGUF_DST)
    print(f"Copied to {GGUF_DST}")
else:
    dst_size = os.path.getsize(GGUF_DST)
    print(f"GGUF already exists at {GGUF_DST} ({dst_size / (1024**3):.1f} GB)")

# Step 2: Verify GGUF magic
print("\n=== Step 2: Verify GGUF ===")
with open(GGUF_DST, "rb") as f:
    magic = f.read(4)
    print(f"Magic bytes: {magic} (hex: {magic.hex()})")
    if magic == b"GGUF":
        print("Valid GGUF file!")
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", f.read(8))[0]
        print(f"GGUF version: {version}")
        print(f"Tensor count: {tensor_count}")
        print(f"Metadata KV count: {metadata_kv_count}")
    else:
        print("WARNING: Not a valid GGUF file!")

# Step 3: Check basis
print("\n=== Step 3: Check basis file ===")
if os.path.exists(BASIS):
    bsize = os.path.getsize(BASIS)
    print(f"Basis: {BASIS} ({bsize / (1024**2):.1f} MB)")
else:
    print(f"ERROR: Basis not found at {BASIS}")

# Step 4: Check llama-cli
print("\n=== Step 4: Check llama-cli ===")
if os.path.exists(LLAMA_CLI):
    print(f"llama-cli found at {LLAMA_CLI}")
    # Check if it has subrotq support
    result = subprocess.run([LLAMA_CLI, "--help"], capture_output=True, text=True)
    if "subrotq" in result.stdout.lower() or "subrotq" in result.stderr.lower():
        print("SubRotQ flags found in help output!")
    else:
        print("WARNING: SubRotQ flags NOT found in help (may still work)")
else:
    print(f"ERROR: llama-cli not found at {LLAMA_CLI}")

print("\n=== Setup Complete ===")
print(f"GGUF: {GGUF_DST}")
print(f"Basis: {BASIS}")
print(f"CLI: {LLAMA_CLI}")
print(f"\nTo run SubRotQ test:")
print(f"  CUDA_VISIBLE_DEVICES=0 {LLAMA_CLI} \\")
print(f'    -m {GGUF_DST} \\')
print(f'    -p "Write a detailed explanation of quantum entanglement" \\')
print(f'    -n 100 --subrotq --subrotq-rank 128 --subrotq-bits 4 \\')
print(f'    --subrotq-basis {BASIS}')

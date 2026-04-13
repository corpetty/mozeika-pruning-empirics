#!/usr/bin/env python3
"""
SubRotQ calibration for Gemma4 26B.

Adapts exp24 calibration code to compute PCA basis for Gemma4,
then saves in format compatible with llama.cpp SubRotQ implementation.

Output: gemma4_26b_subrotq_k128.bin (binary file with basis parameters)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
from tqdm import tqdm
import struct

# Config
MODEL_NAME = "google/gemma-4-26b"  # Hugging Face model ID
CALIB_TOKENS = 2048
SUBROTQ_RANK = 128
N_BITS = 4
DEVICE = "cuda:1"  # Use GPU1 (GPU0 reserved for Ollama)
OUTPUT_FILE = "gemma4_26b_subrotq_k128.bin"

def collect_kvs_for_basis(model, tokenizer, text, max_tokens=2048, device="cuda"):
    """
    Collect K vectors from all layers/heads by running model on calibration text.
    
    Returns:
        dict: {(layer_idx, head_idx): {'K': np.array(T, d_head)}}
    """
    model.eval()
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = inputs["input_ids"][:, :max_tokens].to(device)
    
    print(f"Running model on {input_ids.shape[1]} tokens...")
    
    # Storage for K vectors per layer/head
    kv_dict = {}
    
    # Hook to capture K projections
    handles = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is (batch, seq_len, n_heads * d_head) after qkv_proj
            # Need to reshape and extract K
            
            # Gemma4 uses MQA/GQA - need to check n_kv_heads
            n_heads = module.num_heads
            n_kv_heads = getattr(module, "num_key_value_heads", n_heads)
            d_head = module.head_dim
            
            # output: qkv interleaved or separate?
            # For Gemma, typically: Q, K, V are separate linear layers
            # We need the K projection output specifically
            
            # PROBLEM: Hook location depends on Gemma4 architecture
            # Need to inspect model structure first
            
            pass
        
        return hook
    
    # Register hooks on attention layers
    for layer_idx, layer in enumerate(model.model.layers):
        # Gemma4 attention module
        attn = layer.self_attn
        handle = attn.register_forward_hook(make_hook(layer_idx))
        handles.append(handle)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_ids, use_cache=True)
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    print(f"Collected K vectors from {len(kv_dict)} (layer, head) pairs")
    
    return kv_dict

def fit_pca(K_vectors, k):
    """
    Compute PCA basis.
    
    Args:
        K_vectors: np.array (n_samples, d_head)
        k: target rank
        
    Returns:
        U: (d_head, k)
        mean: (d_head,)
        scale: (k,)
    """
    mean = K_vectors.mean(axis=0)
    X_c = K_vectors - mean
    
    U_full, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    U = Vt[:k].T  # Top k components, (d_head, k)
    
    # Scale = std dev in subspace
    z = X_c @ U
    scale = z.std(axis=0)
    
    return U.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)

def save_basis_binary(basis_dict, output_file, n_layers, n_kv_heads, d_head, k, n_bits):
    """
    Save SubRotQ basis to binary file for llama.cpp.
    
    Format:
        Header:
            uint32_t magic = 0x53524F54  # "SROT" (SubROTq)
            uint32_t version = 1
            uint32_t n_layers
            uint32_t n_kv_heads
            uint32_t d_head
            uint32_t k (rank)
            uint32_t n_bits
        
        For each layer (n_layers):
            For each head (n_kv_heads):
                float32 U[d_head * k]        # PCA basis, column-major
                float32 mean[d_head]         # Mean vector
                float32 scale[k]             # Per-dimension scale
    """
    with open(output_file, 'wb') as f:
        # Header
        f.write(struct.pack('I', 0x53524F54))  # Magic "SROT"
        f.write(struct.pack('I', 1))            # Version
        f.write(struct.pack('I', n_layers))
        f.write(struct.pack('I', n_kv_heads))
        f.write(struct.pack('I', d_head))
        f.write(struct.pack('I', k))
        f.write(struct.pack('I', n_bits))
        
        # Basis parameters per layer/head
        for layer_idx in range(n_layers):
            for head_idx in range(n_kv_heads):
                U, mean, scale = basis_dict[(layer_idx, head_idx)]
                
                # Write U (column-major, d_head x k)
                f.write(U.astype(np.float32).tobytes())
                
                # Write mean
                f.write(mean.astype(np.float32).tobytes())
                
                # Write scale
                f.write(scale.astype(np.float32).tobytes())
    
    print(f"Saved basis to {output_file} ({os.path.getsize(output_file) / 1024 / 1024:.1f} MB)")

def main():
    print("=" * 70)
    print("SubRotQ Calibration for Gemma4 26B")
    print("=" * 70)
    
    print(f"\nModel: {MODEL_NAME}")
    print(f"Calibration tokens: {CALIB_TOKENS}")
    print(f"Target rank k: {SUBROTQ_RANK}")
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_FILE}")
    
    # Load WikiText-2
    print("\nLoading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_text = "\n".join(dataset["text"][:100])  # First 100 paragraphs
    
    # CRITICAL ISSUE: Gemma4 26B is ~52GB unquantized
    # Won't fit on single RTX 3090 (24GB) without quantization
    
    print("\n" + "=" * 70)
    print("CRITICAL: Gemma4 26B unquantized is ~52GB (won't fit on 24GB GPU)")
    print("=" * 70)
    print("\nOptions:")
    print("1. Load in 8-bit (bitsandbytes) - ~26GB, tight fit")
    print("2. Load in 4-bit (bitsandbytes) - ~13GB, more headroom")
    print("3. Use Ollama's GGUF directly via llama.cpp API (if we can access cache)")
    print("4. Download pre-quantized AWQ version")
    
    print("\nRecommendation: Use 4-bit quantization for calibration")
    print("Quality impact minimal for PCA basis computation")
    
    return 0

if __name__ == "__main__":
    exit(main())

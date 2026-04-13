#!/usr/bin/env python3
"""
SubRotQ Basis Calibration Script

Generates PCA basis for K-cache compression from calibration data.
Saves basis in binary format compatible with llama.cpp SubRotQ implementation.

Usage:
    python calibrate_subrotq_basis.py \
        --model mistralai/Mistral-7B-v0.3 \
        --rank 128 \
        --bits 4 \
        --calib-tokens 2048 \
        --output results/subrotq_basis_mistral7b_k128.bin

Output format (.subrotq binary):
    Header (64 bytes):
        - magic: uint32 (0x53524F51 = "SROQ")
        - version: uint32 (1)
        - n_layers: uint32
        - n_kv_heads: uint32
        - d_head: uint32
        - rank: uint32
        - n_bits: uint32
        - reserved: uint32[9]
    
    Per layer (n_layers):
        Per head (n_kv_heads):
            - U: float32[d_head × rank] (PCA basis, column-major)
            - mean: float32[d_head] (centering vector)
            - scale: float32[rank] (per-dimension scale for quantization)
"""

import argparse
import struct
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def collect_kvs_for_basis(model, tokenizer, text, max_tokens=2048, device="cuda"):
    """
    Collect K vectors from all layers/heads for PCA calibration.
    
    Returns:
        dict: {(layer_idx, head_idx): {'K': np.array(T, d_head)}}
    """
    model.eval()
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = inputs["input_ids"].to(device)
    
    if input_ids.shape[1] > max_tokens:
        input_ids = input_ids[:, :max_tokens]
    
    print(f"Calibration tokens: {input_ids.shape[1]}")
    
    # Run forward pass to populate KV cache
    with torch.no_grad():
        output = model(input_ids, use_cache=True, return_dict=True)
    
    # Extract K vectors from DynamicCache
    kv_dict = {}
    past_kv = output.past_key_values
    
    if hasattr(past_kv, 'layers'):
        # DynamicCache format (modern transformers) - has .layers list of DynamicLayer
        n_layers = len(past_kv.layers)
        for layer_idx in range(n_layers):
            layer = past_kv.layers[layer_idx]
            k_state = layer.keys  # (B, n_kv_heads, T, d_head)
            
            # Move to CPU and convert to numpy
            k_np = k_state[0].detach().cpu().numpy()  # (n_kv_heads, T, d_head)
            
            # Store per head
            n_kv_heads, T, d_head = k_np.shape
            for head_idx in range(n_kv_heads):
                key = (layer_idx, head_idx)
                kv_dict[key] = {
                    'K': k_np[head_idx]  # (T, d_head)
                }
    elif isinstance(past_kv, tuple):
        # Tuple format (older transformers)
        n_layers = len(past_kv)
        for layer_idx in range(n_layers):
            if isinstance(past_kv[layer_idx], tuple):
                k_state = past_kv[layer_idx][0]  # (B, n_kv_heads, T, d_head)
                
                # Move to CPU and convert to numpy
                k_np = k_state[0].detach().cpu().numpy()  # (n_kv_heads, T, d_head)
                
                # Store per head
                n_kv_heads, T, d_head = k_np.shape
                for head_idx in range(n_kv_heads):
                    key = (layer_idx, head_idx)
                    kv_dict[key] = {
                        'K': k_np[head_idx]  # (T, d_head)
                    }
    
    return kv_dict


def fit_pca(X, k):
    """
    Fit PCA via SVD on centered data matrix X.
    
    Args:
        X: np.array (n_samples, d) - data matrix
        k: int - target rank
    
    Returns:
        U: np.array (d, k) - PCA basis (top k right singular vectors)
        mean: np.array (d,) - centering vector
        explained_var: float - fraction of variance explained by top k components
    """
    # Convert to float32 for SVD (float16 not supported)
    X = X.astype(np.float32)
    
    # Center data
    mean = X.mean(axis=0)  # (d,)
    X_centered = X - mean  # (n, d)
    
    # SVD: X_centered = U @ S @ Vt
    # We want V (right singular vectors) which are the principal components
    U_left, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # V are the columns we want (principal directions in feature space)
    V = Vt.T  # (d, min(n, d))
    
    # Take top k components
    U_k = V[:, :k]  # (d, k)
    
    # Compute explained variance
    total_var = np.sum(S ** 2)
    explained_var = np.sum(S[:k] ** 2) / total_var if total_var > 0 else 0.0
    
    return U_k, mean, explained_var


def compute_quantization_scale(X, U_k, mean, n_bits=4):
    """
    Compute per-dimension scale factors for quantization.
    
    Projects calibration data onto basis and computes scale such that
    ±3σ maps to the quantization range [0, 2^n_bits - 1].
    
    Args:
        X: np.array (n_samples, d) - calibration data
        U_k: np.array (d, k) - PCA basis
        mean: np.array (d,) - centering vector
        n_bits: int - quantization bit depth
    
    Returns:
        scale: np.array (k,) - per-dimension scale factors
    """
    # Center and project
    X_centered = X - mean  # (n, d)
    Z = X_centered @ U_k  # (n, k)
    
    # Compute per-dimension std dev
    std_per_dim = np.std(Z, axis=0)  # (k,)
    
    # Scale such that ±3σ covers quantization range
    qmax = (1 << n_bits) - 1  # 2^n_bits - 1
    scale = (6.0 * std_per_dim) / qmax  # maps [-3σ, 3σ] -> [0, qmax]
    
    # Avoid division by zero
    scale = np.maximum(scale, 1e-6)
    
    return scale


def save_basis_binary(filepath, basis_dict, n_layers, n_kv_heads, d_head, rank, n_bits):
    """
    Save SubRotQ basis to binary file.
    
    Args:
        filepath: str - output path
        basis_dict: dict - {(layer_idx, head_idx): {'U': (d,k), 'mean': (d,), 'scale': (k,)}}
        n_layers: int
        n_kv_heads: int
        d_head: int
        rank: int
        n_bits: int
    """
    with open(filepath, 'wb') as f:
        # Header (64 bytes)
        magic = 0x53524F51  # "SROQ"
        version = 1
        
        header = struct.pack(
            'I' * 16,  # 16 uint32 = 64 bytes
            magic, version, n_layers, n_kv_heads, d_head, rank, n_bits,
            0, 0, 0, 0, 0, 0, 0, 0, 0  # reserved
        )
        f.write(header)
        
        # Per layer, per head
        for layer_idx in range(n_layers):
            for head_idx in range(n_kv_heads):
                key = (layer_idx, head_idx)
                
                if key not in basis_dict:
                    raise ValueError(f"Missing basis for layer {layer_idx}, head {head_idx}")
                
                U = basis_dict[key]['U']  # (d_head, rank)
                mean = basis_dict[key]['mean']  # (d_head,)
                scale = basis_dict[key]['scale']  # (rank,)
                
                # Validate shapes
                assert U.shape == (d_head, rank), f"U shape mismatch at {key}: {U.shape}"
                assert mean.shape == (d_head,), f"mean shape mismatch at {key}: {mean.shape}"
                assert scale.shape == (rank,), f"scale shape mismatch at {key}: {scale.shape}"
                
                # Write in column-major order (Fortran order) for GPU efficiency
                U_bytes = U.astype(np.float32).tobytes(order='F')
                mean_bytes = mean.astype(np.float32).tobytes()
                scale_bytes = scale.astype(np.float32).tobytes()
                
                f.write(U_bytes)
                f.write(mean_bytes)
                f.write(scale_bytes)
    
    print(f"Saved basis to {filepath}")
    print(f"  File size: {Path(filepath).stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="SubRotQ basis calibration")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--rank", type=int, default=128, help="PCA rank")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bit depth")
    parser.add_argument("--calib-tokens", type=int, default=2048, help="Calibration sequence length")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Calibration dataset")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split")
    parser.add_argument("--output", type=str, required=True, help="Output .bin file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    print(f"=== SubRotQ Basis Calibration ===")
    print(f"Model: {args.model}")
    print(f"Rank: {args.rank}")
    print(f"Bits: {args.bits}")
    print(f"Calibration tokens: {args.calib_tokens}")
    print(f"Device: {args.device}")
    print()
    
    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    
    # Get model architecture info
    config = model.config
    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    d_head = config.hidden_size // config.num_attention_heads
    
    print(f"Architecture: {n_layers} layers, {n_kv_heads} KV heads, d_head={d_head}")
    print()
    
    # Load calibration data
    print("Loading calibration dataset...")
    if args.dataset == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.dataset_split)
        # Concatenate first few articles
        calib_text = "\n\n".join(dataset["text"][:100])
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"Calibration text length: {len(calib_text)} chars")
    print()
    
    # Collect K vectors
    print("Collecting K vectors from calibration data...")
    kv_dict = collect_kvs_for_basis(
        model, tokenizer, calib_text,
        max_tokens=args.calib_tokens,
        device=args.device
    )
    
    print(f"Collected K vectors for {len(kv_dict)} layer-head pairs")
    print()
    
    # Fit PCA per layer-head
    print("Computing PCA basis per layer-head...")
    basis_dict = {}
    
    for (layer_idx, head_idx), data in tqdm(kv_dict.items(), desc="PCA"):
        K = data['K']  # (T, d_head)
        
        # Fit PCA
        U_k, mean, explained_var = fit_pca(K, args.rank)
        
        # Compute quantization scale
        scale = compute_quantization_scale(K, U_k, mean, n_bits=args.bits)
        
        basis_dict[(layer_idx, head_idx)] = {
            'U': U_k,  # (d_head, rank)
            'mean': mean,  # (d_head,)
            'scale': scale,  # (rank,)
            'explained_var': explained_var
        }
    
    # Report statistics
    explained_vars = [v['explained_var'] for v in basis_dict.values()]
    print(f"Explained variance: mean={np.mean(explained_vars):.3f}, "
          f"min={np.min(explained_vars):.3f}, max={np.max(explained_vars):.3f}")
    print()
    
    # Save to binary file
    print("Saving basis to binary file...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_basis_binary(
        args.output, basis_dict,
        n_layers, n_kv_heads, d_head, args.rank, args.bits
    )
    
    print("\n✅ Calibration complete!")


if __name__ == "__main__":
    main()

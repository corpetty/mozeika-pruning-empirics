#!/usr/bin/env python3
"""
Generate PCA basis for Gemma4-E4B using WikiText-2 calibration data.

Standalone script — does NOT depend on collect.py (avoids AWQ/vLLM conflicts).
Uses compress.py only for fit_pca().

Requirements:
    pip install transformers>=4.49 torch datasets numpy tqdm

    For Gemma4 support you need transformers with the Gemma4 model class.
    If your installed version doesn't have it, install from main:
        pip install git+https://github.com/huggingface/transformers.git

Usage:
    python scripts/generate_gemma4_basis.py
    python scripts/generate_gemma4_basis.py --num-samples 100 --k 64
    python scripts/generate_gemma4_basis.py --model google/gemma-4-E4B-it --output results/gemma4_e4b_pca_basis_k128.npz

Isolation (if transformers version conflicts with vLLM):
    python -m venv .venv-gemma4
    source .venv-gemma4/bin/activate
    pip install torch transformers>=4.49 datasets numpy tqdm
    python scripts/generate_gemma4_basis.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import argparse
import gc

from compress import fit_pca


def load_model_standalone(model_name, device, dtype=torch.bfloat16):
    """Load model and tokenizer directly via transformers (no AWQ dependency)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"  Loading model: {model_name} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"":0},  # Force GPU0 only (GPU1 used by Ollama)
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def extract_kv_by_layer(past_key_values):
    """
    Extract (K, V) tensors per layer from past_key_values.

    Handles:
      - DynamicCache with .key_cache/.value_cache lists
      - DynamicCache with .layers list (modern transformers)
      - HybridCache (multimodal models like Gemma4) — skips None entries
      - Legacy tuple-of-tuples format

    Returns:
        list of (layer_idx, K_tensor, V_tensor) — only non-None text layers.
    """
    results = []

    if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
        # DynamicCache / HybridCache — iterate by index, skip None entries
        for i in range(len(past_key_values.key_cache)):
            k = past_key_values.key_cache[i]
            v = past_key_values.value_cache[i]
            if k is not None and v is not None:
                results.append((i, k, v))
    elif hasattr(past_key_values, 'layers'):
        # DynamicCache with .layers attribute
        for i, layer in enumerate(past_key_values.layers):
            k = layer.keys if hasattr(layer, 'keys') else None
            v = layer.values if hasattr(layer, 'values') else None
            if k is not None and v is not None:
                results.append((i, k, v))
    elif isinstance(past_key_values, (tuple, list)):
        # Legacy tuple-of-tuples: each entry is (K, V) or (K, V, ...)
        for i, entry in enumerate(past_key_values):
            if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                results.append((i, entry[0], entry[1]))
    else:
        raise TypeError(
            f"Unsupported past_key_values type: {type(past_key_values)}. "
            f"Attributes: {[a for a in dir(past_key_values) if not a.startswith('_')]}"
        )

    return results


def collect_kvs_for_basis(model, tokenizer, calibration_texts, max_tokens=512, device='cuda'):
    """
    Collect K/V vectors from calibration data for PCA basis fitting.

    Returns:
        dict: {(layer_idx, head_idx): {'K': np.array(T, d_head), 'V': np.array(T, d_head)}}
    """
    kv_storage = {}

    print(f"Collecting KVs from {len(calibration_texts)} calibration samples...")

    for text in tqdm(calibration_texts, desc="Collecting"):
        inputs = tokenizer(text, return_tensors='pt', max_length=max_tokens,
                          truncation=True, padding=False).to(device)

        if inputs['input_ids'].shape[1] < 4:
            continue  # skip very short texts

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_hidden_states=False)
            kv_layers = extract_kv_by_layer(outputs.past_key_values)

            for layer_idx, K, V in kv_layers:
                # K, V shape: (batch=1, num_kv_heads, seq_len, d_head)
                K_np = K.squeeze(0).float().cpu().numpy()  # (num_kv_heads, seq_len, d_head)
                V_np = V.squeeze(0).float().cpu().numpy()

                num_heads, seq_len, d_head = K_np.shape

                for head_idx in range(num_heads):
                    key = (layer_idx, head_idx)
                    if key not in kv_storage:
                        kv_storage[key] = {'K': [], 'V': []}

                    kv_storage[key]['K'].append(K_np[head_idx])  # (seq_len, d_head)
                    kv_storage[key]['V'].append(V_np[head_idx])

        # Free GPU cache periodically
        del outputs
        torch.cuda.empty_cache()

    # Concatenate across all samples
    print("Concatenating collected vectors...")
    for key in tqdm(list(kv_storage.keys()), desc="Concatenating"):
        kv_storage[key]['K'] = np.concatenate(kv_storage[key]['K'], axis=0)  # (T_total, d_head)
        kv_storage[key]['V'] = np.concatenate(kv_storage[key]['V'], axis=0)

    return kv_storage


def compute_explained_variance(X, U_k, mean):
    """Compute fraction of variance explained by top-k PCA components."""
    Xc = X - mean
    total_var = np.sum(Xc ** 2)
    if total_var == 0:
        return 1.0
    projected = Xc @ U_k  # (N, k)
    reconstructed = projected @ U_k.T  # (N, d)
    explained_var = np.sum(reconstructed ** 2)
    return float(explained_var / total_var)


def main():
    parser = argparse.ArgumentParser(
        description="Generate PCA basis for SubRotQ KV compression")
    parser.add_argument('--model', type=str, default='google/gemma-4-E4B-it',
                       help='HuggingFace model name')
    parser.add_argument('--output', type=str, default='results/gemma4_e4b_pca_basis_k128.npz',
                       help='Output NPZ file path')
    parser.add_argument('--k', type=int, default=128,
                       help='PCA rank (subspace dimension)')
    parser.add_argument('--num-samples', type=int, default=500,
                       help='Number of WikiText-2 samples to use')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Max tokens per sample')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--skip-v', action='store_true',
                       help='Skip V basis (SubRotQCache only uses K)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                       choices=['float16', 'bfloat16', 'float32'],
                       help='Model dtype')

    args = parser.parse_args()
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    model_dtype = dtype_map[args.dtype]

    print("=" * 70)
    print("Gemma4-E4B PCA Basis Generation (standalone)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"PCA rank k: {args.k}")
    print(f"Calibration: {args.num_samples} samples x {args.max_tokens} tokens")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"Skip V basis: {args.skip_v}")
    print("=" * 70)

    # [1/4] Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = load_model_standalone(args.model, args.device, dtype=model_dtype)

    config = model.config
    # Handle multimodal models (Gemma4 has text_config)
    text_config = config.text_config if hasattr(config, 'text_config') else config
    num_layers = text_config.num_hidden_layers
    num_kv_heads = text_config.num_key_value_heads
    d_head = text_config.head_dim if hasattr(text_config, 'head_dim') else text_config.hidden_size // text_config.num_attention_heads
    print(f"  Layers: {num_layers}")
    print(f"  KV heads: {num_kv_heads}")
    print(f"  d_head: {d_head}")

    if args.k > d_head:
        print(f"  WARNING: k={args.k} > d_head={d_head}, clamping k to {d_head}")
        args.k = d_head

    # [2/4] Load calibration data
    print("\n[2/4] Loading WikiText-2 calibration data...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 50]
    texts = texts[:args.num_samples]
    print(f"  Selected {len(texts)} samples (from {len(dataset)} total)")

    # [3/4] Collect KVs
    print("\n[3/4] Collecting K/V vectors...")
    kv_data = collect_kvs_for_basis(model, tokenizer, texts,
                                    max_tokens=args.max_tokens,
                                    device=args.device)

    # Free model memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    total_vectors = sum(kv_data[key]['K'].shape[0] for key in kv_data)
    sample_key = next(iter(kv_data))
    actual_d_head = kv_data[sample_key]['K'].shape[1]
    print(f"  Collected {total_vectors:,} K vectors (d_head={actual_d_head})")
    print(f"  Entries: {len(kv_data)} (layer, head) pairs")

    # [4/4] Fit PCA
    print(f"\n[4/4] Fitting PCA (k={args.k})...")
    basis_dict = {}
    explained_variances_k = []

    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        for head_idx in range(num_kv_heads):
            key = (layer_idx, head_idx)
            if key not in kv_data:
                print(f"  Warning: Missing data for L{layer_idx}H{head_idx}")
                continue

            K_vecs = kv_data[key]['K']  # (T, d_head)

            # fit_pca returns (U_k, mean) — 2 values
            U_k, mean_k = fit_pca(K_vecs, args.k)
            basis_dict[f'U_L{layer_idx}_H{head_idx}'] = U_k.astype(np.float32)
            basis_dict[f'mean_L{layer_idx}_H{head_idx}'] = mean_k.astype(np.float32)

            # Compute explained variance separately
            ev = compute_explained_variance(K_vecs, U_k, mean_k)
            explained_variances_k.append(ev)

            if not args.skip_v:
                V_vecs = kv_data[key]['V']
                U_v, mean_v = fit_pca(V_vecs, args.k)
                basis_dict[f'U_V_L{layer_idx}_H{head_idx}'] = U_v.astype(np.float32)
                basis_dict[f'mean_V_L{layer_idx}_H{head_idx}'] = mean_v.astype(np.float32)

            # Free memory as we go
            del kv_data[key]

    # Save metadata
    basis_dict['metadata_k'] = np.array(args.k)
    basis_dict['metadata_d_head'] = np.array(actual_d_head)
    basis_dict['metadata_num_layers'] = np.array(num_layers)
    basis_dict['metadata_num_kv_heads'] = np.array(num_kv_heads)

    # Save
    print(f"\nSaving to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez_compressed(args.output, **basis_dict)

    file_size_mb = os.path.getsize(args.output) / (1024**2)
    num_arrays = len([k for k in basis_dict if not k.startswith('metadata')])
    print(f"  Saved {num_arrays} arrays + metadata ({file_size_mb:.2f} MB)")

    # Statistics
    avg_ev = np.mean(explained_variances_k)
    min_ev = np.min(explained_variances_k)
    max_ev = np.max(explained_variances_k)
    print(f"\nPCA Statistics (K vectors, k={args.k}):")
    print(f"  Explained variance: avg={avg_ev:.4f}, min={min_ev:.4f}, max={max_ev:.4f}")
    print(f"  Layers x KV-heads: {num_layers} x {num_kv_heads} = {num_layers * num_kv_heads}")

    print("\n" + "=" * 70)
    print("Basis generation complete!")
    print(f"Output: {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()

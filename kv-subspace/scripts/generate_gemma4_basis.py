#!/usr/bin/env python3
"""
Generate PCA basis for Gemma4-E4B using WikiText-2 calibration data.
Uses existing collect.py + compress.py infrastructure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import argparse

from collect import get_model_and_tokenizer
from compress import fit_pca

def collect_kvs_for_basis(model, tokenizer, calibration_texts, max_tokens=512, device='cuda'):
    """
    Collect K/V vectors from calibration data for PCA basis fitting.
    
    Returns:
        dict: {(layer_idx, head_idx): {'K': np.array(T, d_head), 'V': np.array(T, d_head)}}
    """
    model.eval()
    kv_storage = {}
    
    print(f"Collecting KVs from {len(calibration_texts)} calibration samples...")
    
    for text in tqdm(calibration_texts, desc="Collecting"):
        inputs = tokenizer(text, return_tensors='pt', max_length=max_tokens, 
                          truncation=True, padding=False).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_hidden_states=False)
            past_key_values = outputs.past_key_values  # Tuple of (K, V) per layer
            
            for layer_idx, (K, V) in enumerate(past_key_values):
                # K, V shape: (batch=1, num_heads, seq_len, d_head)
                K_np = K.squeeze(0).cpu().numpy()  # (num_heads, seq_len, d_head)
                V_np = V.squeeze(0).cpu().numpy()
                
                num_heads, seq_len, d_head = K_np.shape
                
                for head_idx in range(num_heads):
                    key = (layer_idx, head_idx)
                    if key not in kv_storage:
                        kv_storage[key] = {'K': [], 'V': []}
                    
                    # Collect all tokens from this head
                    kv_storage[key]['K'].append(K_np[head_idx])  # (seq_len, d_head)
                    kv_storage[key]['V'].append(V_np[head_idx])
    
    # Concatenate across all samples
    print("Concatenating collected vectors...")
    for key in tqdm(kv_storage.keys(), desc="Concatenating"):
        kv_storage[key]['K'] = np.concatenate(kv_storage[key]['K'], axis=0)  # (T_total, d_head)
        kv_storage[key]['V'] = np.concatenate(kv_storage[key]['V'], axis=0)
    
    return kv_storage

def main():
    parser = argparse.ArgumentParser()
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
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Gemma4-E4B PCA Basis Generation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"PCA rank k: {args.k}")
    print(f"Calibration: {args.num_samples} samples × {args.max_tokens} tokens")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = get_model_and_tokenizer(args.model, awq=False)
    model = model.to(args.device)
    
    # Check model architecture
    config = model.config
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  KV heads: {config.num_key_value_heads}")
    print(f"  Hidden size: {config.hidden_size}")
    
    # Load calibration data
    print("\n[2/4] Loading WikiText-2 calibration data...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 50]
    texts = texts[:args.num_samples]
    print(f"  Loaded {len(texts)} samples")
    
    # Collect KVs
    print("\n[3/4] Collecting K/V vectors...")
    kv_data = collect_kvs_for_basis(model, tokenizer, texts, 
                                    max_tokens=args.max_tokens, 
                                    device=args.device)
    
    total_vectors = sum(kv_data[key]['K'].shape[0] for key in kv_data.keys())
    print(f"  Collected {total_vectors:,} total K/V vectors")
    
    # Fit PCA
    print(f"\n[4/4] Fitting PCA (k={args.k})...")
    basis_dict = {}
    
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    
    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        for head_idx in range(num_kv_heads):
            key = (layer_idx, head_idx)
            if key not in kv_data:
                print(f"  Warning: Missing data for L{layer_idx}H{head_idx}")
                continue
            
            K_vecs = kv_data[key]['K']  # (T, d_head)
            V_vecs = kv_data[key]['V']
            
            # Fit PCA for K
            U_k, mean_k, explained_var_k = fit_pca(K_vecs, args.k)
            basis_dict[f'U_L{layer_idx}_H{head_idx}'] = U_k
            basis_dict[f'mean_L{layer_idx}_H{head_idx}'] = mean_k
            basis_dict[f'explained_var_L{layer_idx}_H{head_idx}'] = explained_var_k
            
            # Fit PCA for V (even though we won't use it, for completeness)
            U_v, mean_v, explained_var_v = fit_pca(V_vecs, args.k)
            basis_dict[f'U_V_L{layer_idx}_H{head_idx}'] = U_v
            basis_dict[f'mean_V_L{layer_idx}_H{head_idx}'] = mean_v
            basis_dict[f'explained_var_V_L{layer_idx}_H{head_idx}'] = explained_var_v
    
    # Save
    print(f"\nSaving to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, **basis_dict)
    
    file_size_mb = os.path.getsize(args.output) / (1024**2)
    print(f"✓ Saved {len(basis_dict)} arrays ({file_size_mb:.2f} MB)")
    
    # Print statistics
    avg_explained_var_k = np.mean([basis_dict[k] for k in basis_dict.keys() 
                                    if k.startswith('explained_var_L')])
    print(f"\nPCA Statistics:")
    print(f"  Average explained variance (K): {avg_explained_var_k:.4f}")
    print(f"  Layers × Heads: {num_layers} × {num_kv_heads} = {num_layers * num_kv_heads}")
    print(f"  Subspace rank k: {args.k}")
    
    print("\n" + "=" * 70)
    print("✓ Basis generation complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()

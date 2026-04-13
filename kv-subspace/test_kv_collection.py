#!/usr/bin/env python3
"""Test KV collection from Mistral."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

model_name = "mistralai/Mistral-7B-v0.3"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✓ Model loaded\n")

# Tokenize
text = "Hello world, this is a test of KV cache collection."
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].to("cuda")

print(f"Input tokens: {input_ids.shape[1]}")

# Run forward pass
with torch.no_grad():
    output = model(input_ids, use_cache=True, return_dict=True)

print(f"\nOutput type: {type(output)}")
print(f"Past KV type: {type(output.past_key_values)}")

past_kv = output.past_key_values

if hasattr(past_kv, 'key_cache'):
    print("\n✓ DynamicCache detected")
    print(f"  Layers: {len(past_kv.key_cache)}")
    print(f"  Layer 0 K shape: {past_kv.key_cache[0].shape}")
    
    # Collect all K vectors
    kv_dict = {}
    for layer_idx in range(len(past_kv.key_cache)):
        k_state = past_kv.key_cache[layer_idx]  # (B, n_kv_heads, T, d_head)
        k_np = k_state[0].detach().cpu().numpy()  # (n_kv_heads, T, d_head)
        
        n_kv_heads, T, d_head = k_np.shape
        for head_idx in range(n_kv_heads):
            key = (layer_idx, head_idx)
            kv_dict[key] = {
                'K': k_np[head_idx]  # (T, d_head)
            }
    
    print(f"\n✓ Collected K vectors for {len(kv_dict)} layer-head pairs")
    print(f"  Sample (L0,H0): {kv_dict[(0,0)]['K'].shape}")
    
else:
    print("\n✗ Not DynamicCache!")

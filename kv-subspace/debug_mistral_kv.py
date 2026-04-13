#!/usr/bin/env python3
"""Debug Mistral KV cache structure."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-v0.3"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model loaded\n")

# Test forward pass
text = "Hello world, this is a test."
inputs = tokenizer(text, return_tensors="pt")

print("Running forward pass with use_cache=True...")
with torch.no_grad():
    output = model(**inputs, use_cache=True, return_dict=True)

print(f"\nOutput type: {type(output)}")
print(f"Output keys: {output.keys()}")
print(f"\nPast key values type: {type(output.past_key_values)}")

# Check if it's DynamicCache
if hasattr(output.past_key_values, 'key_cache'):
    print("✓ Using DynamicCache")
    print(f"  Number of layers: {len(output.past_key_values.key_cache)}")
    print(f"  Layer 0 K shape: {output.past_key_values.key_cache[0].shape}")
    print(f"  Layer 0 V shape: {output.past_key_values.value_cache[0].shape}")
elif isinstance(output.past_key_values, tuple):
    print("✓ Using tuple format")
    print(f"  Number of layers: {len(output.past_key_values)}")
    print(f"  Layer 0 type: {type(output.past_key_values[0])}")
    if isinstance(output.past_key_values[0], tuple):
        print(f"  Layer 0 K shape: {output.past_key_values[0][0].shape}")
        print(f"  Layer 0 V shape: {output.past_key_values[0][1].shape}")

# Test hooking self_attn
print("\n" + "="*60)
print("Testing self_attn hook:")
print("="*60)

captured = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        print(f"Layer {layer_idx} self_attn hook fired!")
        print(f"  Output type: {type(output)}")
        print(f"  Output length: {len(output) if isinstance(output, tuple) else 'N/A'}")
        if isinstance(output, tuple):
            for i, item in enumerate(output):
                print(f"  output[{i}]: {type(item)} {item.shape if hasattr(item, 'shape') else ''}")
        captured[layer_idx] = output
    return hook

# Hook first layer only
handle = model.model.layers[0].self_attn.register_forward_hook(make_hook(0))

print("\nRunning forward pass with hook...")
with torch.no_grad():
    output2 = model(**inputs, use_cache=True, return_dict=True)

handle.remove()

print("\n✓ Done")

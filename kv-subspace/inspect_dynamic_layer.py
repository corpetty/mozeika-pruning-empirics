#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.3",
    device_map="cpu",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

text = "Hello"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs, use_cache=True, return_dict=True)

layer = output.past_key_values.layers[0]
print("All non-callable attributes of DynamicLayer:")
for attr in sorted(dir(layer)):
    if not attr.startswith('_') and not callable(getattr(layer, attr, None)):
        val = getattr(layer, attr)
        print(f"  {attr}: {type(val).__name__}", end='')
        if isinstance(val, torch.Tensor):
            print(f" {val.shape}")
        else:
            print(f" = {val!r}")

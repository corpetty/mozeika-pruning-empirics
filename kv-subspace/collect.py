"""
collect.py — Extract K/V vectors from each attention layer/head via forward hooks.

Usage:
    python collect.py --model Qwen/Qwen3-14B-AWQ --n-tokens 2048 --out results/kvs.npz
"""

import argparse
import numpy as np
import torch
from pathlib import Path


def get_model_and_tokenizer(model_name: str):
    """Load model (AWQ preferred, then standard HF). Returns (model, tokenizer)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use autoawq directly — avoids transformers gptqmodel dependency
    from awq import AutoAWQForCausalLM
    model = AutoAWQForCausalLM.from_quantized(
        model_name, fuse_layers=False, device_map="auto"
    )
    print(f"Loaded AWQ model: {model_name}")
    return model, tokenizer


def find_attention_layers(model):
    """
    Return list of (layer_idx, module) for all attention modules.
    Works for Qwen2/Qwen3 architecture (model.model.layers[i].self_attn).
    Falls back to scanning named_modules for anything with q_proj/k_proj.
    """
    layers = []

    # Qwen2/Qwen3 / LLaMA style
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'self_attn'):
                layers.append((i, layer.self_attn))
        if layers:
            return layers

    # Generic fallback
    for name, module in model.named_modules():
        if hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
            idx = int(name.split('.')[-2]) if name.split('.')[-2].isdigit() else len(layers)
            layers.append((idx, module))

    return layers


def collect_kv_vectors(model, tokenizer, text: str, n_tokens: int, device: str = "cuda"):
    """
    Run a forward pass on `text` (truncated to n_tokens) and collect K/V
    vectors at each attention layer/head via hooks.

    Returns:
        dict: {layer_idx: {'K': np.ndarray (T, n_heads, d_head),
                            'V': np.ndarray (T, n_heads, d_head)}}
    """
    attention_layers = find_attention_layers(model)
    if not attention_layers:
        raise RuntimeError("Could not find attention layers in model")
    print(f"Found {len(attention_layers)} attention layers")

    # Get model config for head dimensions
    model_config = getattr(model, 'config', None)
    config_n_kv_heads = getattr(model_config, 'num_key_value_heads', None)
    config_d_head = getattr(model_config, 'head_dim', None)
    if config_d_head is None and model_config is not None:
        n_heads = getattr(model_config, 'num_attention_heads', None)
        hidden = getattr(model_config, 'hidden_size', None)
        if n_heads and hidden:
            config_d_head = hidden // n_heads

    kv_store = {}  # layer_idx -> {'K': tensor, 'V': tensor}
    hooks = []

    def make_hook(layer_idx, attn_module):
        def hook(module, args, kwargs, output):
            # Qwen3 self_attn.forward signature:
            # The K and V projections happen inside; we hook the output of k_proj/v_proj
            pass
        return hook

    # Hook k_proj and v_proj outputs directly
    kv_raw = {}  # layer_idx -> {'K': [], 'V': []}

    def make_kv_hook(layer_idx, which):
        def hook(module, input, output):
            if layer_idx not in kv_raw:
                kv_raw[layer_idx] = {'K': None, 'V': None}
            # output shape: (batch, seq, n_heads * d_head) or (batch, seq, n_kv_heads * d_head)
            kv_raw[layer_idx][which] = output.detach().cpu().float()
        return hook

    for layer_idx, attn in attention_layers:
        if hasattr(attn, 'k_proj'):
            h = attn.k_proj.register_forward_hook(make_kv_hook(layer_idx, 'K'))
            hooks.append(h)
        if hasattr(attn, 'v_proj'):
            h = attn.v_proj.register_forward_hook(make_kv_hook(layer_idx, 'V'))
            hooks.append(h)

    # Tokenize and run
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=n_tokens)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    actual_tokens = inputs['input_ids'].shape[1]
    print(f"Running forward pass on {actual_tokens} tokens...")

    with torch.no_grad():
        model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Reshape: (batch=1, seq, n_kv_heads * d_head) -> (seq, n_kv_heads, d_head)
    result = {}
    for layer_idx, layer_attn in attention_layers:
        if layer_idx not in kv_raw:
            continue
        raw = kv_raw[layer_idx]
        if raw['K'] is None or raw['V'] is None:
            continue

        # Infer n_kv_heads and d_head from the projection module
        attn = dict(attention_layers)[layer_idx]
        # Prefer model config (most reliable), fall back to attn module attrs
        n_kv_heads = (config_n_kv_heads or
                      getattr(attn, 'num_key_value_heads', None) or
                      getattr(attn, 'num_heads', None))
        d_head = (config_d_head or
                  getattr(attn, 'head_dim', None))

        # Last resort: infer from actual tensor shape
        if n_kv_heads is None or d_head is None:
            total = raw['K'].shape[-1]  # n_kv_heads * d_head
            if d_head is not None:
                n_kv_heads = total // d_head
            elif n_kv_heads is not None:
                d_head = total // n_kv_heads
            else:
                # Assume d_head=128 (Qwen3 default)
                d_head = 128
                n_kv_heads = total // d_head

        K = raw['K'][0]  # (seq, n_kv_heads * d_head)
        V = raw['V'][0]

        T = K.shape[0]
        K = K.reshape(T, n_kv_heads, d_head).numpy()
        V = V.reshape(T, n_kv_heads, d_head).numpy()

        result[layer_idx] = {'K': K, 'V': V}

    print(f"Collected KV vectors from {len(result)} layers")
    if result:
        sample_layer = next(iter(result.values()))
        T, H, D = sample_layer['K'].shape
        print(f"  Shape per layer: T={T} tokens, H={H} heads, d_head={D}")

    return result


def save_kvs(kv_dict: dict, path: str):
    """Save KV dict as compressed npz."""
    flat = {}
    for layer_idx, tensors in kv_dict.items():
        flat[f"layer{layer_idx}_K"] = tensors['K']
        flat[f"layer{layer_idx}_V"] = tensors['V']
    np.savez_compressed(path, **flat)
    print(f"Saved KV vectors to {path}")


def load_kvs(path: str) -> dict:
    """Load KV dict from npz."""
    data = np.load(path)
    result = {}
    for key in data.files:
        parts = key.rsplit('_', 1)
        layer_idx = int(parts[0].replace('layer', ''))
        which = parts[1]  # 'K' or 'V'
        if layer_idx not in result:
            result[layer_idx] = {}
        result[layer_idx][which] = data[key]
    return result


def get_sample_text(n_chars: int = 50000) -> str:
    """Pull a chunk of text from a long document for testing."""
    import urllib.request
    # Project Gutenberg: War and Peace (long, diverse)
    url = "https://www.gutenberg.org/files/2600/2600-0.txt"
    try:
        print("Fetching sample text (War and Peace)...")
        with urllib.request.urlopen(url, timeout=30) as f:
            text = f.read(n_chars).decode('utf-8', errors='ignore')
        return text
    except Exception as e:
        print(f"Download failed ({e}), using fallback text")
        return "The quick brown fox jumps over the lazy dog. " * 2000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-14B-AWQ')
    parser.add_argument('--n-tokens', type=int, default=2048)
    parser.add_argument('--out', default='results/kvs.npz')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = get_model_and_tokenizer(args.model)
    text = get_sample_text(n_chars=args.n_tokens * 8)  # rough chars-to-tokens ratio
    kv_dict = collect_kv_vectors(model, tokenizer, text, args.n_tokens, args.device)
    save_kvs(kv_dict, args.out)


if __name__ == '__main__':
    main()

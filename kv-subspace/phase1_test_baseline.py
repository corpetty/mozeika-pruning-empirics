#!/usr/bin/env python3
"""Phase 1: Test vLLM baseline with Gemma-2-27B on GPU0"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU0 only

from vllm import LLM, SamplingParams
import torch

def test_baseline(max_len=32768):
    """Test Gemma baseline at specified context length"""
    
    print(f"\n{'='*60}")
    print(f"Testing Gemma-2-27B at {max_len} context")
    print(f"{'='*60}\n")
    
    # Reset VRAM tracking
    torch.cuda.reset_peak_memory_stats()
    
    # Initialize model
    print("Loading model...")
    # Use Gemma 4 26B A4B (MoE, 25.2B total / 3.8B active)
    llm = LLM(
        model="google/gemma-4-26B-A4B-it",
        gpu_memory_utilization=0.90,
        max_model_len=max_len,
        tensor_parallel_size=1,
        enforce_eager=True,  # Disable CUDA graphs for cleaner profiling
        trust_remote_code=True
    )
    
    model_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"Model loaded: {model_vram_gb:.2f} GB VRAM\n")
    
    # Generate text to fill KV cache
    prompt = "Explain quantum entanglement in detail:" + " Continue explaining." * 500
    
    print("Generating text...")
    torch.cuda.reset_peak_memory_stats()
    
    outputs = llm.generate(
        prompt,
        SamplingParams(
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9
        )
    )
    
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    kv_cache_gb = peak_vram_gb - model_vram_gb
    
    output_text = outputs[0].outputs[0].text
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Context length: {max_len}")
    print(f"  Model VRAM:     {model_vram_gb:.2f} GB")
    print(f"  Peak VRAM:      {peak_vram_gb:.2f} GB")
    print(f"  KV cache (~):   {kv_cache_gb:.2f} GB")
    print(f"  Output length:  {len(output_text)} chars")
    print(f"{'='*60}\n")
    
    print(f"Sample output:\n{output_text[:500]}...\n")
    
    return {
        "max_len": max_len,
        "model_vram_gb": model_vram_gb,
        "peak_vram_gb": peak_vram_gb,
        "kv_cache_gb": kv_cache_gb,
        "output_chars": len(output_text)
    }

if __name__ == "__main__":
    import json
    
    # Test at 16K and 32K context
    results = []
    
    for ctx_len in [16384, 32768]:
        try:
            result = test_baseline(ctx_len)
            results.append(result)
        except Exception as e:
            print(f"\nFailed at {ctx_len} context: {e}\n")
            results.append({
                "max_len": ctx_len,
                "error": str(e)
            })
    
    # Save results
    output_file = "/home/petty/pruning-research/kv-subspace/results/vllm_baseline.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

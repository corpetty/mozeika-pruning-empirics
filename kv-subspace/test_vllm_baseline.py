#!/usr/bin/env python3
"""
Test baseline vLLM memory usage with varying context lengths.
Establishes upper bound before SubRotQ compression.
"""

import requests
import json
from typing import List, Dict

API_URL = "http://localhost:8000/v1/chat/completions"

def generate_long_prompt(target_tokens: int) -> str:
    """Generate a prompt that will consume ~target_tokens in the KV cache."""
    # Rough estimate: 1 word ~= 1.3 tokens for English
    words_needed = int(target_tokens / 1.3)
    base_text = "The quick brown fox jumps over the lazy dog. " * (words_needed // 10)
    return f"Summarize the following text:\n\n{base_text}\n\nSummary:"

def test_context_length(prompt_tokens: int, max_new_tokens: int = 50) -> Dict:
    """Test generation at a specific context length."""
    prompt = generate_long_prompt(prompt_tokens)
    
    try:
        response = requests.post(API_URL, json={
            "model": "google/gemma-4-E4B-it",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.7
        }, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            usage = data.get('usage', {})
            return {
                'status': 'success',
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
                'response': data['choices'][0]['message']['content'][:100]
            }
        else:
            return {'status': 'error', 'code': response.status_code, 'message': response.text[:200]}
    
    except Exception as e:
        return {'status': 'exception', 'error': str(e)}

def main():
    print("=" * 60)
    print("vLLM Baseline Context Length Test (Gemma4-E4B, 4K max)")
    print("=" * 60)
    
    # Test increasing context lengths
    test_points = [512, 1024, 2048, 3072, 4096, 5120]  # Last one should fail
    
    results = []
    for target_tokens in test_points:
        print(f"\n[TEST] Target context: {target_tokens} tokens")
        result = test_context_length(target_tokens)
        results.append({'target': target_tokens, **result})
        
        if result['status'] == 'success':
            actual = result['prompt_tokens']
            print(f"  ✓ Success: {actual} prompt tokens, {result['completion_tokens']} completion")
            print(f"  Response preview: {result['response']}")
        else:
            print(f"  ✗ Failed: {result.get('error', result.get('message', 'Unknown'))}")
            if target_tokens > 4096:
                print(f"  (Expected - exceeds 4K limit)")
    
    # Save results
    with open('results/vllm_baseline_context_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Results saved to results/vllm_baseline_context_test.json")
    print("=" * 60)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
lm-eval-harness evaluation with SubRotQ compression.

Usage:
    # Baseline (no compression)
    python scripts/eval_lm_harness.py --model google/gemma-4-E4B-it --tasks arc_easy --batch_size 1

    # With SubRotQ compression
    python scripts/eval_lm_harness.py --model google/gemma-4-E4B-it --tasks arc_easy \
        --use_subrotq --basis results/gemma4_e4b_pca_basis_k128_hetero.npz --batch_size 1
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_subrotq_scaling import SubRotQCache
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


class SubRotQHFLM(HFLM):
    """lm-eval-harness model wrapper with SubRotQ KV cache compression."""
    
    def __init__(
        self,
        pretrained: str,
        basis_path: Optional[str] = None,
        k: int = 128,
        n_bits: int = 4,
        device: str = "cuda:0",
        **kwargs
    ):
        """Initialize with optional SubRotQ compression.
        
        Args:
            pretrained: HuggingFace model name or path
            basis_path: Path to SubRotQ PCA basis file (if None, no compression)
            k: PCA rank for compression
            n_bits: Quantization bits
            device: Device to run on
            **kwargs: Passed to parent HFLM
        """
        # Initialize parent (loads model)
        super().__init__(pretrained=pretrained, device=device, **kwargs)
        
        # Setup SubRotQ if basis provided
        self.use_subrotq = basis_path is not None
        if self.use_subrotq:
            print(f"[SubRotQ] Loading basis from {basis_path}")
            self.subrotq_cache = SubRotQCache(
                basis_path=basis_path,
                k=k,
                n_bits=n_bits,
                device=device
            )
            print(f"[SubRotQ] Initialized with k={k}, n_bits={n_bits}")
            
            # Monkey-patch model's forward to use SubRotQ cache
            self._original_forward = self.model.forward
            self.model.forward = self._subrotq_forward
        else:
            print("[Baseline] Running without compression")
    
    def _subrotq_forward(self, *args, **kwargs):
        """Wrapped forward pass that uses SubRotQ cache."""
        
        # Extract past_key_values if provided
        past_kv = kwargs.get('past_key_values', None)
        
        # If we have compressed cache, decompress it
        if past_kv is None and len(self.subrotq_cache.cache) > 0:
            past_kv = self.subrotq_cache.decompress()
            kwargs['past_key_values'] = past_kv
        
        # Run original forward
        outputs = self._original_forward(*args, **kwargs)
        
        # Compress new cache if it was generated
        if outputs.past_key_values is not None:
            self.subrotq_cache.compress_and_store(outputs.past_key_values)
        
        return outputs
    
    def _model_call(self, inps, attn_mask=None, labels=None):
        """Override model call to reset cache between samples."""
        
        # Reset SubRotQ cache for each new sample
        if self.use_subrotq:
            self.subrotq_cache.cache = []
        
        # Call parent implementation
        return super()._model_call(inps, attn_mask=attn_mask, labels=labels)


def main():
    parser = argparse.ArgumentParser(description="Evaluate with lm-eval-harness")
    
    # Model args
    parser.add_argument('--model', type=str, required=True,
                       help='HuggingFace model name or path')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run on')
    
    # SubRotQ args
    parser.add_argument('--use_subrotq', action='store_true',
                       help='Enable SubRotQ compression')
    parser.add_argument('--basis', type=str,
                       default='results/gemma4_e4b_pca_basis_k128_hetero.npz',
                       help='Path to PCA basis file')
    parser.add_argument('--k', type=int, default=128,
                       help='PCA rank')
    parser.add_argument('--n_bits', type=int, default=4,
                       help='Quantization bits')
    
    # Eval args
    parser.add_argument('--tasks', type=str, default='arc_easy',
                       help='Comma-separated list of tasks (e.g., arc_easy,hellaswag)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (use 1 for safety with SubRotQ)')
    parser.add_argument('--num_fewshot', type=int, default=0,
                       help='Number of few-shot examples')
    parser.add_argument('--output_path', type=str, default='results/',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"lm-eval-harness with {'SubRotQ' if args.use_subrotq else 'Baseline'}")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Tasks: {args.tasks}")
    print(f"Batch size: {args.batch_size}")
    
    if args.use_subrotq:
        print(f"SubRotQ config: k={args.k}, n_bits={args.n_bits}, basis={args.basis}")
    
    # Initialize model wrapper
    model_args = {
        'pretrained': args.model,
        'device': args.device,
        'trust_remote_code': True,
        'dtype': 'bfloat16',
        'batch_size': args.batch_size,
    }
    
    if args.use_subrotq:
        model_args.update({
            'basis_path': args.basis,
            'k': args.k,
            'n_bits': args.n_bits,
        })
        lm = SubRotQHFLM(**model_args)
    else:
        lm = HFLM(**model_args)
    
    # Run evaluation
    task_list = args.tasks.split(',')
    
    print(f"\nRunning evaluation on tasks: {task_list}")
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    for task, metrics in results['results'].items():
        print(f"\n{task}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    # Save results
    import json
    output_file = Path(args.output_path) / f"lm_eval_{'subrotq' if args.use_subrotq else 'baseline'}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()

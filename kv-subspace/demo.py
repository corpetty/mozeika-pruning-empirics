#!/usr/bin/env python3
"""
demo.py — KV subspace compression: do something you couldn't do before.

Loads Qwen3-14B-AWQ, patches it with 4x KV compression (k=112/4-bit),
then answers questions over a long document that would OOM at this context
length without compression.

Usage:
    python3 demo.py --doc <path_or_url> [--ctx 32768] [--k 112] [--question "..."]
    python3 demo.py --builtin-long       # built-in 30K token stress test

Options:
    --model       HF model ID (default: Qwen/Qwen3-14B-AWQ)
    --k           Subspace dimension (default: 112, i.e. 87.5% of d_head=128)
    --bits        Quantization bits (default: 4)
    --ctx         Target context length in tokens (default: 32768)
    --basis       Path to a saved basis file (skips calibration)
    --save-basis  Save calibration basis for reuse
    --no-patch    Run without compression (baseline)
    --doc         Path or URL to a text document to use as context
    --question    Question to ask about the document
    --builtin-long  Use internal 30K token test document
"""

import argparse
import os
import sys
import time
import textwrap
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="kvpatch demo — long context on less VRAM")
    p.add_argument("--model",        default="Qwen/Qwen3-14B-AWQ")
    p.add_argument("--k",            type=int, default=112)
    p.add_argument("--bits",         type=int, default=4)
    p.add_argument("--ctx",          type=int, default=32768)
    p.add_argument("--basis",        default=None,  help="Load pre-fitted basis from file")
    p.add_argument("--save-basis",   default=None,  help="Save basis to file after calibration")
    p.add_argument("--no-patch",     action="store_true", help="Run baseline (no compression)")
    p.add_argument("--doc",          default=None,  help="Path or URL to context document")
    p.add_argument("--question",     default="Summarize the key points of this document.")
    p.add_argument("--builtin-long", action="store_true", help="Use built-in 30K stress test")
    p.add_argument("--calib-tokens", type=int, default=2048)
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str):
    from transformers import AutoTokenizer
    from awq import AutoAWQForCausalLM

    print(f"\n[demo] Loading {model_id} ...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_quantized(
        model_id,
        fuse_layers=False,
        trust_remote_code=True,
        safetensors=True,
        max_memory={0: "20GiB", 1: "20GiB", "cpu": "60GiB"},
    )
    model.eval()

    elapsed = time.time() - t0
    print(f"[demo] Loaded in {elapsed:.1f}s")
    return model, tokenizer


# ── Memory stats ──────────────────────────────────────────────────────────────

def gpu_stats():
    if not torch.cuda.is_available():
        return "no GPU"
    lines = []
    for i in range(torch.cuda.device_count()):
        used  = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        lines.append(f"GPU{i}: {used:.1f}/{total:.0f}GB")
    return "  |  ".join(lines)


# ── Document loading ──────────────────────────────────────────────────────────

def load_document(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        import urllib.request
        print(f"[demo] Fetching {path_or_url} ...")
        with urllib.request.urlopen(path_or_url) as r:
            return r.read().decode("utf-8", errors="replace")
    else:
        return Path(path_or_url).read_text(encoding="utf-8", errors="replace")


def builtin_long_doc(n_chars: int = 120_000) -> str:
    """Generate a synthetic long document: rotating scientific paragraphs."""
    paras = [
        ("Thermodynamics is the branch of physics that deals with heat, work, and "
         "temperature, and their relation to energy, entropy, and the physical "
         "properties of matter and radiation. The behavior of these quantities is "
         "governed by the four laws of thermodynamics which convey a quantitative "
         "description using measurable macroscopic physical quantities, but may be "
         "explained in terms of microscopic constituents by statistical mechanics."),
        ("The theory of evolution by natural selection was first formulated in "
         "Charles Darwin's book On the Origin of Species in 1859. According to "
         "natural selection, organisms with heritable traits that help them "
         "adapt to their environment tend to survive and reproduce more "
         "successfully than others of their kind, thereby ensuring the "
         "proliferation of those traits in future generations."),
        ("A neural network is a series of algorithms that endeavors to recognize "
         "underlying relationships in a set of data through a process that mimics "
         "the way the human brain operates. Neural networks can adapt to changing "
         "input so the network generates the best possible result without needing "
         "to redesign the output criteria."),
        ("The Silk Road was a network of trade routes which connected the East "
         "and West, and was central to the economic, cultural, political, and "
         "religious interactions between these regions from the 2nd century BCE "
         "to the 18th century. The Silk Road derives its name from the lucrative "
         "trade in silk carried out along its length, beginning in the Han dynasty "
         "of China."),
        ("In mathematics, the Riemann hypothesis is a conjecture that the Riemann "
         "zeta function has its zeros only at the negative even integers and "
         "complex numbers with real part 1/2. Many consider it to be the most "
         "important unsolved problem in pure mathematics. It is one of the seven "
         "Millennium Prize Problems selected by the Clay Mathematics Institute."),
    ]
    text = ""
    while len(text) < n_chars:
        for p in paras:
            text += p + "\n\n"
            if len(text) >= n_chars:
                break
    return text[:n_chars]


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, tokenizer, document: str, question: str, ctx: int):
    """Build a prompt with document as context and generate an answer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt = (
        f"<|im_start|>system\n"
        f"You are a helpful assistant. Answer questions based solely on the provided document.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"DOCUMENT:\n{document}\n\n"
        f"QUESTION: {question}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx)
    input_ids = enc["input_ids"].to(device)
    actual_ctx = input_ids.shape[1]

    print(f"\n[demo] Prompt tokens: {actual_ctx:,} / {ctx:,} ctx limit")
    print(f"[demo] {gpu_stats()}")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    elapsed = time.time() - t0

    generated = out[0, input_ids.shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    toks_per_sec = len(generated) / elapsed

    print(f"[demo] Generated {len(generated)} tokens in {elapsed:.1f}s "
          f"({toks_per_sec:.1f} tok/s)")
    print(f"[demo] {gpu_stats()}")

    return answer, actual_ctx


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 70)
    print("  kvpatch demo — KV subspace compression")
    print(f"  k={args.k}/4-bit K compression → ~{(128*16)/(args.k*args.bits):.1f}x KV memory reduction")
    print("=" * 70)

    # Load model
    model, tokenizer = load_model(args.model)
    print(f"[demo] After load: {gpu_stats()}")

    # Patch
    if not args.no_patch:
        from kvpatch import patch, KVBasis

        basis = None
        if args.basis:
            basis = KVBasis.load(args.basis)

        basis = patch(
            model, tokenizer,
            basis=basis,
            k=args.k,
            bits=args.bits,
            compress_k=True,
            compress_v=False,   # V: off by default (see RESULTS.md exp20)
            n_tokens=args.calib_tokens,
            verbose=True,
        )

        if args.save_basis:
            basis.save(args.save_basis)
    else:
        print("[demo] Running WITHOUT compression (baseline mode)")

    # Load document
    if args.builtin_long:
        print(f"\n[demo] Using built-in synthetic long document (~30K tokens)")
        doc = builtin_long_doc(n_chars=120_000)
    elif args.doc:
        doc = load_document(args.doc)
    else:
        print("\n[demo] No --doc provided. Using short built-in test.")
        doc = builtin_long_doc(n_chars=10_000)

    # Run inference
    print(f"\n[demo] Question: {args.question}")
    answer, actual_ctx = run_inference(model, tokenizer, doc, args.question, args.ctx)

    print("\n" + "=" * 70)
    print("  ANSWER")
    print("=" * 70)
    print(textwrap.fill(answer, width=70))
    print("=" * 70)

    # Summary
    print(f"\n[demo] Context used: {actual_ctx:,} tokens")
    if not args.no_patch:
        from kvpatch import compression_ratio, memory_delta_gb
        cr = compression_ratio(args.k, args.bits)
        # KV memory at this context without compression
        n_layers = 40  # Qwen3-14B default; ideally read from model
        n_kv = 8
        d = 128
        full_kv_gb = 2 * n_layers * n_kv * actual_ctx * d * 2 / 1024**3
        comp_kv_gb = n_layers * n_kv * actual_ctx * args.k * (args.bits // 8) / 1024**3
        saved = full_kv_gb - comp_kv_gb
        print(f"[demo] Full fp16 K cache would have been: {full_kv_gb:.2f} GB")
        print(f"[demo] Compressed K cache used:           {comp_kv_gb:.2f} GB")
        print(f"[demo] Saved:                             {saved:.2f} GB ({cr:.1f}x)")


if __name__ == "__main__":
    main()

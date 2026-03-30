"""
Experiment 17: Cross-domain calibration robustness.

Answers: Does a PCA basis calibrated on one domain transfer to others?
  - Calibrate on: fiction, code, news, dialogue
  - Eval on the same four domains (+ universal calib set)
  - Compare PPL with k128_4bit and k96_4bit across calibration/eval domain combos

Usage:
    python experiments/exp17_cross_domain.py

Outputs:
    results/exp17_cross_domain.csv
    results/REPORT-17-cross-domain.md
"""

import sys
import csv
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compress import polar_quantize, subspace_polar_quantize, fit_pca
from collect import get_model_and_tokenizer, find_attention_layers

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME   = "Qwen/Qwen3-14B-AWQ"
DATA_FILE    = Path("data/war_and_peace.txt")
CALIB_TOKENS = 1024
EVAL_CTX     = 2048

CONFIGS = {
    "baseline":  (None,       None, None, None,       None, None),
    "k128_4bit": ("subspace", 128,  4,    "subspace", 128,  4),
    "k96_4bit":  ("subspace", 96,   4,    "subspace", 96,   4),
}

# ── Domain text samples ───────────────────────────────────────────────────────
# These are self-contained so the experiment has no external data dependencies.

CODE_TEXT = """\
import torch
import torch.nn as nn
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scale = self.d_head ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

def train_step(model, optimizer, batch, device):
    model.train()
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class KVCache:
    def __init__(self, max_batch_size, max_seq_len, n_kv_heads, head_dim, device, dtype):
        shape = (max_batch_size, n_kv_heads, max_seq_len, head_dim)
        self.cache_k = torch.zeros(shape, device=device, dtype=dtype)
        self.cache_v = torch.zeros(shape, device=device, dtype=dtype)
        self.seq_len = 0

    def update(self, xk, xv):
        bsz, n_heads, seqlen, head_dim = xk.shape
        self.cache_k[:bsz, :, self.seq_len:self.seq_len+seqlen] = xk
        self.cache_v[:bsz, :, self.seq_len:self.seq_len+seqlen] = xv
        self.seq_len += seqlen
        return (self.cache_k[:bsz, :, :self.seq_len],
                self.cache_v[:bsz, :, :self.seq_len])
""" * 6

NEWS_TEXT = """\
WASHINGTON — The Federal Reserve held interest rates steady on Wednesday, signaling caution
about the pace of inflation and economic growth. The decision to maintain rates came after
months of data suggesting consumer prices remain above the Fed's 2 percent target. Chair
Jerome Powell emphasized future adjustments depend on labor markets, spending, and supply
chains.

The Commerce Department reported retail sales rose 0.4 percent in March, beating forecasts
of 0.2 percent growth. Gains were driven by electronics stores and online retailers. Auto
sales declined slightly, reflecting concerns about vehicle prices and financing costs.

Technology stocks rallied after major companies reported stronger-than-expected earnings.
Semiconductor manufacturers saw large gains driven by AI accelerator and data center demand.
Analysts warned valuations may be stretched, with price-to-earnings ratios above historical
norms. Energy prices edged higher following OPEC+ announcements to extend production cuts.

Scientists announced a breakthrough in quantum computing, demonstrating a processor capable
of complex optimization tasks in hours rather than months. The development was hailed as a
milestone toward practical quantum advantage. Industry observers cautioned widespread
commercial applications remain years away, citing error correction and hardware challenges.

The European Central Bank signaled it may cut rates sooner than expected, citing slowing
growth in the eurozone. German manufacturing output fell for the third consecutive quarter,
raising concerns about the bloc's largest economy. Southern European countries showed more
resilience, with Spain and Italy posting positive growth figures for the quarter.

Global shipping costs declined for the fourth consecutive month, suggesting supply chain
pressures that drove inflation during the pandemic era are finally easing. Container rates
on the Asia-to-Europe route fell to their lowest level in three years. Analysts said the
decline reflects both improved capacity and slowing consumer demand in developed markets.

Central banks across Asia maintained divergent policies, with the Bank of Japan cautiously
moving toward policy normalization while others kept rates on hold. Currency volatility
increased as traders repositioned ahead of major economic releases. The dollar strengthened
against most emerging market currencies, raising concerns about dollar-denominated debt
burdens in developing economies.
""" * 10

DIALOGUE_TEXT = """\
ALICE: I've been thinking about the problem with the current architecture.
BOB: Which part specifically? The encoder or the decoder?
ALICE: Both. The encoder isn't capturing long-range dependencies well enough.
BOB: Have you tried increasing the number of attention heads?
ALICE: Yes, from eight to sixteen. It helped a bit, but not enough.
BOB: What about the positional encoding scheme? RoPE versus ALiBi makes a big difference.
ALICE: We're using RoPE, but I think the base frequency is too low. Degradation past 8K.
BOB: You'd want to scale the theta parameter. Some teams use 500K or even 1M for long context.
ALICE: That's what I was thinking. But then we also need to examine the attention mask pattern.
BOB: Are you using full attention or something like sliding window?
ALICE: Full attention right now. Sliding window saves memory but I'm worried about retrieval.
BOB: The hybrid approach might work — full for early layers, sliding for middle, full at top.
ALICE: What's the intuition?
BOB: Early layers build local features, don't need global context. Late layers do global reasoning.
ALICE: That makes sense. We'd save memory in the middle where the savings are biggest.
BOB: Exactly. You can apply different compression ratios per layer.
ALICE: How are you handling calibration? You need representative activations for the subspace.
BOB: We're mixing domains — about 2K tokens each from fiction, code, dialogue, and technical.
ALICE: Does it transfer across domains?
BOB: Mostly yes, but code is a bit of an outlier. The activation distributions differ.
ALICE: That's consistent with our probing experiments. Code representations cluster differently.
BOB: So either calibrate separately for code, or oversample it in the universal calibration set.
ALICE: I'd lean toward oversampling. Simpler deployment story.
BOB: Agreed. One model, one calibration set, ship it.
ALICE: Let me look at the cross-domain numbers and we can decide from there.
BOB: Sounds good. I'll send over the checkpoint when the run finishes.
ALICE: And can you share the perplexity curves too?
BOB: Sure, I'll put them in the results directory.
ALICE: Great. What's our timeline looking like for the paper submission?
BOB: The deadline is in three weeks. We need the ablations done by end of next week at the latest.
ALICE: That's tight. Should we drop the head-level analysis and just do layer-level?
BOB: Probably wise. Layer-level tells most of the story and is way cheaper to run.
ALICE: Agreed. Let's prioritize the cross-domain and context length experiments.
BOB: Those are the ones reviewers will ask about anyway.
ALICE: What about the comparison to other baselines? Do we include H2O and StreamingLLM?
BOB: At minimum H2O, since it's the most cited. StreamingLLM if we have time.
ALICE: I can run those in parallel on the second GPU if SLURM doesn't have it reserved.
BOB: Check the queue first. Alexander submitted some jobs last night.
ALICE: Good call. I'll check before submitting.
""" * 8


def get_domain_texts(fiction_text):
    """Return {domain: text} mapping. Fiction from the real file."""
    return {
        "fiction":  fiction_text,
        "code":     CODE_TEXT,
        "news":     NEWS_TEXT,
        "dialogue": DIALOGUE_TEXT,
    }


def collect_kvs_from_text(model, tokenizer, text, n_tokens, device, n_kv_heads, d_head):
    """Collect KV vectors from arbitrary text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       max_length=n_tokens + 1, add_special_tokens=True)
    input_ids = inputs['input_ids'].to(device)
    if input_ids.shape[1] > n_tokens:
        input_ids = input_ids[:, :n_tokens]

    kv_store = {}
    hooks = []
    for layer_idx, attn in find_attention_layers(model):
        for kv_type, proj_name in [('K', 'k_proj'), ('V', 'v_proj')]:
            def make_capture(li, kvt, nh, dh):
                def hook(module, input, output):
                    x = output.detach().cpu().float()
                    b, s, _ = x.shape
                    x = x.reshape(b, s, nh, dh)[0]
                    key = (li, kvt)
                    if key not in kv_store:
                        kv_store[key] = []
                    kv_store[key].append(x.numpy())
                return hook
            h = getattr(attn, proj_name).register_forward_hook(
                make_capture(layer_idx, kv_type, n_kv_heads, d_head))
            hooks.append(h)

    with torch.no_grad():
        model(input_ids=input_ids)
    for h in hooks:
        h.remove()

    bases_raw = {}
    for (layer_idx, kv_type), arrays in kv_store.items():
        arr = np.concatenate(arrays, axis=0)
        for head_idx in range(arr.shape[1]):
            key = (layer_idx, head_idx)
            if key not in bases_raw:
                bases_raw[key] = {}
            bases_raw[key][kv_type] = arr[:, head_idx, :]
    return bases_raw


def fit_bases(kvs_raw, k):
    bases = {}
    for (layer_idx, head_idx), kv in kvs_raw.items():
        U_k, mean_k = fit_pca(kv['K'], k)
        U_v, mean_v = fit_pca(kv['V'], k)
        bases[(layer_idx, head_idx)] = {
            'U_K': U_k, 'mean_K': mean_k,
            'U_V': U_v, 'mean_V': mean_v,
        }
    return bases


def compress_vec(x_np, method, k, n_bits, U, mean):
    if method == 'subspace':
        return subspace_polar_quantize(x_np, k, n_bits, U, mean)
    elif method == 'full_dim':
        return polar_quantize(x_np, n_bits)
    return x_np


def _get_transformer_body_and_head(model):
    causal_lm = getattr(model, 'model', model)
    if hasattr(causal_lm, 'model') and hasattr(causal_lm, 'lm_head'):
        return causal_lm.model, causal_lm.lm_head
    return causal_lm, model.lm_head


def chunked_cross_entropy(model, input_ids, chunk_size=256):
    transformer_body, lm_head = _get_transformer_body_and_head(model)
    with torch.no_grad():
        outputs = transformer_body(input_ids=input_ids[:, :-1])
        hidden = outputs.last_hidden_state
    labels = input_ids[:, 1:].view(-1)
    total_loss, n_tok = 0.0, 0
    with torch.no_grad():
        for start in range(0, hidden.shape[1], chunk_size):
            end = min(start + chunk_size, hidden.shape[1])
            chunk_logits = lm_head(hidden[:, start:end, :])
            chunk_labels = labels[start:end]
            loss = torch.nn.functional.cross_entropy(
                chunk_logits.view(-1, chunk_logits.size(-1)), chunk_labels)
            total_loss += float(loss) * (end - start)
            n_tok += (end - start)
            del chunk_logits, chunk_labels, loss
            torch.cuda.empty_cache()
    del hidden
    torch.cuda.empty_cache()
    return total_loss / n_tok


def install_hooks(model, cfg, bases, n_kv_heads, d_head):
    K_method, K_k, K_bits, V_method, V_k, V_bits = cfg
    hooks = []
    for layer_idx, attn in find_attention_layers(model):
        for kv_type, proj_name, method, k, bits in [
            ('K', 'k_proj', K_method, K_k, K_bits),
            ('V', 'v_proj', V_method, V_k, V_bits),
        ]:
            if method is None:
                continue
            def make_hook(li, kvt, m, kk, nb):
                def hook(module, input, output):
                    dev, dty = output.device, output.dtype
                    x = output.detach().cpu().float()
                    b, s, _ = x.shape
                    x = x.reshape(b, s, n_kv_heads, d_head)
                    for h in range(n_kv_heads):
                        xh = x[0, :, h, :].numpy()
                        base = bases.get((li, h), {})
                        x[0, :, h, :] = torch.from_numpy(
                            compress_vec(xh, m, kk, nb,
                                         base.get(f'U_{kvt}'), base.get(f'mean_{kvt}')))
                    return x.reshape(b, s, -1).to(device=dev, dtype=dty)
                return hook
            hooks.append(getattr(attn, proj_name).register_forward_hook(
                make_hook(layer_idx, kv_type, method, k, bits)))
    return hooks


def eval_ppl_on_text(model, tokenizer, text, n_tokens, device, bases, cfg, n_kv_heads, d_head):
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       max_length=n_tokens + 1, add_special_tokens=True)
    eval_ids = inputs['input_ids'].to(device)
    if eval_ids.shape[1] > n_tokens + 1:
        eval_ids = eval_ids[:, :n_tokens + 1]

    hooks = install_hooks(model, cfg, bases, n_kv_heads, d_head)
    loss = chunked_cross_entropy(model, eval_ids)
    for h in hooks:
        h.remove()
    torch.cuda.empty_cache()
    del eval_ids
    return float(np.exp(loss))


def main():
    print("=" * 70)
    print("Experiment 17: Cross-Domain Calibration Robustness")
    print("=" * 70)

    device = "cuda"
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    n_kv_heads = model.config.num_key_value_heads
    d_head     = model.config.hidden_size // model.config.num_attention_heads
    n_layers   = len(find_attention_layers(model))
    print(f"n_layers={n_layers}, n_kv_heads={n_kv_heads}, d_head={d_head}")

    fiction_text = DATA_FILE.read_text(encoding="utf-8", errors="replace")
    domain_texts_raw = get_domain_texts(fiction_text)
    domains = list(domain_texts_raw.keys())

    # Split each domain into calib (first 75%) and eval (last 25%) to avoid overlap.
    # This ensures the same eval text is used for a given eval_domain regardless of
    # which domain was used for calibration (fixes Figure 8 data bug where
    # same-domain calib/eval used second half while cross-domain used full text).
    calib_texts = {}
    eval_texts = {}
    for dom, txt in domain_texts_raw.items():
        split = int(len(txt) * 0.75)
        calib_texts[dom] = txt[:split]
        eval_texts[dom]  = txt[split:]

    # Universal calibration = first 75% interleaved from all four domains
    universal_calib_parts = [calib_texts[dom] for dom in domains]
    universal_calib_text = "\n\n".join(universal_calib_parts)

    calib_sources = dict(calib_texts)  # one per domain (first 75% each)
    calib_sources["universal"] = universal_calib_text

    k_values = [96, 128]
    cfg_k = {"baseline": None, "k128_4bit": 128, "k96_4bit": 96}

    csv_path = RESULTS_DIR / "exp17_cross_domain.csv"
    fieldnames = ["calib_domain", "eval_domain", "config", "ppl"]
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add((row["calib_domain"], row["eval_domain"], row["config"]))
        print(f"\nResuming: {len(done)} entries done")

    for calib_domain, calib_text in calib_sources.items():
        needs_eval = any(
            (calib_domain, ed, cfg) not in done
            for ed in domains for cfg in CONFIGS
        )
        if not needs_eval:
            print(f"\n[calib={calib_domain}] all done, skipping")
            continue

        print(f"\n── Calibrating on: {calib_domain} ──")
        calib_kvs = collect_kvs_from_text(model, tokenizer, calib_text,
                                            CALIB_TOKENS, device, n_kv_heads, d_head)
        bases_by_k = {k: fit_bases(calib_kvs, k) for k in k_values}
        print(f"  Fitted {len(bases_by_k[128])} (layer, head) bases")

        for eval_domain in domains:
            # Always use the held-out last 25% of each domain for eval,
            # regardless of which domain was used for calibration.
            eval_text = eval_texts[eval_domain]

            for cfg_name, cfg in CONFIGS.items():
                key = (calib_domain, eval_domain, cfg_name)
                if key in done:
                    continue

                k = cfg_k[cfg_name]
                bases = bases_by_k[k] if k is not None else {}

                print(f"  eval={eval_domain:<10}  cfg={cfg_name:<12}", end="", flush=True)
                ppl = eval_ppl_on_text(model, tokenizer, eval_text, EVAL_CTX,
                                        device, bases, cfg, n_kv_heads, d_head)
                print(f"  PPL={ppl:.4f}")

                row = {"calib_domain": calib_domain, "eval_domain": eval_domain,
                       "config": cfg_name, "ppl": round(ppl, 4)}
                file_exists = csv_path.exists()
                with open(csv_path, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists:
                        w.writeheader()
                    w.writerow(row)

    # Report
    all_rows = []
    if csv_path.exists():
        with open(csv_path) as f:
            all_rows = list(csv.DictReader(f))

    report_path = RESULTS_DIR / "REPORT-17-cross-domain.md"
    calib_order = list(calib_sources.keys())
    with open(report_path, 'w') as f:
        f.write("# Experiment 17: Cross-Domain Calibration Robustness\n\n")
        f.write(f"- Model: Qwen3-14B-AWQ\n")
        f.write(f"- Calibration tokens: {CALIB_TOKENS} per domain\n")
        f.write(f"- Eval context: {EVAL_CTX} tokens\n")
        f.write(f"- Domains: {domains}\n\n")

        for cfg_name in CONFIGS:
            f.write(f"## PPL matrix — {cfg_name}\n\n")
            f.write("| calib↓ / eval→ | " + " | ".join(domains) + " |\n")
            f.write("|---|" + "|".join("---" for _ in domains) + "|\n")
            for calib_dom in calib_order:
                vals = []
                for eval_dom in domains:
                    matches = [r for r in all_rows
                               if r["calib_domain"] == calib_dom
                               and r["eval_domain"] == eval_dom
                               and r["config"] == cfg_name]
                    vals.append(f"{float(matches[0]['ppl']):.3f}" if matches else "N/A")
                f.write(f"| {calib_dom} | " + " | ".join(vals) + " |\n")
            f.write("\n")

    print(f"\nSaved {csv_path}")
    print(f"Wrote {report_path}")
    print("\n" + "=" * 70)
    print("Experiment 17 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

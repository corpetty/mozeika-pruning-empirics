# Peer-Review Revision Tasks — Sub-Agent Briefs

This document is a set of self-contained experiment tasks generated from a peer
review of `paper/main.pdf` (2026-04-14 draft). Each task is scoped so one
sub-agent can execute it end-to-end on a single RTX 3090 (24 GB), produce a CSV
+ REPORT in the standard format, and return for paper integration.

**Before doing anything, every sub-agent should read:**

1. `kv-subspace/experiments/exp24_wikitext2_ppl.py` — the **gold-standard clean
   pipeline template**. Copy its structure. Specifically:
   - WikiText-2 TRAIN split for calibration (first 2048 tokens after dropping
     empty lines), WikiText-2 TEST split for evaluation (first 2048 tokens).
     No overlap possible because splits are disjoint.
   - `chunked_cross_entropy` + `direct_ppl` cross-check (must agree within 5%).
   - Sanity-check baseline PPL is in the "expected range" for the model before
     reporting any compressed numbers. If baseline is wildly off (< 2 or > 25),
     stop and flag it — that is how we caught the W&P memorization bug.
   - Resume-from-CSV pattern so interrupted runs don't re-compute finished rows.
2. `kv-subspace/compress.py` — `subspace_compress(xh, k, n_bits, U, mean, R,
   quantizer='subrotq'|'polarquant')` is the single entry point you call from a
   forward hook. Do not re-implement quantization; import it.
3. `kv-subspace/collect.py` — `get_model_and_tokenizer(MODEL_NAME)` handles
   AWQ / non-AWQ loading, device placement, dtype. Use it.
4. `kv-subspace/results/REPORT-24-wikitext2.md` — the output format every REPORT
   file should follow (purpose, setup, table, findings, implications).

**Common protocol (all tasks):**

- Working directory: `/home/petty/Github/corpetty/mozeika-pruning-empirics/kv-subspace`
- Python: `/home/petty/torch-env/bin/python3`
- Hardware: single RTX 3090. Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- No multi-GPU. If a model + activations doesn't fit on 24 GB, document it and
  stop; don't silently fall back to a smaller context.
- Every task outputs **one CSV and one REPORT markdown**. CSV must include
  a `baseline` row (uncompressed) and all compressed rows. REPORT must include
  the baseline cross-check table and an explicit "sanity pass / fail" line.
- If a task returns surprising numbers (e.g. rel-PPL changes by >0.1× from the
  value cited in the paper), **do not paper over it**. Report the discrepancy
  in the REPORT's "Findings" section. We want to know.
- Seed all randomness. `compress.random_rotation_matrix(d, seed=0)` is already
  seeded. If you add any sampling, seed it explicitly and log the seed.
- When you return results, include: (a) the CSV path, (b) the REPORT path,
  (c) a 3-line summary of the headline number and whether it matches or
  contradicts the paper's claim, (d) total wall-clock GPU time used.

---

## Priority tiers

- **P0** — load-bearing for the paper's headline claims. The paper cannot be
  submitted without these.
- **P1** — closes provenance / transparency gaps flagged in the review.
- **P2** — nice-to-have robustness checks. Do these only if P0/P1 finish with
  time to spare before the deadline.

Tasks are independent unless a "Depends on" line says otherwise; they can be
dispatched in parallel across sub-agents.

---

## Task A1 — Missing baseline: plain 4-bit KV quantization (no PCA, no rotation) — **P0**

**Why this matters.** At the recommended operating point (k=d_head=128, 4-bit),
the PCA basis is mathematically a no-op — it's a rotation composed with another
rotation before uniform quantization. A reviewer will ask: "Is SubRotQ at
k=128/4-bit actually distinguishable from vanilla 4-bit K-quantization?" We
need to answer that directly. Without this number, the paper's "production
configuration" is indistinguishable from the field's default 4-bit baseline
and we have nothing to claim.

**What to measure.** On Qwen3-14B-AWQ, Mistral-7B-v0.3, and Llama-3.1-8B-AWQ,
run three K-only quantization variants at 4-bit, all on the clean WikiText-2
pipeline:

1. **Plain** — per-channel uniform scalar quantization of the raw K vector,
   no centering, no rotation, no PCA. This is the most honest "what does the
   field already do" baseline.
2. **Centered + uniform** — subtract per-(layer,head) mean, then per-channel
   uniform quant, then add mean back. Tests whether centering alone matters.
3. **SubRotQ at k=128** — the paper's recommended config (random rotation +
   uniform quant, PCA is trivially full-rank so it reduces to a rotation).

Also include the uncompressed baseline row.

**Expected output.**

File: `results/expA1_plain_4bit_baseline.csv`
Columns: `model,method,k,bits,ppl,rel_ppl,notes`
One row per (model × method) pair plus three baseline rows (one per model).

File: `results/REPORT-A1-plain-4bit-baseline.md`. Headline table shape:

| Model         | baseline | plain 4-bit | centered 4-bit | SubRotQ k=128/4-bit |
|---------------|----------|-------------|----------------|---------------------|
| Qwen3-14B-AWQ | 6.57     | ?           | ?              | 6.44 (from Exp 24)  |
| Mistral-7B    | 4.26     | ?           | ?              | 4.26 (from Exp 30)  |
| Llama-3.1-8B  | 6.08     | ?           | ?              | 6.13 (from Exp 32)  |

**Protocol.**

1. Copy `experiments/exp24_wikitext2_ppl.py` as a starting point.
2. Replace `eval_ppl_with_hooks` with a function that dispatches on a `method`
   argument: `plain | centered | subrotq_k128`.
3. For `plain`: in the hook, take raw `xh` of shape `(T, d_head)`, apply
   `compress.quantize_uniform(xh, n_bits=4)`, replace. No basis, no rotation.
4. For `centered`: subtract per-(layer,head) mean computed on the calibration
   pass, uniform quant, add mean back.
5. For `subrotq_k128`: call `subspace_compress(xh, k=128, n_bits=4, U, mean, R,
   quantizer='subrotq')` as in exp24. (This is the paper's method at the
   recommended operating point and should reproduce `exp24` k=128/4-bit to
   within 0.01× rel-PPL.)
6. Repeat for all three models. Each model should take ~30–60 min wall-clock
   including basis fit + three eval passes.

**Acceptance criteria.**

- Baseline PPL per model matches within 1% of Exp 24 / Exp 30 / Exp 32.
- SubRotQ k=128/4-bit rel-PPL matches the corresponding row in
  `results/exp24_wikitext2_ppl.csv`, `exp30_mistral_ppl.csv`,
  `exp32_llama3_wikitext2_ppl.csv` within 0.01×.
- The "plain" column gives us a clean number to cite in §6.2 as the field
  baseline against which to compare our contribution.

**Headline question the sub-agent should answer in the REPORT:** does SubRotQ
k=128/4-bit measurably beat plain 4-bit KV quantization on any of the three
models? If yes, by how much. If no, say so plainly — that's an important finding.

---

## Task A2 — Qwen3-1.7B WikiText-2 k×bits sweep — **P0**

**Why this matters.** Paper Table 3 reports Qwen3-1.7B k=128/4-bit rel-PPL =
1.25× (citing it as evidence that smaller models tolerate less aggressive
truncation). This number has no traceable source in the committed data:
`exp11_cross_model.md` shows Qwen3-1.7B k=128/4-bit = 1.13× but that was run on
Project Gutenberg with 512-token evals (pre-bug-fix pipeline). A reviewer will
grep the repo for provenance and find nothing matching 1.25×.

**What to run.** Replicate Exp 24 but on `Qwen/Qwen3-1.7B` (non-AWQ, full
precision). Sweep `k ∈ {64, 96, 112, 128}` × `bits ∈ {4, 8, 16}`. K-only
compression (V uncompressed).

**Output.**

- `results/exp33_qwen17b_wikitext2_ppl.csv` — identical schema to exp24.
- `results/REPORT-33-qwen17b-wikitext2.md`.

**Key setup details.**

- Qwen3-1.7B has **28 layers, 8 KV heads, d_head=128** (confirm with
  `config.num_hidden_layers` etc. at load time — REPORT should print these).
- Expected baseline PPL range on WikiText-2 test: roughly 9–14 for a 1.7B model.
  If you see <3 or >25, stop and flag.
- Not AWQ, so model should load in bf16. Fits comfortably in 24 GB.

**Acceptance criteria.**

- Headline number `rel_ppl(k=128, bits=4)` returned in the sub-agent summary.
- If the number is 1.25× ±0.05, Table 3 is vindicated and we commit the CSV.
- If it differs materially (e.g. 1.13× matching exp11, or 1.05× matching the
  larger-model trend), flag this. We will update Table 3 and the
  "large models compress better" claim accordingly.

**Expected runtime.** 45–90 min on a single 3090.

---

## Task A3 — Qwen3-32B-AWQ WikiText-2 k×bits sweep — **P0**

**Why this matters.** Paper Table 3 reports Qwen3-32B-AWQ k=128/4-bit = 0.96×
and "k=96 at 1.07× PPL, 5.33× CR". Neither number is in any committed CSV.
`exp11_cross_model.md` has Qwen3-32B at 1.04× and 1.09× — close, but measured
on contaminated Project Gutenberg with a different baseline (2.27). Without a
clean replication, the entire "compression tolerance scales with model size"
story in §4.2 / §6.1 / Table 3 rests on uncommitted data.

**What to run.** Replicate Exp 24 but on `Qwen/Qwen3-32B-AWQ`. Sweep
`k ∈ {64, 96, 112, 128}` × `bits ∈ {4, 8, 16}`. K-only compression.

**Output.**

- `results/exp34_qwen32b_wikitext2_ppl.csv`
- `results/REPORT-34-qwen32b-wikitext2.md`

**Key setup details.**

- Qwen3-32B-AWQ should load at ~18–20 GB on the RTX 3090. 24 GB is enough but
  close to the limit. Use `torch_dtype=torch.float16` and AWQ quantized weights;
  set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Qwen3-32B has **64 layers, 8 KV heads, d_head=128** (confirm at load time).
  The basis storage is 2× Qwen3-14B's because of layer count — this is fine for
  calibration but worth printing to the REPORT.
- If 2048 eval tokens on the 64-layer model OOMs, fall back to 1024 eval tokens
  and note it explicitly in the REPORT. Do **not** silently reduce tokens.
- Expected baseline PPL on WikiText-2 test: 5–7 (smaller than 14B would be a
  surprise; slightly larger is plausible).

**Acceptance criteria.**

- Headline numbers at k=128/4-bit and k=96/4-bit in sub-agent summary.
- If these match Table 3 within 0.05×, commit. If not, flag the discrepancy
  so we can update the paper.
- REPORT explicitly compares to Qwen3-14B (k=128/4-bit = 0.98×) and Qwen3-1.7B
  (from Task A2) to check monotonicity in model size.

**Expected runtime.** 2–4 hours on the 3090 (the 64-layer model is slow).

---

## Task A4 — Phi-4 WikiText-2 k×bits sweep *or* formal removal decision — **P0**

**Why this matters.** The paper abstract lists Phi-4 as one of the architectures
tested. Phi-4 appears nowhere in the paper body, and the only Phi-4 data in the
results tree is pre-bug-fix `REPORT-12-cross-arch.md`. This is the single most
flagrant mismatch between claims and evidence in the current draft; a reviewer
will catch it immediately.

**What to run.** Replicate Exp 24 on `microsoft/Phi-4` (or Phi-4-AWQ if the
14B INT4 variant is what Exp 12 used — check and match). Sweep
`k ∈ {64, 96, 112, 128}` × `bits ∈ {4, 8, 16}`. K-only compression.

**Output.**

- `results/exp35_phi4_wikitext2_ppl.csv`
- `results/REPORT-35-phi4-wikitext2.md`

**Key setup details.**

- Phi-4 architecture differs from Qwen3/Mistral/Llama. Confirm at load:
  `n_layers`, `n_kv_heads`, `d_head`. The hook pattern in exp24 assumes the
  `self_attn.k_proj` / `self_attn.v_proj` names; if Phi-4 uses different module
  names, adapt the hook — don't hardcode around it.
- Phi-4 may or may not use QK-norm. Check `config.qk_layernorm` or read the
  model config and record it in the REPORT. This is critical for the V-failure
  cross-architecture narrative in §4.6.
- If model is 14B, expect ~18 GB VRAM in bf16 or ~8 GB in AWQ INT4.
- Expected baseline PPL on WikiText-2: 5–8 (Phi-4 is strong at this size).

**Fallback option.** If Phi-4 doesn't run cleanly on our pipeline within a day
of effort (attention-module structure incompatible with our hooks, non-standard
KV layout, etc.), **return a "do-not-run" recommendation** with a short writeup
of what would be required, and we will remove Phi-4 from the abstract and
refocus on Qwen3/Mistral/Llama.

**Acceptance criteria (if run).**

- Headline k=128/4-bit rel-PPL in sub-agent summary, with QK-norm status.
- V-only sanity check at k=112/4-bit (one row): is Phi-4 V compression
  viable? This feeds directly into the V-failure universality claim.

**Expected runtime.** 1–3 hours depending on size and whether AWQ variant works.

---

## Task B1 — Re-run Exp 22 (SubRotQ vs PolarQuant) on clean WikiText-2 — **P0**

**Why this matters.** The paper's §4.9 / Table 8 quantizer comparison between
SubRotQ and PolarQuant is the only claim the paper makes about one quantization
method beating another. That comparison currently lives in
`exp22_quantizer_comparison.csv`, which was generated from
`experiments/exp22_quantizer_comparison.py` — a script that uses War and Peace
with the broken `EVAL_OFFSET = CALIB_TOKENS + 100` offset bug *and* a baseline
PPL of 1.17 (confirmed memorization regime). In that regime k=64/4-bit gives
rel-PPL 1.04, vs 8.14 on clean WikiText-2 — a completely different operating
point. The quantizer gaps we cite (+0.053 to +0.103) are measured where neither
method is really stressed.

**What to run.** Exact replay of Exp 22 with the clean pipeline:
- Calibration: WikiText-2 TRAIN 2048 tokens.
- Evaluation: WikiText-2 TEST 2048 tokens.
- `k ∈ {64, 96, 112, 128}` × `bits ∈ {4, 8}` × `quantizer ∈ {subrotq, polarquant}`.
- Qwen3-14B-AWQ.

**Output.**

- `results/exp36_quantizer_comparison_wikitext2.csv`
  (don't overwrite exp22; keep both for traceability).
- `results/REPORT-36-quantizer-comparison-clean.md`.

**Protocol.**

1. Copy `experiments/exp24_wikitext2_ppl.py` as the template (not exp22 — its
   data-loading path is the bug source).
2. Extend the hook to dispatch `quantizer='subrotq' | 'polarquant'` via the
   `subspace_compress` argument (already supported).
3. Run both quantizers at all (k, bits) pairs in the same script run so they
   share the same calibration basis. This is load-bearing: basis differences
   between runs would confound the quantizer comparison.
4. Compute "quantizer gap" = `polarquant_rel_ppl - subrotq_rel_ppl` at each
   (k, bits), like Table 8 in the paper.

**Acceptance criteria.**

- Headline answer: on clean WikiText-2, does SubRotQ still beat PolarQuant at
  4-bit, and by how much?
- If the gap shrinks to noise (e.g. <+0.01× across all k), we retract the
  §4.9 claim. That's fine — PolarQuant equivalence at k=128 would be a
  perfectly acceptable result and we update the paper.
- If the gap holds, we commit the clean CSV and cite it in place of exp22.

**Expected runtime.** 1.5–2.5 hours.

---

## Task C1 — Exp 27 downstream tasks at full-benchmark N with margin reporting — **P0**

**Why this matters.** The paper's §4.3 downstream-task claim ("near-lossless
downstream reasoning") currently rests on N=300 samples per task. At that N,
the Wilson 95% CI half-width on ARC-Challenge is ~±5pp, which means the
reported -3pp drop at k=128/4-bit is indistinguishable from zero *and*
indistinguishable from -8pp. The claim is undersupported. Additionally,
`REPORT-27-downstream.md` contains a "Confidence Margin" block showing the
HellaSwag margin flips sign from +0.168 (baseline) to -0.230 (k=128/4-bit),
which indicates the model is now marginally wrong-on-average but still lucky
on accuracy — the paper does not mention this.

**What to run.** Re-run `experiments/exp27_downstream_tasks.py` with `LIMIT`
increased from 300 to each task's full benchmark (or at minimum 1000). Add
the margin data to the summary table in the REPORT. Keep the existing
diagnostics (variance explained, basis stability, per-layer recon error).

Tasks to evaluate (all in the existing script):
- ARC-Challenge (full test: 1172 questions)
- HellaSwag (full validation: 10042 questions — subsample to 3000 for runtime)
- ARC-Easy (full test: 2376 questions)
- WinoGrande (full dev: 1267 questions)

Configs:
- baseline (no compression)
- k=128/4-bit
- k=112/4-bit
- k=96/4-bit (ARC-Challenge only — the paper already documents this as past the
  quality cliff; one data point at full N is enough to confirm the cliff is
  real and not a noise effect)

**Output.**

- `results/exp37_downstream_tasks_full.json`
- `results/exp37_diagnostics.json`
- `results/REPORT-37-downstream-full.md`

**Key protocol points.**

- Keep `per_sample` trajectory data for all configs — we need it to compute
  confidence intervals and do the margin analysis.
- The REPORT **must** include a margin table side-by-side with the accuracy
  table. Call out any sign flips or >0.5 absolute margin drops explicitly.
- Report accuracy with Wilson 95% CI half-widths.
- Report basis stability (first-half vs second-half cos sim). Exp 27 saw 0.396;
  at full benchmark this should be similar, but confirm.

**Acceptance criteria.**

- Headline: at full N, is the k=128/4-bit ARC-Challenge drop statistically
  distinguishable from zero? (p < 0.05 McNemar test, paired on question ID.)
- Margin table included and discussed.
- If accuracy is statistically indistinguishable from baseline *and* margin
  collapse is < 0.3 absolute, we can defensibly say "near-lossless". Otherwise
  we soften the claim in the paper.

**Expected runtime.** 3–6 hours (the 40-layer hook pipeline is slow during
generation; running 4 tasks × 3 configs × 1000–3000 samples each is the bulk).

---

## Task D1 — Long-context stability on a non-contaminated corpus — **P1**

**Why this matters.** §4.4 reports long-context stability on War and Peace.
`REPORT-24-wikitext2.md` explicitly flagged W&P as training-data-contaminated
for Qwen3 (baseline PPL 1.17 is memorization). `REPORT-13-long-context.md`'s
W&P baseline is 10.11 at 40K (less egregious, because the eval offset is
further out), but the corpus is still contamination-adjacent. The relative-PPL
result (1.05–1.11× stable across 512 → 40K) is probably robust, but a reviewer
will note the source and we need a defensible answer.

**What to run.** Replicate `experiments/long_context_scaling.py` with a
non-contaminated long-form corpus. Options (pick the first that works):

1. **PG-19 validation split** (`deepmind/pg19`, validation split). Hugging Face
   dataset, ~50 books, each long enough for 40K-token eval windows. PG-19 is
   Gutenberg-19xx; some overlap with pretraining is possible but much less
   than the obvious classics.
2. **arXiv post-training-cutoff** — papers published after 2024-12 (Qwen3
   cutoff). Harder to source cleanly; use only if PG-19 doesn't work.
3. **Stack Exchange threads** from after the model cutoff, concatenated. Also
   harder to source at 40K token length.

Run configs `k ∈ {128, 96}/4-bit` on Qwen3-14B-AWQ, context lengths
`{512, 1K, 2K, 4K, 8K, 16K, 32K, 40K}`.

**Output.**

- `results/exp38_long_context_pg19.csv`
- `results/exp38_long_context_per_token.csv`
- `results/exp38_long_context_basis_drift.csv`
- `results/REPORT-38-long-context-pg19.md`

**Protocol.**

1. Load one book from PG-19 validation split that has ≥ 50K tokens. Pick a
   book not obviously famous (the idea is to minimize memorization). Record
   the book title + author in the REPORT.
2. Baseline PPL at 512 tokens should be reasonable (5–15 range). If it comes
   in under 3, the book is memorized — pick a different one.
3. Otherwise, replicate the three sub-experiments from exp13 exactly:
   A. PPL vs context length per config.
   B. Per-token loss decile at 16K.
   C. PCA basis drift (K and V) between early / mid / late document positions.

**Acceptance criteria.**

- REPORT's headline: does `k=128/4-bit` rel-PPL stay within 1.05–1.15× across
  512 → 40K tokens on non-memorized text? If yes, §4.4's claim holds. If no
  (e.g. rel-PPL climbs to 1.25× at 40K), we revise the claim.
- Basis drift K / V numbers reported and compared to exp13's 0.825 / 0.702.
- Book title recorded so reviewers can verify independence from training data.

**Expected runtime.** 3–5 hours (long contexts are expensive; 40K on a
40-layer model takes ~20 min per config).

---

## Task D2 — Cross-domain calibration on non-memorized corpora — **P1**

**Why this matters.** `REPORT-17-cross-domain.md` baseline PPLs are:
`fiction=13.25, code=1.28, news=2.53, dialogue=2.86`. Code at PPL 1.28 is
memorization-territory; the other low numbers are suspicious. The cross-domain
heatmap in Figure 7 also has a fiction→fiction cell at 3.55× (k=128) / 5.96×
(k=96) that is a calib/eval overlap artifact per `SUMMARY.md` line 242 but is
not labelled as such in the paper caption. The entire §4.8 claim rests on
numbers from contaminated corpora.

**What to run.** Re-run cross-domain calibration × evaluation at
`k=128/4-bit` and `k=96/4-bit` on Qwen3-14B-AWQ, using four non-contaminated
domains:

1. **Fiction** — BookCorpus (check for Qwen3 memorization; probably partial)
   or PG-19 validation split with a non-famous book.
2. **Code** — The Stack v2 subset from after Qwen3 cutoff, or a small Python
   project by a post-cutoff author. Verify baseline PPL is > 2 before running.
3. **News** — CommonCrawl news from after Qwen3 cutoff, or a recent article
   from a reputable source.
4. **Dialogue** — SHP or OpenAssistant after cutoff, or a recent
   podcast transcript.

All four corpora must pass a memorization sanity check: baseline PPL > 3 on a
512-token held-out segment. If a domain's baseline is lower, replace it.

**Output.**

- `results/exp39_cross_domain_clean.csv`
- `results/REPORT-39-cross-domain-clean.md`

**Protocol.**

1. For each domain, tokenize 4096 tokens total, split into `calib[:2048]`
   (for basis) and `eval[2048:4096]` (for evaluation). This guarantees no
   overlap.
2. Compute baseline PPL on each domain's eval segment. Any baseline < 3 → stop
   and swap the corpus.
3. Run the full 4×4 calibration × evaluation matrix at `k=128/4-bit` and
   `k=96/4-bit`.
4. Compute **relative PPL vs same-domain baseline**, not raw PPL, so the
   heatmap is interpretable regardless of per-domain PPL levels.
5. Mark the diagonal cells that are same-domain (not "calib overlap").

**Acceptance criteria.**

- At k=128/4-bit, maximum cross-domain rel-PPL ≤ 1.10× (the paper currently
  claims "within 0.05× of same-domain"). If the new experiment confirms this,
  §4.8 stands. If cross-domain penalty is larger on clean data, we revise.
- At k=96/4-bit, we can confirm or refute the brittleness claim.
- REPORT lists corpus sources + baseline PPL per domain so reviewers can
  verify non-memorization.

**Expected runtime.** 2–3 hours.

---

## Task E1 — Verify Algorithm 1 math: what exactly is stored, and how many bytes? — **P1**

**Why this matters.** The paper says basis storage is "~45 MB for Qwen3-14B"
but the arithmetic gives `40 × 8 × 128² × 4 bytes = 20.97 MB` for a full-rank
basis in float32, or `≈ 18 MB` for k=112. Something in the accounting is off.
This is the kind of small error a reviewer uses to assess whether the authors
actually checked their own numbers.

**What to compute.** A short scripted accounting, no model inference needed.

1. Load the existing calibration basis for Qwen3-14B from wherever it's
   cached (or recompute via `exp24`'s `collect_kvs_for_basis` + `fit_bases`).
2. Print and sum the actual byte-size of every stored tensor per (layer, head):
   `U_K`, `mean_K`, `U_V`, `mean_V`, any rotation `R`, any scale/offset tables.
3. Report the total in MB for three configs: (full basis float32), (full basis
   float16), (k=112 truncated basis float16).
4. Include the per-(layer, head) quantization scale/offset footprint at
   inference time — for 8 heads × 40 layers × (scale+offset per dimension or
   per head?) × 4 bytes. This is the *per-token* state, distinct from the
   *per-(layer, head)* basis. The paper conflates them.

**Output.**

- `results/expE1_basis_storage_audit.md` (single markdown file, no CSV needed).
- `results/expE1_basis_storage_audit.json` (structured numbers for the LaTeX).

**Acceptance criteria.**

- Three numbers the paper can cite directly:
  - Per-(layer, head) **offline** basis storage (MB, for each of the three
    candidate configs).
  - Per-token **online** state added to the KV cache (bytes per token, so we
    can say e.g. "adds X bytes/token of scale+offset overhead").
  - Total overhead of the method at 32K context (MB).
- Correct formula written out in the REPORT so the number reproduces.

**Expected runtime.** 15–30 minutes. No inference needed; can run on CPU.

---

## Task E2 — Per-channel 4-bit KVQuant-style comparison — **P2**

**Why this matters.** Beyond plain per-channel quant (Task A1), the closest
published baseline at 4 bits is KVQuant (Hooper et al., 2024), which uses
per-channel quantization with outlier preservation (top-k% outliers kept in
fp16, rest at 4-bit). The paper cites KVQuant but does not benchmark against
it. Task A1 gives us the "no PCA, no rotation, no outliers" floor; this task
gives us a fairer published baseline.

**What to run.** On Qwen3-14B-AWQ + WikiText-2 clean pipeline, implement a
minimal KVQuant-style K-only compression:

1. Per-channel scale/offset (same as plain 4-bit).
2. Identify top 1% of channels by magnitude variance and store those in fp16
   instead of quantizing.
3. Rest at 4-bit uniform.
4. Report rel-PPL and effective bit-budget (accounting for the fp16 outlier
   channels).

Compare directly to SubRotQ k=128/4-bit.

**Output.** `results/expE2_kvquant_comparison.csv` +
`results/REPORT-E2-kvquant-comparison.md`.

**This is P2.** Only pursue if A1–D2 all land and we have time before the
deadline. If forced to cut, cut this first.

**Expected runtime.** 2–4 hours including implementation.

---

## Task F1 — Mechanism ablation: does random rotation at k=128 actually help? — **P1**

**Why this matters.** §5 of the paper offers hand-wavy explanations ("random
rotation decorrelates," "quantization noise averages out") for why k=128/4-bit
works. None of these is backed by a measurement. A cheap, directly testable
sub-claim is: **is the random rotation step doing any work at k=128, or does
centered uniform quantization alone suffice?** If rotation is doing nothing,
the paper should say so; if it's doing a lot, we have a real mechanism story.

**What to run.** On Qwen3-14B-AWQ + WikiText-2, at k=128/4-bit, compare four
variants:

1. No rotation, no centering, per-channel uniform quant (same as Task A1
   "plain").
2. Centering only, no rotation.
3. Rotation only, no centering (pre-rotate, quantize, unrotate).
4. Centering + rotation (= SubRotQ k=128).

Sweep `n_bits ∈ {2, 3, 4, 6, 8}` to see where rotation matters most. At 8-bit
we expect all four to be equivalent; at 2-bit we expect rotation to help a lot
(this is the whole point of QuIP / QuaRot-style approaches).

**Output.** `results/expF1_rotation_ablation.csv` +
`results/REPORT-F1-rotation-ablation.md`.

**Acceptance criteria.**

- A plot-ready table showing rel-PPL for the 4 variants × 5 bit depths.
- Headline: "At k=128/4-bit, rotation buys X× PPL improvement over plain
  per-channel quant. At 2-bit, rotation buys Y×. At 8-bit, no difference."
- This gives §5 a real mechanism sentence instead of hand-waving.

**Expected runtime.** 1.5–2.5 hours.

---

## Task G1 — Commit existing uncommitted data (if any) — **P1**

**Why this matters.** The paper's Table 3 numbers for Qwen3-1.7B (1.25×) and
Qwen3-32B (0.96× / 1.07×) don't appear in any committed CSV, but there's some
chance they came from a local run that was never saved. Before running Tasks A2
and A3 from scratch, check the author's working directories for any notebook
output, scratch script, or stale CSV that might contain the Table 3 numbers.
If found, we need to (a) commit it, (b) document which experiment script
generated it, and (c) decide whether the pipeline was clean or needs re-run.

**What to do.**

1. `grep -r "1.25" kv-subspace/ --include="*.csv" --include="*.json" --include="*.md"`
2. `grep -r "0.96" kv-subspace/ --include="*.csv" --include="*.json" --include="*.md"`
3. Check `~/Downloads`, `/tmp`, Jupyter checkpoint dirs, SLURM log dirs.
4. Check git stash.

**Output.** A short memo: either "found it, here's the file, here's the
experiment" or "confirmed nothing exists, A2 and A3 must run from scratch."

**This is a no-compute task.** 15 minutes of disk search.

---

## Recommended sub-agent dispatch

Assume 3 sub-agents and a 48-hour budget for P0 work.

**Sub-agent 1 (~16h):**
- Task G1 (15 min)
- Task A1 (~3h: 3 models × 1h each)
- Task A2 (~1.5h: Qwen3-1.7B)
- Task A3 (~3h: Qwen3-32B, the slow one)
- Task A4 (~2h: Phi-4 if feasible)

**Sub-agent 2 (~12h):**
- Task B1 (~2h: quantizer comparison)
- Task C1 (~6h: full-N downstream tasks)
- Task F1 (~2h: rotation ablation)

**Sub-agent 3 (~10h):**
- Task E1 (~30 min: storage audit, CPU only)
- Task D1 (~4h: PG-19 long context)
- Task D2 (~3h: cross-domain clean)

**Reserve for P2:**
- Task E2 (KVQuant comparison) — only if there's time after P0+P1 land.

All tasks return via the "common protocol" block at the top: CSV path,
REPORT path, 3-line summary, wall time. I will integrate the results into a
single revision pass on the paper.

---

## Integration checklist (for me, not sub-agents)

Once results land, the writing-side tasks that do **not** need sub-agents:

- Reframe abstract and §3 around the "PCA is a no-op at k=d" observation
  (no new experiments needed).
- Fix "~6×" → "~4×" (arithmetic: 7.16 / 1.89 = 3.79).
- Fix "45 MB" → whatever Task E1 reports.
- Fix "10 GB" → "5.24 GB" (K-only at 32K, per Exp 29).
- Fix "5 architectures" to "4 architecture families, N model variants" where
  N is determined after Task A4's feasibility decision.
- Fill in placeholder arXiv IDs (SQuat, SVDq).
- Verify Kaushik et al. UWSH citation is a real paper (arxiv 2512.05117 has a
  suspicious ID).
- Soften "production-ready" → "production-viable pending fused kernel."
- Promote the Exp 28 → Exp 18 self-correction to a named paragraph in §4.7
  (self-correction is a credibility signal, not a footnote).
- Rewrite §5 as "Hypotheses" unless Task F1's results give us real mechanism
  data, in which case rewrite §5 as "Mechanism" with F1 cited.
- Add a Limitations bullet acknowledging N=300 was underpowered pre-Task-C1.

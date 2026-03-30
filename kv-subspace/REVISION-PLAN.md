# Subspace PolarQuant — Revision Plan

**Paper:** Subspace PolarQuant: KV Cache Compression via PCA Projection and Rotation Quantization
**Author:** Corey Petty, Institute of Free Technology
**Date of review:** March 30, 2026
**Repository:** github.com/corpetty/mozeika-pruning-empirics/tree/master/kv-subspace

This document consolidates all findings from a code-grounded peer review of the paper and experimental codebase. Issues are organized by severity, each with a diagnosis, evidence, and concrete remediation path.

---

## Summary Verdict

The experimental infrastructure is strong: 21 experiments, 5 architectures, clean reproducible code, real hardware measurements on an RTX 3090. The truncation-dominance finding and cross-architecture K-V asymmetry documentation are genuine contributions. However, the paper has five critical problems that would each independently warrant rejection at a competitive venue: the implementation is not PolarQuant, the headline number contradicts the paper's own long-context data, major prior art is missing, Figure 8 contains a data bug, and Experiment 19 is non-functional. All are fixable with the existing codebase. Below is the full remediation plan.

---

## CRITICAL — Must Fix Before Publication

### C1. The Implementation Is Not PolarQuant

**Diagnosis:** The code in `compress.py` (lines 13–42) implements random orthogonal rotation via QR decomposition of Gaussian noise followed by uniform scalar quantization. The actual PolarQuant paper (Han et al., arXiv:2502.02617) implements a fundamentally different algorithm: recursive polar coordinate conversion across log₂(d) levels, followed by 1D k-means++ codebook quantization per angle dimension. These are not the same algorithm.

**Compounding issue:** Two independent papers named "PolarQuant" appeared in February 2025. The one at arXiv:2502.02617 (Han et al., KAIST/Google/Yale, AISTATS 2026) uses random preconditioning + polar transformation + k-means codebooks. A second at arXiv:2502.00527 (Wu et al., NeurIPS 2025) exploits 2D RoPE-rotated dimension pairs using lookup tables. The paper's bibliography cites arXiv:2502.02617 but the description matches neither.

**Evidence:**
- `compress.py` line 13: `random_rotation_matrix()` via `np.linalg.qr(A)` — correct that rotation is random, but the quantization step is wrong
- `compress.py` lines 45–58: `quantize_uniform()` — per-dimension min/max uniform scalar quantization, not polar coordinates or k-means codebooks
- `paper/main.tex` lines 95–96: describes PolarQuant as "a learned-rotation quantization scheme" — incorrect on both counts (rotation is random, and the algorithm is different)
- `EXPLAINER.md` line 52, `SUMMARY.md` line 13: same "learned rotation" mischaracterization propagated to all user-facing docs

**Remediation — choose one of:**

Option A (recommended, lower effort): Rename the method to "Subspace Rotation Quantization" or "PCA + Random Rotation Quantization." Describe the rotation as random orthogonal preconditioning. Cite both PolarQuant papers accurately and explain that your simplified pipeline uses the same random rotation principle but replaces the polar coordinate transform and k-means codebooks with uniform scalar quantization. This is honest and reviewable.

Option B (higher effort, stronger paper): Implement the actual PolarQuant algorithm (polar coordinate conversion + k-means codebooks) and re-run the experiments. This would let you keep the name and potentially improve compression quality, since k-means codebooks adapt to the actual distribution better than uniform quantization.

**Files to modify:** `compress.py`, `paper/main.tex` (lines 52–53, 94–96, 128–132), `EXPLAINER.md` (line 52), `SUMMARY.md` (line 13), `RESULTS.md` (lines 20, 52), `kvpatch/` docstrings.

---

### C2. Headline Number Contradicts Own Long-Context Data

**Diagnosis:** The abstract claims "k=112/4-bit achieves 4.27× compression at 14% perplexity cost." This number comes from Table 1 / Experiment 9, which evaluates on three hardcoded encyclopedic passages at 512 tokens with baseline PPL ~2.6. The paper's own Table 4 (War and Peace, realistic context lengths) shows k=112/4-bit fails the 20% viability threshold at every single context length tested: rel-PPL ranges from 1.35× at 8K to 1.68× at 512 and 40K tokens. The only configuration that passes across all context lengths is k=128/4-bit (full rank, no subspace truncation).

**Evidence from the data:**

Table 1 (short encyclopedic passages, 512 tokens): k=112/4-bit → 1.14× rel-PPL
Table 4 (War and Peace): k=112/4-bit at 512 tokens → 1.68× rel-PPL, at 8K → 1.35×, at 40K → 1.68×
Table 4: k=128/4-bit passes 20% threshold at all 8 context lengths (range: 1.05–1.11×)

The 1.14× vs 1.68× discrepancy at the same context length on different text indicates the encyclopedic passages are unusually compression-friendly.

**Source files:**
- `results/bitrate_k_sweep.csv` — Table 1 data (512-token eval passages)
- `results/long_context_ppl.csv` — Table 4 data (War and Peace, 512–40K tokens)
- `experiments/bitrate_k_sweep.py` lines 30–150 — hardcoded eval passages

**Remediation:**

Lead with k=128/4-bit as the primary recommended configuration (4.00× CR, <12% PPL cost, stable across all context lengths and text types). Reposition k=112/4-bit as an architecture-dependent aggressive mode that works for some models (Mistral, Phi) but not reliably for Qwen3 on natural text. Add WikiText-2 perplexity as a standard reference point. Either reconcile the Table 1 vs Table 4 discrepancy explicitly in the paper or replace Table 1's eval passages with a standard benchmark.

The abstract should read something like: "At k=128/4-bit, Qwen3-14B achieves 4.00× KV cache compression with <12% perplexity cost, stable across 40K context lengths." The subspace contribution becomes the analysis of when and why truncation is viable, rather than a blanket recommendation.

---

### C3. Missing Major Prior Art

**Diagnosis:** The related work section (Section 2) cites only 7 papers and misses the most directly overlapping published work. The paper's core approach — PCA projection combined with quantization for KV caches — has been explored extensively in 2024–2025.

**Papers that must be cited and discussed:**

KVTC (Staniszewski & Łańcucki, NVIDIA) — Published Nov 2025, accepted at ICLR 2026. Combines PCA-based feature decorrelation with dynamic-programming-optimal bit allocation and DEFLATE entropy coding. Achieves 20× compression with <1 point accuracy drop on reasoning and long-context benchmarks. This is the single most important missing reference: it does essentially the same thing (PCA + quantization) but achieves 5× better compression.

Palu (Chang et al.) — ICLR 2025. Decomposes KV projection weight matrices via SVD, applies Walsh-Hadamard transforms, then quantizes latent representations. Achieves 11.4× compression. Notably, Palu successfully compresses V vectors by fusing the reconstruction matrix into the output projection — directly relevant to the V intractability claim.

MatryoshkaKV — ICLR 2025. Trains orthogonal projections via knowledge distillation for 60–75% compression of both K and V. Uses a Matryoshka-style nesting that enables adaptive per-layer compression at runtime.

Eigen Attention — EMNLP 2024. Performs attention computation directly in PCA-reduced space rather than reconstructing to full dimension. Provides a different decomposition of the truncation-vs-quantization tradeoff.

Additional relevant work: xKV (2025, cross-layer SVD sharing), LoRC (NeurIPS 2024, progressive low-rank compression), ReCalKV (2025, head reordering + offline calibration), SVDq (Feb 2025, SVD basis + importance-aware mixed-precision), SQuat (2025, subspace-orthogonal KV cache quantization), KIVI (ICML 2024, asymmetric K-vs-V quantization), GEAR (NeurIPS 2024 Workshop, low-rank error correction).

**Remediation:**

Expand Section 2 significantly. Add a "Dimensionality Reduction" paragraph covering KVTC, Palu, Eigen Attention, MatryoshkaKV, and xKV. Add a "Combined Approaches" paragraph covering SVDq, SQuat, and GEAR. Clearly articulate what the paper adds beyond KVTC (if anything — this is the key question a reviewer will ask). The honest answer may be: "We provide a simpler pipeline with lower engineering complexity and a systematic empirical characterization of the truncation-vs-quantization tradeoff across more architectures than prior work."

Update `paper/refs.bib` with all missing references.

---

### C4. Figure 8 (Cross-Domain Transfer) Has a Data Bug

**Diagnosis:** The fiction column in both heatmaps of Figure 8 shows relative PPL values of ~0.40 for most calibration domains, implying compression improves quality — physically impossible for lossy compression.

**Root cause:** In `experiments/exp17_cross_domain.py` (lines 411–413), when `calib_domain == eval_domain`, the code uses the second half of the text for evaluation. For fiction (War and Peace), this produces a baseline PPL of 10.53 (second half of the novel). When calibrating on any other domain and evaluating on fiction, the code uses the full text starting from the beginning, producing a baseline PPL of 1.23. The `make_figures.py` script (line 334) computes the fiction baseline as the mean across all calibration domains: (10.53 + 1.23 + 1.23 + 1.23 + 1.23) / 5 = 3.09. Dividing the non-fiction-calibrated fiction PPL (~1.25) by this inflated mean (~3.09) produces the impossible 0.40× values.

**Secondary concern:** The fiction baseline PPL of 1.23 (beginning of War and Peace) is suspiciously low and may indicate the model has memorized the opening passages, which would compromise the evaluation further.

**Evidence:** From `results/exp17_cross_domain.csv`, all baseline fiction PPL values except fiction→fiction are 1.2318, while fiction→fiction is 10.5302 — an 8.5× spread for the same eval domain.

**Source files:**
- `experiments/exp17_cross_domain.py` lines 411–413 — the conditional eval text split
- `paper/make_figures.py` lines 325–355 — the averaging that produces the bug
- `results/exp17_cross_domain.csv` — the raw data

**Remediation:**

Fix the eval logic so that every (calib, eval) pair uses the same eval text for a given eval domain, regardless of calibration domain. The cleanest approach is to always use a held-out portion of each domain's text for evaluation (e.g., last 25%) and use only the first portion for calibration. Then re-run the experiment and regenerate Figure 8. Also investigate the 1.23 fiction baseline — if the model has memorized early War and Peace, use a different eval offset.

---

### C5. Experiment 19 Is Non-Functional

**Diagnosis:** Every single row in `results/REPORT-19-online-basis.md` produces identical PPL (11.5819) regardless of update strategy (window vs EMA) or update interval (256 to 2048 tokens). This is the signature of a no-op: the online basis updates are not being applied to subsequent compression operations.

**Likely bugs in `experiments/exp19_online_basis.py`:**

1. Line 301: The drift logging compares the current V basis against the K basis (`U_init, _ = k_bases[key2].get()`) rather than the initial V basis. This means the drift metric measures the wrong thing.

2. Lines 279–292: The basis update counter increments only when `li == 0` (layer 0), but the update mechanism relies on `v_capture` which accumulates vectors across all layers. The interaction between the layer-0 gating and the cross-layer capture buffer may result in updates that don't propagate correctly.

3. All update intervals produce exactly 1 basis update regardless of interval length, suggesting the counter logic fires exactly once and then either the update doesn't stick or subsequent evaluations don't pick up the new basis.

**Impact:** The paper cites Experiment 19 as evidence that "online basis updating cannot close [the V compression] gap" and "the failure is structural, not a drift problem" (REPORT-20, line 81). This conclusion is unsupported if the updating code never actually executed.

**Remediation:**

1. Fix the drift comparison: compare current V basis against the initial V basis, not the K basis.
2. Add instrumentation to verify updates are actually occurring: log the norm difference between pre- and post-update V bases, and verify the updated basis is used in subsequent compression calls.
3. Re-run with the fixed code and updated instrumentation.
4. If online updating still doesn't help, the structural conclusion stands but with stronger evidence. If it does help, the V intractability claim needs revision.

---

## MAJOR — Significantly Weakens the Paper

### M1. V Intractability Claim Is Overstated

**Diagnosis:** The paper claims V vectors "resist subspace compression at any k < d_head, regardless of whether the architecture uses QK-norm" and calls this "architecture-independent failure." While the experimental evidence for V being harder to compress than K is strong and consistent across Qwen3 and Llama, the claim of universal intractability is contradicted by multiple published methods.

**Literature that successfully compresses V vectors:**
- Palu compresses V to ~d_head/2 rank by fusing the V reconstruction matrix into the output projection Wo
- KVTC applies PCA to both K and V, achieving 20× total compression
- MatryoshkaKV trains orthogonal projections for 60–75% compression of both K and V
- DeepSeek MLA jointly compresses K and V into a shared latent with 93.3% cache reduction

**What the data actually supports:** V vectors resist offline PCA + uniform scalar quantization compression at k < d_head across Qwen3 and Llama architectures. Whether V compression is viable through alternative quantization schemes, learned projections, or weight fusion remains an open question addressed by other work.

**Note:** This claim is further weakened by the Experiment 19 bug (C5 above), since that experiment was supposed to rule out basis drift as the failure mode.

**Remediation:** Rewrite Section 4.6 and the abstract to frame V compression as "harder" rather than "intractable." Cite Palu, KVTC, and MatryoshkaKV as methods that achieve V compression through different techniques. Acknowledge that the failure is specific to offline PCA + uniform quantization, not universal.

---

### M2. Evaluation Protocol Is Below 2025–2026 Standards

**Diagnosis:** The paper evaluates on three hardcoded encyclopedic passages at 512 tokens (Table 1) plus needle-in-a-haystack (Table 5). This is insufficient for a KV cache compression paper submitted in 2026.

**What the field expects (based on KVTC, PolarQuant, Palu, MatryoshkaKV):**

Tier 1 (necessary): WikiText-2 and/or C4 perplexity as standard baseline. This takes ~10 minutes to run and provides a universally comparable reference point.

Tier 2 (expected at top venues): Downstream reasoning tasks via lm-eval-harness — MMLU, GSM8K, ARC-Challenge, HellaSwag, BoolQ. KVTC evaluates on AIME25, LiveCodeBench, GSM8K, MMLU, Qasper, RULER, and MATH-500.

Tier 3 (differentiating): Long-context benchmarks — LongBench (most widely used), RULER (subsumes NIAH with additional complex retrieval patterns), InfiniteBench for 100K+ contexts.

Tier 4 (system-level): Throughput (tokens/second), time-to-first-token latency, peak memory, max batch size.

**Remediation:** At minimum, add WikiText-2 perplexity and 3–5 downstream tasks (MMLU, GSM8K, ARC-C, HellaSwag). LongBench would be a strong addition. Wall-clock latency/throughput measurements are necessary to contextualize the 13× hook overhead.

---

### M3. Needle-in-Haystack Has n=3 Per Cell

**Diagnosis:** Each cell in the 5×4 NIAH grid (Figure 5) contains exactly 3 trials (one per needle phrase). The possible accuracy values per cell are 0%, 33%, 67%, or 100%. The headline comparison of 97% (k=128) vs 93% (baseline) is a difference of 2 correct answers out of 60 total trials. A binomial confidence interval at n=60 is approximately ±6%, making the difference statistically insignificant.

**Evidence:** `results/exp15_needle.csv` contains 180 rows (3 configs × 4 context lengths × 5 depths × 3 needles).

**Remediation:** Increase to at least n=10 needle phrases per cell (200 trials per config). Alternatively, use RULER, which provides a standardized multi-pattern retrieval benchmark that subsumes simple NIAH.

---

### M4. No Latency Measurements for Practical Assessment

**Diagnosis:** The `kvpatch/patcher.py` hook implementation does `.detach().cpu().float()` on every K/V projection output at every layer for every token, processes in NumPy, then converts back to device dtype and CUDA. This CPU roundtrip produces the ~13× decode overhead noted in Experiment 14. The paper claims "a fused CUDA kernel would reduce this to ~1.2×" but provides no profiling data to support this estimate.

**Impact:** Without latency measurements, the memory savings claims exist in a vacuum. A 4× memory reduction that makes decoding 13× slower is not a practical win for most deployment scenarios.

**Remediation — choose one of:**

Option A (minimum): Profile the 13× overhead to break it down into: CPU→GPU transfer time, NumPy compute time, GPU→CPU transfer time, and kernel launch overhead. This would let you estimate the fused-kernel improvement with some rigor.

Option B (stronger): Implement a minimal CUDA kernel for the project-rotate-quantize path and measure actual overhead. Even a naive Triton kernel would provide a realistic estimate.

Option C (honest minimum): Remove the "~1.2× estimated" claim, state the 13× overhead honestly as a limitation, and note that production deployment requires a fused kernel without estimating the improvement.

---

## MINOR — Should Fix but Not Blocking

### m1. "Learned Rotation" Appears in All Documentation

Every user-facing document (paper, EXPLAINER.md, SUMMARY.md, RESULTS.md) consistently describes the rotation as "learned." The code correctly uses a random orthogonal matrix. Replace "learned rotation" with "random orthogonal rotation" throughout. This is a quick find-and-replace but is important because "learned" implies a training step that doesn't exist, which would confuse anyone trying to reproduce the work.

### m2. kvpatch Default Calibration Overlaps Eval Passage

The biology passage ("The mitochondria are membrane-bound organelles...") appears both in `kvpatch/calibration.py` (default calibration corpus) and in `experiments/bitrate_k_sweep.py` (eval passage 0). While Experiment 9 calibrates on War and Peace (separate from eval), anyone using the `kvpatch` library out of the box and evaluating on the paper's passages would get artificially favorable results. Replace the default calibration text with non-overlapping passages.

### m3. Adaptive Scheduling Improvement Is Within Noise

Table 7 reports rank-proportional scheduling at 1.132× vs uniform at 1.153× — a 1.8% improvement from a single evaluation run across three 512-token passages. With no error bars, confidence intervals, or repeated seeds, this is not distinguishable from run-to-run variance. Either add confidence intervals (run 5+ seeds) or reframe as a trend rather than a finding.

### m4. Compression Ratio Does Not Account for Quantization Metadata

The CR formula `(d × 16) / (k × bits)` doesn't include the per-channel scale and offset required by uniform quantization. For k=112 this overhead is negligible (<1%), but it should be noted as a theoretical upper bound.

### m5. Cross-Architecture Results Untested at Long Contexts

Table 3 (Mistral-7B, Phi-4) evaluates on the same three 512-token encyclopedic passages. The finding that Mistral tolerates k=64 may not survive evaluation on longer contexts, given how different the short-passage and long-context results are for Qwen3. At minimum, note this limitation. Better: run k=64/4-bit on Mistral at 4K+ context.

### m6. UWSH Extension Is a Stretch

The UWSH (Kaushik et al., 2025) is about cross-model weight-level universality, not within-model activation structure. Extending it to KV activation subspaces conflates two distinct phenomena. Reframe the UWSH connection as motivational context ("UWSH provides a theoretical explanation for why KV compression works") rather than claiming UWSH directly extends to KV caches. The existing KV cache literature already provides independent empirical evidence for low-rank KV structure.

### m7. DeepSeek MLA Relationship Should Be Discussed

MLA's learned down-projection is mathematically analogous to a learned PCA basis, but MLA is an architectural modification requiring training from scratch while the paper's approach is post-hoc. This distinction is valuable and worth discussing explicitly, since MLA demonstrates the viability of aggressive low-rank KV compression when the model is trained for it. Also cite the MHA2MLA paper (ACL 2025) which bridges the two paradigms.

---

## Suggested Reframing of the Paper

The data supports a more modest but defensible contribution than what the current draft claims. Here is a suggested repositioning:

**Title:** Could keep the current title or adjust to "Empirical Characterization of Subspace KV Cache Compression Across Transformer Architectures"

**Core contribution 1 — Truncation dominance:** The truncation-vs-quantization decomposition (Experiment 9) is clean, novel in its comprehensiveness, and actionable. Lead with: "The critical design variable for KV cache compression is subspace dimension k, not bit depth. Practitioners should maximize k before minimizing bits."

**Core contribution 2 — Architecture-specific compression profiles:** The cross-architecture sweep (Qwen3 family, Mistral, Phi-4, Llama) provides genuinely useful guidance. Different architectures have dramatically different compression thresholds (Mistral tolerates k=64; Qwen3-14B needs k=112+; Llama needs k=128). This hasn't been documented at this breadth before.

**Core contribution 3 — K-V asymmetry characterization:** The observation that V vectors are substantially harder to compress than K vectors, documented across more architectures and with more controls than prior work, is a useful contribution when framed appropriately (as relative difficulty, not impossibility).

**Recommended operating point:** k=128/4-bit (4.00× CR) as the safe default, with k=112/4-bit (4.27× CR) for Mistral and Phi architectures. K-only subspace compression with V full-dimensional quantization.

**What to let go of:** The claim that subspace projection at k=112 is the recommended config (it fails on long context), the V intractability absolute (contradict by literature), and the PolarQuant branding (it's a different algorithm).

---

## Action Items in Priority Order

1. Decide on PolarQuant naming (Option A: rename, or Option B: reimplement). This affects the entire paper and should be settled first.
2. Fix Experiment 19 bugs (C5). Re-run. This determines whether the V claim needs softening beyond what the literature already requires.
3. Fix Figure 8 data bug (C4). Re-run Experiment 17 with correct eval logic.
4. Revise the abstract and headline numbers (C2). Lead with k=128/4-bit.
5. Expand related work section (C3). Add all missing citations.
6. Add WikiText-2 perplexity and 3–5 downstream benchmarks (M2).
7. Increase NIAH trials or replace with RULER (M3).
8. Add latency profiling or remove the 1.2× estimate (M4).
9. Soften V intractability claim (M1).
10. Fix all minor issues (m1–m7).
11. Re-run cross-architecture experiments at longer context (m5).
12. Write revised paper with new framing.

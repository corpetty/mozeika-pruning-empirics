# Active Research Threads

## 1. Pruning & Sparsity (Theoretical)

**Entry point:** Mozeika & Pizzoferrato 2026 (in `papers/`)

**Core question:** Can we get principled sparse networks that run efficiently locally without sacrificing capability?

**Why it matters for Corey's setup:** The gap between local 70B models and cloud frontier models is partly architectural, partly about inference cost. Sparsity techniques that are principled (not just magnitude heuristics) could close that gap faster. MoE is already a form of learned sparsity — this framework could inform better gating.

**Open threads:**
- Follow-up work since the Mozeika/Pizzoferrato paper posted today
- NeurIPS 2024 fundamental limits paper — get and summarize
- How does this connect to the Lottery Ticket Hypothesis work? (Frankle & Carlin 2019 + follow-ups)
- Renormalization Group + winning tickets (Universality paper cited in the paper)

---

## 2. Architecture Beyond Transformers

**Core question:** What replaces/extends the transformer for tasks where LLMs hit walls?

**Corey's framing:** LLM architecture is limiting. Weight alteration is hard. RAG is fundamentally bounded by embedding space size.

**Directions worth tracking:**
- State space models (Mamba, S4) — continuous-time dynamics, better long-range, O(n) inference
- Neuromorphic / spiking networks — different compute paradigm entirely
- Hyperdimensional computing — fast, energy-efficient, very different representational space
- Memory-augmented networks (differentiable neural computers) — explicit external memory vs. baked-in weights
- Hopfield networks / modern Hopfield (as cited in the paper) — energy-based associative memory, connects to the stat mech framework

**Key insight from today's conversation:** The weight-alteration / catastrophic forgetting problem is fundamental to the transformer/gradient-descent paradigm. Biological systems don't have this. Something with more local, sparse, dynamic updates is probably part of the answer.

---

## 3. Daemon.ai / IFT AI Initiative

See `daemon/daemon-context.md` — fully documented.

**What it is:** Vertically integrated local/cloud/uncensored AI platform. Qt6/QML client (desktop + Android), anonymous cloud routing, frontier model abliteration service, decentralized training (planned). Single author building solo. Website live, cloud router live, models disabled for now.

**The connection to the pruning research:**
- Abliteration (surgical removal of safety behavior) = targeted parameter intervention — the Mozeika/Pizzoferrato framework for principled weight selection is directly relevant
- Decentralized training uses "Protocol Models" (~100× communication compression) — structurally the same problem as principled sparsity
- The "Safety Tax" framing (Huang et al. 2025) is the product/market thesis; the stat mech framework is the theoretical backing

**Key open question:** Is Logos/IFT funding the Mozeika/Pizzoferrato work specifically to support Daemon's technical roadmap, or are these parallel efforts? The timing and institutional overlap (both IFT-connected) suggests coordination.

---

## 4. Local Inference Trajectory

**Current state (March 2026):** 70B models local but noticeably behind frontier. Good for 70-80% of tasks.

**Factors that could close the gap:**
- Better quantization (beyond AWQ/GPTQ — learned quantization)
- Principled pruning (→ this research thread)
- Better architectures (MoE, SSMs)
- Hardware (more GPUs — Corey's constraint)

**Corey's hardware:** Dual RTX 3090 (48GB total VRAM). Currently maxed with qwen3.5-35B-A3B across both cards. Next meaningful upgrade: A100/H100 or more 3090s.

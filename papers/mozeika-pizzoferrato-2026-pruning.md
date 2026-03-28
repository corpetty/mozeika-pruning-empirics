# Statistical Mechanics of Learning and Pruning in Neural Networks

**Authors:** Alexander Mozeika (independent), Andrea Pizzoferrato (Univ. Bath + Alan Turing Institute)  
**Date:** March 23, 2026  
**Context:** Logos internal research collaboration with the Alan Turing Institute  
**PDF:** `~/.openclaw/media/inbound/Statistical_mechanics_of_learning_and_pruning_in_neural_netw---0a42ca21-dc64-46cc-a68a-ed2ace300918.pdf`

---

## Core Contribution

Unifies weight learning and network pruning as **coupled stochastic dynamics** using a two-temperature, two-timescale Langevin framework. Rather than treating pruning as a post-hoc heuristic (magnitude threshold, etc.), this derives it from first principles via statistical mechanics and Bayesian inference.

The key move: introduce binary mask variables `h ∈ {0,1}` alongside weights `w`, and couple them through a shared energy function. Analyze the equilibrium/posterior behavior across different relative timescales and temperatures.

---

## Key Equations

**Coupled energy function:**
```
Ẽ(w,h | D) = L(w ◦ h | D) + (η/2)||w||² + Σᵢ V(hᵢ)
```
where `V(h) = α h²(h-1)² + (ρ/2)h²` — double-well potential forcing h binary as α→∞, ρ controls sparsity.

**Coupled Langevin dynamics:**
```
τ_w dw/dt = -∂_w Ẽ + noise(T_w)
τ_h dh/dt = -∂_h Ẽ + noise(T_h)
```
Separate temperatures and timescales for weights vs. masks.

---

## Analyzed Regimes

| Regime | Condition | What you get |
|--------|-----------|-------------|
| Equal timescales | τ_w ≈ τ_h, T_w ≈ T_h | Joint Boltzmann posterior P(w,h\|D) |
| Fast learning | τ_w ≪ τ_h | Marginalize over w → mask selection ≈ Bayesian model selection |
| Fast pruning | τ_h ≪ τ_w | Marginalize over h → weight training under mask ensemble |
| Low temperature | β→∞ | MAP objectives → alternating optimization algorithms |

**Replica trick appears naturally:** when β_h/β_w = n (integer), replica copies of the model arise from the partition function. Ensembling is baked into the theory, not added ad hoc.

---

## Practical Implications

- **ρ** = explicit sparsity knob (ℓ0 surrogate in the energy)
- **β_w, β_h ratio** = controls ensemble behavior vs. MAP exploitation
- **τ_w ≪ τ_h** (fast learning) → mask selection via marginal likelihood — superior to magnitude pruning
- **Low-T limit** → justifies iterative alternating prune-train procedures with an exact objective
- **Layer-specific τ, β** → principled way to avoid layer-collapse (convolution layers more sensitive to pruning, sensitivity drops deeper)

---

## Connections to Existing Literature

The paper's intro explicitly cites and positions against:
- Synaptic flow conservation pruning (no data needed, avoids layer-collapse)
- Lottery Ticket Hypothesis (iterative magnitude pruning → 10% original size, no accuracy loss)  
- Sparsity in Deep Learning review (structured vs unstructured)
- Hopfield networks / dilute Hopfield models (storage capacity with dilution)
- Statistical physics of inference (teacher-student scenario)
- Renormalization Group perspective on "winning tickets"

**Key gap filled:** Most prior work either uses ad hoc heuristics (magnitude) or optimization-based methods with computational intractability tradeoffs. This gives a principled energy formulation that derives both.

---

## Open Questions / Follow-up Directions

- Does this framework extend cleanly to transformers (attention weights + MLP weights have very different roles)?
- Replica calculation assumes integer temperature ratios — what are good practical approximations for non-integer n?
- Layer-specific temperature assignment: how to set β per layer in practice? Any empirical guidance?
- Connection to MoE architectures — sparse gating is essentially a learned mask problem

---

## Related Recent Work (2024-2025)

- **"How Sparse Can We Prune A Deep Network: A Fundamental Limit Perspective"** — NeurIPS 2024 poster. First-principles sparsity limits. Direct complement to this paper.
- **arXiv:2006.16617** — "Statistical Mechanical Analysis of Neural Network Pruning" — earlier work in this vein; sparse (edge-pruned) networks generalize better than node-pruned for fixed parameter count.
- **u-MoE (MERL 2025)** — test-time pruning as micro-grained mixture-of-experts. Connects pruning to adaptive inference.

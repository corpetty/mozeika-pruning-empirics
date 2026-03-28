# Explainer: Statistical Mechanics of Learning and Pruning in Neural Networks

*A guide to the concepts in Mozeika & Pizzoferrato (2026)*

---

## What This Paper Is Actually About

Neural networks are too big. A modern language model might have billions of parameters, but empirically we know that most of those parameters are redundant — you can remove 90% of the weights with little or no accuracy loss. The question is: *which* weights can you safely remove, and *when* during training should you remove them?

Most existing approaches to this problem (called **pruning**) are essentially heuristics: remove the weights with the smallest absolute values, or remove weights that don't change much during training. These work surprisingly well in practice, but nobody has a principled reason for *why* they work, or how to make them better.

This paper's contribution is to reframe the entire problem using the language of **statistical mechanics** — the branch of physics that deals with systems of many interacting particles. It turns out this gives you not just intuition but exact mathematical tools: equilibrium distributions, thermodynamic limits, and Bayesian interpretations. The result is a framework that derives pruning from first principles rather than guessing.

---

## Part 1: The Two-Variable Picture

### Weights and Masks

Standard deep learning has one set of variables: the **weights** `w` — the numbers that determine how the network transforms its inputs. Pruning asks: which weights should be set to zero?

The paper introduces a second set of variables: **masks** `h`. Think of each mask as a gate on a weight:
- `h_i = 1` → weight `w_i` is active, contributes to computation
- `h_i = 0` → weight `w_i` is silenced, as if it doesn't exist

The effective weight used in computation is `w ◦ h` — the Hadamard (element-wise) product of the two vectors. So if `w_i = 0.7` and `h_i = 0`, that weight contributes nothing. If `h_i = 1`, it contributes its full value.

Now the problem has two coupled questions:
1. What should the weight values `w` be? (learning)
2. Which weights should be active? (pruning, i.e. which `h_i = 1`)

The paper treats both as dynamical variables that evolve together.

---

## Part 2: Energy Functions — The Physics Analogy

### What is an Energy Function?

In physics, a system evolves to minimize its energy. A ball rolls downhill. A magnet aligns its poles. The configuration with lowest energy is the equilibrium state.

In machine learning, the equivalent concept is the **loss function** — the network "wants" to minimize prediction error. The paper extends this by defining a single **energy function** that captures everything: learning quality, weight regularization, and the cost of keeping weights active.

### The Coupled Energy Function

```
Ẽ(w, h | D) = L(w ◦ h | D)  +  (η/2)||w||²  +  Σᵢ V(hᵢ)
```

Three terms:

**Term 1: `L(w ◦ h | D)` — the prediction loss**  
How well does the masked network predict the training data `D`? For regression this is typically `(1/2) Σ_μ (y^μ - F[w◦h, x^μ])²` — squared error summed over training examples. This is the standard loss function.

**Term 2: `(η/2)||w||²` — weight regularization**  
The L2 penalty on weight magnitude. Prevents weights from growing arbitrarily large. Standard in virtually every neural network training setup. `η` controls the strength.

**Term 3: `Σᵢ V(hᵢ)` — the mask potential**  
This is the novel piece. Each mask has a potential energy:

```
V(h) = α h²(h-1)²  +  (ρ/2)h²
```

Let's unpack this:

- The first part, `α h²(h-1)²`, is a **double-well potential**. It has two minima: one at `h=0` and one at `h=1`. As α→∞, the mask is forced to be exactly binary — either completely off or completely on. This is the mathematical device that makes masks behave like switches.

- The second part, `(ρ/2)h²`, **tilts the double well**. It adds an extra cost for `h=1` (being active). The parameter `ρ > 0` means there's a baseline energetic preference for masks to be off — i.e., a preference for sparsity. Higher `ρ` → more aggressive pruning.

Visually: imagine two valleys in a landscape. One valley is at `h=0` (weight off), the other at `h=1` (weight on). The `ρ` term makes the `h=0` valley slightly deeper — so unless a weight is genuinely useful, the system prefers to turn it off.

---

## Part 3: Stochastic Dynamics — How the System Evolves

### Langevin Dynamics

Rather than gradient descent (which deterministically follows the gradient downhill), the paper uses **Langevin dynamics** — gradient descent plus random noise:

```
τ_w  dw/dt  =  -∂Ẽ/∂w  +  noise with magnitude ∝ T_w
τ_h  dh/dt  =  -∂Ẽ/∂h  +  noise with magnitude ∝ T_h
```

The noise is the key addition. Why noise?

1. **Physical realism**: In stat mech, temperature represents thermal fluctuations. Particles don't just roll downhill — they jiggle. High temperature = large fluctuations = exploration. Low temperature = small fluctuations = exploitation.

2. **Avoiding local minima**: Pure gradient descent gets stuck. Noise lets the system escape local minima and explore the loss landscape.

3. **Bayesian connection**: At equilibrium, Langevin dynamics samples from the Boltzmann distribution `P(w,h) ∝ exp(-Ẽ/T)`. This is exactly Bayesian posterior inference — the "temperature" T plays the role of uncertainty.

### Two Temperatures, Two Timescales

Crucially, weights and masks have **separate** temperatures and timescales:
- `T_w` = temperature for weights (controls weight noise)
- `T_h` = temperature for masks (controls mask noise)
- `τ_w` = timescale for weight updates
- `τ_h` = timescale for mask updates

This separation is what makes the theory rich. It lets you analyze what happens when weights update much faster than masks, or vice versa.

---

## Part 4: Three Regimes — The Core Analysis

The paper's main contribution is analyzing what happens in different limits of these timescales and temperatures.

### Regime 1: Equal Timescales (τ_w ≈ τ_h, T_w ≈ T_h)

Both weights and masks update at the same rate, at the same temperature. The system reaches a **joint equilibrium**:

```
P(w, h | D)  ∝  exp(-β Ẽ(w, h | D))
```

where `β = 1/T` is the inverse temperature.

This is the most natural case. At equilibrium, the system samples weight-mask pairs according to the Boltzmann distribution — configurations with low energy (good predictions + sparse masks) are exponentially more probable.

**Low-temperature limit (β→∞):** The distribution concentrates around the minimum energy configuration. This is MAP (Maximum A Posteriori) inference — find the single best (w,h) pair. The optimization problem becomes:

```
minimize over (w,h):  L(w◦h | D)  +  (η/2)||w||²  +  (ρ/2) Σᵢ hᵢ
```

The last term `(ρ/2) Σ hᵢ` is the sparsity penalty — each active weight costs `ρ/2`. This is exactly an L0-regularized regression: find the sparse weight configuration that minimizes loss.

---

### Regime 2: Fast Learning (τ_w ≪ τ_h)

Weights update much faster than masks. The weights equilibrate (reach their optimal values for the current mask configuration) before the masks change appreciably.

**What this means:** For any given mask configuration `h`, the weights quickly converge to their optimal values. The masks then evolve slowly based on the *marginal likelihood* — the loss after integrating out (averaging over) all possible weight values.

Mathematically, you integrate out the weights to get an effective free energy for the masks:

```
F[h]  =  -T_w log ∫ dw  exp(-β_w Ẽ(w,h|D))
```

The masks then evolve to minimize `F[h]`. This is **Bayesian model selection**: choose the mask (model structure) that has the highest marginal likelihood, accounting for the full uncertainty over weight values.

**Why this is better than magnitude pruning:** Magnitude pruning removes weights based on their current values — a local, myopic criterion. Fast-learning regime pruning removes weights based on how much their removal *affects the loss after reoptimizing the remaining weights* — a global, principled criterion.

**The replica trick appears here:** When the temperature ratio `n = β_h/β_w` is an integer, the mathematical expression for `F[h]` involves `Z[h]^n` — the partition function raised to a power. To handle this, the paper introduces `n` copies ("replicas") of the model. This is the **replica trick** from statistical physics: compute quantities involving `Z^n` by working with n independent but coupled systems.

The physical interpretation: the replica parameter n controls ensemble behavior. n=1 is standard learning. n>1 is like training an ensemble of n models that share the same mask structure — the masks are chosen to be good for the *average model in the ensemble*, not just a single model. This provides implicit regularization.

---

### Regime 3: Fast Pruning (τ_h ≪ τ_w)

The reverse: masks equilibrate much faster than weights. For any given weight configuration `w`, the masks quickly find their optimal values before the weights change.

**What this means:** The weights evolve under a *mask-averaged* effective loss. At each step, the weights are trained against the expected loss over all possible mask configurations.

This is the dual of fast learning: instead of finding the best mask for uncertain weights, you're finding the best weights for uncertain masks.

The practical effect: the weight training automatically adapts to handle sparse versions of the network. Weights that are needed for multiple sparse configurations get reinforced. Weights that are redundant across configurations get suppressed.

---

### Regime 4: Low Temperature (β → ∞ for both)

In all regimes, taking the temperature to zero produces **MAP optimization** — find the single best configuration. The stochastic dynamics become deterministic alternating optimization:

**Given current masks h, optimize weights:**
```
ŵ[h]  =  argmin_w  { L(w◦h | D)  +  (η/2)||w||² }
```
(Standard regularized regression on the pruned network)

**Given current weights w, optimize masks:**
```
ĥ[w]  =  argmin_h  { L(w◦h | D)  +  (ρ/2) Σᵢ hᵢ }
```
(Find the sparsest mask that keeps loss low, given current weights)

This **alternating optimization** procedure is exactly what modern pruning algorithms do in practice (e.g., iterative magnitude pruning with retraining). The paper derives it as the zero-temperature limit of a principled thermodynamic theory — giving it a formal justification it previously lacked.

---

## Part 5: The Bayesian Interpretation

### Connecting Physics to Probability

The Boltzmann distribution `P(w,h) ∝ exp(-βẼ)` is mathematically identical to a Bayesian posterior:

```
P(w, h | D)  ∝  P(D | w, h) × P(w) × P(h)
```

where:
- `P(D | w, h)` = likelihood: how probable is the data given this network? (∝ exp(-β L(w◦h|D)))
- `P(w)` = prior over weights: `∝ exp(-β(η/2)||w||²)` — a Gaussian prior favoring small weights
- `P(h)` = prior over masks: `∝ exp(-β Σ V(hᵢ))` — a prior favoring sparse masks

**Fast learning regime Bayesian interpretation:**  
Marginalizing over weights = computing the **marginal likelihood** (evidence). Selecting masks to maximize marginal likelihood is **Bayesian model selection** — the principled way to choose model structure while accounting for parameter uncertainty.

**Fast pruning regime Bayesian interpretation:**  
Marginalizing over masks = training weights against an **expected likelihood** averaged over sparse model structures. This is a form of variational inference.

**Replica interpretation:**  
The n replicas in the fast-learning regime correspond to drawing n independent weight samples and averaging. Practically: the mask is chosen to be good for an *ensemble* of n models, providing robustness against the specific weight values.

For squared loss, the paper derives an explicit generative model: labels are drawn from a Gaussian centered on the *average prediction across replicas*. This makes the ensemble interpretation concrete and gives a recipe for simulation.

---

## Part 6: What You Can Control — The Design Knobs

The framework provides four tunable parameters that directly correspond to design decisions in a pruning algorithm:

### ρ — Sparsity Pressure
The cost per active weight. **Higher ρ → more aggressive pruning.** This is the cleanest knob: set ρ to control how sparse you want the final network. Unlike ad hoc pruning rates ("remove 50% of weights"), ρ has a physical meaning — it's the energetic cost of keeping a connection active.

### T_w (β_w) — Weight Temperature
How much noise in weight updates. **Higher T_w → more weight exploration.** Low T_w = weights converge precisely to their optimal values. High T_w = weights are uncertain, stochastic. In the fast-learning regime, T_w controls how thoroughly the weights are marginalized over when evaluating masks.

### T_h (β_h) — Mask Temperature
How much noise in mask updates. **Higher T_h → more mask exploration.** High T_h = masks flip frequently, exploring many network structures. Low T_h = masks converge to a fixed configuration. Annealing T_h from high to low is a principled pruning schedule.

### τ_w / τ_h — Relative Timescales
The ratio of update speeds. This determines which regime you're in:
- `τ_w ≪ τ_h` (weights fast) → fast-learning regime → Bayesian model selection for masks
- `τ_h ≪ τ_w` (masks fast) → fast-pruning regime → weight training with mask averaging
- `τ_w ≈ τ_h` → joint regime → full posterior over (w,h)

**Practical mapping to hyperparameters:**
- Learning rate ratio ≈ τ ratio
- Weight decay = η (prior strength)
- L1 penalty on mask logits ≈ ρ
- Training noise / stochastic gradient variance ≈ temperature

---

## Part 7: Why Layer-Collapse Doesn't Happen

A practical problem with naive pruning: you can accidentally remove an entire layer, which catastrophically breaks the network. This is "layer collapse."

The framework handles this naturally through **layer-specific temperatures and timescales**. Empirically (from the pruning literature, cited in the paper), convolution layers are more sensitive to pruning than fully connected layers, and deeper conv layers are less sensitive than shallower ones.

By assigning different `(T_w, T_h, τ_w, τ_h)` per layer, you can allocate pruning "budget" where the network can afford it and protect layers where collapse would be harmful. The theory gives you the right objective to optimize — you're not guessing.

---

## Part 8: The Big Picture

Here's what makes this paper significant beyond its technical content:

**Pruning used to be:** "Remove the small weights, retrain, repeat. It works empirically but we don't know why."

**Pruning now is:** "Find the energy minimum of a thermodynamic system with coupled weight and mask dynamics. The algorithm you run is the zero-temperature limit of principled physics."

The practical difference:
1. You have a **justified objective**, not a heuristic
2. You have **principled hyperparameters** (ρ, T_w, T_h, τ) instead of "try different pruning rates"
3. You have a **Bayesian interpretation** that connects to the broader literature on inference and model selection
4. You have **regime analysis** that tells you what different algorithms are actually computing

The deeper implication: if neural network training and pruning are thermodynamic processes, then tools from statistical physics — phase transitions, universality classes, renormalization group methods — become available for analyzing why deep learning works. This paper is one step in that direction.

---

## Connections to Adjacent Ideas

**→ Universal Weight Subspace Hypothesis (Kaushik et al. 2025):**  
UWSH says all trained networks converge to the same low-dimensional subspace. The Mozeika framework might explain *why*: the universal subspace is the energy minimum of the thermodynamic system, and all training trajectories — regardless of initialization — converge to the same basin. The fast-learning regime's Bayesian model selection preferentially preserves directions in this subspace. Nobody has made this connection explicitly yet.

**→ Lottery Ticket Hypothesis:**  
The "winning ticket" — the small subnetwork that can be trained from scratch to full accuracy — is the low-temperature MAP solution of the mask variable in Mozeika's framework. The iterative magnitude pruning procedure that finds it is approximately the alternating optimization derived from β→∞.

**→ LoRA and PEFT:**  
LoRA adapts models within a low-rank subspace. Mozeika's mask variables are a learned discrete version of the same idea: instead of low-rank weight updates, you get sparse weight selection. EigenLoRAx (the UWSH follow-up) bridges these — subspace + sparsity.

**→ Daemon's abliteration:**  
Abliteration removes safety behavior by modifying model weights. The Mozeika framework suggests this should be done via principled mask selection (fast-learning regime), not magnitude heuristics — find the masks that minimize safety-refusal loss while minimizing damage to general capability. The KL divergence and TopKJS damage metrics in the Heretic tool are approximating the thermodynamic damage cost in the energy function.

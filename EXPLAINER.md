# What We're Actually Doing Here: A Plain-Language Guide

*For people who want to understand the research without needing a PhD in physics.*

---

## The Big Picture

Neural networks — the AI systems behind image recognition, language models, translation, and most modern AI — are made of millions or billions of "weights." Think of weights like the dials on a mixing board: each one controls how strongly a signal gets passed from one part of the network to the next. During training, the system adjusts all these dials until it gets good at its task.

The problem: most of those dials barely matter. In a typical large network, you can set 90%+ of the weights to zero and the network still works just as well — sometimes better. Those dead weights aren't doing useful work; they're just taking up space and compute.

**Pruning** is the process of finding and removing those useless weights. This research is about finding smarter ways to do that.

---

## Why It Matters

- **Speed:** Fewer weights = fewer calculations = faster inference. Important for running AI on your laptop or phone instead of a data center.
- **Cost:** Fewer parameters = smaller models = less GPU memory and electricity.
- **Understanding:** If 90% of a network's weights can be removed without hurting performance, that tells us something deep about what the network actually learned and what the extra weights were doing (or not doing).

---

## The Two Methods We Tested

### Method 1: Glauber Dynamics (the physics-inspired approach)

This method comes from a paper by Mozeika & Pizzoferrato (2026) that borrows ideas from **statistical mechanics** — the physics of how large collections of particles behave at different temperatures.

**The intuition:** Imagine the network's weights as a bag of magnets on a table, vibrating. At high temperature, magnets flip randomly — lots of chaos. As you slowly cool the system, magnets settle into stable low-energy arrangements. The system "finds" a good configuration by exploring randomly at first, then settling down.

In the pruning context:
- Each weight has a "switch" (called a mask) that can be ON or OFF (keep or remove)
- At high "temperature," we randomly flip these switches, trying lots of different combinations
- As we gradually cool down, we keep the flips that improve the network and reject ones that hurt it
- By the time temperature reaches zero, we've settled on a good sparse configuration

This is called **Glauber dynamics** (named after a physicist). The "energy" being minimized is a combination of how well the network performs and how sparse its weights are.

**What we found:** This works well on small networks (LeNet on MNIST), achieving 95% sparsity with barely any accuracy loss. The pruning process goes through three distinct phases:
1. *Rapid pruning* — lots of weights die quickly
2. *Restructuring* — the network reorganizes, some weights come back, others go; accuracy holds steady
3. *Collapse* — as temperature approaches zero, the last marginal weights die off

**The catch:** The original paper's theoretical predictions about *when* the phase transition happens were off by a factor of 100–20,000× for real multi-layer networks. The physics works beautifully in theory but the mean-field approximation breaks down in practice.

---

### Method 2: Fisher Information Pruning (the practical engineering approach)

This method is more straightforward: rank every weight by how much damage removing it would cause, then remove the least important ones.

**The intuition:** Not all weights are equal. Some weights, if zeroed out, would barely change the output. Others are critical load-bearing parts of the network. Fisher information is a statistical measure of "how sensitive is the network's output to this particular weight?" — essentially, how much does this dial actually matter?

The process:
1. Run a batch of training data through the network
2. Measure how much each weight influenced the output (via backpropagation)
3. Square those influence scores → that's the Fisher saliency
4. Remove the bottom X% of weights (lowest saliency = least important)
5. Fine-tune the remaining weights to compensate
6. Repeat until you hit your sparsity target

**What we found on VGG16 + CIFAR-10:** Starting from a 89.9% accurate network, after removing 90% of weights the network hit **93.1% accuracy** — *better than where we started.* The pruning process acts as a form of regularization, forcing the network to rely on the weights that actually matter and discarding the noise.

---

## The Surprising Result

Both experiments showed the same counterintuitive pattern: **removing weights can improve accuracy.**

How is this possible? The networks we're working with are massively overparameterized for their tasks:
- **VGG16** was designed to classify 1,000 categories of ImageNet images. We're asking it to classify 10 categories of CIFAR-10 (basically: planes, cars, birds, cats, etc.). It has roughly 40× more parameters than it needs.
- **LeNet-300-100** on MNIST is closer to the right size, so pruning causes a small accuracy loss at very high sparsity.

Overparameterized networks generalize worse because they "memorize" noise in the training data. Pruning forces them to find more essential representations. It's like editing: the first draft has filler words; the final draft has only what matters, and it's clearer.

---

## What the Phase Transition Means (Without the Equations)

The Mozeika paper talks a lot about a "phase transition" — a critical threshold in pruning pressure (called ρ_c) where the network's behavior suddenly changes qualitatively, like water turning to ice.

In practice, you can observe this:
- **Below ρ_c:** The network is dense-ish, and pruning causes roughly proportional accuracy loss
- **Near ρ_c:** The network is in a critical state — highly sensitive, accuracy can swing
- **Above ρ_c (forced too sparse):** The network collapses — you've removed too many load-bearing weights and accuracy falls off a cliff

The Glauber dynamics approach uses temperature as a way to explore the space *around* this transition rather than blindly pushing past it. The analogy: water doesn't suddenly turn to ice in a controlled way; you have to cool it slowly to get good crystals (snowflakes, not slush).

**What our experiments showed:** The phase transition is real and observable — but only clearly in simple linear networks. In deep networks (MLPs, CNNs), the math gets messy and the theoretical predictions break down. The "transition" is more of a gradual cliff than a sharp edge.

---

## Why This Matters for the Future

### For running AI locally
State-of-the-art language models have hundreds of billions of parameters. Even with 4-bit quantization, a 70B model takes ~40GB of VRAM. Principled pruning could shrink that to 7GB with minimal quality loss — the difference between needing a $10,000 GPU cluster and running on a decent gaming PC.

### For understanding what AI "knows"
If you can remove 90% of a network's parameters and it still works, the question becomes: *what are the remaining 10% representing?* Research suggests (see the UWSH paper in our notes) that all large trained networks converge to similar low-dimensional structures regardless of how they were trained. Pruning might be a way to *extract* that structure directly.

### For efficient AI systems
A network that's 90% sparse can be computed 10× faster with the right hardware. This matters for:
- Mobile/edge AI (phones, embedded systems)
- Real-time applications (robotics, video analysis)
- Decentralized AI (running AI over many small devices rather than one big server)

---

## The Honest Limitations

1. **The theoretical predictions don't work in practice.** The beautiful physics of the Mozeika paper applies cleanly to a single-layer linear network. Real networks are non-linear and deep — the math gives the right intuitions but wrong numbers.

2. **The results are task-specific.** VGG16 on CIFAR-10 can be pruned aggressively because the model is massively overparameterized for the task. You can't necessarily prune a model by 90% for a hard task it was designed for.

3. **We're testing small/medium models.** The techniques work well on LeNet and VGG16. Whether they scale to billion-parameter transformers without significant engineering work is an open question.

4. **Pruning doesn't compress storage by default.** A "sparse" model still stores zeros. You need specialized sparse data structures (or the "compress" step we implemented) to actually reduce memory and compute. This is an active engineering area.

---

## What's Next

The most promising next direction is **Belief Propagation pruning** (Krzakala & Zdeborová, 2019) — a method from the same statistical mechanics tradition that handles non-linear networks more rigorously than the mean-field approach in the Mozeika paper. It's computationally heavier but might give us the predictive power the Mozeika method promises but doesn't deliver on real architectures.

In short: the physics intuition is right, the theoretical tools need updating for modern deep learning, and the empirical results are promising enough to keep pushing.

---

*Document maintained by NickMolty. Last updated: 2026-03-28.*
*Technical details: see GLAUBER_RESULTS.md, VGG16_RESULTS.md, and the scripts in this repo.*

#!/usr/bin/env python3
"""
Exp 22: LeNet-300-100 on MNIST — First Real Architecture Test

Apply Mozeika energy-based pruning layer-by-layer to a real FC network on
MNIST, then compare to magnitude pruning at matched sparsity levels.

Architecture: 784 → 300 (relu) → 100 (relu) → 10 (softmax/CE)
Total params: ~266K

Pruning method — Mozeika energy scoring:
  In the fast-learning (τ_w ≪ τ_h), low-temperature (β→∞) limit, the
  Glauber acceptance rule for flipping h_j from 1→0 reduces to:

      accept prune  iff  ΔLoss_j < ρ/2

  where ρ is the Mozeika sparsity pressure and ΔLoss_j is the loss increase
  from removing weight j.  First-order Taylor gives:

      ΔLoss_j ≈ |∂L/∂w_j · w_j|

  This is the sensitivity-based importance score (related to SNIP). Binary
  search on ρ controls the target sparsity per layer.

  For the smallest layer (100→10, 1000 params) we also run actual Glauber
  coordinate sweeps as validation that the scoring agrees.

Output:
  results/cnn_mnist_mozeika.csv
  results/cnn_mnist_magnitude.csv
"""
import numpy as np
import gzip
import os
import sys
import struct
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pruning_core.optimizers import AdamOptimizer

# ─── MNIST loading ───────────────────────────────────────────────────────────

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def download_mnist():
    """Download MNIST .gz files if not present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            url = f"{MNIST_URL}/{fname}"
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, path)


def load_images(path):
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float64) / 255.0


def load_labels(path):
    with gzip.open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist():
    download_mnist()
    X_train = load_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz"))
    y_train = load_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz"))
    X_test = load_images(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"))
    y_test = load_labels(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"))
    return X_train, y_train, X_test, y_test


# ─── MLP with softmax / cross-entropy ───────────────────────────────────────

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def one_hot(y, K=10):
    oh = np.zeros((len(y), K), dtype=np.float64)
    oh[np.arange(len(y)), y] = 1.0
    return oh


class MLP:
    """LeNet-300-100: 784→300→100→10 with ReLU hidden + softmax output."""

    def __init__(self, layer_sizes=(784, 300, 100, 10), seed=42):
        rng = np.random.default_rng(seed)
        self.W = []
        self.b = []
        self.masks = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            std = np.sqrt(2.0 / fan_in)
            self.W.append(rng.standard_normal((fan_in, fan_out)) * std)
            self.b.append(np.zeros(fan_out))
            self.masks.append(np.ones((fan_in, fan_out)))

    def forward(self, X):
        """Return (activations, pre_activations)."""
        acts = [X]
        pres = []
        a = X
        for i, (W, b, m) in enumerate(zip(self.W, self.b, self.masks)):
            z = a @ (W * m) + b
            pres.append(z)
            a = relu(z) if i < len(self.W) - 1 else softmax(z)
            acts.append(a)
        return acts, pres

    def predict(self, X):
        acts, _ = self.forward(X)
        return acts[-1].argmax(axis=1)

    def accuracy(self, X, y):
        return (self.predict(X) == y).mean()

    def loss(self, X, y):
        acts, _ = self.forward(X)
        probs = acts[-1]
        return -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()

    def backward(self, X, y):
        """Backprop for CE loss.  Returns (grad_W, grad_b) lists."""
        acts, pres = self.forward(X)
        n = len(y)
        L = len(self.W)
        gW = [None] * L
        gb = [None] * L

        # output layer: softmax + CE  ⇒  δ = p − one_hot(y)
        delta = acts[-1].copy()
        delta[np.arange(n), y] -= 1.0
        delta /= n

        for l in range(L - 1, -1, -1):
            gW[l] = acts[l].T @ delta
            gW[l] *= self.masks[l]
            gb[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ (self.W[l] * self.masks[l]).T) * (pres[l - 1] > 0)

        return gW, gb

    def total_params(self):
        return sum(w.size for w in self.W)

    def total_active(self):
        return sum(int((m > 0.5).sum()) for m in self.masks)

    def sparsity(self):
        return 1.0 - self.total_active() / self.total_params()

    def layer_sparsities(self):
        return [1.0 - float((m > 0.5).mean()) for m in self.masks]

    def clone(self):
        import copy
        return copy.deepcopy(self)


# ─── Training ────────────────────────────────────────────────────────────────

def train_mlp(model, X_train, y_train, X_val, y_val,
              lr=0.001, epochs=10, batch_size=256, verbose=True):
    L = len(model.W)
    aW = [AdamOptimizer(model.W[l].size, lr=lr) for l in range(L)]
    ab = [AdamOptimizer(model.b[l].size, lr=lr) for l in range(L)]
    n = len(X_train)

    for ep in range(epochs):
        perm = np.random.permutation(n)
        eloss = 0.0
        nb = 0
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            gW, gb = model.backward(X_train[idx], y_train[idx])
            for l in range(L):
                w = aW[l].step(model.W[l].flatten(), gW[l].flatten())
                model.W[l] = w.reshape(model.W[l].shape)
                b = ab[l].step(model.b[l].flatten(), gb[l].flatten())
                model.b[l] = b.reshape(model.b[l].shape)
            eloss += model.loss(X_train[idx], y_train[idx])
            nb += 1

        if verbose:
            vacc = model.accuracy(X_val, y_val)
            print(f"  Epoch {ep+1}/{epochs}  loss={eloss/nb:.4f}  val_acc={vacc:.4f}")

    return model


def finetune(model, X_train, y_train, lr=0.0005, epochs=3, batch_size=256):
    """Fine-tune with masks frozen (pruned weights stay zero)."""
    return train_mlp(model, X_train, y_train,
                     X_train[:1000], y_train[:1000],
                     lr=lr, epochs=epochs, batch_size=batch_size, verbose=True)


# ─── Mozeika energy-based pruning ───────────────────────────────────────────

def mozeika_importance(model, X_calib, y_calib):
    """
    Mozeika importance score per weight (fast-learning, T→0 limit).

    In the Glauber framework, weight j is pruned when:
        ΔLoss_j < ρ/2
    where ΔLoss_j ≈ |∂L/∂w_j · w_j|  (first-order Taylor).

    Returns list of score arrays, one per layer.
    """
    gW, _ = model.backward(X_calib, y_calib)
    return [np.abs(gW[l] * model.W[l]) for l in range(len(model.W))]


def mozeika_prune(model, target_sparsity, X_calib, y_calib,
                  eta=0.0001, alpha=1.0):
    """
    Mozeika energy-based pruning with layer-wise binary search on ρ.

    For each layer independently:
      • importance_j = |∂L/∂w_j · w_j| + η·w_j²/2   (loss sensitivity + L2)
      • prune h_j = 0  if  importance_j < ρ/2
      • binary search ρ to hit target_sparsity ± 3%
    """
    pruned = model.clone()
    scores = mozeika_importance(pruned, X_calib, y_calib)

    for l in range(len(pruned.W)):
        # Full Mozeika energy score: loss contribution + L2 regularization
        s = scores[l] + 0.5 * eta * pruned.W[l] ** 2
        flat = s.flatten()

        # Binary search on ρ: prune if score < ρ/2
        rho_lo, rho_hi = 0.0, 2.0 * flat.max() + 1e-6
        best_mask = pruned.masks[l].copy()
        best_diff = 999.0

        for _ in range(30):
            rho = (rho_lo + rho_hi) / 2.0
            mask = (s >= rho / 2.0).astype(float)
            sp = 1.0 - mask.mean()

            diff = abs(sp - target_sparsity)
            if diff < best_diff:
                best_diff = diff
                best_mask = mask.copy()

            if diff < 0.005:
                break
            if sp < target_sparsity:
                rho_lo = rho
            else:
                rho_hi = rho

        pruned.masks[l] = best_mask
        sp = 1.0 - best_mask.mean()
        print(f"  Layer {l} ({pruned.W[l].shape}): sparsity={sp:.3f}")

    return pruned


# ─── Glauber coordinate sweep (for small layer validation) ──────────────────

def glauber_prune_layer3(model, X_calib, y_calib, target_sparsity,
                         eta=0.0001, alpha=1.0, T=15, K=20, seed=42):
    """
    Actual Glauber coordinate sweep on the output layer (100→10 = 1000 params).
    Feasible because the layer is small.  Uses CE loss for the output layer.

    rho range is calibrated from the energy-score importance values to match
    the scale of per-weight loss contributions.
    """
    pruned = model.clone()
    l = len(pruned.W) - 1  # output layer
    rng = np.random.default_rng(seed)

    # Get input to this layer
    acts, _ = pruned.forward(X_calib)
    X_in = acts[l]

    W = pruned.W[l].copy()
    mask = pruned.masks[l].copy()
    b = pruned.b[l].copy()
    n_in, n_out = W.shape
    M = len(X_calib)

    # Calibrate rho range from importance scores
    gW, _ = pruned.backward(X_calib, y_calib)
    imp = np.abs(gW[l] * pruned.W[l])
    rho_max = 2.0 * np.percentile(imp, 95)  # cover up to 95th pctile
    print(f"    Importance range: [{imp.min():.6f}, {imp.max():.6f}], rho_max={rho_max:.6f}")

    # Binary search on rho
    rho_lo, rho_hi = 0.0, rho_max
    best_mask = mask.copy()
    best_W = W.copy()
    best_diff = 999.0

    for bs in range(8):
        rho = (rho_lo + rho_hi) / 2.0
        W_cur = pruned.W[l].copy()
        mask_cur = np.ones((n_in, n_out))
        rng_inner = np.random.default_rng(seed + bs * 100)

        for t in range(T):
            indices = rng_inner.permutation(n_in * n_out)
            flips = 0
            for idx in indices:
                i, j = divmod(idx, n_out)
                old_val = mask_cur[i, j]

                # Energy with current mask
                z0 = X_in @ (W_cur * mask_cur) + b
                p0 = softmax(z0)
                ce0 = -np.log(p0[np.arange(M), y_calib] + 1e-12).mean()
                h_flat = mask_cur.flatten()
                V0 = np.sum(alpha * h_flat**2 * (h_flat - 1)**2 + (rho / 2) * h_flat)
                E0 = ce0 + 0.5 * eta * np.sum(W_cur**2) + V0

                # Flip
                mask_cur[i, j] = 1.0 - old_val

                # Quick weight re-opt (K Adam steps on this layer)
                W_try = W_cur.copy()
                opt = AdamOptimizer(N=W_try.size, lr=0.01)
                for _ in range(K):
                    z = X_in @ (W_try * mask_cur) + b
                    p = softmax(z)
                    delta = p.copy()
                    delta[np.arange(M), y_calib] -= 1.0
                    delta /= M
                    grad = X_in.T @ delta
                    grad = grad * mask_cur + eta * W_try
                    W_try = opt.step(W_try.flatten(), grad.flatten()).reshape(W_try.shape)

                z1 = X_in @ (W_try * mask_cur) + b
                p1 = softmax(z1)
                ce1 = -np.log(p1[np.arange(M), y_calib] + 1e-12).mean()
                h_flat = mask_cur.flatten()
                V1 = np.sum(alpha * h_flat**2 * (h_flat - 1)**2 + (rho / 2) * h_flat)
                E1 = ce1 + 0.5 * eta * np.sum(W_try**2) + V1

                if E1 < E0:
                    W_cur = W_try
                    flips += 1
                else:
                    mask_cur[i, j] = old_val

            sp = 1.0 - mask_cur.mean()
            if t % 5 == 0 or t == T - 1:
                print(f"    Sweep {t+1}/{T}: flips={flips} sparsity={sp:.3f}")
            if flips == 0 and t > 2:
                break

        sp = 1.0 - mask_cur.mean()
        diff = abs(sp - target_sparsity)
        print(f"    BS iter {bs}: rho={rho:.6f} → sparsity={sp:.3f}")

        if diff < best_diff:
            best_diff = diff
            best_mask = mask_cur.copy()
            best_W = W_cur.copy()

        if diff < 0.03:
            break
        if sp < target_sparsity:
            rho_lo = rho
        else:
            rho_hi = rho

    pruned.masks[l] = best_mask
    pruned.W[l] = best_W
    return pruned


# ─── Magnitude pruning ──────────────────────────────────────────────────────

def magnitude_prune(model, target_sparsity):
    """Global unstructured magnitude pruning."""
    pruned = model.clone()
    all_mags = np.concatenate([np.abs(w).flatten() for w in pruned.W])
    n_prune = int(len(all_mags) * target_sparsity)
    if n_prune == 0:
        return pruned
    threshold = np.sort(all_mags)[n_prune]
    for l in range(len(pruned.W)):
        pruned.masks[l] = (np.abs(pruned.W[l]) >= threshold).astype(float)
    return pruned


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    np.random.seed(42)
    os.makedirs("results", exist_ok=True)

    # ── 1. Load MNIST ──
    print("=" * 60)
    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ── 2. Train dense baseline ──
    print("\n" + "=" * 60)
    print("Training dense LeNet-300-100...")
    model = MLP(layer_sizes=(784, 300, 100, 10), seed=42)
    model = train_mlp(model, X_train, y_train, X_test, y_test,
                      lr=0.001, epochs=10, batch_size=256)
    dense_acc = model.accuracy(X_test, y_test)
    dense_loss = model.loss(X_test, y_test)
    print(f"\nDense baseline: acc={dense_acc:.4f} ({dense_acc*100:.2f}%)  loss={dense_loss:.4f}")
    print(f"Total params: {model.total_params()}")

    # Calibration subset
    rng = np.random.default_rng(42)
    calib_idx = rng.choice(len(X_train), 1000, replace=False)
    X_calib, y_calib = X_train[calib_idx], y_train[calib_idx]

    # ── 3. Sparsity sweep ──
    target_sparsities = [0.0, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80]

    mozeika_rows = []
    magnitude_rows = []

    for tsp in target_sparsities:
        print("\n" + "=" * 60)
        print(f"Target sparsity: {tsp*100:.0f}%")

        # ── Magnitude pruning ──
        print("\n--- Magnitude pruning ---")
        mag = magnitude_prune(model, tsp)
        mag_sp = mag.sparsity()
        mag_acc0 = mag.accuracy(X_test, y_test)
        print(f"  Actual sparsity: {mag_sp:.3f}  acc_before_ft: {mag_acc0:.4f}")

        if tsp > 0:
            mag = finetune(mag, X_train, y_train)
        mag_acc1 = mag.accuracy(X_test, y_test)
        print(f"  acc_after_ft: {mag_acc1:.4f}")

        lsp = mag.layer_sparsities()
        magnitude_rows.append(dict(
            target_sparsity=tsp, actual_sparsity=mag_sp,
            acc_before_finetune=mag_acc0, acc_after_finetune=mag_acc1,
            layer1_sparsity=lsp[0], layer2_sparsity=lsp[1], layer3_sparsity=lsp[2],
        ))

        # ── Mozeika (energy-score) pruning ──
        if tsp == 0.0:
            mozeika_rows.append(dict(
                target_sparsity=0.0, actual_sparsity=0.0,
                acc_before_finetune=dense_acc, acc_after_finetune=dense_acc,
                layer1_sparsity=0.0, layer2_sparsity=0.0, layer3_sparsity=0.0,
            ))
            continue

        print("\n--- Mozeika (energy-score) pruning ---")
        moz = mozeika_prune(model, tsp, X_calib, y_calib, eta=0.0001, alpha=1.0)
        moz_sp = moz.sparsity()
        moz_acc0 = moz.accuracy(X_test, y_test)
        print(f"  Total sparsity: {moz_sp:.3f}  acc_before_ft: {moz_acc0:.4f}")

        moz = finetune(moz, X_train, y_train)
        moz_acc1 = moz.accuracy(X_test, y_test)
        print(f"  acc_after_ft: {moz_acc1:.4f}")

        lsp = moz.layer_sparsities()
        mozeika_rows.append(dict(
            target_sparsity=tsp, actual_sparsity=moz_sp,
            acc_before_finetune=moz_acc0, acc_after_finetune=moz_acc1,
            layer1_sparsity=lsp[0], layer2_sparsity=lsp[1], layer3_sparsity=lsp[2],
        ))

    # ── 4. Glauber validation on output layer ──
    print("\n" + "=" * 60)
    print("Glauber validation: actual coordinate sweep on layer 3 (100×10)")
    print("Running at 50% target sparsity...")
    glauber_val = glauber_prune_layer3(
        model, X_calib, y_calib, target_sparsity=0.50,
        eta=0.0001, alpha=1.0, T=15, K=20, seed=42,
    )
    l3_sp = glauber_val.layer_sparsities()[2]
    glauber_acc = glauber_val.accuracy(X_test, y_test)
    # Compare to score-based layer 3 sparsity at 50%
    moz50 = [r for r in mozeika_rows if abs(r["target_sparsity"] - 0.5) < 0.01]
    if moz50:
        score_l3_sp = moz50[0]["layer3_sparsity"]
        print(f"  Glauber layer3 sparsity: {l3_sp:.3f}  (score-based: {score_l3_sp:.3f})")
    print(f"  Glauber full-model acc (only layer3 pruned): {glauber_acc:.4f}")

    # ── 5. Save results ──
    cols = ["target_sparsity", "actual_sparsity", "acc_before_finetune",
            "acc_after_finetune", "layer1_sparsity", "layer2_sparsity",
            "layer3_sparsity"]

    def write_csv(path, rows):
        with open(path, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(f"{r[c]:.4f}" for c in cols) + "\n")
        print(f"  Saved {path}")

    write_csv("results/cnn_mnist_mozeika.csv", mozeika_rows)
    write_csv("results/cnn_mnist_magnitude.csv", magnitude_rows)

    # ── 6. Summary ──
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Dense baseline: {dense_acc:.4f}")
    print()

    header = f"{'Target':>8s}  {'Actual':>8s}  {'Before FT':>10s}  {'After FT':>10s}  {'L1':>6s}  {'L2':>6s}  {'L3':>6s}"
    print("Magnitude pruning:")
    print(header)
    for r in magnitude_rows:
        print(f"  {r['target_sparsity']:>6.0%}  {r['actual_sparsity']:>8.3f}"
              f"  {r['acc_before_finetune']:>10.4f}  {r['acc_after_finetune']:>10.4f}"
              f"  {r['layer1_sparsity']:>6.3f}  {r['layer2_sparsity']:>6.3f}"
              f"  {r['layer3_sparsity']:>6.3f}")

    print()
    print("Mozeika (energy-score) pruning:")
    print(header)
    for r in mozeika_rows:
        print(f"  {r['target_sparsity']:>6.0%}  {r['actual_sparsity']:>8.3f}"
              f"  {r['acc_before_finetune']:>10.4f}  {r['acc_after_finetune']:>10.4f}"
              f"  {r['layer1_sparsity']:>6.3f}  {r['layer2_sparsity']:>6.3f}"
              f"  {r['layer3_sparsity']:>6.3f}")

    print(f"\nElapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()

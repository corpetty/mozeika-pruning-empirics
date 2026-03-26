"""
Experiment 27: Synthetic Teacher Network — OBD Saliency Recovery

Protocol (Mozeika's suggestion):
1. Generate sparse teacher NN (single hidden layer, random sparse weights)
2. Generate dataset: Gaussian inputs X ~ N(0,I), outputs y = f_teacher(X)
3. Train student with zero-temperature fast-learning (OBD saliency rule):
   - Inner loop: optimize w with h fixed (Adam + L2)
   - Outer loop: prune active weights where S_i = 0.5 * F_ii * w_i^2 < rho/2
4. Measure Hamming(h_student, h_teacher) as function of rho

KEY: Hamming must be computed permutation-corrected (Hungarian matching on rows of W1)
because the hidden layer has permutation symmetry — any permutation of neurons gives
the same function, so the student may learn the right structure in a different order.

Correct metric: for each student neuron, find the best-matching teacher neuron
(by cosine similarity of weight vectors), then compare masks.
"""

import numpy as np
import csv
from pathlib import Path
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─── Adam optimizer ───────────────────────────────────────────────────────────

class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2; self.eps = eps
        self.m = {}; self.v = {}; self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ─── Teacher generation ───────────────────────────────────────────────────────

def generate_sparse_teacher(N_in, N_h, N_out, sparsity, sigma_w, seed):
    """Random sparse single-hidden-layer teacher. ReLU activation."""
    rng = np.random.default_rng(seed)
    h1 = (rng.random((N_h, N_in)) > sparsity).astype(np.float32)
    W1 = rng.normal(0, sigma_w, (N_h, N_in)).astype(np.float32) * h1
    b1 = np.zeros(N_h, dtype=np.float32)
    W2 = rng.normal(0, sigma_w / np.sqrt(N_h), (N_out, N_h)).astype(np.float32)
    b2 = np.zeros(N_out, dtype=np.float32)
    return {"W1": W1, "h1": h1, "b1": b1, "W2": W2, "b2": b2}


def teacher_forward(t, X):
    z1 = np.maximum(X @ t["W1"].T + t["b1"], 0)
    return z1 @ t["W2"].T + t["b2"]


def generate_dataset(teacher, M, seed):
    rng = np.random.default_rng(seed + 9999)
    X = rng.normal(0, 1, (M, teacher["W1"].shape[1])).astype(np.float32)
    y = teacher_forward(teacher, X)
    return X, y


# ─── Student network ──────────────────────────────────────────────────────────

def student_forward(W1, mask1, b1, W2, b2, X):
    W1_eff = W1 * mask1
    z1 = np.maximum(X @ W1_eff.T + b1, 0)
    return z1, z1 @ W2.T + b2


def student_grads(W1, mask1, b1, W2, b2, X, y, eta):
    M = X.shape[0]
    W1_eff = W1 * mask1
    z1 = np.maximum(X @ W1_eff.T + b1, 0)
    out = z1 @ W2.T + b2
    diff = out - y
    mse = np.mean(diff ** 2)
    l2 = 0.5 * eta * (np.sum(W1_eff ** 2) + np.sum(W2 ** 2))
    loss = mse + l2

    d_out = 2.0 * diff / M
    dW2 = d_out.T @ z1 + eta * W2
    db2 = d_out.sum(axis=0)
    d_z1 = (d_out @ W2) * (z1 > 0).astype(np.float32)
    dW1_eff = d_z1.T @ X
    dW1 = (dW1_eff + eta * W1_eff) * mask1
    db1 = d_z1.sum(axis=0)
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def train_student(W1, mask1, b1, W2, b2, X, y, eta, lr, epochs, batch_size=256, seed=0):
    M = X.shape[0]
    rng = np.random.default_rng(seed)
    params = {"W1": W1.copy(), "b1": b1.copy(), "W2": W2.copy(), "b2": b2.copy()}
    opt = Adam(lr=lr)
    for _ in range(epochs):
        perm = rng.permutation(M)
        for start in range(0, M, batch_size):
            idx = perm[start:start + batch_size]
            _, grads = student_grads(
                params["W1"], mask1, params["b1"], params["W2"], params["b2"],
                X[idx], y[idx], eta)
            opt.step(params, grads)
            params["W1"] *= mask1
    return params


# ─── Fisher + OBD ─────────────────────────────────────────────────────────────

def estimate_fisher_diag_W1(W1, mask1, b1, W2, b2, X, y, n_batches=30, batch_size=256, seed=0):
    M = X.shape[0]
    rng = np.random.default_rng(seed)
    F_diag = np.zeros_like(W1)
    for _ in range(n_batches):
        idx = rng.choice(M, size=min(batch_size, M), replace=False)
        _, grads = student_grads(W1, mask1, b1, W2, b2, X[idx], y[idx], eta=0.0)
        F_diag += grads["W1"] ** 2
    return F_diag / n_batches


def obd_prune_step(W1, mask1, F_diag, rho):
    sal = 0.5 * F_diag * (W1 * mask1) ** 2
    active = mask1.astype(bool)
    prune = active & (sal < rho / 2.0)
    new_mask = mask1.copy()
    new_mask[prune] = 0.0
    return new_mask, int(prune.sum())


# ─── Permutation-corrected Hamming distance ───────────────────────────────────

def hamming_permutation_corrected(W1_student, mask1_student, W1_teacher, mask1_teacher):
    """
    Find the best permutation of student neurons to teacher neurons (Hungarian),
    then compute Hamming distance between matched masks.

    Similarity metric: cosine similarity of effective weight vectors (W * mask).
    """
    N_h = W1_student.shape[0]

    # Effective weight vectors (L2 normalized)
    W1s = W1_student * mask1_student  # (N_h, N_in)
    W1t = W1_teacher * mask1_teacher

    # Normalize rows
    norm_s = np.linalg.norm(W1s, axis=1, keepdims=True) + 1e-10
    norm_t = np.linalg.norm(W1t, axis=1, keepdims=True) + 1e-10
    W1s_n = W1s / norm_s
    W1t_n = W1t / norm_t

    # Cosine similarity matrix: (N_h, N_h)
    cos_sim = W1s_n @ W1t_n.T  # student rows × teacher rows

    # Hungarian algorithm: maximize similarity (minimize negative)
    row_ind, col_ind = linear_sum_assignment(-cos_sim)

    # Reorder student mask according to matching
    mask_student_matched = mask1_student[col_ind]  # match student[col_ind] to teacher[row_ind]

    # Hamming on matched masks
    # row_ind is always [0,1,...,N_h-1] when matrix is square
    ham_matched = float(np.mean(mask_student_matched != mask1_teacher[row_ind]))

    # Also raw Hamming for comparison
    ham_raw = float(np.mean(mask1_student != mask1_teacher))

    return ham_matched, ham_raw


def hamming_raw(mask_student, mask_teacher):
    return float(np.mean(mask_student != mask_teacher))


# ─── Single trial ─────────────────────────────────────────────────────────────

def run_single(cfg, rho, seed):
    rng_init = np.random.default_rng(seed * 1000)

    teacher = generate_sparse_teacher(
        cfg.N_in, cfg.N_h, cfg.N_out, cfg.teacher_sparsity, cfg.sigma_w, seed)
    X_train, y_train = generate_dataset(teacher, cfg.M_train, seed)
    X_test, y_test = generate_dataset(teacher, cfg.M_test, seed + 42)

    # Init student
    scale = np.sqrt(2.0 / cfg.N_in)
    W1 = rng_init.normal(0, scale, (cfg.N_h, cfg.N_in)).astype(np.float32)
    b1 = np.zeros(cfg.N_h, dtype=np.float32)
    W2 = rng_init.normal(0, scale / np.sqrt(cfg.N_h), (cfg.N_out, cfg.N_h)).astype(np.float32)
    b2 = np.zeros(cfg.N_out, dtype=np.float32)
    mask1 = np.ones((cfg.N_h, cfg.N_in), dtype=np.float32)

    # Pretrain
    params = train_student(W1, mask1, b1, W2, b2, X_train, y_train,
                           cfg.eta, cfg.lr, cfg.pretrain_epochs, cfg.batch_size, seed=seed)
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    # Iterative prune + finetune
    total_pruned = 0
    for round_idx in range(cfg.max_prune_rounds):
        F_diag = estimate_fisher_diag_W1(W1, mask1, b1, W2, b2, X_train, y_train,
                                          cfg.fisher_batches, cfg.batch_size, seed=seed + round_idx)
        mask1, n_pruned = obd_prune_step(W1, mask1, F_diag, rho)
        total_pruned += n_pruned
        W1 *= mask1
        if n_pruned > 0:
            params = train_student(W1, mask1, b1, W2, b2, X_train, y_train,
                                   cfg.eta, cfg.lr, cfg.finetune_epochs, cfg.batch_size,
                                   seed=seed + 100 + round_idx)
            W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
            W1 *= mask1
        else:
            break

    # Evaluate
    _, pred_test = student_forward(W1, mask1, b1, W2, b2, X_test)
    test_mse = float(np.mean((pred_test - y_test) ** 2))

    # Hamming: both raw and permutation-corrected
    ham_perm, ham_raw = hamming_permutation_corrected(
        W1, mask1, teacher["W1"], teacher["h1"])
    active_frac = float(mask1.mean())
    teacher_active = float(teacher["h1"].mean())

    return {
        "rho": rho, "seed": seed,
        "hamming_perm": ham_perm,
        "hamming_raw": ham_raw,
        "active_frac": active_frac,
        "teacher_active_frac": teacher_active,
        "test_mse": test_mse,
        "total_pruned": total_pruned,
        "rounds": round_idx + 1,
    }


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class Config:
    N_in: int = 80
    N_h: int = 40
    N_out: int = 1
    teacher_sparsity: float = 0.5
    sigma_w: float = 1.0

    M_train: int = 4000
    M_test: int = 1000

    eta: float = 1e-4
    lr: float = 1e-3
    pretrain_epochs: int = 30
    finetune_epochs: int = 15
    max_prune_rounds: int = 40
    batch_size: int = 256
    fisher_batches: int = 30

    rho_values: list = field(default_factory=lambda: [
        0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2
    ])
    n_seeds: int = 5


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_experiment(cfg: Config):
    results = []
    total = len(cfg.rho_values) * cfg.n_seeds
    done = 0

    print(f"Exp 27: Synthetic teacher — {cfg.N_in}→{cfg.N_h}→{cfg.N_out}, "
          f"teacher_sparsity={cfg.teacher_sparsity}, M_train={cfg.M_train}")
    print(f"Hamming: permutation-corrected (Hungarian matching) + raw")
    print(f"Sweeping {len(cfg.rho_values)} rho × {cfg.n_seeds} seeds = {total} trials\n")

    for rho in cfg.rho_values:
        perm_hams, raw_hams = [], []
        for seed in range(cfg.n_seeds):
            r = run_single(cfg, rho, seed)
            results.append(r)
            perm_hams.append(r["hamming_perm"])
            raw_hams.append(r["hamming_raw"])
            done += 1
            print(f"  [{done:3d}/{total}] rho={rho:.1e}  seed={seed}  "
                  f"ham_perm={r['hamming_perm']:.3f}  ham_raw={r['hamming_raw']:.3f}  "
                  f"active={r['active_frac']:.3f}  teacher={r['teacher_active_frac']:.3f}  "
                  f"mse={r['test_mse']:.4f}")
        print(f"  ─── rho={rho:.1e}  mean_ham_perm={np.mean(perm_hams):.4f}  "
              f"mean_ham_raw={np.mean(raw_hams):.4f}\n")

    # Save
    out_path = RESULTS_DIR / "synthetic_teacher.csv"
    keys = ["rho", "seed", "hamming_perm", "hamming_raw", "active_frac",
            "teacher_active_frac", "test_mse", "total_pruned", "rounds"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"Results saved → {out_path}")

    print("\n=== Summary ===")
    print(f"{'rho':>10}  {'ham_perm':>10}  {'ham_raw':>10}  {'active':>8}  {'mse':>10}")
    for rho in cfg.rho_values:
        rows = [r for r in results if r["rho"] == rho]
        print(f"{rho:>10.1e}  "
              f"{np.mean([r['hamming_perm'] for r in rows]):>10.4f}  "
              f"{np.mean([r['hamming_raw'] for r in rows]):>10.4f}  "
              f"{np.mean([r['active_frac'] for r in rows]):>8.4f}  "
              f"{np.mean([r['test_mse'] for r in rows]):>10.4f}")
    return results


if __name__ == "__main__":
    cfg = Config()
    run_experiment(cfg)

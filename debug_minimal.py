"""Minimal test of replicas implementation."""
import numpy as np
from pruning_core.replicas import MultiReplicaGlauber

# Minimal setup
M = 10
d = 5
seed = 42

rng = np.random.RandomState(seed)
X = rng.randn(M, d)
h_true = (rng.rand(d) < 0.5).astype(float)
w_true = rng.randn(d)
y = X @ (w_true * h_true) + rng.randn(M) * 0.1

print("Data: X shape =", X.shape)
print("y shape:", y.shape)

replica = MultiReplicaGlauber(n_replicas=1, eta_val=0.001, alpha=1.0)
w_init = rng.randn(d, 1) * 0.1
h_init = np.ones((d, 1), dtype=float)

# Check forward pass manually
print("\nManual forward pass check:")
print("X @ (w*h):", (X @ (w_init * h_init)).shape)  # Should be (M, 1)

try:
    w_final, h_final, losses = replica.run(
        [[w_init]], [h_init], X, y, [0.001], [0.01], [1.0],
        T=5, T_h=1.0, rng=np.random.default_rng(seed)
    )
    print("Success! h_final shape:", h_final[0].shape)
except Exception as e:
    print("Error:", e)

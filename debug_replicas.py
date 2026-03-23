"""Debug script to understand dimension requirements."""
import numpy as np

# Simulate what replicas.py does
M = 300
d_in = 100
d_out = 1

# Forward pass
X = np.random.randn(M, d_in)  # (300, 100)
h = np.ones((d_in, d_out))  # (100, 1)
w = np.random.randn(d_in, d_out) * 0.1  # (100, 1)
y = np.random.randn(M)  # (300,)

# Forward
print("Forward pass:")
print(f"  X shape: {X.shape}")
print(f"  h shape: {h.shape}")
print(f"  w shape: {w.shape}")
print(f"  w*h shape: {(w*h).shape}")
a1 = X @ (w * h)
print(f"  a1 = X @ (w*h) shape: {a1.shape}")
y_pred = a1.astype(float)
print(f"  y_pred shape: {y_pred.shape}")

# Backward delta
delta = y_pred - y.ravel()
print(f"\nDelta (output - y): {delta.shape}")
print(f"  y.ravel().shape: {y.ravel().shape}")

# Try the reshapes
delta_reshaped = delta.reshape(-1, 1)
print(f"\nDelta.reshape(-1,1) shape: {delta_reshaped.shape}")

# a_prev from forward = X
a_prev = X
print(f"a_prev = X shape: {a_prev.shape}")
print(f"a_prev.T shape: {a_prev.T.shape}")

# Gradient computation
print(f"\nGradient: a_prev.T @ delta_2d")
print(f"  a_prev.T shape: {a_prev.T.shape}")
print(f"  delta_2d shape: {delta_reshaped.shape}")

try:
    grad = a_prev.T @ delta_reshaped
    print(f"  Resulting gradient shape: {grad.shape}")
except ValueError as e:
    print(f"  Error: {e}")

# Check what shape we need
print(f"\nWeight shape: {w.shape}")
print(f"For backprop, we need grad shape = w.shape = {w.shape}")

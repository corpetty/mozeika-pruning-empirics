"""
Experiment 11: What does pruning remove spectrally?
"""
import numpy as np
import os


def mlp_sample(M, layer_sizes, sigma=0.01, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = np.random.randn(M, layer_sizes[0])
    w0_list = []
    h0_list = []
    for N_in, N_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        w = np.random.randn(N_in, N_out) * sigma
        h = (np.random.rand(N_in, N_out) > 0.5).astype(float)
        w0_list.append(w.copy())
        h0_list.append(h.copy())
    h0_list[-1] = np.ones_like(h0_list[-1])
    a = X.copy()
    for k in range(len(w0_list)):
        w, h = w0_list[k], h0_list[k]
        z = a @ (w * h)
        a = np.tanh(z) if k < len(h0_list) - 1 else z
    y = a.flatten().reshape(-1, 1)
    return X, y, w0_list, h0_list


def grad_mlp_loss_w(w_list, h_list, X, y):
    M = X.shape[0]
    n_layers = len(w_list)
    a = X.copy()
    a_list = [X.copy()]
    z_list = []
    for k in range(n_layers):
        w, h = w_list[k], h_list[k]
        z = a @ (w * h)
        z_list.append(z.copy())
        if k < n_layers - 1:
            a = np.tanh(z)
        else:
            a = z
        a_list.append(a.copy())
    grads = []
    delta = a_list[-1] - y
    for l in range(n_layers - 1, -1, -1):
        a_prev = a_list[l]
        delta_2d = delta
        grad = a_prev.T @ delta_2d
        grad = grad * h_list[l]
        grads.insert(0, grad)
        if l > 0:
            delta_prev = delta_2d @ w_list[l].T
            delta_prev = delta_prev * (1 - z_list[l - 1] ** 2)
            delta = delta_prev
    return grads


def optimize_adam(w_list, h_list, X, y, eta_list, K=50, lr=1e-2):
    ms = [np.zeros_like(w.flatten()) for w in w_list]
    vs = [np.zeros_like(w.flatten()) for w in w_list]
    for k in range(K):
        grads = grad_mlp_loss_w(w_list, h_list, X, y)
        for l in range(len(w_list)):
            grads[l] = grads[l] + eta_list[l] * w_list[l]
        for l in range(len(w_list)):
            w_flat = w_list[l].flatten()
            g = grads[l].flatten()
            ms[l] = 0.9 * ms[l] + 0.1 * g
            vs[l] = 0.99 * vs[l] + 0.01 * g ** 2
            m_hat = ms[l] / (1 - 0.9 ** (k + 1))
            v_hat = vs[l] / (1 - 0.99 ** (k + 1))
            w_new = w_flat - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            w_list[l] = w_new.reshape(w_list[l].shape)
    return w_list


def total_energy(w_list, h_list, X, y, eta_list, alpha, rho_list):
    n_layers = len(w_list)
    a = X
    for k in range(n_layers):
        z = a @ (w_list[k] * h_list[k])
        a = np.tanh(z) if k < n_layers - 1 else z
    L = np.mean((y - a.flatten()) ** 2) / 2
    reg = sum(eta * np.sum(w ** 2) / 2 for w, eta in zip(w_list, eta_list))
    V = sum(alpha * np.sum((h ** 2) * ((h - 1) ** 2)) + rho * np.sum(h) / 2
            for h, rho in zip(h_list, rho_list))
    return L + reg + V


def run_glauber(w_init_list, h_init_list, X, y, eta_list, alpha, rho_list, T=50, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    w = [wi.copy() for wi in w_init_list]
    h = [hi.copy() for hi in h_init_list]
    n_layers = len(w)
    for t in range(T):
        for l in range(n_layers):
            in_dim, out_dim = h[l].shape
            order = rng.permutation(in_dim * out_dim)
            for idx in order:
                row = idx // out_dim
                col = idx % out_dim
                h_try = [hi.copy() for hi in h]
                h_try[l][row, col] = 1 - h_try[l][row, col]
                w_try = [wi.copy() for wi in w]
                for _ in range(3):
                    grads = grad_mlp_loss_w(w_try, h_try, X, y)
                    for ll in range(n_layers):
                        w_try[ll] = w_try[ll] - 0.01 * (grads[ll] + eta_list[ll] * w_try[ll])
                E_curr = total_energy(w, h, X, y, eta_list, alpha, rho_list)
                E_try = total_energy(w_try, h_try, X, y, eta_list, alpha, rho_list)
                if E_try < E_curr:
                    w = w_try
                    h = h_try
        w = optimize_adam(w, h, X, y, eta_list, K=20)
    return w, h


def variance_in_top_k(S, k):
    p = (S ** 2) / np.sum(S ** 2)
    p_sorted = np.sort(p)[::-1]
    return np.sum(p_sorted[:k])


def main():
    print("Experiment 11: Variance concentration analysis")
    layer_sizes = [20, 16, 1]
    M = 100
    n_seeds = 5
    alpha = 1.0
    eta_val = 0.0001
    eta_list = [eta_val] * (len(layer_sizes) - 1)
    rhos = [0, 0.0003, 0.0005, 0.001]
    os.makedirs('results', exist_ok=True)
    
    results = []
    for rho in rhos:
        print(f"rho={rho}...")
        rho_list = [rho] * (len(layer_sizes) - 1)
        S_all = []
        for seed in range(n_seeds):
            X, y, w0, h0 = mlp_sample(M, layer_sizes, sigma=0.01, seed=9000 + seed)
            for h in h0:
                h[:] = 1.0
            w = [wi.copy() for wi in w0]
            h = h0
            w = optimize_adam(w, h, X, y, eta_list, K=300)
            w_final, h_final = run_glauber(
                w, h.copy(), X, y, eta_list, alpha, rho_list, T=50,
                rng=np.random.default_rng(8000 + seed)
            )
            W_layer0 = w_final[0]
            S_all.append(W_layer0)
        
        W_stacked = np.array(S_all)
        W_flat = W_stacked.reshape(n_seeds, -1)
        U, S, Vt = np.linalg.svd(W_flat, full_matrices=False)
        results.append({
            'rho': rho,
            'var_top_1': variance_in_top_k(S, 1),
            'var_top_3': variance_in_top_k(S, 3),
            'var_top_5': variance_in_top_k(S, 5),
        })
        print(f"  top-1={variance_in_top_k(S, 1):.3%}, top-3={variance_in_top_k(S, 3):.3%}, top-5={variance_in_top_k(S, 5):.3%}")
    
    csv_path = 'results/variance_concentration.csv'
    with open(csv_path, 'w') as f:
        f.write('rho,var_top_1,var_top_3,var_top_5\n')
        for r in results:
            f.write(f"{r['rho']},{r['var_top_1']:.6f},{r['var_top_3']:.6f},{r['var_top_5']:.6f}\n")
    
    print(f"\nSaved to {csv_path}")
    return results


if __name__ == "__main__":
    main()

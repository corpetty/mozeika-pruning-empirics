"""
Experiment 10: Debug version with explicit shape checking.
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
    y = a.flatten()
    return X, y, w0_list, h0_list


def grad_mlp_loss_w(w_list, h_list, X, y):
    M = X.shape[0]
    n_layers = len(w_list)
    
    # Forward with caching
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
    
    # Backward
    grads = []
    delta = a_list[-1] - y  # (M,) or (M, 1) - output is linear so same
    
    print(f"  grad: delta.init.shape={delta.shape}")
    
    for l in range(n_layers - 1, -1, -1):
        a_prev = a_list[l]
        print(f"  grad l={l}: a_prev.shape={a_prev.shape}, delta.shape={delta.shape}, w_list[{l}].shape={w_list[l].shape}")
        
        delta_2d = delta.reshape(-1, 1)
        print(f"  grad l={l}: delta_2d.shape={delta_2d.shape}")
        
        grad = a_prev.T @ delta_2d
        print(f"  grad l={l}: grad.shape={grad.shape}, target={w_list[l].shape}")
        
        grad = grad * h_list[l]
        grads.insert(0, grad)
        
        if l > 0:
            print(f"  grad l={l}: computing delta_prev, w_list[{l}].T.shape={w_list[l].T.shape}")
            delta_prev = delta_2d @ w_list[l].T
            print(f"  grad l={l}: delta_prev.shape={delta_prev.shape}")
            delta_prev = delta_prev * (1 - z_list[l - 1] ** 2)
            print(f"  grad l={l}: after tanh', delta_prev.shape={delta_prev.shape}")
            delta = delta_prev.flatten()
            print(f"  grad l={l}: after flatten, delta.shape={delta.shape}")
    
    return grads


def optimize_adam(w_list, h_list, X, y, eta_list, K=50, lr=1e-2):
    for k in range(K):
        grads = grad_mlp_loss_w(w_list, h_list, X, y)
        for l in range(len(w_list)):
            grads[l] = grads[l] + eta_list[l] * w_list[l]
        
        for l in range(len(w_list)):
            w_flat = w_list[l].flatten()
            g = grads[l].flatten()
            ms = np.zeros_like(w_flat)
            vs = np.zeros_like(w_flat)
            ms = 0.9 * ms + 0.1 * g
            vs = 0.99 * vs + 0.01 * g ** 2
            m_hat = ms / (1 - 0.9 ** (k + 1))
            v_hat = vs / (1 - 0.99 ** (k + 1))
            w_flat = w_flat - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            w_list[l] = w_flat.reshape(w_list[l].shape)
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
    losses = []
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
        E = total_energy(w, h, X, y, eta_list, alpha, rho_list)
        losses.append(E)
    return w, h, losses


def spectral_analysis(W_matrix):
    num_seeds, in_dim, out_dim = W_matrix.shape
    W_flat = W_matrix.reshape(num_seeds, -1)
    U, S, Vt = np.linalg.svd(W_flat, full_matrices=False)
    eff_rank = np.sum(S) / np.max(S) if np.max(S) > 0 else 0
    p = (S ** 2) / np.sum(S ** 2)
    spec_entropy = -np.sum(p * np.log(p + 1e-10))
    return S, eff_rank, spec_entropy


def main():
    np.random.seed(42)
    os.makedirs('results', exist_ok=True)
    
    layer_sizes = [20, 16, 1]
    M = 100
    n_seeds = 5
    
    alpha = 1.0
    eta_val = 0.0001
    eta_list = [eta_val] * (len(layer_sizes) - 1)
    
    rhos = [0, 0.0003, 0.0005, 0.001]
    
    print(f"Exp 10: MLP {layer_sizes}, M={M}, seeds={n_seeds}, rhos={rhos}\n")
    
    results = []
    
    for rho in rhos:
        print(f"rho={rho}...")
        rho_list = [rho] * (len(layer_sizes) - 1)
        
        all_W_layer0 = []
        losses_all = []
        
        for seed in range(n_seeds):
            X, y, w0, h0 = mlp_sample(M, layer_sizes, sigma=0.01, seed=9000 + seed)
            print(f"seed={seed}: X.shape={X.shape}, y.shape={y.shape}")
            
            for h in h0:
                h[:] = 1.0
            
            w = [wi.copy() for wi in w0]
            h = h0
            print(f"Starting optimization batch...")
            w = optimize_adam(w, h, X, y, eta_list, K=50)  # Just test K=50
            
            w_final, h_final, losses = run_glauber(
                w, h.copy(), X, y, eta_list, alpha, rho_list, T=50,
                rng=np.random.default_rng(8000 + seed)
            )
            
            all_W_layer0.append(w_final[0])
            losses_all.append(np.mean(losses))
            print(f"  seed {seed}: loss = {np.mean(losses):.4f}, w0.shape={w_final[0].shape}\n")
        
        W_stacked = np.array(all_W_layer0)
        S, eff_rank, spec_entropy = spectral_analysis(W_stacked)
        
        print(f"  rank={eff_rank:.4f}, entropy={spec_entropy:.4f}\n")
        
        for seed in range(n_seeds):
            results.append({
                'rho': rho,
                'seed': seed,
                'effective_rank': eff_rank,
                'spectral_entropy': spec_entropy,
                'loss': losses_all[seed]
            })
    
    csv_path = 'results/spectral_structure.csv'
    with open(csv_path, 'w') as f:
        f.write('rho,seed,effective_rank,spectral_entropy,loss\n')
        for r in results:
            f.write(f"{r['rho']},{r['seed']},{r['effective_rank']:.6f},{r['spectral_entropy']:.6f},{r['loss']:.6f}\n")
    
    print(f"Saved: {csv_path}")
    
    print("\nSummary:")
    for rho in rhos:
        rho_res = [r for r in results if r['rho'] == rho]
        print(f"  rho={rho}: rank={np.mean([r['effective_rank'] for r in rho_res]):.4f}, "
              f"entropy={np.mean([r['spectral_entropy'] for r in rho_res]):.4f}")
    
    return results


if __name__ == "__main__":
    main()

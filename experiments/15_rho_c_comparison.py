"""
Stage 5b: Extract empirical rho_c from phase_diagram.csv and compare with theory.

From Mozeika theory: rho_c ≈ 2*sqrt(alpha * eta) in the single-replica case.
"""
import numpy as np
import os

# Load phase_diagram.csv
csv_path = 'results/phase_diagram.csv'
data = []
with open(csv_path, 'r') as f:
    next(f)  # Skip header
    for line in f:
        eta, rho, hamming, rho_c_est = line.strip().split(',')
        data.append({
            'eta': float(eta),
            'rho': float(rho),
            'hamming': float(hamming),
            'rho_c_est': float(rho_c_est)
        })

# Extract empirical rho_c (where hamming drops below 0.1)
print("=== Empirical rho_c extraction ===")
empirical_results = []
for eta in np.unique([d['eta'] for d in data]):
    rows = [d for d in data if abs(d['eta'] - eta) < 0.00001]
    # Sort by rho
    rows.sort(key=lambda x: x['rho'])
    
    # Find first rho where hamming < 0.1
    for d in rows:
        if d['hamming'] < 0.1:
            empirical_rho_c = d['rho']
            break
    else:
        # No crossing found, estimate from last row
        empirical_rho_c = rows[-1]['rho']
    
    empirical_results.append({'eta': eta, 'empirical_rho_c': empirical_rho_c})
    print(f"eta={eta:.6f}: empirical rho_c = {empirical_rho_c:.6f}")

# Theoretical prediction: rho_c = 2*sqrt(alpha * eta)
# From experiment 03, alpha=1.0
ALPHA = 1.0
theoretical_results = []
print("\n=== Theoretical vs Empirical ===")
for er in empirical_results:
    eta = er['eta']
    rho_c_empirical = er['empirical_rho_c']
    rho_c_theory = 2 * np.sqrt(ALPHA * eta)
    ratio = rho_c_empirical / rho_c_theory if rho_c_theory > 0 else np.nan
    
    theoretical_results.append({
        'source': 'empirical',
        'rho_c': rho_c_empirical,
        'alpha': ALPHA,
        'eta': eta
    })
    theoretical_results.append({
        'source': 'theoretical',
        'rho_c': rho_c_theory,
        'alpha': ALPHA,
        'eta': eta
    })
    
    print(f"eta={eta:.6f}: empirical={rho_c_empirical:.6f}, theory={rho_c_theory:.6f}, ratio={ratio:.3f}")

os.makedirs('results', exist_ok=True)

# Save comparison
csv_path = 'results/rho_c_comparison.csv'
with open(csv_path, 'w') as f:
    f.write('source,rho_c,alpha,eta\n')
    for r in theoretical_results:
        f.write(f"{r['source']},{r['rho_c']:.6f},{r['alpha']},{r['eta']:.6f}\n")

print(f"\nSaved to {csv_path}")

# Save detailed comparison
detailed_path = 'results/rho_c_detailed.txt'
with open(detailed_path, 'w') as f:
    f.write("Empirical vs Theoretical rho_c Comparison\n")
    f.write("=" * 50 + "\n\n")
    f.write("From Mozeika theory: rho_c ≈ 2*sqrt(alpha * eta)\n\n")
    for er in empirical_results:
        eta = er['eta']
        empirical_rho_c = er['empirical_rho_c']
        theoretical_rho_c = 2 * np.sqrt(ALPHA * eta)
        ratio = empirical_rho_c / theoretical_rho_c
        f.write(f"eta={eta:.6f}:\n")
        f.write(f"  Empirical rho_c: {empirical_rho_c:.6f}\n")
        f.write(f"  Theoretical rho_c: {theoretical_rho_c:.6f}\n")
        f.write(f"  Ratio (empirical/theory): {ratio:.3f}\n")
        f.write(f"  Assessment: {'Good match' if 0.8 < ratio < 1.2 else 'Discrepancy'}\n\n")

print(f"Detailed report saved to {detailed_path}")

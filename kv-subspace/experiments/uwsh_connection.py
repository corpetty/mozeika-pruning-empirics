"""
uwsh_connection.py — Test UWSH (Universal Weight Subspace Hypothesis) in KV cache.

Measures whether principal subspaces of KV vectors are shared across:
  (a) Different layers (cross-layer subspace alignment)
  (b) Different attention heads within a layer (cross-head subspace alignment)
  (c) Different text domains (cross-domain subspace stability)

Uses principal angles (scipy.linalg.subspace_angles) to quantify subspace overlap.
A subspace overlap of 1.0 means identical subspaces; 0.0 means orthogonal.

Usage:
    /home/petty/torch-env/bin/python3 experiments/uwsh_connection.py
"""

import sys
import os
import csv
import numpy as np
from pathlib import Path
from scipy.linalg import subspace_angles

# Add parent dir to path so we can import collect.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collect import load_kvs, get_model_and_tokenizer, collect_kv_vectors

np.random.seed(42)

# ── Config ───────────────────────────────────────────────────────────────────

KVS_PATH = "results/kvs.npz"
KVS_DOMAIN2_PATH = "results/kvs_domain2.npz"
OUTPUT_CSV = "results/uwsh_connection.csv"
VARIANCE_THRESHOLD = 0.90  # fraction of variance for choosing k
D_HEAD = 128
N_HEADS = 8
N_LAYERS = 40

# Domain 2 text: Wikipedia-style factual content (contrasting with Project Gutenberg fiction)
DOMAIN2_TEXT = """
Quantum computing is a type of computation whose operations can harness the phenomena of quantum
mechanics, such as superposition, interference, and entanglement. Devices that perform quantum
computations are known as quantum computers. Though current quantum computers may be too small to
outperform usual computers for practical applications, larger realizations are believed to be
capable of solving certain computational problems, such as integer factorization, substantially
faster than classical computers. The study of quantum computing is a subfield of quantum
information science. The basic unit of information in quantum computing is the qubit, similar to
the bit in traditional digital electronics. Unlike a classical bit, a qubit can exist in a
superposition of its two basic states. When measuring a qubit, the result is a probabilistic
outcome of the two states. A quantum gate can put a qubit into superposition. However,
a measurement of any number of qubits in superposition would not be useful in quantum computing
without quantum entanglement. Entanglement provides a correlation between the quantum states of
two or more qubits. Quantum algorithms exploit this entanglement to solve problems. As the
number of entangled qubits grows, the number of possible states in superposition grows
exponentially, ideally allowing the quantum computer to simultaneously consider far more states
than a classical computer can do. This has significance because many hard computational
problems require searching through a large number of possible solutions.

Machine learning is a branch of artificial intelligence and computer science which focuses on
the use of data and algorithms to imitate the way that humans learn, gradually improving its
accuracy. Machine learning is an important component of the growing field of data science.
Through the use of statistical methods, algorithms are trained to make classifications or
predictions, and to uncover key insights in data mining projects. These insights subsequently
drive decision making within applications and businesses, ideally impacting key growth metrics.
As big data continues to expand and grow, the market demand for data scientists will increase.
They will be required to help identify the most relevant business questions and the data to
answer them. Machine learning algorithms are typically created using frameworks that accelerate
solution development, such as TensorFlow and PyTorch.

The transformer is a deep learning architecture developed by researchers at Google and based on
the multi-head attention mechanism, proposed in the 2017 paper Attention Is All You Need. Text
is converted to numerical representations called tokens, and each token is converted into a
vector via looking up from a word embedding table. At each layer, each token is then
contextualized within the scope of the context window with other unmasked tokens via a parallel
multi-head attention mechanism, allowing the signal for key tokens to be amplified and less
important tokens to be diminished. Transformers have the advantage of having no recurrent units
and therefore require less training time than earlier recurrent neural architectures such as
long short-term memory. Later variations have been widely adopted for training large language
models on large language datasets, such as the Wikipedia corpus and Common Crawl.

Neural networks are computing systems inspired by the biological neural networks that
constitute animal brains. An artificial neural network is based on a collection of connected
units or nodes called artificial neurons, which loosely model the neurons in a biological
brain. Each connection, like the synapses in a biological brain, can transmit a signal to other
neurons. An artificial neuron receives signals then processes them and can signal neurons
connected to it. The signal at a connection is a real number, and the output of each neuron is
computed by some non-linear function of the sum of its inputs. The connections are called edges.
Neurons and edges typically have a weight that adjusts as learning proceeds. The weight
increases or decreases the strength of the signal at a connection. Neurons may have a threshold
such that a signal is sent only if the aggregate signal crosses that threshold.

Cryptography is the practice and study of techniques for secure communication in the presence
of adversarial behavior. More generally, cryptography is about constructing and analyzing
protocols that prevent third parties or the public from reading private messages. Modern
cryptography exists at the intersection of the disciplines of mathematics, computer science,
information security, electrical engineering, digital signal processing, physics, and others.
Core concepts in cryptography include encryption, decryption, cipher, and cryptographic hash
functions. Applications of cryptography include electronic commerce, chip-based payment cards,
digital currencies, computer passwords, and military communications.

The theory of general relativity describes gravity as a geometric property of space and time,
or four-dimensional spacetime. In particular, the curvature of spacetime is directly related to
the energy and momentum of whatever matter and radiation are present. The relation is specified
by the Einstein field equations, a system of second order partial differential equations. Some
predictions of general relativity differ significantly from those of classical physics,
especially concerning the passage of time, the geometry of space, the motion of bodies in free
fall, and the propagation of light. Examples of such differences include gravitational time
dilation, gravitational lensing, the gravitational redshift of light, the Shapiro time delay
and singularities or black holes.

Protein structure prediction is the inference of the three-dimensional structure of a protein
from its amino acid sequence—that is, the prediction of its secondary and tertiary structure
from primary structure. Structure prediction is different from the inverse problem of protein
design. Protein structure prediction is one of the most important goals pursued by
computational biology and bioinformatics. It is highly important in medicine for drug design
and in biotechnology for the design of novel enzymes. Every two years, the performance of
current methods is assessed in the CASP experiment which stands for Critical Assessment of
protein Structure Prediction. A continuous evaluation of protein structure prediction web
servers is performed by the community project CAMEO3D.

Climate change includes both human-driven global warming and its larger effects on Earth's
weather patterns. There have been previous periods of climate change, but the current changes
are distinctly more rapid and not due to natural causes. Instead, they are caused by the
emission of greenhouse gases, mostly carbon dioxide and methane. Burning fossil fuels for
energy use creates most of these emissions. Certain agricultural practices, industrial
processes, and forest loss are additional sources. Greenhouse gases are transparent to sunlight,
allowing it through to heat the Earth's surface. When the Earth emits that heat as infrared
radiation the gases absorb it, trapping the heat near the Earth's surface and causing global
warming.

Database management systems provide efficient mechanisms for storing, retrieving, and
manipulating data. Relational database management systems use structured query language for
database creation and data manipulation. SQL statements are used for both interactive queries
and embedding in application programs. SQL supports a wide range of operations including
selection of specific records, joining tables, aggregation functions, subqueries, and
transaction management. Modern databases also support indexing techniques such as B-trees and
hash indexes for efficient data retrieval. NoSQL databases have emerged as alternatives to
relational databases for certain use cases, offering flexible schemas, horizontal scalability,
and optimized performance for specific data models such as document, key-value, columnar, and
graph databases.

Distributed systems are computing systems in which components located on networked computers
communicate and coordinate their actions by passing messages. The components interact with one
another in order to achieve a common goal. Three significant challenges of distributed systems
are maintaining concurrency of components, overcoming the lack of a global clock, and managing
the independent failure of components. When a component of one system fails, the entire system
does not fail. Examples of distributed systems vary from SOA-based systems to massively
multiplayer online games to peer-to-peer applications. A computer program that runs within a
distributed system is called a distributed program. Distributed computing also refers to the
use of distributed systems to solve computational problems.
""" * 6  # Repeat to get enough tokens


def pca_basis(X, variance_threshold=VARIANCE_THRESHOLD):
    """Compute PCA basis keeping enough components for variance_threshold.
    Returns (U_k, k, explained_variance_ratios)."""
    mean = X.mean(axis=0)
    Xc = X - mean
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    cumvar = np.cumsum(var) / var.sum()
    k = int(np.searchsorted(cumvar, variance_threshold) + 1)
    k = min(k, len(S))
    return Vt[:k].T, k  # (d, k), k


def subspace_overlap(U1, U2):
    """Compute subspace overlap from principal angles.
    Returns mean cosine of principal angles (1 = identical, 0 = orthogonal).
    U1: (d, k1), U2: (d, k2) — orthonormal column matrices."""
    angles = subspace_angles(U1, U2)
    # Cosines of principal angles
    cosines = np.cos(angles)
    return float(np.mean(cosines))


def compute_cross_layer(kvs, kv_type="K"):
    """Cross-layer subspace alignment: compare layer i with layer i+1."""
    print(f"\n  Cross-layer alignment ({kv_type})...", end=""); sys.stdout.flush()
    rows = []
    layers = sorted(kvs.keys())

    # Pre-compute PCA bases per layer (average across heads for cross-layer)
    bases = {}
    for li in layers:
        # Stack all heads: (T * n_heads, d_head)
        X = kvs[li][kv_type]  # (T, n_heads, d_head)
        T, H, D = X.shape
        X_flat = X.reshape(T * H, D)
        U, k = pca_basis(X_flat)
        bases[li] = (U, k)

    for i in range(len(layers) - 1):
        la, lb = layers[i], layers[i + 1]
        U_a, k_a = bases[la]
        U_b, k_b = bases[lb]
        # Use min dimension for comparison
        k_min = min(k_a, k_b)
        overlap = subspace_overlap(U_a[:, :k_min], U_b[:, :k_min])
        rows.append({
            "layer_a": la, "layer_b": lb,
            "head_a": -1, "head_b": -1,
            "kv_type": kv_type,
            "metric_type": "cross_layer",
            "k_a": k_a, "k_b": k_b,
            "subspace_overlap": overlap,
        })

    print(f" done ({len(rows)} pairs)")
    sys.stdout.flush()
    return rows


def compute_cross_head(kvs, kv_type="K"):
    """Cross-head subspace alignment: compare head 0 with head h within each layer."""
    print(f"\n  Cross-head alignment ({kv_type})...", end=""); sys.stdout.flush()
    rows = []
    layers = sorted(kvs.keys())
    count = 0

    for li in layers:
        X = kvs[li][kv_type]  # (T, n_heads, d_head)
        T, H, D = X.shape

        # PCA basis per head
        head_bases = {}
        for h in range(H):
            U, k = pca_basis(X[:, h, :])
            head_bases[h] = (U, k)

        # Compare head 0 to all others
        U_0, k_0 = head_bases[0]
        for h in range(1, H):
            U_h, k_h = head_bases[h]
            k_min = min(k_0, k_h)
            overlap = subspace_overlap(U_0[:, :k_min], U_h[:, :k_min])
            rows.append({
                "layer_a": li, "layer_b": li,
                "head_a": 0, "head_b": h,
                "kv_type": kv_type,
                "metric_type": "cross_head",
                "k_a": k_0, "k_b": k_h,
                "subspace_overlap": overlap,
            })
            count += 1

        # Also compare all pairs for summary stats
        for ha in range(H):
            for hb in range(ha + 1, H):
                if ha == 0:
                    continue  # already done above
                U_a, k_a = head_bases[ha]
                U_b, k_b = head_bases[hb]
                k_min = min(k_a, k_b)
                overlap = subspace_overlap(U_a[:, :k_min], U_b[:, :k_min])
                rows.append({
                    "layer_a": li, "layer_b": li,
                    "head_a": ha, "head_b": hb,
                    "kv_type": kv_type,
                    "metric_type": "cross_head",
                    "k_a": k_a, "k_b": k_b,
                    "subspace_overlap": overlap,
                })
                count += 1

    print(f" done ({len(rows)} pairs)")
    sys.stdout.flush()
    return rows


def compute_cross_domain(kvs_domain1, kvs_domain2, kv_type="K"):
    """Cross-domain subspace stability: compare PCA subspaces from two different texts."""
    print(f"\n  Cross-domain alignment ({kv_type})...", end=""); sys.stdout.flush()
    rows = []
    layers = sorted(set(kvs_domain1.keys()) & set(kvs_domain2.keys()))

    for li in layers:
        X1 = kvs_domain1[li][kv_type]  # (T, n_heads, d_head)
        X2 = kvs_domain2[li][kv_type]

        T1, H, D = X1.shape
        T2 = X2.shape[0]

        for h in range(H):
            U1, k1 = pca_basis(X1[:, h, :])
            U2, k2 = pca_basis(X2[:, h, :])
            k_min = min(k1, k2)
            overlap = subspace_overlap(U1[:, :k_min], U2[:, :k_min])
            rows.append({
                "layer_a": li, "layer_b": li,
                "head_a": h, "head_b": h,
                "kv_type": kv_type,
                "metric_type": "cross_domain",
                "k_a": k1, "k_b": k2,
                "subspace_overlap": overlap,
            })

    print(f" done ({len(rows)} pairs)")
    sys.stdout.flush()
    return rows


def collect_domain2_kvs(n_tokens=2048):
    """Run forward pass on domain 2 text to get KV vectors from a different domain."""
    if Path(KVS_DOMAIN2_PATH).exists():
        print("Loading cached domain 2 KV vectors...")
        sys.stdout.flush()
        return load_kvs(KVS_DOMAIN2_PATH)

    print("Collecting domain 2 KV vectors (loading model, ~2 min)...")
    sys.stdout.flush()

    model, tokenizer = get_model_and_tokenizer("Qwen/Qwen3-14B-AWQ")
    kvs2 = collect_kv_vectors(model, tokenizer, DOMAIN2_TEXT, n_tokens)

    # Save for reuse
    from collect import save_kvs
    Path(KVS_DOMAIN2_PATH).parent.mkdir(parents=True, exist_ok=True)
    save_kvs(kvs2, KVS_DOMAIN2_PATH)

    # Free GPU memory
    import torch, gc
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return kvs2


def print_summary(rows):
    """Print summary tables."""
    print("\n" + "=" * 80)
    print("UWSH CONNECTION — SUBSPACE ALIGNMENT SUMMARY")
    print("=" * 80)
    sys.stdout.flush()

    for kv_type in ["K", "V"]:
        print(f"\n--- {kv_type} vectors ---")

        # Cross-layer
        cl = [r for r in rows if r["metric_type"] == "cross_layer" and r["kv_type"] == kv_type]
        if cl:
            overlaps = [r["subspace_overlap"] for r in cl]
            print(f"\n  Cross-layer (adjacent layers):")
            print(f"    Mean overlap: {np.mean(overlaps):.4f}")
            print(f"    Min:  {np.min(overlaps):.4f} (L{cl[np.argmin(overlaps)]['layer_a']}-L{cl[np.argmin(overlaps)]['layer_b']})")
            print(f"    Max:  {np.max(overlaps):.4f} (L{cl[np.argmax(overlaps)]['layer_a']}-L{cl[np.argmax(overlaps)]['layer_b']})")
            # Grouped by layer range
            early = [r["subspace_overlap"] for r in cl if r["layer_a"] < 10]
            mid = [r["subspace_overlap"] for r in cl if 10 <= r["layer_a"] < 30]
            late = [r["subspace_overlap"] for r in cl if r["layer_a"] >= 30]
            if early: print(f"    Early (L0-9):  {np.mean(early):.4f}")
            if mid:   print(f"    Mid (L10-29):  {np.mean(mid):.4f}")
            if late:  print(f"    Late (L30-39): {np.mean(late):.4f}")

        # Cross-head
        ch = [r for r in rows if r["metric_type"] == "cross_head" and r["kv_type"] == kv_type]
        if ch:
            overlaps = [r["subspace_overlap"] for r in ch]
            print(f"\n  Cross-head (all head pairs, within layers):")
            print(f"    Mean overlap: {np.mean(overlaps):.4f}")
            print(f"    Min:  {np.min(overlaps):.4f}")
            print(f"    Max:  {np.max(overlaps):.4f}")
            # Per-layer
            for l_range, lo, hi in [("Early (L0-9)", 0, 10), ("Mid (L10-29)", 10, 30), ("Late (L30-39)", 30, 40)]:
                subset = [r["subspace_overlap"] for r in ch if lo <= r["layer_a"] < hi]
                if subset:
                    print(f"    {l_range}: {np.mean(subset):.4f}")

        # Cross-domain
        cd = [r for r in rows if r["metric_type"] == "cross_domain" and r["kv_type"] == kv_type]
        if cd:
            overlaps = [r["subspace_overlap"] for r in cd]
            print(f"\n  Cross-domain (same head, different text):")
            print(f"    Mean overlap: {np.mean(overlaps):.4f}")
            print(f"    Min:  {np.min(overlaps):.4f}")
            print(f"    Max:  {np.max(overlaps):.4f}")
            for l_range, lo, hi in [("Early (L0-9)", 0, 10), ("Mid (L10-29)", 10, 30), ("Late (L30-39)", 30, 40)]:
                subset = [r["subspace_overlap"] for r in cd if lo <= r["layer_a"] < hi]
                if subset:
                    print(f"    {l_range}: {np.mean(subset):.4f}")

    sys.stdout.flush()


def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    Path("results").mkdir(exist_ok=True)

    print("=" * 80)
    print("UWSH Connection Experiment")
    print("Testing KV cache subspace alignment across layers, heads, and text domains")
    print("=" * 80)
    sys.stdout.flush()

    # Load domain 1 KV vectors
    print(f"\nLoading domain 1 KV vectors from {KVS_PATH}...")
    sys.stdout.flush()
    kvs1 = load_kvs(KVS_PATH)
    print(f"  Loaded {len(kvs1)} layers")
    sys.stdout.flush()

    all_rows = []

    # (a) Cross-layer alignment
    print("\n--- (a) Cross-layer subspace alignment ---")
    sys.stdout.flush()
    all_rows.extend(compute_cross_layer(kvs1, "K"))
    all_rows.extend(compute_cross_layer(kvs1, "V"))

    # (b) Cross-head alignment
    print("\n--- (b) Cross-head subspace alignment ---")
    sys.stdout.flush()
    all_rows.extend(compute_cross_head(kvs1, "K"))
    all_rows.extend(compute_cross_head(kvs1, "V"))

    # (c) Cross-domain alignment (requires second forward pass)
    print("\n--- (c) Cross-domain subspace stability ---")
    sys.stdout.flush()
    kvs2 = collect_domain2_kvs(n_tokens=2048)
    all_rows.extend(compute_cross_domain(kvs1, kvs2, "K"))
    all_rows.extend(compute_cross_domain(kvs1, kvs2, "V"))

    # Save CSV
    fieldnames = ["layer_a", "layer_b", "head_a", "head_b", "kv_type",
                  "metric_type", "k_a", "k_b", "subspace_overlap"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {OUTPUT_CSV}")
    sys.stdout.flush()

    # Print summary
    print_summary(all_rows)

    print("\nDone.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

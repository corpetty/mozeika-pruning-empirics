#!/usr/bin/env python3
"""
Calibrate SubRotQ for Gemma4 26B using llama.cpp Python bindings.

Collects K vectors from WikiText-2 calibration set and computes per-layer,
per-head PCA basis for SubRotQ compression.

Output: gemma4_26b_subrotq_basis.npz with U, mean, scale for each layer/head
"""

import numpy as np
from llama_cpp import Llama
import os
from pathlib import Path

# Config
MODEL_PATH = "/usr/share/ollama/.ollama/models/blobs/sha256-7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df"
CALIB_TOKENS = 2048  # Calibration set size
SUBROTQ_RANK = 128   # Target rank k
N_BITS = 4           # Quantization bits

# WikiText-2 calibration text (first 2K tokens worth)
CALIB_TEXT = """
The following is a snippet from the WikiText-2 dataset used for language model calibration.

= Valkyria Chronicles III = 

 Senjō no Valkyria 3 : <unk> Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . <unk> the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " <unk> Raven " . 
 The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more <unk> for series newcomers . Character designer <unk> Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n . 
 It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 .

= = Gameplay = =

 As with previous <unk> Chronicles games , Valkyria Chronicles III is a tactical role @-@ playing game where players take control of a military unit and take part in missions against enemy forces . Stories are told through comic book @-@ like panels with animated character portraits , with characters speaking partially through voiced speech bubbles and partially through <unk> text . The player progresses through a series of linear missions , gradually unlocked as maps that can be freely <unk> through and replayed as they are unlocked . The route to each story location on the map varies depending on an individual player 's approach : when one option is selected , the other is sealed off to the player . Outside missions , the player characters rest in a camp , where units can be customized and character growth occurs . Alongside the main story missions are character @-@ specific sub missions relating to different squad members . After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty than those found in the rest of the game . There are also love simulation elements related to the game 's two main <unk> , although they take a very minor role .
"""

def collect_kv_cache(model, text, max_tokens=2048):
    """
    Run model on text and extract K cache from all layers/heads.
    
    Returns:
        dict: {(layer_idx, head_idx): {'K': np.array(seq_len, d_head)}}
    """
    print(f"Tokenizing calibration text...")
    tokens = model.tokenize(text.encode('utf-8'), add_bos=True)
    tokens = tokens[:max_tokens]
    print(f"Using {len(tokens)} tokens for calibration")
    
    # Run model to populate KV cache
    print("Running model to collect K vectors...")
    model.reset()  # Clear any existing cache
    
    # Process in one shot (llama.cpp will populate cache)
    _ = model(tokens, max_tokens=0)  # Just prefill, no generation
    
    # Extract K cache via llama.cpp internals
    # Note: llama-cpp-python doesn't expose KV cache directly
    # We need to use the C API via ctypes
    from llama_cpp import llama_cpp
    
    ctx = model._ctx.ctx
    n_layers = llama_cpp.llama_n_layer(model._model.model)
    
    print(f"Model has {n_layers} layers")
    
    # Get KV cache structure
    # This is tricky - llama.cpp doesn't expose this easily
    # We'll need to use low-level API or dump from CUDA memory
    
    # WORKAROUND: Since llama-cpp-python doesn't expose KV cache,
    # we'll need to use our PyTorch-based collection from the research repo
    print("\nWARNING: llama-cpp-python doesn't expose KV cache.")
    print("Falling back to PyTorch-based collection...")
    
    return None

def fit_pca_basis(K_vectors, k):
    """
    Compute PCA basis for K vectors.
    
    Args:
        K_vectors: np.array of shape (n_samples, d_head)
        k: target rank
        
    Returns:
        U: np.array (d_head, k) - PCA basis (top k components)
        mean: np.array (d_head,) - mean vector
        scale: np.array (k,) - std dev of projected coefficients
    """
    # Center the data
    mean = K_vectors.mean(axis=0)
    X_centered = K_vectors - mean
    
    # SVD
    U_full, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Take top k components
    U = Vt[:k].T  # (d_head, k)
    
    # Compute scale (std dev in subspace)
    z = X_centered @ U  # Project to subspace
    scale = z.std(axis=0)  # Per-dimension std dev
    
    return U, mean, scale

def main():
    print("=" * 60)
    print("SubRotQ Calibration for Gemma4 26B")
    print("=" * 60)
    
    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return 1
    
    print(f"\nModel: {MODEL_PATH}")
    print(f"Calibration tokens: {CALIB_TOKENS}")
    print(f"Target rank k: {SUBROTQ_RANK}")
    print(f"Quantization bits: {N_BITS}")
    
    # PROBLEM: llama-cpp-python doesn't expose KV cache
    # Need to use PyTorch approach instead
    
    print("\n" + "=" * 60)
    print("PIVOT: llama-cpp-python doesn't expose KV cache")
    print("Switching to PyTorch-based calibration approach...")
    print("=" * 60)
    print("\nThis requires:")
    print("1. Load Gemma4 26B in transformers (Hugging Face)")
    print("2. Use existing collect_kvs_for_basis() from exp24")
    print("3. Save basis to .npz file")
    print("4. Convert to llama.cpp format")
    
    print("\nNext: Download Gemma4 26B from Hugging Face and adapt exp24 code.")
    
    return 0

if __name__ == "__main__":
    exit(main())

"""
mmd.py
======
Maximum Mean Discrepancy (MMD) computation for layer-wise domain shift
analysis in DSA-LoRA: Domain-Shift-Aware Parameter-Efficient Fine-Tuning
for Malaria Detection in Low-Resource Clinical Settings.

MMD measures the distance between two probability distributions in a
reproducing kernel Hilbert space using an RBF (Gaussian) kernel.

Interpretation:
    MMD ≈ 0   → distributions are similar (low domain shift at this layer)
    MMD >> 0  → distributions are different (high domain shift at this layer)

DSA-LoRA uses per-layer MMD scores to select which transformer blocks
to apply LoRA adapters to, concentrating adaptation capacity on layers
most affected by the HIC → LMIC domain shift.
"""

import numpy as np
import torch


# ── KERNEL FUNCTION ───────────────────────────────────────────────────────────

def rbf_kernel(X, Y, sigma=None):
    """
    Compute the RBF (Gaussian) kernel matrix between X and Y.

    K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

    Uses the median heuristic for bandwidth selection if sigma is None.
    The median heuristic sets sigma to the median pairwise distance,
    which is a standard data-driven choice for kernel methods.

    Args:
        X:     array of shape [n, d]
        Y:     array of shape [m, d]
        sigma: kernel bandwidth (None = median heuristic)

    Returns:
        K: kernel matrix of shape [n, m]
    """
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    XY = X @ Y.T
    dist_sq = XX + YY.T - 2 * XY

    if sigma is None:
        sigma = np.median(np.sqrt(np.abs(dist_sq) + 1e-8))
        sigma = max(sigma, 1e-2)  # avoid near-zero bandwidth

    K = np.exp(-dist_sq / (2 * sigma ** 2))
    return K


# ── MMD COMPUTATION ───────────────────────────────────────────────────────────

def compute_mmd(X, Y, n_subsample=200, seed=42):
    """
    Compute the unbiased MMD^2 estimate between distributions X and Y.

    MMD^2(X, Y) = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    where k is the RBF kernel and expectations are over i.i.d. samples.

    Subsampling is used for computational efficiency since MMD is O(n^2).
    Features are L2-normalised before kernel computation for stability.

    Args:
        X:           source domain features [n, d] (e.g. NIH)
        Y:           target domain features [m, d] (e.g. Lacuna)
        n_subsample: number of samples per domain for MMD estimate
        seed:        random seed for subsampling reproducibility

    Returns:
        mmd_score: scalar MMD value (non-negative)
    """
    np.random.seed(seed)

    n = min(n_subsample, len(X))
    m = min(n_subsample, len(Y))
    idx_x = np.random.choice(len(X), n, replace=False)
    idx_y = np.random.choice(len(Y), m, replace=False)

    X_s = X[idx_x].astype(np.float32)
    Y_s = Y[idx_y].astype(np.float32)

    # L2 normalise for numerical stability
    X_s = X_s / (np.linalg.norm(X_s, axis=1, keepdims=True) + 1e-8)
    Y_s = Y_s / (np.linalg.norm(Y_s, axis=1, keepdims=True) + 1e-8)

    K_XX = rbf_kernel(X_s, X_s)
    K_YY = rbf_kernel(Y_s, Y_s)
    K_XY = rbf_kernel(X_s, Y_s)

    # Unbiased estimate: exclude diagonal terms
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    mmd = (K_XX.sum() / (n * (n - 1)) +
           K_YY.sum() / (m * (m - 1)) -
           2 * K_XY.mean())

    return float(max(mmd, 0.0))  # MMD^2 >= 0 by definition


def compute_mmd_stable(X, Y, n_runs=5, n_subsample=200):
    """
    Compute a stable MMD estimate by averaging over multiple runs.

    Reduces variance from random subsampling by running MMD
    computation n_runs times with different random seeds.

    Args:
        X:           source domain features [n, d]
        Y:           target domain features [m, d]
        n_runs:      number of independent MMD estimates to average
        n_subsample: samples per domain per run

    Returns:
        mean_mmd: mean MMD score across runs
        std_mmd:  standard deviation across runs
    """
    scores = [
        compute_mmd(X, Y, n_subsample=n_subsample, seed=i)
        for i in range(n_runs)
    ]
    return float(np.mean(scores)), float(np.std(scores))


# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────

def extract_layer_features(model, loader, device, n_layers=12):
    """
    Extract CLS token features from all ViT transformer blocks
    using forward hooks.

    The CLS token at position 0 of each block's output serves as the
    global image representation at that layer. Comparing CLS token
    distributions between source and target domains gives a per-layer
    measure of how much the domain shift affects each block.

    ViT-B/16 architecture:
        - 12 transformer blocks (blocks[0] to blocks[11])
        - Hidden dimension: 768
        - CLS token shape per block: [batch_size, 768]

    Args:
        model:    ViT-B/16 model (Stage 1 source domain model)
        loader:   DataLoader for the probe dataset
        device:   torch device
        n_layers: number of transformer blocks (12 for ViT-B/16)

    Returns:
        features: dict {layer_idx: np.array of shape [N, 768]}
                  where N is the total number of probe images
    """
    features = {i: [] for i in range(n_layers)}
    hooks    = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output: [batch, seq_len, hidden_dim]
            # CLS token is at sequence position 0
            cls_token = output[:, 0, :].detach().cpu()
            features[layer_idx].append(cls_token)
        return hook

    # Register hooks on each transformer block
    for i in range(n_layers):
        h = model.blocks[i].register_forward_hook(make_hook(i))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)
            _      = model(images)

    # Remove all hooks after extraction
    for h in hooks:
        h.remove()

    # Concatenate batches into single feature matrix per layer
    for i in range(n_layers):
        features[i] = torch.cat(features[i], dim=0).numpy()

    return features


# ── LAYER SELECTION ───────────────────────────────────────────────────────────

def select_lora_layers(mmd_scores, strategy='median'):
    """
    Select which transformer layers to apply LoRA to based on MMD scores.

    Strategy 'median': apply LoRA to layers with MMD >= median MMD score.
    This selects approximately half the layers — those with above-average
    domain shift — while freezing the rest.

    Args:
        mmd_scores: list of MMD scores, one per layer
        strategy:   selection strategy ('median' supported)

    Returns:
        lora_layers:   list of layer indices to apply LoRA to
        frozen_layers: list of layer indices to freeze
        threshold:     MMD threshold used for selection
    """
    scores    = np.array(mmd_scores)
    threshold = float(np.median(scores))

    lora_layers   = [i for i, s in enumerate(mmd_scores) if s >= threshold]
    frozen_layers = [i for i, s in enumerate(mmd_scores) if s < threshold]

    print(f"MMD threshold ({strategy}): {threshold:.6f}")
    print(f"LoRA layers  ({len(lora_layers)}/12): {lora_layers}")
    print(f"Frozen layers ({len(frozen_layers)}/12): {frozen_layers}")

    return lora_layers, frozen_layers, threshold

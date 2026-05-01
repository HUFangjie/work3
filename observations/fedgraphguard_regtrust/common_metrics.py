"""Shared utilities for FedGraphGuard (aka ReG-Trust) observation experiments.

This module is intentionally self-contained and does not depend on the runtime
of ``defense_fedgraphguard.py`` so observations can be reproduced independently.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def _to_numpy_logits(logits_list: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
    """Convert inputs to float64 numpy arrays and validate consistent shapes."""
    if not logits_list:
        raise ValueError("logits_list must not be empty.")

    shapes = []
    converted = []
    for logits in logits_list:
        arr = np.asarray(logits, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Each logits tensor must have shape (N_pub, C); got {arr.shape}.")
        shapes.append(arr.shape)
        converted.append(arr)

    if len(set(shapes)) != 1:
        raise ValueError(f"All client logits must have the same shape, got {set(shapes)}.")
    return converted


def topk_indices_from_logits(logits: np.ndarray, kappa: int = 5) -> np.ndarray:
    """Return top-k class index sets for each public sample.

    Args:
        logits: shape (N_pub, C)
        kappa: top-k parameter used in Jaccard similarity.
    """
    if kappa <= 0:
        raise ValueError("kappa must be positive.")
    n_pub, n_cls = logits.shape
    kappa = min(kappa, n_cls)

    # argpartition is O(C) and enough for set-based overlap.
    part = np.argpartition(logits, kth=n_cls - kappa, axis=1)
    return part[:, -kappa:]


def jaccard_similarity_matrix(logits_list: Sequence[np.ndarray], kappa: int = 5) -> np.ndarray:
    """Compute client-client Jaccard similarity matrix from logits list.

    Similarity is averaged across the public dataset:
      S[i,j] = mean_n |topk_i(n) ∩ topk_j(n)| / |topk_i(n) ∪ topk_j(n)|
    """
    clean_logits = _to_numpy_logits(logits_list)
    k_clients = len(clean_logits)

    topk = [topk_indices_from_logits(z, kappa=kappa) for z in clean_logits]
    s = np.eye(k_clients, dtype=np.float64)

    for i in range(k_clients):
        set_i = topk[i]
        for j in range(i + 1, k_clients):
            set_j = topk[j]
            # N_pub is usually small enough to do direct per-sample set ops.
            overlaps = []
            for row_i, row_j in zip(set_i, set_j):
                a = set(row_i.tolist())
                b = set(row_j.tolist())
                union = len(a | b)
                inter = len(a & b)
                overlaps.append(inter / union if union > 0 else 0.0)
            sim = float(np.mean(overlaps))
            s[i, j] = sim
            s[j, i] = sim
    return s


def singular_spectrum_metrics(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return singular values, cumulative energy ratio, and effective rank."""
    mat = np.asarray(s, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Input matrix must be square.")

    sing = np.linalg.svd(mat, compute_uv=False)
    energy = np.square(sing)
    cum_energy = np.cumsum(energy) / np.sum(energy)

    probs = sing / np.sum(sing)
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps))
    effective_rank = float(np.exp(entropy))
    return sing, cum_energy, effective_rank


def row_energy(e: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute per-row L2 energy for an error matrix."""
    mat = np.asarray(e, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("E must be 2D.")
    vals = np.linalg.norm(mat, ord=2, axis=1)
    if normalize:
        denom = np.sum(vals)
        if denom > 0:
            vals = vals / denom
    return vals


def byzantine_energy_concentration(e: np.ndarray, byzantine_ids: Iterable[int]) -> float:
    """Energy share captured by known Byzantine rows."""
    mat = np.asarray(e, dtype=np.float64)
    re = row_energy(mat, normalize=False)
    denom = np.sum(re)
    if denom <= 0:
        return 0.0
    idx = np.array(sorted(set(int(i) for i in byzantine_ids)), dtype=np.int64)
    return float(np.sum(re[idx]) / denom)


def sparsity_ratio(e: np.ndarray, threshold: float = 1e-3) -> float:
    """Compute ||E||_0 / K^2 using thresholding for numerical stability."""
    mat = np.asarray(e, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("E must be 2D.")
    non_zero = np.sum(np.abs(mat) > threshold)
    return float(non_zero / mat.size)


def apply_attack(
    logits: np.ndarray,
    attack_type: str,
    rng: Optional[np.random.Generator] = None,
    target_class: int = 0,
    sigma: float = 1.0,
    benign_mean: Optional[np.ndarray] = None,
    benign_std: Optional[np.ndarray] = None,
    z_scale: float = 1.5,
) -> np.ndarray:
    """Apply attacks used by Observation-2 design.

    Supported: gaussian, label_flip, targeted, alie.
    """
    z = np.asarray(logits, dtype=np.float64)
    rng = rng or np.random.default_rng()
    attack = attack_type.lower().strip()

    if attack == "gaussian":
        return z + rng.normal(loc=0.0, scale=sigma, size=z.shape)

    if attack == "label_flip":
        adv = z.copy()
        top = np.argmax(z, axis=1)
        bot = np.argmin(z, axis=1)
        rows = np.arange(z.shape[0])
        adv[rows, top], adv[rows, bot] = z[rows, bot], z[rows, top]
        return adv

    if attack == "targeted":
        adv = np.zeros_like(z)
        adv[:, target_class] = 10.0
        return adv

    if attack == "alie":
        if benign_mean is None or benign_std is None:
            raise ValueError("ALIE attack requires benign_mean and benign_std.")
        return np.asarray(benign_mean, dtype=np.float64) + z_scale * np.asarray(benign_std, dtype=np.float64)

    raise ValueError(f"Unsupported attack_type: {attack_type}")


def low_rank_rpca(
    s: np.ndarray,
    lam: Optional[float] = None,
    rho: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inexact ADMM solver for RPCA: min ||L||_* + lam||E||_1 s.t. L + E = S."""
    mat = np.asarray(s, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("S must be a square matrix.")

    k = mat.shape[0]
    lam = lam if lam is not None else 1.0 / np.sqrt(k)

    l = np.zeros_like(mat)
    e = np.zeros_like(mat)
    y = np.zeros_like(mat)

    inv_rho = 1.0 / rho

    for _ in range(max_iter):
        # Update L via singular value thresholding.
        u, sig, vt = np.linalg.svd(mat - e + inv_rho * y, full_matrices=False)
        sig_shrink = np.maximum(sig - inv_rho, 0.0)
        l = (u * sig_shrink) @ vt

        # Update E via soft thresholding.
        residual = mat - l + inv_rho * y
        e = np.sign(residual) * np.maximum(np.abs(residual) - lam * inv_rho, 0.0)

        # Dual update.
        primal = mat - l - e
        y = y + rho * primal

        if np.linalg.norm(primal, ord="fro") / (np.linalg.norm(mat, ord="fro") + 1e-12) < tol:
            break

    return l, e


def spectral_gap(w: np.ndarray) -> float:
    """Fiedler value λ2 of normalized Laplacian."""
    mat = np.asarray(w, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("W must be square.")

    deg = np.sum(mat, axis=1)
    inv_sqrt = np.where(deg > 1e-12, 1.0 / np.sqrt(deg), 0.0)
    d_is = np.diag(inv_sqrt)
    lap = np.eye(mat.shape[0]) - d_is @ mat @ d_is

    eig = np.linalg.eigvalsh(lap)
    eig = np.sort(np.real(eig))
    if len(eig) < 2:
        return 0.0
    return float(eig[1])


def personalized_pagerank(
    w: np.ndarray,
    beta: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 200,
    seed: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute PPR trust scores using power iteration on row-normalized graph."""
    mat = np.asarray(w, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("W must be square.")

    k = mat.shape[0]
    row_sum = np.sum(mat, axis=1, keepdims=True)
    row_sum = np.where(row_sum > 1e-12, row_sum, 1.0)
    w_norm = mat / row_sum

    if seed is None:
        seed_vec = np.ones(k, dtype=np.float64) / k
    else:
        seed_vec = np.asarray(seed, dtype=np.float64)
        seed_vec = seed_vec / np.sum(seed_vec)

    v = seed_vec.copy()
    for _ in range(max_iter):
        v_new = (1.0 - beta) * seed_vec + beta * (w_norm.T @ v)
        if np.linalg.norm(v_new - v, ord=2) < tol:
            v = v_new
            break
        v = v_new

    return v / np.sum(v)


def connectivity_stats(graph: np.ndarray, benign_ids: Sequence[int], byzantine_ids: Sequence[int]) -> dict:
    """Compute W_benign, W_cross and connectivity gap."""
    w = np.asarray(graph, dtype=np.float64)
    b = np.array(benign_ids, dtype=np.int64)
    z = np.array(byzantine_ids, dtype=np.int64)

    w_benign = float(np.mean(w[np.ix_(b, b)])) if len(b) > 0 else 0.0
    w_cross = float(np.mean(w[np.ix_(z, b)])) if len(b) > 0 and len(z) > 0 else 0.0
    gap = w_benign - w_cross

    return {"W_benign": w_benign, "W_cross": w_cross, "connectivity_gap": gap}

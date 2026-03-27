"""
Evaluation metrics for classification and calibration.

Includes:
  - accuracy
  - loss (via evaluate_model)
  - confidence-on-error
  - Expected Calibration Error (ECE)
  - average confidence
  - KS statistic between confidence distributions of correct vs. error

NEW:
  - evaluate_with_calibration_and_raw(): returns metrics + per-sample raw data
    for reliability diagrams (confidence, pred, target, correctness) and
    per-bin stats (bin_acc, bin_conf, bin_counts, bin_edges).
"""

from typing import Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


# ----------------------------------------------------------------------
# Basic accuracy
# ----------------------------------------------------------------------
def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / float(total + 1e-12)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            if y.ndim > 1:
                y = y.view(-1)
            logits = model(x)
            loss = criterion(logits, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / float(total_samples + 1e-12)
    avg_acc = total_correct / float(total_samples + 1e-12)
    return {"loss": avg_loss, "accuracy": avg_acc}


# ----------------------------------------------------------------------
# Confidence & calibration helpers
# ----------------------------------------------------------------------
def compute_confidence_and_predictions(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=-1)
    conf, preds = torch.max(probs, dim=-1)
    return conf, preds


def compute_confidence_on_error(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    confidences, preds = compute_confidence_and_predictions(logits)
    errors = preds.ne(targets)
    if errors.sum() == 0:
        return 0.0
    return float(confidences[errors].mean().item())


def compute_avg_confidence(logits: torch.Tensor) -> float:
    confidences, _ = compute_confidence_and_predictions(logits)
    if confidences.numel() == 0:
        return 0.0
    return float(confidences.mean().item())


def compute_calibration_bins_from_raw(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    num_bins: int = 15,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-bin confidence/accuracy/counts from raw per-sample values.

    Args:
        confidences: [N] float in [0,1]
        correctness: [N] float/bool in {0,1}
    """
    device = confidences.device
    conf = confidences.view(-1)
    corr = correctness.view(-1).float()

    bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=device)
    bin_confidence = torch.zeros(num_bins, device=device)
    bin_accuracy = torch.zeros(num_bins, device=device)
    bin_counts = torch.zeros(num_bins, device=device)

    for b in range(num_bins):
        lower = bin_edges[b]
        upper = bin_edges[b + 1]
        if b == num_bins - 1:
            in_bin = (conf >= lower) & (conf <= upper)
        else:
            in_bin = (conf >= lower) & (conf < upper)

        count = int(in_bin.sum().item())
        if count > 0:
            bin_counts[b] = float(count)
            bin_confidence[b] = conf[in_bin].mean()
            bin_accuracy[b] = corr[in_bin].mean()
        else:
            bin_counts[b] = 0.0
            bin_confidence[b] = 0.0
            bin_accuracy[b] = 0.0

    return {
        "bin_edges": bin_edges,
        "bin_confidence": bin_confidence,
        "bin_accuracy": bin_accuracy,
        "bin_counts": bin_counts,
    }


def compute_ece_from_bins(
    bin_confidence: torch.Tensor,
    bin_accuracy: torch.Tensor,
    bin_counts: torch.Tensor,
) -> float:
    N = float(bin_counts.sum().item())
    if N <= 0:
        return 0.0
    abs_diff = torch.abs(bin_accuracy - bin_confidence)
    ece = (bin_counts * abs_diff).sum() / N
    return float(ece.item())


def compute_ks_confidence_correct_vs_error_from_raw(
    confidences: np.ndarray,
    correctness: np.ndarray,
) -> float:
    """
    KS between confidence distributions of correct vs error, using raw numpy.
    """
    correct_conf = confidences[correctness.astype(bool)]
    error_conf = confidences[~correctness.astype(bool)]
    if correct_conf.size == 0 or error_conf.size == 0:
        return 0.0

    correct_conf = np.sort(correct_conf)
    error_conf = np.sort(error_conf)
    all_vals = np.concatenate([correct_conf, error_conf])

    cdf_correct = np.searchsorted(correct_conf, all_vals, side="right") / float(len(correct_conf))
    cdf_error = np.searchsorted(error_conf, all_vals, side="right") / float(len(error_conf))
    return float(np.max(np.abs(cdf_correct - cdf_error)))


# ----------------------------------------------------------------------
# NEW: one-pass eval that also returns raw reliability data
# ----------------------------------------------------------------------
def evaluate_with_calibration_and_raw(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_bins: int = 15,
    criterion: Optional[nn.Module] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Returns:
      metrics: loss/acc/auc/ece/coe/avg_conf/ks
      raw: dict with:
        - confidence: np.ndarray [N]
        - pred: np.ndarray [N]
        - target: np.ndarray [N]
        - correct: np.ndarray [N]  (0/1)
        - bin_edges, bin_confidence, bin_accuracy, bin_counts: np.ndarray
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    conf_list = []
    probs_list = []
    pred_list = []
    tgt_list = []
    corr_list = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            if y.ndim > 1:
                y = y.view(-1)

            logits = model(x)
            loss = criterion(logits, y)

            probs = F.softmax(logits, dim=-1)
            conf, preds = torch.max(probs, dim=-1)

            correct = preds.eq(y).long()

            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_correct += int(correct.sum().item())
            total_samples += bs

            conf_list.append(conf.detach().cpu())
            probs_list.append(probs.detach().cpu())
            pred_list.append(preds.detach().cpu())
            tgt_list.append(y.detach().cpu())
            corr_list.append(correct.detach().cpu())

    if total_samples == 0:
        metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "auc": float("nan"),
            "ece": 0.0,
            "confidence_on_error": 0.0,
            "coe": 0.0,
            "avg_confidence": 0.0,
            "ks_confidence": 0.0,
        }
        raw = {
            "confidence": np.zeros((0,), dtype=np.float32),
            "pred": np.zeros((0,), dtype=np.int64),
            "target": np.zeros((0,), dtype=np.int64),
            "correct": np.zeros((0,), dtype=np.int64),
            "bin_edges": np.linspace(0, 1, num_bins + 1).astype(np.float32),
            "bin_confidence": np.zeros((num_bins,), dtype=np.float32),
            "bin_accuracy": np.zeros((num_bins,), dtype=np.float32),
            "bin_counts": np.zeros((num_bins,), dtype=np.float32),
        }
        return metrics, raw

    conf_all = torch.cat(conf_list, dim=0).float()         # [N]
    pred_all = torch.cat(pred_list, dim=0).long()          # [N]
    tgt_all = torch.cat(tgt_list, dim=0).long()            # [N]
    corr_all = torch.cat(corr_list, dim=0).float()         # [N]
    probs_all = torch.cat(probs_list, dim=0).float().numpy()  # [N, C]
    # AUROC (binary or multi-class OVR macro). If undefined (single-class), set to NaN.
    try:
        if probs_all.shape[1] == 2:
            auc = float(roc_auc_score(tgt_all.numpy().astype(int), probs_all[:, 1]))
        else:
            auc = float(roc_auc_score(tgt_all.numpy().astype(int), probs_all, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")

    bins = compute_calibration_bins_from_raw(conf_all.to(device), corr_all.to(device), num_bins=num_bins)
    ece = compute_ece_from_bins(bins["bin_confidence"], bins["bin_accuracy"], bins["bin_counts"])

    # confidence-on-error
    err_mask = (corr_all < 0.5)
    coe = float(conf_all[err_mask].mean().item()) if int(err_mask.sum().item()) > 0 else 0.0
    avg_conf = float(conf_all.mean().item())

    # KS on CPU numpy
    conf_np = conf_all.numpy().astype(np.float32)
    corr_np = corr_all.numpy().astype(np.int64)
    ks = compute_ks_confidence_correct_vs_error_from_raw(conf_np, corr_np)

    metrics = {
        "loss": float(total_loss / float(total_samples)),
        "accuracy": float(total_correct / float(total_samples)),
        "auc": float(auc),
        "ece": float(ece),
        "confidence_on_error": float(coe),
        "coe": float(coe),
        "avg_confidence": float(avg_conf),
        "ks_confidence": float(ks),
    }

    raw = {
        "confidence": conf_np,
        "pred": pred_all.numpy().astype(np.int64),
        "target": tgt_all.numpy().astype(np.int64),
        "correct": corr_np,
        "bin_edges": bins["bin_edges"].detach().cpu().numpy().astype(np.float32),
        "bin_confidence": bins["bin_confidence"].detach().cpu().numpy().astype(np.float32),
        "bin_accuracy": bins["bin_accuracy"].detach().cpu().numpy().astype(np.float32),
        "bin_counts": bins["bin_counts"].detach().cpu().numpy().astype(np.float32),
    }
    return metrics, raw


# ----------------------------------------------------------------------
# Backward-compatible wrapper
# ----------------------------------------------------------------------
def evaluate_with_calibration(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_bins: int = 15,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    metrics, _ = evaluate_with_calibration_and_raw(
        model=model,
        dataloader=dataloader,
        device=device,
        num_bins=num_bins,
        criterion=criterion,
    )
    return metrics

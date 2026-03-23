from __future__ import annotations

from typing import Any

import numpy as np


def logits_to_unstable_probs(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    probs = np.empty_like(logits, dtype=np.float32)
    positive_mask = logits >= 0

    probs[positive_mask] = 1.0 / (1.0 + np.exp(-logits[positive_mask]))
    exp_logits = np.exp(logits[~positive_mask])
    probs[~positive_mask] = exp_logits / (1.0 + exp_logits)
    return probs


def binary_logloss(labels: np.ndarray, unstable_probs: np.ndarray, eps: float = 1e-6) -> float:
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    unstable_probs = np.clip(np.asarray(unstable_probs, dtype=np.float32).reshape(-1), eps, 1 - eps)
    stable_probs = 1.0 - unstable_probs
    pred = np.stack([stable_probs, unstable_probs], axis=1)
    true = np.stack([1.0 - labels, labels], axis=1)
    return float(np.mean(-np.sum(true * np.log(pred), axis=1)))


def binary_accuracy(labels: np.ndarray, logits: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    preds = (logits_to_unstable_probs(logits) >= 0.5).astype(np.int64)
    return float(np.mean(preds == labels))


def binary_auc(labels: np.ndarray, logits: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = logits_to_unstable_probs(logits).reshape(-1)
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_rank_sum = ranks[positives].sum()
    auc = (pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def summarize_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    unstable_probs = logits_to_unstable_probs(logits)
    return {
        "logloss": binary_logloss(labels, unstable_probs),
        "accuracy": binary_accuracy(labels, logits),
        "auc": binary_auc(labels, logits),
    }

"""Classification and calibration metrics.

Conventions
-----------
* Labels are 0..K-1 (class index = ground-truth code - 1).
* AA = mean per-class recall over classes *present in y_true*; absent classes are excluded and
  listed (a class with no test pixels has undefined recall).
* Per-class precision/recall/F1 use ``zero_division=0``: a class never predicted gets precision 0,
  so it is penalised rather than silently dropped. Macro-F1 averages over classes present in
  y_true (same set as AA).
* ECE: 15 equal-width bins on the max-softmax confidence; ECE = sum_b (n_b / N) |acc_b - conf_b|.
  Empty bins contribute nothing.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

N_BINS = 15


def ece_score(probs: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> dict:
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    acc, cf, cnt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        cnt.append(int(m.sum()))
        acc.append(float(correct[m].mean()) if m.any() else np.nan)
        cf.append(float(conf[m].mean()) if m.any() else np.nan)
    return {"edges": edges.tolist(), "acc": acc, "conf": cf, "count": cnt}


def evaluate(y: np.ndarray, probs: np.ndarray, n_classes: int) -> dict:
    pred = probs.argmax(1)
    labels = np.arange(n_classes)
    present = np.unique(y)
    cm = confusion_matrix(y, pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {
        "oa": float((pred == y).mean()),
        "aa": float(r[present].mean()),
        "kappa": float(cohen_kappa_score(y, pred, labels=labels)),
        "macro_f1": float(f[present].mean()),
        "ece": ece_score(probs, y),
        "per_class_precision": p.tolist(), "per_class_recall": r.tolist(),
        "per_class_f1": f.tolist(), "support": s.tolist(),
        "classes_absent_from_test": [int(c) for c in labels if c not in present],
        "confusion": cm.tolist(),
        "reliability": reliability_bins(probs, y),
        "n_test": int(len(y)),
    }

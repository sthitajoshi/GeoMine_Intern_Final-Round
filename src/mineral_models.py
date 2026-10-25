"""Synthetic-trained mineral classifier operating on continuum-removed absorption vectors."""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def features(cr: np.ndarray) -> np.ndarray:
    """1 - CR over the SWIR window: absorption-depth vector (0 on the continuum).
    Continuum removal makes it invariant to multiplicative illumination."""
    f = 1.0 - cr
    return np.nan_to_num(f, nan=0.0)


def train_mlp(F: np.ndarray, y: np.ndarray, seed: int = 0):
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300, early_stopping=True,
                        validation_fraction=0.15, n_iter_no_change=15, random_state=seed)
    return make_pipeline(StandardScaler(), clf).fit(F, y)


def predict_with_reject(model, F: np.ndarray, unknown_idx: int, reject_prob: float):
    """Argmax class; "unknown" if argmax is the unknown class or max probability < reject_prob."""
    P = model.predict_proba(F)
    lab = P.argmax(1)
    conf = P.max(1)
    lab = np.where((lab == unknown_idx) | (conf < reject_prob), unknown_idx, lab)
    return lab, conf, P

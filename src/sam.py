"""Spectral Angle Mapper and a simple feature-fit (least-squares band-depth scaling) check."""
from __future__ import annotations

import numpy as np


def spectral_angle(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """X (N, b), R (M, b) -> angles (N, M) in radians. Scale-invariant (illumination-robust)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    return np.arccos(np.clip(Xn @ Rn.T, -1, 1))


def sam_classify(X, refs: dict, threshold: float):
    """refs: mineral -> (variants, b). Pixel gets the mineral of its smallest angle over all
    variants, or -1 (unknown / none) if that angle exceeds ``threshold`` radians.
    Returns labels, best angle, second-best angle among *other* minerals, names."""
    names = list(refs)
    ang = np.stack([spectral_angle(X, refs[m]).min(1) for m in names], 1)   # (N, M)
    order = np.argsort(ang, 1)
    best = ang[np.arange(len(X)), order[:, 0]]
    second = ang[np.arange(len(X)), order[:, 1]]
    lab = np.where(best <= threshold, order[:, 0], -1)
    return lab, best, second, names, ang


def feature_fit(cr_pix: np.ndarray, cr_ref: np.ndarray):
    """Fit (1 - cr_pix) ~ a * (1 - cr_ref) by least squares over the window.
    Returns scale a (relative band-depth/abundance proxy), RMS residual and R^2."""
    yp = 1 - cr_pix
    yr = 1 - cr_ref[None, :]
    a = np.nansum(yp * yr, 1) / (np.nansum(yr * yr) + 1e-12)
    res = yp - a[:, None] * yr
    rms = np.sqrt(np.nanmean(res ** 2, 1))
    ss_tot = np.nansum((yp - np.nanmean(yp, 1, keepdims=True)) ** 2, 1)
    r2 = 1 - np.nansum(res ** 2, 1) / (ss_tot + 1e-12)
    return a, rms, r2

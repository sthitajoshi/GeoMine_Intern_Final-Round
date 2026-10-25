"""Convex-hull continuum removal and diagnostic absorption-feature parameters.

CR(lambda) = R(lambda) / C(lambda), where C is the upper convex hull of the spectrum over the
window (default 2.0-2.5 um, bands sorted by wavelength). CR = 1 on the hull, < 1 in absorptions.
Band depth D = 1 - CR at the feature minimum; position = wavelength of that minimum.
"""
from __future__ import annotations

import numpy as np

# Diagnostic SWIR features (centre nm, search half-width nm). Sources: Clark et al. 1990/2003,
# Hunt 1977; positions are typical for the USGS Cuprite spectra.
FEATURES = {
    "alunite": (2165, 25),        # Al-OH + S-O combination, with 2.32 shoulder
    "kaolinite": (2165, 25),      # doublet 2.165 / 2.205 - shares the 2.16 position with alunite
    "muscovite": (2205, 20),      # Al-OH single feature at ~2.20-2.21, plus 2.35
    "buddingtonite": (2120, 25),  # NH4 feature at ~2.12 (plus 2.02)
    "calcite": (2340, 25),        # CO3 overtone at ~2.335-2.345
    "chalcedony": (2255, 35),     # broad Si-OH ~2.26
}


def upper_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Upper convex hull of (x, y) evaluated at x (x must be strictly increasing)."""
    hull = []
    for i in range(len(x)):
        while len(hull) >= 2:
            i0, i1 = hull[-2], hull[-1]
            # remove i1 if it lies on or below the segment i0 -> i
            if (y[i1] - y[i0]) * (x[i] - x[i0]) <= (y[i] - y[i0]) * (x[i1] - x[i0]):
                hull.pop()
            else:
                break
        hull.append(i)
    return np.interp(x, x[hull], y[hull])


def continuum_remove(spectra: np.ndarray, centres: np.ndarray, lo=2000.0, hi=2500.0,
                     band_ok: np.ndarray | None = None):
    """spectra: (N, B). Returns (CR (N, b) for the selected bands sorted by wavelength, band index)."""
    sel = (centres >= lo) & (centres <= hi)
    if band_ok is not None:
        sel &= band_ok
    idx = np.flatnonzero(sel)
    idx = idx[np.argsort(centres[idx])]
    x = centres[idx]
    keep = np.concatenate([[True], np.diff(x) > 0])  # drop duplicate centres at detector joins
    idx, x = idx[keep], x[keep]
    S = np.asarray(spectra, np.float64)[:, idx]
    out = np.ones_like(S)
    for n in range(S.shape[0]):
        y = S[n]
        if not np.all(np.isfinite(y)) or y.max() <= 0:
            out[n] = np.nan
            continue
        c = upper_hull(x, y)
        out[n] = np.where(c > 0, y / c, np.nan)
    return np.clip(out, 0, 1.0), idx


def feature_params(cr: np.ndarray, x: np.ndarray, centre: float, half: float):
    """Depth (1 - min CR) and position (nm) of the minimum within centre +- half."""
    m = (x >= centre - half) & (x <= centre + half)
    sub = cr[:, m]
    k = np.nanargmin(np.where(np.isfinite(sub), sub, 9), axis=1)
    depth = 1 - sub[np.arange(len(sub)), k]
    pos = x[m][k]
    return depth, pos

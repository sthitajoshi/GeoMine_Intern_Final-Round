"""Band importance, <=15-band selection and Sentinel-2 band simulation.

Importance = permutation importance of a fitted classifier on the *validation* pixels (never
test): shuffle one band across validation pixels, measure the drop in OA, average over repeats.
Neighbouring AVIRIS bands are strongly correlated, so a single permuted band can be partly
"covered" by its neighbours; we therefore also report importance of contiguous 5-band windows
and select bands greedily with a minimum spacing so the 15 picks spread across features.
"""
from __future__ import annotations

import numpy as np

# Sentinel-2A MSI bands: (name, central nm, bandwidth nm), ESA S2 User Handbook / SRF document.
S2_BANDS = [("B1", 442.7, 21), ("B2", 492.4, 66), ("B3", 559.8, 36), ("B4", 664.6, 31),
            ("B5", 704.1, 15), ("B6", 740.5, 15), ("B7", 782.8, 20), ("B8", 832.8, 106),
            ("B8A", 864.7, 21), ("B9", 945.1, 20), ("B10", 1373.5, 31), ("B11", 1613.7, 91),
            ("B12", 2202.4, 175)]


def permutation_importance(predict, X_val, y_val, n_repeats=3, seed=0, window=1):
    """predict: callable (N, B) -> labels. Returns (n_bands,) mean OA drop.
    window > 1 permutes ``window`` contiguous bands together and assigns the drop to the centre."""
    rng = np.random.default_rng(seed)
    base = (predict(X_val) == y_val).mean()
    nb = X_val.shape[1]
    imp = np.zeros(nb)
    half = window // 2
    for b in range(nb):
        cols = list(range(max(0, b - half), min(nb, b + half + 1)))
        drops = []
        for _ in range(n_repeats):
            Xp = X_val.copy()
            perm = rng.permutation(len(X_val))
            Xp[:, cols] = X_val[perm][:, cols]
            drops.append(base - (predict(Xp) == y_val).mean())
        imp[b] = np.mean(drops)
    return imp, float(base)


def select_bands(importance, centres, k=15, min_sep_nm=30.0):
    """Greedy: take bands by decreasing importance, skipping ones within ``min_sep_nm`` of a pick."""
    order = np.argsort(-importance)
    picks = []
    for b in order:
        if all(abs(centres[b] - centres[p]) >= min_sep_nm for p in picks):
            picks.append(int(b))
        if len(picks) == k:
            break
    return sorted(picks)


def simulate_s2(X: np.ndarray, centres: np.ndarray):
    """Average the AVIRIS bands whose centres lie within each S2 band's [c - bw/2, c + bw/2].

    Returns (X_s2, used_band_names, report). S2 bands with no AVIRIS centre inside the range
    (e.g. B10 cirrus at 1.375 um, removed with the water-vapour bands) are reported, not invented.
    """
    cols, names, report = [], [], []
    for name, c, bw in S2_BANDS:
        m = (centres >= c - bw / 2) & (centres <= c + bw / 2)
        report.append({"band": name, "centre_nm": c, "bandwidth_nm": bw, "n_aviris_bands": int(m.sum()),
                       "aviris_idx": np.flatnonzero(m).tolist()})
        if m.any():
            cols.append(X[..., m].mean(-1))
            names.append(name)
    return np.stack(cols, -1), names, report


def forward_select(X_tr, y_tr, X_val, y_val, k=15):
    """Greedy forward selection: at each step add the band that most improves validation OA of a
    shrinkage LDA (fast, closed-form). Handles band correlation directly, unlike per-band
    permutation importance. Returns (picks, validation OA after each step)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    picks, trace = [], []
    remaining = list(range(X_tr.shape[1]))
    for _ in range(k):
        best_b, best_s = None, -1.0
        for b in remaining:
            cols = picks + [b]
            m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X_tr[:, cols], y_tr)
            s = (m.predict(X_val[:, cols]) == y_val).mean()
            if s > best_s:
                best_b, best_s = b, s
        picks.append(best_b)
        remaining.remove(best_b)
        trace.append(float(best_s))
    return sorted(picks), trace

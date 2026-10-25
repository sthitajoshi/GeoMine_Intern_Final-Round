"""Spectral preprocessing whose learned parameters are fitted on training pixels only.

Band policy (crop scenes): the EHU "corrected" cubes already drop the water-vapour bands
(~1.36-1.41 um, ~1.81-1.95 um) and the last, noisiest channel. We keep the remaining bands by
default and let ``drop_ranges_nm`` remove more (e.g. ``[[1300, 1500]]`` for the live-change
"remove the 1.3-1.5 um bands"). Every retained / dropped band is written to the experiment log.

Normalisation: per-band z-score (mean / std from training pixels). Raw AVIRIS DN-like radiance
units differ by band by orders of magnitude, so without it the RBF kernel distance is dominated
by the brightest visible/NIR bands. PCA for the CNN is likewise fitted on training pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from .wavelengths import mask_ranges


@dataclass
class SpectralPreprocessor:
    drop_ranges_nm: list | None = None
    n_pca: int | None = None
    keep_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    pca_: PCA | None = None

    def fit(self, X_train: np.ndarray, centres: np.ndarray, seed: int = 0) -> "SpectralPreprocessor":
        """X_train: (N, B) training pixels only."""
        self.keep_ = np.flatnonzero(mask_ranges(centres, self.drop_ranges_nm))
        Xs = X_train[:, self.keep_].astype(np.float64)
        self.mean_ = Xs.mean(0)
        self.std_ = Xs.std(0) + 1e-6
        if self.n_pca:
            self.pca_ = PCA(self.n_pca, random_state=seed).fit((Xs - self.mean_) / self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X[..., self.keep_] - self.mean_) / self.std_
        if self.pca_ is not None:
            shp = Z.shape[:-1]
            Z = self.pca_.transform(Z.reshape(-1, Z.shape[-1])).reshape(*shp, -1)
        return Z.astype(np.float32)

    def describe(self, centres: np.ndarray) -> dict:
        dropped = np.setdiff1d(np.arange(len(centres)), self.keep_)
        return {"n_bands_in": int(len(centres)), "n_bands_kept": int(len(self.keep_)),
                "dropped_band_idx": dropped.tolist(),
                "dropped_band_nm": np.round(centres[dropped], 1).tolist(),
                "drop_ranges_nm": self.drop_ranges_nm, "n_pca": self.n_pca,
                "pca_explained_var": (None if self.pca_ is None
                                      else float(self.pca_.explained_variance_ratio_.sum()))}

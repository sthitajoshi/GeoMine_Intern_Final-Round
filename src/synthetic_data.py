"""Synthetic training spectra generated only from library spectra.

Generative model for one synthetic pixel (all choices seeded and logged):

* Class c in TARGETS: target fraction f ~ U(f_min, 1); the remaining 1 - f is split (Dirichlet)
  between 1-2 background components.
* Class "unknown": either (a) backgrounds only, (b) a non-target library mineral
  (montmorillonite, jarosite, dickite, opal, hematite) with f ~ U(0.3, 1) plus background, or
  (c) a target at a sub-detection fraction f ~ U(0.02, f_min / 2).
* Backgrounds (library-derived, no image pixels): hematite (Fe3+, ferric soils/varnish), and
  spectrally featureless "albedo" components = constant reflectance 0.05-0.6 with a linear slope
  of +-0.05 per um, standing in for quartz/feldspar-rich, SWIR-neutral rocks.
* Mixing: with prob. 0.5 linear (areal/checkerboard); otherwise intimate, by mixing single-
  scattering albedo w = 1 - ((1 - R) / (1 + R))^2 (isotropic two-stream inversion of Hapke)
  and converting back R = (1 - sqrt(1 - w)) / (1 + sqrt(1 - w)).
* Illumination: multiplicative factor U(0.6, 1.4) (slope/topography, BRDF) plus additive
  offset U(-0.01, 0.01) (residual path radiance after atmospheric correction).
* Noise: zero-mean Gaussian with the per-band sigma estimated from the Cuprite image itself
  (``estimate_noise``), optionally scaled by ``noise_scale``.
"""
from __future__ import annotations

import numpy as np

from .mineral_library import OTHERS, TARGETS


def estimate_noise(cube: np.ndarray, block: int = 8, pct: float = 10.0) -> np.ndarray:
    """Per-band noise sigma from horizontal shift differences, taken in the most homogeneous
    blocks (lower ``pct`` percentile of block-wise estimates) so scene texture is excluded as far
    as possible. sigma_noise = std(x[i, j+1] - x[i, j]) / sqrt(2)."""
    H, W, B = cube.shape
    d = (cube[:, 1:, :] - cube[:, :-1, :]) / np.sqrt(2)
    hb, wb = H // block, (W - 1) // block
    d = d[:hb * block, :wb * block].reshape(hb, block, wb, block, B)
    s = d.std(axis=(1, 3)).reshape(-1, B)
    return np.percentile(s, pct, axis=0)


def _to_w(R):
    R = np.clip(R, 1e-4, 0.999)
    g = (1 - R) / (1 + R)
    return 1 - g ** 2


def _from_w(w):
    g = np.sqrt(np.clip(1 - w, 0, 1))
    return (1 - g) / (1 + g)


def _mix(comps: np.ndarray, fr: np.ndarray, intimate: bool):
    if intimate:
        return _from_w((fr[:, None] * _to_w(comps)).sum(0))
    return (fr[:, None] * comps).sum(0)


def _background(rng, centres, lib):
    if rng.random() < 0.35:
        v = lib["hematite"]
        return v[rng.integers(len(v))]
    level = rng.uniform(0.05, 0.6)
    slope = rng.uniform(-0.05, 0.05)
    return level + slope * (centres - 2250.0) / 1000.0


def generate(lib: dict, centres: np.ndarray, noise_sigma: np.ndarray, n_per_class: int = 3000,
             seed: int = 0, f_min: float = 0.3, noise_scale: float = 1.0, p_intimate: float = 0.5):
    """Returns X (N, B) reflectance, y (N,) with 0..len(TARGETS)-1 targets and len(TARGETS)=unknown,
    and per-sample target fraction."""
    rng = np.random.default_rng(seed)
    classes = TARGETS + ["unknown"]
    X, y, frac = [], [], []
    for ci, cname in enumerate(classes):
        for _ in range(n_per_class):
            n_bg = rng.integers(1, 3)
            bgs = [_background(rng, centres, lib) for _ in range(n_bg)]
            if cname != "unknown":
                v = lib[cname]
                comp, f = v[rng.integers(len(v))], rng.uniform(f_min, 1.0)
            else:
                mode = rng.random()
                if mode < 0.3:
                    comp, f = None, 0.0
                elif mode < 0.75:
                    v = lib[OTHERS[rng.integers(len(OTHERS))]]
                    comp, f = v[rng.integers(len(v))], rng.uniform(0.3, 1.0)
                else:
                    v = lib[TARGETS[rng.integers(len(TARGETS))]]
                    comp, f = v[rng.integers(len(v))], rng.uniform(0.02, f_min / 2)
            bfr = rng.dirichlet(np.ones(n_bg)) * (1 - f)
            comps = np.array(([comp] if comp is not None else []) + bgs)
            fr = np.concatenate([[f] if comp is not None else [], bfr])
            fr = fr / fr.sum()
            s = _mix(comps, fr, rng.random() < p_intimate)
            s = s * rng.uniform(0.6, 1.4) + rng.uniform(-0.01, 0.01)
            s = s + rng.normal(0, 1, s.shape) * noise_sigma * noise_scale
            X.append(s)
            y.append(ci)
            frac.append(f)
    return np.array(X), np.array(y), np.array(frac), classes

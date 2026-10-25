"""Vegetation indices from the nearest AVIRIS bands.

NDVI = (R800 - R670) / (R800 + R670)                 (chlorophyll absorption vs NIR plateau)
NDRE = (R790 - R720) / (R790 + R720)                 (red-edge; less saturated in dense canopy)
REP  = red-edge inflection by linear 4-point method (Guyot & Baret 1988), nm.
The crop cubes are at-sensor values, not surface reflectance, so absolute index values are
relative; spatial contrasts between fields are what the maps are for.
"""
from __future__ import annotations

import numpy as np

from .wavelengths import nearest_band


def _b(cube, centres, nm):
    return cube[..., nearest_band(centres, nm)].astype(np.float64)


def ndvi(cube, centres):
    n, r = _b(cube, centres, 800), _b(cube, centres, 670)
    return np.where(n + r != 0, (n - r) / (n + r + 1e-12), np.nan)


def ndre(cube, centres):
    n, e = _b(cube, centres, 790), _b(cube, centres, 720)
    return np.where(n + e != 0, (n - e) / (n + e + 1e-12), np.nan)


def red_edge_position(cube, centres):
    r670, r700, r740, r780 = (_b(cube, centres, w) for w in (670, 700, 740, 780))
    rre = (r670 + r780) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        rep = 700 + 40 * (rre - r700) / (r740 - r700)
    return np.clip(rep, 680, 760)

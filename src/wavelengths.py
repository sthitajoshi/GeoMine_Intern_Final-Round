"""Band-centre wavelengths for the three AVIRIS scenes.

The .mat files carry no spectral metadata, so we recover it from sensor calibration tables:

Indian Pines (AVIRIS, 12 June 1992)
    ``92AV3C.spc`` is the channel calibration file distributed by Purdue/MultiSpec with the
    original 220-band 92AV3C image (columns: centre nm, FWHM nm, centre-std, FWHM-std, channel).
    The EHU "corrected" cube removed bands [104-108], [150-163] and 220 (1-based, of the 220),
    leaving 200. We apply the same removal to the calibration table.

Salinas (AVIRIS, 1998)
    The USGS splib07 release includes the AVIRIS-Classic 1998 channel-centre table
    (``s07_AV98_AVIRIS_1998_Wavelengths_(um)_224_chans.txt``). EHU removed bands
    [108-112], [154-167] and 224 (1-based, of 224), leaving 204.

Cuprite (AVIRIS, 19 June 1997, f970619t01p02_r02_sc03)
    All 224 channels are present in the cube. splib07 ships per-year AVIRIS tables, but the
    1997 table is only inside the 5.5 GB ScienceBase zip, which we could not retrieve. We use
    the 1998 table as a documented proxy. Sanity check (see tests): the channels the 1997 cube
    zeroes out (atmospheric water at 1.38 and 1.88 um) fall exactly on 1.36-1.39 and 1.83-1.93 um
    in the 1998 table. Year-to-year AVIRIS-Classic centre drift is of order 1-2 nm, i.e. well
    under the ~10 nm FWHM, so feature positions are reliable to about one channel.

AVIRIS has four spectrometers; at the joins the centres go *backwards* (e.g. 675 -> 664 nm,
1269 -> 1255 nm, 1882 -> 1878 nm). The table is therefore not monotonic, and code must never
assume a linear or sorted wavelength axis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
AV98_FILE = RAW / "splib07_AV98" / "s07_AV98_AVIRIS_1998_Wavelengths_(um)_224_chans.txt"
SPC_FILE = RAW / "92AV3C.spc"

# 1-based band numbers removed by EHU to make the "corrected" cubes
IP_REMOVED_1B = list(range(104, 109)) + list(range(150, 164)) + [220]
SALINAS_REMOVED_1B = list(range(108, 113)) + list(range(154, 168)) + [224]


def _keep(n_total: int, removed_1b: list[int]) -> np.ndarray:
    removed = set(b - 1 for b in removed_1b)
    return np.array([i for i in range(n_total) if i not in removed])


def read_av98() -> np.ndarray:
    """AVIRIS-Classic 1998 channel centres in nm (224,)."""
    w = np.loadtxt(AV98_FILE, skiprows=1)
    assert w.shape == (224,), w.shape
    return w * 1000.0


def fwhm_from_centres(c: np.ndarray) -> np.ndarray:
    """Approximate FWHM by the local channel spacing (AVIRIS is roughly Nyquist sampled)."""
    order = np.argsort(c)
    cs = c[order]
    d = np.gradient(cs)
    out = np.empty_like(c)
    out[order] = np.clip(d, 5.0, 20.0)
    return out


def indian_pines() -> dict:
    spc = np.loadtxt(SPC_FILE, skiprows=2)
    assert spc.shape[0] == 220
    keep = _keep(220, IP_REMOVED_1B)
    assert keep.size == 200
    return dict(centres=spc[keep, 0], fwhm=spc[keep, 1], source_band_index=keep,
                source="92AV3C.spc (Purdue MultiSpec AVIRIS 1992 calibration)", exact=True)


def salinas() -> dict:
    w = read_av98()
    keep = _keep(224, SALINAS_REMOVED_1B)
    assert keep.size == 204
    return dict(centres=w[keep], fwhm=fwhm_from_centres(w)[keep], source_band_index=keep,
                source="splib07 AVIRIS-Classic 1998 table", exact=True)


def cuprite() -> dict:
    w = read_av98()
    return dict(centres=w, fwhm=fwhm_from_centres(w), source_band_index=np.arange(224),
                source="splib07 AVIRIS-Classic 1998 table used as proxy for the 1997 flight", exact=False)


def get(scene: str) -> dict:
    return {"indian_pines": indian_pines, "salinas": salinas, "cuprite": cuprite}[scene]()


def nearest_band(centres: np.ndarray, nm: float) -> int:
    return int(np.argmin(np.abs(np.asarray(centres) - nm)))


def mask_ranges(centres: np.ndarray, ranges_nm) -> np.ndarray:
    """Boolean mask that is True for bands *outside* every [lo, hi] range in ``ranges_nm``."""
    keep = np.ones(len(centres), bool)
    for lo, hi in ranges_nm or []:
        keep &= ~((centres >= lo) & (centres <= hi))
    return keep

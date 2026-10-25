"""USGS splib07 reference spectra resampled to the Cuprite AVIRIS bands.

Two routes, compared against each other in validation:

1. ``convolved``  (primary) - native-resolution splib07a laboratory spectra (Beckman / ASD) with
   their own wavelength and bandpass tables, convolved by us with Gaussian spectral response
   functions centred on the adopted AVIRIS centres, FWHM = local channel spacing (~10 nm).
   This is what the brief asks for when no matching-year convolution is available.
2. ``av98``  - splib07b spectra convolved by USGS to AVIRIS-Classic 1998. Not the 1997 flight
   year, so used only as a cross-check of our convolution (they share the same centre table).

Because the 1997 channel table was not retrievable, both routes inherit the 1998-centre proxy
(see ``wavelengths.py``). Library deleted values (-1.23e34) are treated as missing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "data" / "raw" / "splib07a"
AV98 = ROOT / "data" / "raw" / "splib07_AV98"

# record name -> (mineral label, native instrument code)
RECORDS = {
    "Alunite_GDS84_Na03_BECKa_AREF": ("alunite", "BECK"),
    "Alunite_HS295_ASDFRa_AREF": ("alunite", "ASD"),
    "Kaolinite_CM9_BECKb_AREF": ("kaolinite", "BECK"),
    "Kaolinite_KGa-1_(wxl)_BECKb_AREF": ("kaolinite", "BECK"),
    "Muscovite_GDS107_BECKa_AREF": ("muscovite", "BECK"),
    "Buddingtonite_NHB2301_BECKb_AREF": ("buddingtonite", "BECK"),
    "Calcite_WS272_BECKa_AREF": ("calcite", "BECK"),
    "Chalcedony_CU91-6A_ASDNGa_AREF": ("chalcedony", "ASDNG"),
    # not targets: used only as "other" spectra for the unknown class / mixtures
    "Montmorillonite_SWy-1_BECKb_AREF": ("montmorillonite", "BECK"),
    "Hematite_GDS27_BECKa_AREF": ("hematite", "BECK"),
    "Jarosite_GDS99_K_200C_Syn_BECKa_AREF": ("jarosite", "BECK"),
    "Dickite_NMNH106242_BECKb_AREF": ("dickite", "BECK"),
    "Opal_WS732_BECKa_AREF": ("opal", "BECK"),
}
TARGETS = ["alunite", "kaolinite", "muscovite", "buddingtonite", "calcite", "chalcedony"]
OTHERS = ["montmorillonite", "hematite", "jarosite", "dickite", "opal"]

_WL = {"BECK": "splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt",
       "ASD": "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt",
       "ASDNG": "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"}
_BP = {"BECK": "splib07a_Bandpass_(FWHM)_BECK_Beckman_in_microns.txt",
       "ASD": "splib07a_Bandpass_(FWHM)_ASDFR_StandardResolution.txt",
       "ASDNG": "splib07a_Bandpass_(FWHM)_ASDNG_High-Res_NextGen.txt"}


def _read_col(path: Path) -> np.ndarray:
    v = np.loadtxt(path, skiprows=1)
    v[v < -1e30] = np.nan
    return v


def load_native(record: str):
    """Returns (wavelength nm, reflectance, fwhm nm) at the lab instrument's resolution."""
    inst = RECORDS[record][1]
    wl = _read_col(NATIVE / _WL[inst]) * 1000
    refl = _read_col(NATIVE / f"splib07a_{record}.txt")
    fw = _read_col(NATIVE / _BP[inst]) * 1000
    assert wl.size == refl.size, (record, wl.size, refl.size)
    return wl, refl, fw


def gaussian_convolve(wl, refl, centres, fwhm):
    """Resample a spectrum to sensor bands with Gaussian SRFs (normalised; NaNs excluded).
    The lab spectrum already has a finite bandpass of its own (see the splib07a bandpass tables);
    we do not deconvolve it, so the effective response is slightly broader than the sensor FWHM
    where the lab bandpass is not negligible."""
    ok = np.isfinite(refl) & np.isfinite(wl)
    wl, refl = wl[ok], refl[ok]
    out = np.full(len(centres), np.nan)
    sig = fwhm / (2 * np.sqrt(2 * np.log(2)))
    for i, (c, s) in enumerate(zip(centres, sig)):
        w = np.exp(-0.5 * ((wl - c) / s) ** 2)
        if w.sum() > 1e-6 and (wl.min() <= c <= wl.max()):
            out[i] = (w * refl).sum() / w.sum()
    return out


def load_av98(record: str) -> np.ndarray:
    return _read_col(AV98 / f"s07_AV98_{record}.txt")


def build_library(centres, fwhm, route="convolved"):
    """Dict mineral -> (n_variants, B) spectra on the image bands, plus record names."""
    lib, names = {}, {}
    for rec, (mineral, _) in RECORDS.items():
        if route == "convolved":
            wl, r, _fw = load_native(rec)
            s = gaussian_convolve(wl, r, centres, fwhm)
        else:
            s = load_av98(rec)
        lib.setdefault(mineral, []).append(s)
        names.setdefault(mineral, []).append(rec)
    return {k: np.array(v) for k, v in lib.items()}, names

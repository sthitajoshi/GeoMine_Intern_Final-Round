"""Load the three scenes as float32 cubes with validated shapes and class tables."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio

from . import wavelengths

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

IP_CLASSES = ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees",
              "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
              "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"]
SALINAS_CLASSES = ["Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow", "Fallow_rough_plow",
                   "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop",
                   "Corn_senesced_green_weeds", "Lettuce_romaine_4wk", "Lettuce_romaine_5wk",
                   "Lettuce_romaine_6wk", "Lettuce_romaine_7wk", "Vinyard_untrained",
                   "Vinyard_vertical_trellis"]
# Labelled-pixel counts published on the EHU page; used as a load-time integrity check.
IP_COUNTS = [46, 1428, 830, 237, 483, 730, 28, 478, 20, 972, 2455, 593, 205, 1265, 386, 93]
SALINAS_COUNTS = [2009, 3726, 1976, 1394, 2678, 3959, 3579, 11271, 6203, 3278, 1068, 1927, 916,
                  1070, 7268, 1807]
# Classes that are actual crops/vegetation (used to mask the vegetation-index maps).
IP_CROP = {"Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-pasture-mowed",
           "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat"}
SALINAS_CROP = {c for c in SALINAS_CLASSES if not c.startswith(("Fallow", "Stubble", "Soil"))}

SCENES = ("indian_pines", "salinas", "cuprite")


@dataclass
class Scene:
    name: str
    cube: np.ndarray            # (H, W, B) float32 reflectance-like values
    gt: np.ndarray | None       # (H, W) int, 0 = unlabelled, 1..K = classes
    centres: np.ndarray         # (B,) nm, NOT necessarily monotonic
    fwhm: np.ndarray
    class_names: list = field(default_factory=list)
    crop_classes: set = field(default_factory=set)
    wavelength_source: str = ""
    wavelength_exact: bool = True
    gsd_m: float | None = None
    valid: np.ndarray | None = None  # (H, W) bool pixels with finite, non-zero spectra

    @property
    def shape(self):
        return self.cube.shape


def _load(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python scripts/download_data.py` first")
    return sio.loadmat(path)[key]


def load_scene(name: str) -> Scene:
    if name not in SCENES:
        raise ValueError(f"unknown scene {name!r}; choose from {SCENES}")
    wl = wavelengths.get(name)
    if name == "indian_pines":
        cube = _load(RAW / "Indian_pines_corrected.mat", "indian_pines_corrected").astype(np.float32)
        gt = _load(RAW / "Indian_pines_gt.mat", "indian_pines_gt").astype(np.int64)
        expect, names, counts, crops, gsd = (145, 145, 200), IP_CLASSES, IP_COUNTS, IP_CROP, 20.0
    elif name == "salinas":
        cube = _load(RAW / "Salinas_corrected.mat", "salinas_corrected").astype(np.float32)
        gt = _load(RAW / "Salinas_gt.mat", "salinas_gt").astype(np.int64)
        expect, names, counts, crops, gsd = (512, 217, 204), SALINAS_CLASSES, SALINAS_COUNTS, SALINAS_CROP, 3.7
    else:
        # int16 reflectance scaled by 10000 (max 12000 = bright specular/saturated pixels)
        cube = _load(RAW / "Cuprite_f970619t01p02_r02_sc03.a.rfl.mat", "X").astype(np.float32) / 10000.0
        gt, expect, names, counts, crops, gsd = None, (512, 614, 224), [], None, set(), 20.0
    if cube.shape != expect:
        raise ValueError(f"{name}: cube shape {cube.shape} != expected {expect}")
    if cube.shape[2] != wl["centres"].size:
        raise ValueError(f"{name}: {cube.shape[2]} bands but {wl['centres'].size} wavelengths")
    cube = np.where(np.isfinite(cube), cube, 0.0).astype(np.float32)
    if gt is not None:
        if gt.shape != cube.shape[:2]:
            raise ValueError("ground-truth shape mismatch")
        got = np.bincount(gt.ravel(), minlength=len(names) + 1)[1:]
        if list(got) != counts:
            raise ValueError(f"{name}: class counts {list(got)} differ from published {counts}")
    valid = np.any(cube != 0, axis=2)
    return Scene(name, cube, gt, wl["centres"].astype(np.float64), wl["fwhm"].astype(np.float64),
                 names, crops, wl["source"], wl["exact"], gsd, valid)


def true_colour(scene: Scene, rgb_nm=(640, 550, 470), pct=(2, 98)) -> np.ndarray:
    """Percentile-stretched RGB composite (H, W, 3) in [0, 1]."""
    idx = [wavelengths.nearest_band(scene.centres, w) for w in rgb_nm]
    img = scene.cube[:, :, idx].astype(np.float64)
    out = np.zeros_like(img)
    for k in range(3):
        lo, hi = np.percentile(img[..., k][scene.valid], pct)
        out[..., k] = np.clip((img[..., k] - lo) / max(hi - lo, 1e-9), 0, 1)
    return out

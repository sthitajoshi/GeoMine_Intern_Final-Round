"""Full-scene inference with saved models (shared by generate_outputs.py and the app)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from . import continuum_removal as crm
from . import crop_models as cm
from . import mineral_models as mm
from .data_loader import Scene
from .utils import MODELS


def model_paths(scene: str, split: str = "spatial") -> dict:
    return {"svm": MODELS / f"{scene}_{split}_svm.pkl",
            "cnn3d": MODELS / f"{scene}_{split}_cnn3d.pt",
            "cnn3d_prep": MODELS / f"{scene}_{split}_cnn3d_prep.pkl",
            "mlp": MODELS / "cuprite_mlp.pkl"}


def _need(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"model artefact {p.name} not found in models/ - run "
                                "`python scripts/reproduce_all.py` (or train_crop_models.py) first")


def predict_crop(scene: Scene, model: str = "cnn3d", split: str = "spatial", calibrated: bool = True):
    """Returns (class map (H, W) 0..K-1, probability cube (H, W, K))."""
    H, W, B = scene.cube.shape
    paths = model_paths(scene.name, split)
    if model == "svm":
        _need(paths["svm"])
        d = pickle.load(open(paths["svm"], "rb"))
        X = d["prep"].transform(scene.cube.reshape(-1, B))
        P = cm.svm_proba(d["model"], X, len(scene.class_names))
    elif model == "cnn3d":
        _need(paths["cnn3d"])
        _need(paths["cnn3d_prep"])
        ck = torch.load(paths["cnn3d"], map_location="cpu", weights_only=False)
        prep = pickle.load(open(paths["cnn3d_prep"], "rb"))["prep"]
        net = cm.CNN3D(ck["n_bands"], ck["n_classes"], ck["patch"], ck["dropout"])
        net.load_state_dict(ck["state_dict"])
        Z = prep.transform(scene.cube)
        rows, cols = np.unravel_index(np.arange(H * W), (H, W))
        logits = np.concatenate([cm.cnn_logits(net, cm.extract_patches(Z, rows[i:i + 8192], cols[i:i + 8192], ck["patch"]))
                                 for i in range(0, H * W, 8192)])
        P = cm.softmax(logits, ck["temperature"] if calibrated else 1.0)
    else:
        raise ValueError(model)
    P = P.reshape(H, W, -1)
    return P.argmax(-1), P


def predict_minerals(scene: Scene):
    """Re-runs the synthetic-trained MLP on the scene; returns label map (-2 invalid, -1 unknown),
    confidence map and probabilities."""
    p = model_paths("cuprite")["mlp"]
    _need(p)
    d = pickle.load(open(p, "rb"))
    H, W, B = scene.cube.shape
    pix = scene.cube.reshape(-1, B)
    lo, hi = d["cr_window_nm"]
    win = (scene.centres >= lo) & (scene.centres <= hi) & d["band_ok"]
    valid = scene.valid.ravel() & (pix[:, win].mean(1) > 0.02) & (pix[:, win].max(1) < 1.0)
    cr, _ = crm.continuum_remove(pix[valid], scene.centres, lo, hi, d["band_ok"])
    U = len(d["classes"]) - 1
    lab, conf, P = mm.predict_with_reject(d["model"], mm.features(cr), U, d["reject_prob"])
    out = np.full(H * W, -2)
    out[valid] = np.where(lab == U, -1, lab)
    cmap = np.full(H * W, np.nan, np.float32)
    cmap[valid] = conf
    return out.reshape(H, W), cmap.reshape(H, W), d["classes"]

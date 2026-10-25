"""Evidence for the random-vs-spatial gap (Indian Pines, seed-0 SVM and CNN models).

1. Distance from each test pixel to the nearest training pixel (Chebyshev, px) under each split.
2. Test accuracy binned by that distance, for the model trained on each split.
3. Spatial autocorrelation of spectra: mean spectral angle between labelled pixels and their
   neighbours at increasing lag, within the same class.

Usage: python scripts/gap_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_cdt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402

from src import inference  # noqa: E402
from src.data_loader import load_scene  # noqa: E402
from src.splits import TEST, TRAIN  # noqa: E402
from src.utils import EXP, OUT, Logger, save_json  # noqa: E402


def main():
    log = Logger("gap_analysis")
    scene = load_scene("indian_pines")
    gt = scene.gt
    res = {"distance": {}, "acc_by_distance": {}}
    bins = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 20), (21, 200)]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    for kind in ("random", "spatial"):
        sp = np.load(EXP / "splits" / f"indian_pines_{kind}_s0.npz")["split"]
        dist = distance_transform_cdt(sp != TRAIN, metric="chessboard")
        test = sp == TEST
        d = dist[test]
        res["distance"][kind] = {"median": float(np.median(d)), "mean": float(d.mean()),
                                 "frac_adjacent": float((d == 1).mean())}
        axs[0].hist(np.clip(d, 0, 30), bins=30, alpha=0.6, label=f"{kind} (median {np.median(d):.0f} px)")
        for model in ("svm", "cnn3d"):
            path = inference.model_paths("indian_pines", kind)
            if not path[model].exists():
                continue
            lab, _ = inference.predict_crop(scene, model, kind)
            correct = (lab == gt - 1)[test]
            accs = []
            for lo, hi in bins:
                m = (d >= lo) & (d <= hi)
                accs.append(float(correct[m].mean()) if m.sum() >= 20 else np.nan)
            res["acc_by_distance"][f"{kind}_{model}"] = {"bins": bins, "acc": accs,
                                                         "n": [int(((d >= lo) & (d <= hi)).sum()) for lo, hi in bins]}
            axs[1].plot(range(len(bins)), accs, "o-", label=f"{model} trained on {kind} split")
    axs[0].set_xlabel("test pixel: distance to nearest training pixel (px)")
    axs[0].set_ylabel("count")
    axs[0].legend(fontsize=7)
    axs[1].set_xticks(range(len(bins)))
    axs[1].set_xticklabels([f"{a}-{b}" if a != b else f"{a}" for a, b in bins], fontsize=7)
    axs[1].set_xlabel("distance to nearest training pixel (px)")
    axs[1].set_ylabel("test accuracy")
    axs[1].legend(fontsize=6)
    # spectral autocorrelation: angle to same-class pixel at lag L (horizontal), vs different fields
    X = scene.cube / np.linalg.norm(scene.cube, axis=2, keepdims=True)
    lags = [1, 2, 4, 8, 16, 32]
    same = []
    for L in lags:
        a, b = X[:, :-L], X[:, L:]
        m = (gt[:, :-L] > 0) & (gt[:, :-L] == gt[:, L:])
        same.append(float(np.degrees(np.arccos(np.clip((a * b).sum(-1)[m], -1, 1))).mean()))
    rng = np.random.default_rng(0)
    lab_idx = np.argwhere(gt > 0)
    i, j = lab_idx[rng.integers(len(lab_idx), size=20000)].T
    k, l = lab_idx[rng.integers(len(lab_idx), size=20000)].T
    sc = gt[i, j] == gt[k, l]
    far = np.maximum(abs(i - k), abs(j - l)) > 40
    ang = np.degrees(np.arccos(np.clip((X[i, j] * X[k, l]).sum(-1), -1, 1)))
    res["spectral_angle_same_class_by_lag_deg"] = dict(zip(map(str, lags), same))
    res["spectral_angle_same_class_far_deg"] = float(ang[sc & far].mean())
    res["spectral_angle_diff_class_deg"] = float(ang[~sc].mean())
    axs[2].plot(lags, same, "o-", label="same class, lag L (same field mostly)")
    axs[2].axhline(res["spectral_angle_same_class_far_deg"], color="C1", ls="--", label="same class, >40 px apart")
    axs[2].axhline(res["spectral_angle_diff_class_deg"], color="C3", ls=":", label="different class")
    axs[2].set_xscale("log", base=2)
    axs[2].set_xlabel("lag L (px)")
    axs[2].set_ylabel("mean spectral angle (deg)")
    axs[2].legend(fontsize=6)
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "gap_evidence.png", dpi=150)
    plt.close(fig)
    save_json(res, OUT / "reports" / "gap_analysis.json")
    log(str(res))


if __name__ == "__main__":
    main()

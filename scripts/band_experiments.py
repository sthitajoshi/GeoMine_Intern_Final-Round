"""Band importance, <=15-band and Sentinel-2-like experiments, and the block-size study.

All on Indian Pines, SVM, spatial split (the honest protocol), 3 seeds. For each seed the split is
rebuilt exactly as in train_crop_models.py (same seed -> same split), importance is computed on
that seed's VALIDATION pixels, bands are chosen from it, the SVM is retuned on validation, and the
test set is used once.

Usage: python scripts/band_experiments.py [--skip_blocks]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_crop_models import make_split  # noqa: E402
from src import band_selection as bs  # noqa: E402
from src import crop_models as cm  # noqa: E402
from src import splits as sp  # noqa: E402
from src.data_loader import load_scene  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.preprocessing import SpectralPreprocessor  # noqa: E402
from src.utils import EXP, Logger, load_config, load_json, save_json  # noqa: E402


def svm_eval(X, y, fit_idx, val_idx, test_idx, train_idx, K, cfg, seed):
    s = cfg["svm"]
    best, _ = cm.tune_svm(X[fit_idx], y[fit_idx], X[val_idx], y[val_idx], s["grid_C"], s["grid_gamma"], seed)
    model = cm.fit_svm(X[train_idx], y[train_idx], best["C"], best["gamma"], seed)
    return evaluate(y[test_idx], cm.svm_proba(model, X[test_idx], K), K), best, model


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_blocks", action="store_true")
    ap.add_argument("--reuse_importance", action="store_true", help="reuse permutation importances from a previous run")
    a = ap.parse_args()
    cfg = load_config()
    log = Logger("band_experiments")
    scene = load_scene("indian_pines")
    H, W, B = scene.cube.shape
    K = 16
    Xall = scene.cube.reshape(-1, B)
    y = scene.gt.ravel() - 1
    prev = load_json(EXP / "bands" / "band_experiments.json") if a.reuse_importance else None
    res = {"seeds": {}, "s2_report": None}
    for seed in cfg["seeds"]:
        split, fit_idx, val_idx, test_idx, info, _ = make_split(scene, "spatial", seed, cfg)
        train_idx = np.flatnonzero(split.ravel() == sp.TRAIN)
        prep = SpectralPreprocessor(None, None).fit(Xall[train_idx], scene.centres)
        Xs = prep.transform(Xall)
        full, best, model = svm_eval(Xs, y, fit_idx, val_idx, test_idx, train_idx, K, cfg, seed)
        log(f"seed {seed} full 200 bands OA={full['oa']:.4f}")
        if prev:
            ps = prev["seeds"][str(seed)]
            imp, imp5, base = np.array(ps["importance"]), ps["importance_window5"], ps["val_base_oa"]
            imp5 = None if imp5 is None else np.array(imp5)
        else:
            imp, base = bs.permutation_importance(model.predict, Xs[val_idx], y[val_idx], cfg["bands"]["n_repeats"], seed)
            imp5, _ = bs.permutation_importance(model.predict, Xs[val_idx], y[val_idx], 1, seed, window=5) if seed == 0 else (None, None)
        picks = bs.select_bands(imp, scene.centres, cfg["bands"]["k"], cfg["bands"]["min_sep_nm"])
        sel, _, _ = svm_eval(Xs[:, picks], y, fit_idx, val_idx, test_idx, train_idx, K, cfg, seed)
        log(f"seed {seed} 15 selected bands {np.round(scene.centres[picks]).astype(int).tolist()} OA={sel['oa']:.4f}")
        fwd, trace = bs.forward_select(Xs[train_idx], y[train_idx], Xs[val_idx], y[val_idx], cfg["bands"]["k"])
        selF, _, _ = svm_eval(Xs[:, fwd], y, fit_idx, val_idx, test_idx, train_idx, K, cfg, seed)
        log(f"seed {seed} 15 forward-selected bands {np.round(scene.centres[fwd]).astype(int).tolist()} OA={selF['oa']:.4f}")
        # Sentinel-2-like: average raw bands inside each S2 band, then z-score on train
        Xs2, names, rep = bs.simulate_s2(Xall, scene.centres)
        p2 = SpectralPreprocessor(None, None).fit(Xs2[train_idx], np.arange(Xs2.shape[1], dtype=float))
        Z2 = p2.transform(Xs2)
        s2, _, _ = svm_eval(Z2, y, fit_idx, val_idx, test_idx, train_idx, K, cfg, seed)
        res["s2_report"] = rep
        log(f"seed {seed} S2-like bands {names} OA={s2['oa']:.4f}")
        # random 15 bands baseline (is selection better than chance?)
        rng = np.random.default_rng(seed)
        rnd = sorted(rng.choice(B, 15, replace=False).tolist())
        rr, _, _ = svm_eval(Xs[:, rnd], y, fit_idx, val_idx, test_idx, train_idx, K, cfg, seed)
        res["seeds"][seed] = {
            "importance": imp.tolist(), "importance_window5": None if imp5 is None else imp5.tolist(),
            "val_base_oa": base, "picks": picks, "picks_nm": scene.centres[picks].tolist(),
            "full": _short(full), "selected15": _short(sel), "forward15": _short(selF),
            "forward_picks": fwd, "forward_picks_nm": scene.centres[fwd].tolist(), "forward_val_trace": trace, "s2": _short(s2), "random15": _short(rr),
            "s2_bands": names, "svm_params_full": best}
    res["centres"] = scene.centres.tolist()
    save_json(res, EXP / "bands" / "band_experiments.json")

    if not a.skip_blocks:
        blocks = {}
        for b in cfg["block_size_study"]:
            rows = []
            for seed in cfg["seeds"]:
                split, fit_idx, val_idx, test_idx, info, _ = make_split(scene, "spatial", seed, cfg, block=b)
                train_idx = np.flatnonzero(split.ravel() == sp.TRAIN)
                prep = SpectralPreprocessor(None, None).fit(Xall[train_idx], scene.centres)
                m, _, _ = svm_eval(prep.transform(Xall), y, fit_idx, val_idx, test_idx, train_idx, K, cfg, seed)
                big = np.bincount(scene.gt.ravel(), minlength=17)[1:] >= 50
                rows.append({**_short(m), "n_train": info["n_train"], "n_val": info["n_val"], "n_test": info["n_test"],
                             "frac_train": info["frac_train"], "frac_test": info["frac_test"],
                             "classes_missing_train": int(np.sum(big & (np.array(info["class_counts_train"]) == 0)))})
                log(f"block {b} seed {seed} OA={m['oa']:.4f} train={info['n_train']} test={info['n_test']}")
            blocks[b] = rows
        save_json(blocks, EXP / "bands" / "block_size_study.json")


def _short(m):
    return {k: m[k] for k in ("oa", "aa", "kappa", "macro_f1", "ece")}


if __name__ == "__main__":
    main()

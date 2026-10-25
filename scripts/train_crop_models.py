"""Part A main experiment: {indian_pines, salinas} x {random, spatial} x {svm, cnn3d} x 3 seeds.

For every run: build the split, fit preprocessing on training pixels, tune on validation pixels,
evaluate once on test, save metrics to experiments/runs/<scene>_<split>_<model>_s<seed>.json.
Seed-0 models (+ preprocessing) are saved to models/ for the app and full-scene maps.

Usage: python scripts/train_crop_models.py [--scenes indian_pines salinas] [--models svm cnn3d]
                                            [--splits random spatial] [--block 20] [--tag ""]
"""
from __future__ import annotations

import argparse
import warnings
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import crop_models as cm  # noqa: E402
from src import splits as sp  # noqa: E402
from src.data_loader import load_scene  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.preprocessing import SpectralPreprocessor  # noqa: E402
from src.utils import EXP, MODELS, Logger, environment, load_config, save_json  # noqa: E402


def make_split(scene, kind, seed, cfg, block=None):
    gt = scene.gt
    if kind == "random":
        split = sp.random_split(gt, cfg["random_split"]["train_frac"], seed)
        sp.verify_random(split, gt, cfg["random_split"]["train_frac"])
        # validation for tuning = stratified 20 % of the training pixels (no test pixel involved)
        rng = np.random.default_rng(1000 + seed)
        tr = np.flatnonzero(split.ravel() == sp.TRAIN)
        fit_idx, val_idx = [], []
        for c in np.unique(gt.ravel()[tr]):
            idx = tr[gt.ravel()[tr] == c]
            rng.shuffle(idx)
            nv = int(round(cfg["random_split"]["val_frac_of_train"] * idx.size)) if idx.size >= 5 else 0
            val_idx += list(idx[:nv])
            fit_idx += list(idx[nv:])
        info = sp.split_summary(split, gt)
        info.update(kind="random", seed=seed, n_fit=len(fit_idx), n_tune_val=len(val_idx))
        return split, np.array(fit_idx), np.array(val_idx), np.flatnonzero(split.ravel() == sp.TEST), info, None
    s = cfg["spatial_split"]
    block = block or s["block"]
    split, assign, info = sp.spatial_split(gt, block, seed, s["buffer"], tuple(s["targets"]),
                                           s["n_trials"], s["min_class_px"])
    info["verification"] = sp.verify_spatial(split, assign, gt, block, s["buffer"], s["min_class_px"])
    info["kind"] = "spatial"
    f = split.ravel()
    return (split, np.flatnonzero(f == sp.TRAIN), np.flatnonzero(f == sp.VAL),
            np.flatnonzero(f == sp.TEST), info, assign)


def run_svm(scene, X, y, idx, cfg, seed, log, prep_desc):
    fit_idx, val_idx, test_idx = idx
    s = cfg["svm"]
    cap = s["max_train"] if len(fit_idx) > s["max_train"] else None
    t0 = time.time()
    best, table = cm.tune_svm(X[fit_idx], y[fit_idx], X[val_idx], y[val_idx], s["grid_C"],
                              s["grid_gamma"], seed, cap)
    log(f"    svm best {best} val_oa={max(t['val_oa'] for t in table):.3f}")
    return best, table, time.time() - t0


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["indian_pines", "salinas"])
    ap.add_argument("--splits", nargs="+", default=["random", "spatial"])
    ap.add_argument("--models", nargs="+", default=["svm", "cnn3d"])
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--drop_ranges_nm", type=str, default=None, help='e.g. "1300,1500;1800,1950"')
    ap.add_argument("--tag", default="")
    ap.add_argument("--no_save_models", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    seeds = a.seeds or cfg["seeds"]
    drop = cfg["preprocessing"]["drop_ranges_nm"]
    if a.drop_ranges_nm:
        drop = [[float(v) for v in r.split(",")] for r in a.drop_ranges_nm.split(";")]
    log = Logger("train_crop_models" + (f"_{a.tag}" if a.tag else ""))
    log(f"env {environment()}")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    for scene_name in a.scenes:
        scene = load_scene(scene_name)
        H, W, B = scene.cube.shape
        K = len(scene.class_names)
        Xall = scene.cube.reshape(-1, B)
        yall = scene.gt.ravel() - 1                     # -1 = unlabelled
        rows, cols = np.unravel_index(np.arange(H * W), (H, W))
        for kind in a.splits:
            for seed in seeds:
                split, fit_idx, val_idx, test_idx, info, assign = make_split(scene, kind, seed, cfg, a.block)
                tag = f"{scene_name}_{kind}_s{seed}" + (f"_{a.tag}" if a.tag else "")
                np.savez_compressed(_mk(EXP / "splits") / f"{tag}.npz",
                                    split=split, assign=assign if assign is not None else np.zeros(1))
                n_train = int((split == sp.TRAIN).sum())
                log(f"{tag}: train={n_train} (fit {len(fit_idx)}) val={len(val_idx)} test={len(test_idx)}")
                prep = SpectralPreprocessor(drop, None).fit(Xall[fit_idx], scene.centres, seed)
                Xs = prep.transform(Xall)
                for model_name in a.models:
                    t0 = time.time()
                    out = {"scene": scene_name, "split": kind, "model": model_name, "seed": seed,
                           "split_info": info, "preprocessing": prep.describe(scene.centres),
                           "n_train_px": n_train, "n_test_px": int(len(test_idx))}
                    if model_name == "svm":
                        best, table, t_tune = run_svm(scene, Xs, yall, (fit_idx, val_idx, test_idx), cfg, seed, log, None)
                        # final fit: random split refits on all training pixels; spatial on train only
                        final_idx = np.flatnonzero(split.ravel() == sp.TRAIN)
                        cap = cfg["svm"]["max_train"] if len(final_idx) > cfg["svm"]["max_train"] else None
                        t1 = time.time()
                        model = cm.fit_svm(Xs[final_idx], yall[final_idx], best["C"], best["gamma"], seed, cap)
                        t_fit = time.time() - t1
                        t1 = time.time()
                        P = cm.svm_proba(model, Xs[test_idx], K)
                        t_inf = time.time() - t1
                        out.update(hyperparams=best, tuning_table=table, train_time_s=t_tune + t_fit,
                                   inference_time_s=t_inf, svm_train_cap=cap,
                                   n_svm_fit_px=int(min(len(final_idx), cap or 10**9)))
                        if seed == seeds[0] and not a.no_save_models:
                            _mk(MODELS)
                            with open(MODELS / f"{scene_name}_{kind}_svm{'_' + a.tag if a.tag else ''}.pkl", "wb") as f:
                                pickle.dump({"model": model, "prep": prep, "centres": scene.centres,
                                             "class_names": scene.class_names}, f)
                    else:
                        c = cfg["cnn"]
                        prep_c = SpectralPreprocessor(drop, cfg["preprocessing"]["n_pca_cnn"]).fit(
                            Xall[fit_idx], scene.centres, seed)
                        Z = prep_c.transform(scene.cube)
                        P_tr = cm.extract_patches(Z, rows[fit_idx], cols[fit_idx], c["patch"])
                        P_va = cm.extract_patches(Z, rows[val_idx], cols[val_idx], c["patch"])
                        best_m, best_loss, tuning = None, np.inf, []
                        for lr in c["grid_lr"]:
                            m, hist = cm.train_cnn(P_tr, yall[fit_idx], P_va, yall[val_idx], K, seed, lr,
                                                   c["epochs"], c["batch"], c["patience"], c["weight_decay"],
                                                   c["dropout"], log=log)
                            vl = -max(h["val_oa"] - 1e-3 * h["val_loss"] for h in hist)
                            tuning.append({"lr": lr, "best_val_loss": vl, "epochs_run": len(hist),
                                           "best_val_oa": max(h["val_oa"] for h in hist)})
                            if vl < best_loss:
                                best_m, best_loss, best_lr, best_hist = m, vl, lr, hist
                        t_train = time.time() - t0
                        T = cm.fit_temperature(cm.cnn_logits(best_m, P_va), yall[val_idx])
                        t1 = time.time()
                        P_te = cm.extract_patches(Z, rows[test_idx], cols[test_idx], c["patch"])
                        logits = cm.cnn_logits(best_m, P_te)
                        t_inf = time.time() - t1
                        P = cm.softmax(logits)
                        cal = evaluate(yall[test_idx], cm.softmax(logits, T), K)
                        out.update(hyperparams={"lr": best_lr, "patch": c["patch"], "n_pca": cfg["preprocessing"]["n_pca_cnn"],
                                                "dropout": c["dropout"], "weight_decay": c["weight_decay"]},
                                   tuning_table=tuning, history=best_hist, temperature=T,
                                   calibrated={"ece": cal["ece"], "reliability": cal["reliability"]},
                                   train_time_s=t_train, inference_time_s=t_inf,
                                   cnn_preprocessing=prep_c.describe(scene.centres),
                                   n_params=int(sum(p.numel() for p in best_m.parameters())))
                        if seed == seeds[0] and not a.no_save_models:
                            _mk(MODELS)
                            torch.save({"state_dict": best_m.state_dict(), "n_bands": Z.shape[2], "n_classes": K,
                                        "patch": c["patch"], "dropout": c["dropout"], "temperature": T},
                                       MODELS / f"{scene_name}_{kind}_cnn3d{'_' + a.tag if a.tag else ''}.pt")
                            with open(MODELS / f"{scene_name}_{kind}_cnn3d_prep{'_' + a.tag if a.tag else ''}.pkl", "wb") as f:
                                pickle.dump({"prep": prep_c, "centres": scene.centres, "class_names": scene.class_names}, f)
                    met = evaluate(yall[test_idx], P, K)
                    out["metrics"] = met
                    log(f"  {tag} {model_name}: OA={met['oa']:.4f} AA={met['aa']:.4f} k={met['kappa']:.4f} "
                        f"F1={met['macro_f1']:.4f} ECE={met['ece']:.4f} ({time.time()-t0:.0f}s)")
                    save_json(out, EXP / "runs" / f"{scene_name}_{kind}_{model_name}_s{seed}"
                              f"{'_' + a.tag if a.tag else ''}.json")


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


if __name__ == "__main__":
    main()

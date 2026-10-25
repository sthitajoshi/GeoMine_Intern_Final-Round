"""Part B: Cuprite mineral mapping without ground truth.

Pipeline: library convolution -> continuum removal (2.00-2.45 um) -> band depth / position ->
SAM on absorption shape (+ feature-fit residual) -> synthetic-trained MLP with unknown class ->
no-label validation (agreement, example spectra, residuals, sensitivity, convolution cross-check).

Outputs: experiments/mineral/*.json|npz and models/cuprite_mlp.pkl.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import continuum_removal as crm  # noqa: E402
from src import mineral_library as ml  # noqa: E402
from src import mineral_models as mm  # noqa: E402
from src import sam as samm  # noqa: E402
from src import synthetic_data as syn  # noqa: E402
from src.data_loader import load_scene  # noqa: E402
from src.utils import EXP, MODELS, Logger, load_config, save_json  # noqa: E402

OUTD = EXP / "mineral"


def band_quality(scene) -> np.ndarray:
    """Bands usable for mapping: not zero-filled by the reflectance retrieval, not in deep water
    absorptions, and above 400 nm."""
    mean = scene.cube[scene.valid].mean(0)
    ok = mean > 0.01
    c = scene.centres
    for lo, hi in [(1340, 1440), (1790, 1960), (0, 400), (2460, 3000)]:
        ok &= ~((c >= lo) & (c <= hi))
    return ok


def sam_on_cr(F, ref_F, names, thr, min_depth):
    """SAM on absorption vectors (1 - CR); pixels with no absorption > min_depth are 'none'."""
    lab, best, second, _, ang = samm.sam_classify(F, ref_F, thr)
    weak = F.max(1) < min_depth
    lab = np.where(weak, -1, lab)
    return lab, best, second, ang


def main():
    cfg = load_config()["mineral"]
    log = Logger("run_mineral_mapping")
    OUTD.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    scene = load_scene("cuprite")
    H, W, B = scene.cube.shape
    ok = band_quality(scene)
    lo, hi = cfg["cr_window_nm"]
    log(f"cuprite {scene.cube.shape}; usable bands {ok.sum()}; wavelength source: {scene.wavelength_source}")

    # ---------------------------------------------------------------- library
    lib, rec_names = ml.build_library(scene.centres, scene.fwhm, "convolved")
    lib98, _ = ml.build_library(scene.centres, scene.fwhm, "av98")
    win = (scene.centres >= lo) & (scene.centres <= hi) & ok
    conv_check = {}
    for m in lib:
        d = lib[m][:, win] - lib98[m][:, win]
        conv_check[m] = {"rms_diff_reflectance": float(np.sqrt(np.nanmean(d ** 2))),
                         "max_abs_diff": float(np.nanmax(np.abs(d)))}
    log(f"own Gaussian convolution vs USGS AV98 convolution (SWIR window): {conv_check}")

    # ---------------------------------------------------------------- noise / SNR
    sigma = syn.estimate_noise(scene.cube)
    mean_sig = scene.cube[scene.valid].mean(0)
    snr = np.where(sigma > 0, mean_sig / np.maximum(sigma, 1e-6), 0)
    log(f"estimated SNR median in window: {np.median(snr[win]):.0f} (sigma median {np.median(sigma[win]):.4f})")

    # ---------------------------------------------------------------- continuum removal
    cache = EXP / "cache" / "cuprite_cr.npz"
    pix = scene.cube.reshape(-1, B)
    valid = scene.valid.ravel() & (pix[:, win].mean(1) > 0.02) & (pix[:, win].max(1) < 1.0)
    if cache.exists():
        z = np.load(cache)
        CR, idx = z["cr"], z["idx"]
        log("loaded cached continuum removal")
    else:
        t1 = time.time()
        cr_v, idx = crm.continuum_remove(pix[valid], scene.centres, lo, hi, ok)
        CR = np.full((H * W, cr_v.shape[1]), np.nan, np.float32)
        CR[valid] = cr_v
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, cr=CR, idx=idx)
        log(f"continuum removal of {valid.sum()} pixels: {time.time()-t1:.0f}s")
    x = scene.centres[idx]
    F = mm.features(CR)

    # band depth / position maps
    depth, pos = {}, {}
    for m, (c, h) in crm.FEATURES.items():
        d, p = crm.feature_params(np.nan_to_num(CR, nan=1.0), x, c, h)
        d[~valid] = np.nan
        depth[m], pos[m] = d.reshape(H, W), np.where(valid, p, np.nan).reshape(H, W)

    # library in the same feature space
    ref_cr = {m: crm.continuum_remove(lib[m], scene.centres, lo, hi, ok)[0] for m in lib}
    ref_F = {m: mm.features(ref_cr[m]) for m in ml.TARGETS}
    lib_angles = {a: {b: float(samm.spectral_angle(ref_F[a], ref_F[b]).min()) for b in ml.TARGETS}
                  for a in ml.TARGETS}

    # ---------------------------------------------------------------- synthetic data
    sig_scale = 1.0
    Xs, ys, fs, classes = syn.generate(lib, scene.centres, sigma * sig_scale, cfg["synthetic_n_per_class"], 0,
                                       cfg["synthetic_f_min"])
    Xt, yt, ft, _ = syn.generate(lib, scene.centres, sigma * sig_scale, 800, 1, cfg["synthetic_f_min"])
    Fs = mm.features(crm.continuum_remove(Xs, scene.centres, lo, hi, ok)[0])
    Ft = mm.features(crm.continuum_remove(Xt, scene.centres, lo, hi, ok)[0])
    U = len(ml.TARGETS)

    # SAM threshold chosen on *synthetic* labelled data (macro-F1), never on the real scene
    sweep = []
    for thr in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
        lab_s, *_ = sam_on_cr(Fs, ref_F, ml.TARGETS, thr, cfg["min_band_depth"])
        lab_s = np.where(lab_s < 0, U, lab_s)
        sweep.append({"thr": thr, "synthetic_macro_f1": float(f1_score(ys, lab_s, average="macro"))})
    # knee rule: the smallest angle reaching 95 % of the best synthetic macro-F1. Synthetic data
    # is cleaner than the real scene, so the arg-max sits at the loosest threshold; a tighter angle
    # trades a little synthetic recall for fewer false matches on real pixels.
    top = max(d["synthetic_macro_f1"] for d in sweep)
    thr = min(d["thr"] for d in sweep if d["synthetic_macro_f1"] >= 0.95 * top)
    log(f"SAM threshold sweep (synthetic train set): {sweep} -> {thr} rad")
    lab_t, *_ = sam_on_cr(Ft, ref_F, ml.TARGETS, thr, cfg["min_band_depth"])
    lab_t = np.where(lab_t < 0, U, lab_t)
    sam_syn = {"macro_f1": float(f1_score(yt, lab_t, average="macro")), "oa": float((lab_t == yt).mean()),
               "confusion": confusion_matrix(yt, lab_t, labels=range(U + 1)).tolist()}

    # ---------------------------------------------------------------- ML model
    t1 = time.time()
    model = mm.train_mlp(Fs, ys, 0)
    lab_ml_t, conf_t, _ = mm.predict_with_reject(model, Ft, U, cfg["ml_reject_prob"])
    ml_syn = {"macro_f1": float(f1_score(yt, lab_ml_t, average="macro")), "oa": float((lab_ml_t == yt).mean()),
              "confusion": confusion_matrix(yt, lab_ml_t, labels=range(U + 1)).tolist(),
              "per_class_f1": f1_score(yt, lab_ml_t, average=None, labels=range(U + 1)).tolist()}
    # detection rate vs target fraction (synthetic): where does the model stop seeing the mineral?
    frac_bins = [0.3, 0.45, 0.6, 0.8, 1.01]
    det = []
    for a_, b_ in zip(frac_bins[:-1], frac_bins[1:]):
        m = (yt < U) & (ft >= a_) & (ft < b_)
        det.append({"frac": f"{a_:.2f}-{min(b_,1):.2f}", "recall": float((lab_ml_t[m] == yt[m]).mean())})
    log(f"MLP trained in {time.time()-t1:.0f}s; synthetic test OA={ml_syn['oa']:.3f} macroF1={ml_syn['macro_f1']:.3f}; SAM synthetic OA={sam_syn['oa']:.3f}")
    with open(MODELS / "cuprite_mlp.pkl", "wb") as f:
        pickle.dump({"model": model, "classes": classes, "window_idx": idx, "band_ok": ok,
                     "cr_window_nm": [lo, hi], "reject_prob": cfg["ml_reject_prob"]}, f)

    # noise-robustness of the synthetic model (sim-to-real proxy): test on 2x and 3x noise
    noise_rob = []
    for k in [0.5, 2.0, 3.0]:
        Xk, yk, _, _ = syn.generate(lib, scene.centres, sigma * k, 400, 7, cfg["synthetic_f_min"])
        Fk = mm.features(crm.continuum_remove(Xk, scene.centres, lo, hi, ok)[0])
        lk, _, _ = mm.predict_with_reject(model, Fk, U, cfg["ml_reject_prob"])
        noise_rob.append({"noise_x": k, "oa": float((lk == yk).mean()), "macro_f1": float(f1_score(yk, lk, average="macro"))})

    # ---------------------------------------------------------------- apply to the real scene
    vi = np.flatnonzero(valid)
    lab_sam = np.full(H * W, -2, np.int64)       # -2 = invalid / masked, -1 = unknown
    best_ang = np.full(H * W, np.nan, np.float32)
    margin = np.full(H * W, np.nan, np.float32)
    ls, b_, s_, ang = sam_on_cr(F[vi], ref_F, ml.TARGETS, thr, cfg["min_band_depth"])
    lab_sam[vi], best_ang[vi], margin[vi] = ls, b_, s_ - b_
    lab_ml = np.full(H * W, -2, np.int64)
    conf_ml = np.full(H * W, np.nan, np.float32)
    P_ml = np.zeros((H * W, U + 1), np.float32)
    lm, cf, P = mm.predict_with_reject(model, F[vi], U, cfg["ml_reject_prob"])
    lab_ml[vi] = np.where(lm == U, -1, lm)
    conf_ml[vi], P_ml[vi] = cf, P

    # feature-fit residual / relative abundance for the SAM best match
    fit_scale = np.full(H * W, np.nan, np.float32)
    fit_rms = np.full(H * W, np.nan, np.float32)
    fit_r2 = np.full(H * W, np.nan, np.float32)
    best_variant = {}
    for k, m in enumerate(ml.TARGETS):
        sel = lab_sam == k
        if sel.any():
            angs = samm.spectral_angle(F[sel], ref_F[m])
            v = angs.argmin(1)
            for j in np.unique(v):
                s2 = np.flatnonzero(sel)[v == j]
                a_, r_, q_ = samm.feature_fit(CR[s2], ref_cr[m][j])
                fit_scale[s2], fit_rms[s2], fit_r2[s2] = a_, r_, q_
            best_variant[m] = np.bincount(v, minlength=len(ref_F[m])).tolist()

    # ---------------------------------------------------------------- no-label validation
    both = (lab_sam > -2)
    a_s = np.where(lab_sam[both] < 0, U, lab_sam[both])
    a_m = np.where(lab_ml[both] < 0, U, lab_ml[both])
    agree = {"overall_agreement": float((a_s == a_m).mean()),
             "cohen_kappa_sam_vs_ml": float(cohen_kappa_score(a_s, a_m)),
             "confusion_sam_rows_ml_cols": confusion_matrix(a_s, a_m, labels=range(U + 1)).tolist(),
             "labels": ml.TARGETS + ["unknown"]}
    mapped = a_s < U
    agree["agreement_where_sam_mapped"] = float((a_s[mapped] == a_m[mapped]).mean()) if mapped.any() else None
    counts = {"sam": {n: int((a_s == i).sum()) for i, n in enumerate(agree["labels"])},
              "ml": {n: int((a_m == i).sum()) for i, n in enumerate(agree["labels"])}}
    log(f"SAM vs ML agreement {agree['overall_agreement']:.3f}, kappa {agree['cohen_kappa_sam_vs_ml']:.3f}")
    log(f"pixel counts {counts}")

    # threshold / gate sensitivity on the real scene (how much the map changes, not accuracy)
    sens = []
    for t_ in [thr * 0.67, thr, thr * 1.33]:
        for d_ in [0.01, cfg["min_band_depth"], 0.04]:
            l2, *_ = sam_on_cr(F[vi], ref_F, ml.TARGETS, t_, d_)
            sens.append({"thr": round(float(t_), 3), "min_depth": d_,
                         "frac_mapped": float((l2 >= 0).mean()),
                         "per_class_px": {n: int((l2 == i).sum()) for i, n in enumerate(ml.TARGETS)},
                         "agreement_with_default": float((l2 == ls).mean())})

    # example pixels: highest-confidence ML pixel per mineral that SAM agrees with
    examples = {}
    for k, m in enumerate(ml.TARGETS):
        cand = np.flatnonzero((lab_ml == k) & (lab_sam == k))
        if cand.size == 0:
            cand = np.flatnonzero(lab_ml == k)
        if cand.size:
            j = cand[np.argsort(-conf_ml[cand])[:5]]
            examples[m] = {"pixels_rc": [list(map(int, np.unravel_index(q, (H, W)))) for q in j],
                           "ml_conf": conf_ml[j].tolist(), "sam_angle": best_ang[j].tolist(),
                           "fit_rms": fit_rms[j].tolist()}

    # residual summary per class
    resid = {m: {"median_fit_rms": float(np.nanmedian(fit_rms[lab_sam == k])) if (lab_sam == k).any() else None,
                 "median_r2": float(np.nanmedian(fit_r2[lab_sam == k])) if (lab_sam == k).any() else None,
                 "median_depth": float(np.nanmedian(depth[m].ravel()[lab_sam == k])) if (lab_sam == k).any() else None,
                 "median_position_nm": float(np.nanmedian(pos[m].ravel()[lab_sam == k])) if (lab_sam == k).any() else None}
             for k, m in enumerate(ml.TARGETS)}

    np.savez_compressed(OUTD / "cuprite_maps.npz", lab_sam=lab_sam.reshape(H, W), lab_ml=lab_ml.reshape(H, W),
                        sam_angle=best_ang.reshape(H, W), sam_margin=margin.reshape(H, W),
                        ml_conf=conf_ml.reshape(H, W), ml_proba=P_ml.reshape(H, W, U + 1).astype(np.float16),
                        fit_scale=fit_scale.reshape(H, W), fit_rms=fit_rms.reshape(H, W), fit_r2=fit_r2.reshape(H, W),
                        **{f"depth_{m}": depth[m].astype(np.float32) for m in depth},
                        **{f"pos_{m}": pos[m].astype(np.float32) for m in pos},
                        cr_x=x, valid=valid.reshape(H, W))
    np.savez_compressed(OUTD / "library.npz", centres=scene.centres, band_ok=ok,
                        **{f"conv_{m}": lib[m] for m in lib}, **{f"av98_{m}": lib98[m] for m in lib98},
                        **{f"cr_{m}": ref_cr[m] for m in ref_cr}, sigma=sigma, snr=snr)
    save_json({"wavelength_source": scene.wavelength_source, "wavelength_exact": scene.wavelength_exact,
               "usable_bands": int(ok.sum()), "cr_window_nm": [lo, hi], "n_cr_bands": int(len(idx)),
               "records": rec_names, "convolution_check": conv_check,
               "snr_window_median": float(np.median(snr[win])), "sigma_window_median": float(np.median(sigma[win])),
               "library_angle_between_targets_rad": lib_angles, "sam_threshold_rad": thr,
               "sam_threshold_sweep_synthetic": sweep, "min_band_depth": cfg["min_band_depth"],
               "sam_synthetic": sam_syn, "ml_synthetic": ml_syn, "ml_detection_vs_fraction": det,
               "ml_noise_robustness": noise_rob, "agreement": agree, "pixel_counts": counts,
               "sensitivity": sens, "examples": examples, "residuals": resid, "best_variant_counts": best_variant,
               "n_valid_pixels": int(valid.sum()), "runtime_s": time.time() - t0},
              OUTD / "mineral_results.json")
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

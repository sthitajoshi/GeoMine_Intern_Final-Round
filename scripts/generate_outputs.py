"""Aggregate experiment JSON into results.csv and render every map and figure under outputs/.

Usage: python scripts/generate_outputs.py
"""
from __future__ import annotations

import csv
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402

from src import indices, inference  # noqa: E402
from src import visualization as vz  # noqa: E402
from src.data_loader import load_scene, true_colour  # noqa: E402
from src.mineral_library import TARGETS  # noqa: E402
from src.utils import EXP, OUT, ROOT, Logger, load_json, save_json  # noqa: E402

REQUIRED = [("indian_pines", "random", "svm"), ("indian_pines", "spatial", "svm"),
            ("indian_pines", "spatial", "cnn3d"), ("salinas", "spatial", "cnn3d")]
COLS = ["scene", "split", "model", "oa_mean", "oa_std", "aa_mean", "kappa_mean", "macro_f1_mean",
        "ece_mean", "n_train_px", "n_test_px"]
FIG = OUT / "figures"


def load_runs():
    runs = {}
    for f in glob.glob(str(EXP / "runs" / "*.json")):
        if Path(f).stem.count("_s") and not Path(f).stem.split("_s")[-1].isdigit():
            continue                                         # tagged ablation runs
        d = json.load(open(f))
        runs.setdefault((d["scene"], d["split"], d["model"]), []).append(d)
    return runs


def n_fit(r):
    """Pixels the model was actually fitted on: the SVM may be capped (max_train) below the split's
    training-pixel count; everything else uses the full training split."""
    return int(r.get("n_svm_fit_px") or r["n_train_px"])


def write_results(runs, log):
    keys = REQUIRED + sorted(k for k in runs if k not in REQUIRED)
    rows, ext, per_seed = [], [], []
    for k in keys:
        rs = sorted(runs.get(k, []), key=lambda d: d["seed"])
        if len(rs) < 3:
            log(f"WARNING {k}: only {len(rs)} seeds - row left blank")
            rows.append(dict(zip(COLS, list(k) + [""] * 8)))
            continue
        m = {n: np.array([r["metrics"][n] for r in rs]) for n in ("oa", "aa", "kappa", "macro_f1", "ece")}
        rows.append({"scene": k[0], "split": k[1], "model": k[2],
                     "oa_mean": f"{m['oa'].mean():.4f}", "oa_std": f"{m['oa'].std(ddof=1):.4f}",
                     "aa_mean": f"{m['aa'].mean():.4f}", "kappa_mean": f"{m['kappa'].mean():.4f}",
                     "macro_f1_mean": f"{m['macro_f1'].mean():.4f}", "ece_mean": f"{m['ece'].mean():.4f}",
                     "n_train_px": int(round(np.mean([n_fit(r) for r in rs]))),
                     "n_test_px": int(round(np.mean([r["n_test_px"] for r in rs])))})
        e = {"scene": k[0], "split": k[1], "model": k[2]}
        for n, v in m.items():
            e[f"{n}_mean"], e[f"{n}_std"] = round(float(v.mean()), 4), round(float(v.std(ddof=1)), 4)
        if k[2] == "cnn3d":
            ec = np.array([r["calibrated"]["ece"] for r in rs])
            e["ece_temp_scaled_mean"], e["ece_temp_scaled_std"] = round(float(ec.mean()), 4), round(float(ec.std(ddof=1)), 4)
        e["train_time_s_mean"] = round(float(np.mean([r["train_time_s"] for r in rs])), 1)
        e["inference_time_s_mean"] = round(float(np.mean([r["inference_time_s"] for r in rs])), 2)
        e["n_train_px_per_seed"] = [n_fit(r) for r in rs]
        e["n_train_split_px_per_seed"] = [r["n_train_px"] for r in rs]
        e["n_test_px_per_seed"] = [r["n_test_px"] for r in rs]
        e["hyperparams_per_seed"] = [r["hyperparams"] for r in rs]
        ext.append(e)
        for r in rs:
            per_seed.append({"scene": k[0], "split": k[1], "model": k[2], "seed": r["seed"],
                             **{n: round(r["metrics"][n], 4) for n in ("oa", "aa", "kappa", "macro_f1", "ece")},
                             "n_train_px": n_fit(r), "n_train_split_px": r["n_train_px"], "n_test_px": r["n_test_px"],
                             "hyperparams": json.dumps(r["hyperparams"])})
    with open(ROOT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        w.writerows(rows)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    with open(OUT / "reports" / "results_per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(per_seed[0]))
        w.writeheader()
        w.writerows(per_seed)
    save_json(ext, OUT / "reports" / "results_extended.json")
    log("results.csv written:\n" + (ROOT / "results.csv").read_text())
    return ext


def per_class_table(runs, scene):
    names = load_scene(scene).class_names
    out = []
    for k, rs in runs.items():
        if k[0] != scene:
            continue
        f1 = np.array([r["metrics"]["per_class_f1"] for r in rs], float)
        sups = np.array([r["metrics"]["support"] for r in rs])
        f1[sups == 0] = np.nan                       # class absent from that seed's test set: undefined, not 0
        sup = sups[0]
        for i, n in enumerate(names):
            v = f1[:, i][np.isfinite(f1[:, i])]
            out.append({"scene": scene, "split": k[1], "model": k[2], "class": n,
                        "f1_mean": round(float(v.mean()), 4) if v.size else "",
                        "f1_std": round(float(v.std(ddof=1)), 4) if v.size > 1 else "",
                        "n_seeds_in_test": int(v.size), "test_support_seed0": int(sup[i])})
    return out


def fig_gap(ext):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels, x = [], 0
    for scene in ("indian_pines", "salinas"):
        for model in ("svm", "cnn3d"):
            for j, split in enumerate(("random", "spatial")):
                e = next((e for e in ext if (e["scene"], e["split"], e["model"]) == (scene, split, model)), None)
                if e:
                    ax.bar(x + j * 0.38, e["oa_mean"], 0.36, yerr=e["oa_std"], color=["#4c72b0", "#dd8452"][j],
                           label=split if x == 0 else None, capsize=3)
            labels.append(f"{scene.replace('_', ' ')}\n{model}")
            x += 1
    ax.set_xticks(np.arange(x) + 0.19)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Overall accuracy (mean ± sd, 3 seeds)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "random_vs_spatial_gap.png", dpi=150)
    plt.close(fig)


def fig_reliability(runs):
    keys = [k for k in REQUIRED if k in runs] + [("indian_pines", "random", "cnn3d")]
    keys = [k for k in keys if k in runs]
    fig, axs = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.2))
    for ax, k in zip(np.atleast_1d(axs), keys):
        r = sorted(runs[k], key=lambda d: d["seed"])[0]
        for rel, lab, c in [(r["metrics"]["reliability"], f"raw ECE={r['metrics']['ece']:.3f}", "C0")] + (
                [(r["calibrated"]["reliability"], f"temp-scaled ECE={r['calibrated']['ece']:.3f}", "C3")] if "calibrated" in r else []):
            e = np.array(rel["edges"])
            ax.plot((e[:-1] + e[1:]) / 2, np.array(rel["acc"], float), "o-", ms=3, color=c, label=lab)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(" / ".join(k) + " (seed 0)", fontsize=8)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "reliability_diagrams.png", dpi=150)
    plt.close(fig)


def fig_confusions(runs):
    for k in [("indian_pines", "spatial", "svm"), ("indian_pines", "spatial", "cnn3d"),
              ("indian_pines", "random", "svm"), ("salinas", "spatial", "cnn3d")]:
        if k not in runs:
            continue
        names = load_scene(k[0]).class_names
        cm = np.sum([np.array(r["metrics"]["confusion"]) for r in runs[k]], 0)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                if cm[i, j]:
                    ax.text(j, i, f"{cmn[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=5,
                            color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticklabels([f"{n} ({cm[i].sum()})" for i, n in enumerate(names)], fontsize=6)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true (test px, summed over 3 seeds)")
        ax.set_title("Row-normalised confusion: " + " / ".join(k), fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / f"confusion_{'_'.join(k)}.png", dpi=150)
        plt.close(fig)


def fig_splits():
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    cm_ = plt.matplotlib.colors.ListedColormap(["#ffffff", "#2ca02c", "#1f77b4", "#d62728"])
    for ax, (scene, kind) in zip(axs, [("indian_pines", "random"), ("indian_pines", "spatial"),
                                        ("salinas", "random"), ("salinas", "spatial")]):
        p = EXP / "splits" / f"{scene}_{kind}_s0.npz"
        if p.exists():
            ax.imshow(np.load(p)["split"], cmap=cm_, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(f"{scene} {kind} (seed 0)", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=l) for l, c in zip(["unused / buffer", "train", "val", "test"],
                                                                  ["#dddddd", "#2ca02c", "#1f77b4", "#d62728"])],
               loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "split_maps.png", dpi=150)
    plt.close(fig)


def crop_maps(scene_name, log):
    scene = load_scene(scene_name)
    K = len(scene.class_names)
    pal = vz.class_palette(K)
    legend = [(n, tuple(pal[i])) for i, n in enumerate(scene.class_names)]
    cdir, fdir = OUT / "class_maps", OUT / "confidence_maps"
    idir = OUT / "index_maps"
    for d in (cdir, fdir, idir):
        d.mkdir(parents=True, exist_ok=True)
    rgb = true_colour(scene)
    plt.imsave(OUT / "class_maps" / f"{scene_name}_true_colour.png", rgb)
    vz.save_map_png(cdir / f"{scene_name}_ground_truth.png", vz.colourise(scene.gt - 1, pal), f"{scene_name}: ground truth (labelled pixels only)", legend)
    summary = {}
    for model in ("cnn3d", "svm"):
        lab, P = inference.predict_crop(scene, model, "spatial")
        conf = P.max(-1)
        ent = -(P * np.log(P + 1e-12)).sum(-1) / np.log(K)
        note = "Model trained on the spatial split, seed 0. Unlabelled pixels are extrapolation (no ground truth)."
        vz.save_map_png(cdir / f"{scene_name}_{model}_class_map.png", vz.colourise(lab, pal),
                        f"{scene_name}: {model} full-scene class map", legend, note=note)
        vz.save_map_png(fdir / f"{scene_name}_{model}_confidence.png", vz.scalar_rgb(conf, "magma", 0, 1),
                        f"{scene_name}: {model} max-probability confidence", cbar=("magma", 0, 1, "confidence"),
                        note="CNN: temperature-scaled softmax (T fitted on validation). SVM: Platt-scaled probabilities.")
        vz.save_map_png(fdir / f"{scene_name}_{model}_entropy.png", vz.scalar_rgb(ent, "viridis", 0, 1),
                        f"{scene_name}: {model} normalised predictive entropy", cbar=("viridis", 0, 1, "uncertainty"))
        np.savez_compressed(cdir / f"{scene_name}_{model}_maps.npz", labels=lab.astype(np.int16),
                            confidence=conf.astype(np.float32), entropy=ent.astype(np.float32))
        lab_mask = scene.gt > 0
        summary[model] = {"mean_conf_all": float(conf.mean()), "mean_conf_labelled": float(conf[lab_mask].mean()),
                          "pixel_counts": np.bincount(lab.ravel(), minlength=K).tolist()}
        if model == "cnn3d":
            crop_ids = [i for i, n in enumerate(scene.class_names) if n in scene.crop_classes]
            crop_mask = np.isin(lab, crop_ids)
    index_formulas = {}
    for name, fn, cmap_, vr in [("ndvi", indices.ndvi, "RdYlGn", (-0.1, 0.9)), ("ndre", indices.ndre, "RdYlGn", (-0.05, 0.6)),
                                ("red_edge_position", indices.red_edge_position, "viridis", (700, 740))]:
        v = fn(scene.cube, scene.centres)
        formula = indices.describe(name, scene.centres)
        vz.save_map_png(idir / f"{scene_name}_{name}_all.png", vz.scalar_rgb(v, cmap_, *vr),
                        f"{scene_name}: {name.upper()} (all pixels)", cbar=(cmap_, *vr, name),
                        note=f"{formula} (R<nm> = band centred at that wavelength). At-sensor values: relative, not absolute.")
        vm = np.where(crop_mask, v, np.nan)
        vz.save_map_png(idir / f"{scene_name}_{name}_crop_classes.png", vz.scalar_rgb(vm, cmap_, *vr),
                        f"{scene_name}: {name.upper()} on pixels predicted as crop classes", cbar=(cmap_, *vr, name),
                        note=f"{formula} (R<nm> = band centred at that wavelength).\n"
                             "Masked with the CNN3D class map (crop classes only). At-sensor values: relative, not absolute.")
        index_formulas[name] = formula
        np.save(idir / f"{scene_name}_{name}.npy", v.astype(np.float32))
    summary["index_formulas"] = index_formulas
    log(f"{scene_name} maps written")
    return summary


def fig_bands(log):
    p = EXP / "bands" / "band_experiments.json"
    if not p.exists():
        log("band experiments missing - skipping")
        return None
    d = load_json(p)
    c = np.array(d["centres"])
    o = np.argsort(c)
    fig, ax = plt.subplots(figsize=(9, 3.3))
    imps = np.array([d["seeds"][s]["importance"] for s in d["seeds"]])
    ax.plot(c[o], imps.mean(0)[o], "k-", lw=1, label="per-band permutation importance (mean of 3 seeds, validation)")
    w5 = d["seeds"]["0"]["importance_window5"]
    if w5:
        ax.plot(c[o], np.array(w5)[o], "C0-", lw=1, alpha=0.7, label="5-band window importance (seed 0)")
    for s in d["seeds"]:
        ax.plot(d["seeds"][s]["forward_picks_nm"], [-0.002 * (int(s) + 1)] * 15, "|", ms=10, color=f"C{int(s)+1}",
                label=f"15 forward-selected bands, seed {s}")
    for lo, hi, lab in [(680, 750, "red edge"), (950, 990, "water 970"), (1150, 1250, "water 1200"),
                        (2050, 2150, "cellulose/lignin"), (1400, 1450, "")]:
        ax.axvspan(lo, hi, color="g", alpha=0.08)
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.95 if lab else 0, lab, ha="center", fontsize=7, color="g")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("OA drop")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "band_importance.png", dpi=150)
    plt.close(fig)
    tab = {}
    for key in ("full", "selected15", "forward15", "s2", "random15"):
        v = np.array([[d["seeds"][s][key][m] for m in ("oa", "aa", "kappa", "macro_f1")] for s in d["seeds"]])
        tab[key] = {"oa_mean": v[:, 0].mean(), "oa_std": v[:, 0].std(ddof=1), "aa_mean": v[:, 1].mean(),
                    "kappa_mean": v[:, 2].mean(), "macro_f1_mean": v[:, 3].mean()}
    blocks = None
    bp = EXP / "bands" / "block_size_study.json"
    if bp.exists():
        b = load_json(bp)
        blocks = {k: {"oa_mean": float(np.mean([r["oa"] for r in v])), "oa_std": float(np.std([r["oa"] for r in v], ddof=1)),
                      "aa_mean": float(np.mean([r["aa"] for r in v])), "n_train_mean": float(np.mean([r["n_train"] for r in v])),
                      "missing_train_classes": [r["classes_missing_train"] for r in v]} for k, v in b.items()}
    save_json({"band_table": tab, "s2_report": d["s2_report"], "block_size": blocks}, OUT / "reports" / "band_summary.json")
    log(f"band table {tab}\nblock size {blocks}")
    return tab


def mineral_outputs(log):
    z = np.load(EXP / "mineral" / "cuprite_maps.npz")
    L = np.load(EXP / "mineral" / "library.npz")
    R = load_json(EXP / "mineral" / "mineral_results.json")
    md = OUT / "mineral_maps"
    md.mkdir(parents=True, exist_ok=True)
    scene = load_scene("cuprite")
    rgb = true_colour(scene)
    plt.imsave(md / "cuprite_true_colour.png", rgb)
    legend = [(m, vz.matplotlib.colors.to_rgb(vz.MINERAL_COLOURS[m])) for m in TARGETS] + [("unknown / none", (0.85,) * 3), ("masked", (0.15,) * 3)]
    vz.save_map_png(md / "cuprite_sam_class_map.png", vz.mineral_rgb(z["lab_sam"], TARGETS),
                    f"Cuprite: SAM on continuum-removed 2.0-2.45 um (thr {R['sam_threshold_rad']} rad)", legend)
    vz.save_map_png(md / "cuprite_ml_class_map.png", vz.mineral_rgb(z["lab_ml"], TARGETS),
                    "Cuprite: synthetic-trained MLP (reject p<0.5)", legend)
    vz.save_map_png(OUT / "confidence_maps" / "cuprite_ml_confidence.png", vz.scalar_rgb(z["ml_conf"], "magma", 0, 1),
                    "Cuprite: MLP max probability (NOT calibrated on real data)", cbar=("magma", 0, 1, "p(max)"))
    vz.save_map_png(OUT / "confidence_maps" / "cuprite_sam_match_quality.png",
                    vz.scalar_rgb(np.where(z["lab_sam"] >= 0, 1 - z["sam_angle"] / R["sam_threshold_rad"], np.nan), "viridis", 0, 1),
                    "Cuprite: SAM match quality = 1 - angle/threshold (mapped px)", cbar=("viridis", 0, 1, "match quality"))
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    for ax, m in zip(axs.flat, TARGETS):
        im = ax.imshow(z[f"depth_{m}"], cmap="inferno", vmin=0, vmax=0.25)
        c, h = __import__("src.continuum_removal", fromlist=["FEATURES"]).FEATURES[m]
        ax.set_title(f"{m}: band depth at {c}±{h} nm", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(md / "cuprite_band_depth_maps.png", dpi=120)
    plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    fe = __import__("src.continuum_removal", fromlist=["FEATURES"]).FEATURES
    for ax, m in zip(axs.flat, TARGETS):
        c, h = fe[m]
        v = np.where(z[f"depth_{m}"] >= 0.03, z[f"pos_{m}"], np.nan)   # position is meaningless without a feature
        im = ax.imshow(v, cmap="coolwarm", vmin=c - h, vmax=c + h)
        ax.set_title(f"{m}: minimum position (nm), depth >= 0.03", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(md / "cuprite_band_position_maps.png", dpi=120)
    plt.close(fig)
    for m in TARGETS:
        vz.save_map_png(md / f"cuprite_band_depth_{m}.png", vz.scalar_rgb(z[f"depth_{m}"], "inferno", 0, 0.25),
                        f"Cuprite: {m} diagnostic band depth", cbar=("inferno", 0, 0.25, "1 - CR"))
    ab = np.where(z["lab_sam"] >= 0, z["fit_scale"], np.nan)
    vz.save_map_png(md / "cuprite_relative_abundance.png", vz.scalar_rgb(ab, "viridis", 0, 1.2),
                    "Cuprite: feature-fit scale for the SAM mineral (relative abundance proxy)",
                    cbar=("viridis", 0, 1.2, "fitted depth / library depth"),
                    note="Not a calibrated abundance: depends on grain size, mixing type and the library sample.")
    # agreement matrix
    a = np.array(R["agreement"]["confusion_sam_rows_ml_cols"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(np.log1p(a), cmap="Blues")
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ax.text(j, i, str(a[i, j]), ha="center", va="center", fontsize=6)
    labs = R["agreement"]["labels"]
    ax.set_xticks(range(len(labs)))
    ax.set_yticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=60, fontsize=7)
    ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlabel("MLP (synthetic-trained)")
    ax.set_ylabel("SAM (physics)")
    ax.set_title(f"Pixel agreement, kappa={R['agreement']['cohen_kappa_sam_vs_ml']:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "mineral_agreement.png", dpi=150)
    plt.close(fig)
    # example spectra
    cr = np.load(EXP / "cache" / "cuprite_cr.npz")["cr"]
    x = z["cr_x"]
    H, W = z["lab_sam"].shape
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, m in zip(axs.flat, TARGETS):
        for r_ in L[f"cr_{m}"]:
            ax.plot(x, r_, "k--", lw=1, label="library (convolved, CR)")
        ex = R["examples"].get(m)
        if ex:
            for (i, j), cf, ang in list(zip(ex["pixels_rc"], ex["ml_conf"], ex["sam_angle"]))[:3]:
                ax.plot(x, cr[i * W + j], lw=1.2, label=f"px ({i},{j}) p={cf:.2f} angle={ang:.2f}")
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("nm")
        ax.legend(fontsize=6)
    fig.suptitle("Continuum-removed example pixels (highest MLP confidence, agreeing with SAM where possible) vs library", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "mineral_example_spectra.png", dpi=150)
    plt.close(fig)
    # library: own convolution vs USGS AV98
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    c = L["centres"]
    win = (c > 1950) & (c < 2500) & L["band_ok"]
    for ax, m in zip(axs.flat, TARGETS):
        ax.plot(c[win], L[f"conv_{m}"][0][win], label="our Gaussian convolution")
        ax.plot(c[win], L[f"av98_{m}"][0][win], "--", label="USGS AVIRIS-1998 convolution")
        ax.set_title(m)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "library_convolution_check.png", dpi=130)
    plt.close(fig)
    # synthetic confusion
    S = np.array(R["ml_synthetic"]["confusion"])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(S / S.sum(1, keepdims=True), cmap="Greens", vmin=0, vmax=1)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            ax.text(j, i, str(S[i, j]), ha="center", va="center", fontsize=6)
    labs = TARGETS + ["unknown"]
    ax.set_xticks(range(7)), ax.set_yticks(range(7))
    ax.set_xticklabels(labs, rotation=60, fontsize=7)
    ax.set_yticklabels(labs, fontsize=7)
    ax.set_title(f"MLP on held-out SYNTHETIC spectra (OA={R['ml_synthetic']['oa']:.3f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "mineral_synthetic_confusion.png", dpi=150)
    plt.close(fig)
    # USGS comparison: side-by-side with the published map if it has been placed in data/raw
    usgs = sorted((ROOT / "data" / "raw").glob("usgs_cuprite_map.*"))
    fig, axs = plt.subplots(1, 3 if usgs else 2, figsize=(15 if usgs else 10, 6))
    axs[0].imshow(vz.mineral_rgb(z["lab_sam"], TARGETS))
    axs[0].set_title("ours: SAM")
    axs[1].imshow(vz.mineral_rgb(z["lab_ml"], TARGETS))
    axs[1].set_title("ours: synthetic-trained MLP")
    if usgs:
        axs[2].imshow(plt.imread(usgs[0]))
        axs[2].set_title("USGS published AVIRIS mineral map (different extent/projection)")
    for a_ in axs:
        a_.set_xticks([]), a_.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "mineral_vs_usgs.png", dpi=120)
    plt.close(fig)
    log("mineral outputs written")


def main():
    warnings.filterwarnings("ignore")
    log = Logger("generate_outputs")
    for d in ("figures", "reports", "class_maps", "confidence_maps", "mineral_maps"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    ext = write_results(runs, log)
    pc = per_class_table(runs, "indian_pines") + per_class_table(runs, "salinas")
    with open(OUT / "reports" / "per_class_f1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(pc[0]))
        w.writeheader()
        w.writerows(pc)
    fig_gap(ext)
    fig_reliability(runs)
    fig_confusions(runs)
    fig_splits()
    summ = {s: crop_maps(s, log) for s in ("indian_pines", "salinas")}
    save_json(summ, OUT / "reports" / "map_summary.json")
    fig_bands(log)
    mineral_outputs(log)


if __name__ == "__main__":
    main()

"""Independent programmatic checks used by reports/final_audit.md.

Recomputes metrics from stored confusion matrices, re-aggregates results.csv from per-seed runs,
re-verifies every saved split, checks wavelength/band consistency and artefact presence.
Writes outputs/reports/audit_checks.json and exits non-zero if any check fails.

Usage: python scripts/audit_checks.py
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import splits as sp  # noqa: E402
from src import wavelengths as wl  # noqa: E402
from src.data_loader import load_scene  # noqa: E402
from src.utils import save_json  # noqa: E402

RES = {}
FAIL = []


def check(name, ok, detail):
    RES[name] = {"pass": bool(ok), "detail": detail}
    if not ok:
        FAIL.append(name)
    print(("PASS " if ok else "FAIL ") + name + ": " + (json.dumps(detail, default=str)[:300]))


def metrics_from_confusion():
    worst = 0.0
    n = 0
    for f in glob.glob(str(ROOT / "experiments" / "runs" / "*.json")):
        d = json.load(open(f))
        m = d["metrics"]
        cm = np.array(m["confusion"], float)
        tot = cm.sum()
        oa = np.trace(cm) / tot
        present = cm.sum(1) > 0
        rec = np.diag(cm)[present] / cm.sum(1)[present]
        aa = rec.mean()
        pe = (cm.sum(0) * cm.sum(1)).sum() / tot ** 2
        kappa = (oa - pe) / (1 - pe)
        prec = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(len(cm)), where=cm.sum(0) > 0)
        recall_all = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(len(cm)), where=cm.sum(1) > 0)
        f1 = np.divide(2 * prec * recall_all, prec + recall_all, out=np.zeros(len(cm)), where=(prec + recall_all) > 0)
        mf1 = f1[present].mean()
        diffs = [abs(oa - m["oa"]), abs(aa - m["aa"]), abs(kappa - m["kappa"]), abs(mf1 - m["macro_f1"])]
        worst = max(worst, max(diffs))
        # reliability bins must add up to the test count and their weighted gap equals the ECE
        rel = m["reliability"]
        cnt = np.array(rel["count"])
        acc = np.nan_to_num(np.array(rel["acc"], float))
        cf = np.nan_to_num(np.array(rel["conf"], float))
        ece = (cnt / cnt.sum() * np.abs(acc - cf)).sum()
        worst = max(worst, abs(ece - m["ece"]))
        assert cnt.sum() == tot == d["n_test_px"], f
        n += 1
    check("metrics_recomputed_from_confusion", worst < 1e-9, {"runs_checked": n, "max_abs_diff": worst})


def results_csv_from_runs():
    rows = list(csv.DictReader(open(ROOT / "results.csv")))
    hdr = open(ROOT / "results.csv").readline().strip().split(",")
    want = ["scene", "split", "model", "oa_mean", "oa_std", "aa_mean", "kappa_mean", "macro_f1_mean",
            "ece_mean", "n_train_px", "n_test_px"]
    check("results_csv_header_exact", hdr == want, hdr)
    bad = []
    for r in rows:
        runs = sorted(glob.glob(str(ROOT / "experiments" / "runs" / f"{r['scene']}_{r['split']}_{r['model']}_s[0-9].json")))
        seeds = sorted(json.load(open(f))["seed"] for f in runs)
        ms = [json.load(open(f))["metrics"] for f in runs]
        oa = np.array([m["oa"] for m in ms])
        exp = {"oa_mean": oa.mean(), "oa_std": oa.std(ddof=1), "aa_mean": np.mean([m["aa"] for m in ms]),
               "kappa_mean": np.mean([m["kappa"] for m in ms]), "macro_f1_mean": np.mean([m["macro_f1"] for m in ms]),
               "ece_mean": np.mean([m["ece"] for m in ms])}
        for k, v in exp.items():
            if abs(float(r[k]) - v) > 6e-5:
                bad.append((r["scene"], r["split"], r["model"], k, r[k], v))
        if seeds != [0, 1, 2]:
            bad.append((r["scene"], r["split"], r["model"], "seeds", seeds))
    check("results_csv_matches_three_seed_runs", not bad, {"rows": len(rows), "mismatches": bad})
    req = {("indian_pines", "random", "svm"), ("indian_pines", "spatial", "svm"),
           ("indian_pines", "spatial", "cnn3d"), ("salinas", "spatial", "cnn3d")}
    check("results_csv_required_rows", req <= {(r["scene"], r["split"], r["model"]) for r in rows},
          sorted(req))


def splits_verified():
    cfg = json.load(open(ROOT / "configs" / "default.json"))
    gts = {s: load_scene(s).gt for s in ("indian_pines", "salinas")}
    out = {}
    for f in sorted(glob.glob(str(ROOT / "experiments" / "splits" / "*.npz"))):
        name = Path(f).stem
        scene = "indian_pines" if name.startswith("indian_pines") else "salinas"
        z = np.load(f)
        gt = gts[scene]
        if "_random_" in name:
            sp.verify_random(z["split"], gt, cfg["random_split"]["train_frac"])
            out[name] = "random: stratified 10% verified"
            continue
        block = 30 if name.endswith("_b30") else cfg["spatial_split"]["block"]
        r = cfg["spatial_split"]["buffer"]
        sp.verify_spatial(z["split"], z["assign"], gt, block, r, 50)
        s = sp.split_summary(z["split"], gt)
        out[name] = {k: round(s[k], 3) for k in ("frac_train", "frac_val", "frac_test")}
        # independent brute-force patch check on every training pixel
        test_area = z["assign"] == sp.TEST
        for i, j in np.argwhere(z["split"] == sp.TRAIN):
            assert not test_area[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1].any(), (name, i, j)
    check("all_saved_splits_verified", True, out)
    fr = [v for k, v in out.items() if isinstance(v, dict) and not k.endswith("_b30")]
    tol = max(max(abs(v["frac_train"] - .6), abs(v["frac_val"] - .2), abs(v["frac_test"] - .2)) for v in fr)
    check("spatial_proportions_within_0.10_of_60_20_20", tol <= 0.10, {"max_deviation": round(tol, 3)})


def wavelengths_consistent():
    spc = np.loadtxt(wl.SPC_FILE, skiprows=2)
    ip_rm = spc[[b - 1 for b in wl.IP_REMOVED_1B], 0]
    av = wl.read_av98()
    sa_rm = av[[b - 1 for b in wl.SALINAS_REMOVED_1B]]

    def in_water(c):
        return ((c > 1340) & (c < 1420)) | ((c > 1800) & (c < 1960)) | (c > 2480)
    check("removed_bands_are_water_bands", in_water(ip_rm).all() and in_water(sa_rm).all(),
          {"ip_removed_nm": np.round(ip_rm).tolist(), "salinas_removed_nm": np.round(sa_rm).tolist()})
    cup = load_scene("cuprite")
    zero = cup.cube[cup.valid].mean(0) <= 0
    zc = cup.centres[zero]
    check("cuprite_zeroed_bands_fall_in_water_windows_of_proxy_table",
          bool(np.all((zc < 400) | ((zc > 1350) & (zc < 1400)) | ((zc > 1820) & (zc < 1940)))), np.round(zc).tolist())


def artefacts():
    need = ["results.csv", "reports/technical_report.pdf", "README.md", "app.py", "run_app.py", "requirements.txt",
            "outputs/class_maps/indian_pines_cnn3d_class_map.png", "outputs/class_maps/salinas_cnn3d_class_map.png",
            "outputs/confidence_maps/indian_pines_cnn3d_confidence.png", "outputs/confidence_maps/salinas_cnn3d_confidence.png",
            "outputs/mineral_maps/cuprite_sam_class_map.png", "outputs/mineral_maps/cuprite_ml_class_map.png",
            "outputs/mineral_maps/cuprite_band_depth_maps.png", "outputs/mineral_maps/cuprite_relative_abundance.png",
            "outputs/confidence_maps/cuprite_ml_confidence.png", "outputs/index_maps/indian_pines_ndvi_crop_classes.png",
            "outputs/index_maps/salinas_ndre_crop_classes.png", "models/indian_pines_spatial_cnn3d.pt",
            "models/salinas_spatial_cnn3d.pt", "models/cuprite_mlp.pkl"]
    missing = [p for p in need if not (ROOT / p).exists()]
    check("required_artefacts_present", not missing, {"missing": missing})
    try:
        import pymupdf
        n = len(pymupdf.open(ROOT / "reports" / "technical_report.pdf"))
        body = [i + 1 for i, p in enumerate(pymupdf.open(ROOT / "reports" / "technical_report.pdf")) if "Appendix" in p.get_text()][0] - 1
        check("report_body_le_6_pages", body <= 6, {"total_pages": n, "body_pages": body})
    except ImportError:
        RES["report_body_le_6_pages"] = {"pass": None, "detail": "pymupdf not installed"}
    vid = ROOT / "outputs" / "demo" / "demo.webm"
    RES["demo_video_local"] = {"pass": vid.exists(), "detail": f"{vid.stat().st_size / 1e6:.1f} MB" if vid.exists() else "missing"}


if __name__ == "__main__":
    metrics_from_confusion()
    results_csv_from_runs()
    splits_verified()
    wavelengths_consistent()
    artefacts()
    save_json(RES, ROOT / "outputs" / "reports" / "audit_checks.json")
    print("FAILED:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)

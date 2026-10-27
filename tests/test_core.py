"""Unit tests on small synthetic fixtures (no download needed), plus data tests that are skipped
when data/raw is absent."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import band_selection, continuum_removal as crm, metrics, sam, splits, synthetic_data  # noqa: E402
from src.preprocessing import SpectralPreprocessor  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
HAVE_DATA = (RAW / "Indian_pines_corrected.mat").exists()


# ----------------------------------------------------------------------------- fixtures
def toy_gt(H=90, W=90, K=4, seed=0):
    """Field-like ground truth: rectangular patches of K classes plus unlabelled gaps."""
    rng = np.random.default_rng(seed)
    gt = np.zeros((H, W), np.int64)
    for i in range(0, H, 15):
        for j in range(0, W, 15):
            if rng.random() < 0.85:
                gt[i:i + 12, j:j + 12] = rng.integers(1, K + 1)
    return gt


# ----------------------------------------------------------------------------- splits
def test_random_split_is_stratified_10pct():
    gt = toy_gt()
    s = splits.random_split(gt, 0.1, seed=3)
    splits.verify_random(s, gt, 0.1)
    for c in range(1, gt.max() + 1):
        n = (gt == c).sum()
        assert ((gt == c) & (s == splits.TRAIN)).sum() == max(1, round(0.1 * n))
    assert np.all(s[gt == 0] == 0)


@pytest.mark.parametrize("block,r", [(15, 3), (20, 3), (30, 2)])
def test_spatial_split_no_leakage(block, r):
    gt = toy_gt()
    s, assign, info = splits.spatial_split(gt, block, seed=1, buffer=r, n_trials=200, min_class_px=50)
    rep = splits.verify_spatial(s, assign, gt, block, r, 50)
    assert rep["train_patch_clear_of_test"]
    # independent brute-force check of the patch condition
    H, W = gt.shape
    tr = np.argwhere(s == splits.TRAIN)
    test_area = assign == splits.TEST
    for i, j in tr[:: max(1, len(tr) // 300)]:
        win = test_area[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1]
        assert not win.any()
    assert 0.3 < info["frac_train"] < 0.8


def test_verify_spatial_detects_leak():
    gt = toy_gt()
    s, assign, _ = splits.spatial_split(gt, 15, seed=0, buffer=3, n_trials=50)
    # inject a training pixel right next to a test block
    ti = np.argwhere((assign == splits.TEST) & (gt > 0))[0]
    s2 = s.copy()
    a2 = assign.copy()
    i, j = ti
    nb = (i, j + 1) if j + 1 < gt.shape[1] and assign[i, j + 1] != splits.TEST else (i, j - 1)
    s2[nb] = splits.TRAIN
    a2[nb] = splits.TRAIN
    with pytest.raises(AssertionError):
        splits.verify_spatial(s2, a2, gt, 15, 3)


def test_blocks_are_pure():
    gt = toy_gt()
    _, assign, _ = splits.spatial_split(gt, 20, seed=2, buffer=3, n_trials=30)
    bid = splits.block_ids(gt.shape, 20)
    for b in np.unique(bid):
        assert np.unique(assign[bid == b]).size == 1


# ----------------------------------------------------------------------------- metrics
def test_metrics_known_values():
    y = np.array([0, 0, 1, 1, 2, 2])
    P = np.eye(3)[[0, 0, 1, 2, 2, 2]] * 0.9 + 0.1 / 3
    m = metrics.evaluate(y, P, 3)
    assert m["oa"] == pytest.approx(5 / 6)
    assert m["aa"] == pytest.approx((1 + 0.5 + 1) / 3)
    assert m["confusion"][1] == [0, 1, 1]
    assert 0 < m["kappa"] < 1


def test_metrics_absent_class_not_in_aa():
    y = np.array([0, 0, 1, 1])
    P = np.eye(3)[[0, 0, 1, 2]]
    m = metrics.evaluate(y, P, 3)
    assert m["classes_absent_from_test"] == [2]
    assert m["aa"] == pytest.approx(0.75)


def test_ece_perfect_and_overconfident():
    y = np.array([0, 1, 0, 1])
    P_perfect = np.eye(2)[y]
    assert metrics.ece_score(P_perfect, y) == pytest.approx(0.0)
    P_over = np.array([[0.99, 0.01]] * 4)            # always confident class 0, 50 % right
    assert metrics.ece_score(P_over, y) == pytest.approx(0.49, abs=1e-6)


# ----------------------------------------------------------------------------- preprocessing
def test_preprocessor_fits_on_train_only_and_drops_ranges():
    rng = np.random.default_rng(0)
    centres = np.linspace(400, 2500, 50)
    Xtr = rng.normal(5, 2, (100, 50))
    Xte = rng.normal(50, 2, (100, 50))       # very different: must not affect the fitted mean
    p = SpectralPreprocessor([[1300, 1500]], None).fit(Xtr, centres)
    assert np.all((centres[p.keep_] < 1300) | (centres[p.keep_] > 1500))
    Z = p.transform(Xtr)
    assert np.allclose(Z.mean(0), 0, atol=1e-5)
    assert p.transform(Xte).mean() > 10


# ----------------------------------------------------------------------------- continuum / SAM
def test_continuum_removal_gaussian_feature():
    x = np.linspace(2000, 2450, 46)
    cont = 0.5 + 0.0002 * (x - 2000)
    feat = 1 - 0.3 * np.exp(-0.5 * ((x - 2200) / 15) ** 2)
    R = (cont * feat)[None]
    cr, idx = crm.continuum_remove(R, x, 2000, 2450)
    assert cr.shape == (1, 46)
    assert np.isclose(cr[0, 0], 1) and np.isclose(cr[0, -1], 1)
    d, p = crm.feature_params(cr, x[idx], 2200, 30)
    assert d[0] == pytest.approx(0.3, abs=0.01)
    assert abs(p[0] - 2200) <= 10


def test_continuum_handles_unsorted_and_invalid():
    x = np.array([2000, 2010, 2005, 2020, 2030.0])     # detector overlap: not monotonic
    R = np.array([[0.5, 0.4, 0.45, 0.5, 0.5], [np.nan] * 5])
    cr, idx = crm.continuum_remove(R, x, 2000, 2030)
    assert np.all(np.diff(x[idx]) > 0)
    assert np.all(np.isnan(cr[1]))


def test_sam_scale_invariant_and_unknown():
    R = {"a": np.array([[1, 0, 0.0]]), "b": np.array([[0, 1, 0.0]])}
    X = np.array([[5, 0.1, 0], [0.2, 3, 0], [0, 0, 1.0]])
    lab, best, second, names, _ = sam.sam_classify(X, R, threshold=0.2)
    assert list(lab[:2]) == [0, 1]
    assert lab[2] == -1                       # orthogonal to both refs -> unknown
    assert sam.spectral_angle(np.array([[1, 2, 3.0]]), np.array([[2, 4, 6.0]]))[0, 0] < 1e-6


def test_feature_fit_recovers_scale():
    ref = 1 - 0.4 * np.exp(-0.5 * ((np.arange(40) - 20) / 3.0) ** 2)
    pix = 1 - 0.5 * (1 - ref)
    a, rms, r2 = sam.feature_fit(pix[None], ref)
    assert a[0] == pytest.approx(0.5, abs=1e-6) and rms[0] < 1e-9 and r2[0] > 0.999


# ----------------------------------------------------------------------------- synthetic data
def test_synthetic_generation_shapes_labels_and_unknown():
    centres = np.linspace(2000, 2450, 40)
    base = {m: np.array([0.5 - 0.2 * np.exp(-0.5 * ((centres - c) / 10) ** 2)])
            for m, c in zip(["alunite", "kaolinite", "muscovite", "buddingtonite", "calcite", "chalcedony",
                             "montmorillonite", "hematite", "jarosite", "dickite", "opal"],
                            [2165, 2205, 2200, 2120, 2340, 2260, 2210, 2400, 2265, 2180, 2250])}
    X, y, f, classes = synthetic_data.generate(base, centres, np.full(40, 0.002), n_per_class=20, seed=0)
    assert X.shape == (140, 40) and classes[-1] == "unknown"
    assert set(np.unique(y)) == set(range(7))
    assert np.all(f[y < 6] >= 0.3) and np.all(np.isfinite(X))
    X2, *_ = synthetic_data.generate(base, centres, np.full(40, 0.002), n_per_class=20, seed=0)
    assert np.allclose(X, X2)                  # deterministic


def test_noise_estimate_recovers_sigma():
    rng = np.random.default_rng(0)
    cube = 0.3 + rng.normal(0, 0.01, (64, 64, 3))
    s = synthetic_data.estimate_noise(cube, block=8, pct=50)
    assert np.allclose(s, 0.01, rtol=0.15)


# ----------------------------------------------------------------------------- bands
def test_s2_simulation_reports_missing_bands():
    centres = np.array([440, 450, 490, 560, 665, 705, 740, 783, 830, 865, 945, 1610, 2200.0])
    X = np.ones((5, len(centres)))
    Xs, names, rep = band_selection.simulate_s2(X, centres)
    assert "B10" not in names and Xs.shape == (5, len(names))
    assert [r for r in rep if r["band"] == "B10"][0]["n_aviris_bands"] == 0


def test_select_bands_spacing():
    imp = np.arange(100)[::-1].astype(float)
    centres = np.linspace(400, 2500, 100)
    picks = band_selection.select_bands(imp, centres, k=15, min_sep_nm=50)
    assert len(picks) <= 15
    c = np.sort(centres[picks])
    assert np.all(np.diff(c) >= 50)


# ----------------------------------------------------------------------------- real data
@pytest.mark.skipif(not HAVE_DATA, reason="data not downloaded")
def test_real_scenes_shapes_and_wavelengths():
    from src.data_loader import load_scene
    for name, shape in [("indian_pines", (145, 145, 200)), ("salinas", (512, 217, 204))]:
        s = load_scene(name)
        assert s.cube.shape == shape and s.centres.size == shape[2]
        assert s.gt.max() == 16 and len(s.class_names) == 16
        assert 360 < s.centres.min() < 420 and 2450 < s.centres.max() < 2520
    c = load_scene("cuprite")
    assert c.cube.shape == (512, 614, 224) and c.centres.size == 224
    # bands zeroed by the 1997 retrieval must sit in the water-vapour windows of the adopted table
    zero = c.cube[c.valid].mean(0) <= 0.0
    zc = c.centres[zero]
    assert np.all((zc < 400) | ((zc > 1350) & (zc < 1400)) | ((zc > 1820) & (zc < 1940)))


@pytest.mark.skipif(not HAVE_DATA, reason="data not downloaded")
def test_saved_split_files_leak_free():
    """Every spatial split actually used in experiments/ passes the formal leakage check."""
    import json
    from src.data_loader import load_scene
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "default.json").read_text())
    sdir = Path(__file__).resolve().parents[1] / "experiments" / "splits"
    files = sorted(sdir.glob("*_spatial_s*.npz")) if sdir.exists() else []
    if not files:
        pytest.skip("no saved splits yet")
    gts = {}
    for f in files:
        scene = "indian_pines" if f.name.startswith("indian_pines") else "salinas"
        gts.setdefault(scene, load_scene(scene).gt)
        z = np.load(f)
        block = cfg["spatial_split"]["block"]
        for part in f.stem.split("_"):
            if part.startswith("b") and part[1:].isdigit():
                block = int(part[1:])
        splits.verify_spatial(z["split"], z["assign"], gts[scene], block, cfg["spatial_split"]["buffer"])


# ----------------------------------------------------------------------------- exports / models / config
def test_exports_png_tiff_pdf():
    from src import exports, reporting
    img = np.random.default_rng(0).random((40, 50, 3))
    assert exports.png_bytes(img, "t", [("a", (1, 0, 0))])[:4] == b"\x89PNG"
    import io
    import tifffile
    arr = np.arange(20, dtype=np.int16).reshape(4, 5)
    back = tifffile.imread(io.BytesIO(exports.tiff_bytes(arr, "labels")))
    assert np.array_equal(back, arr)
    pdf = reporting.summary_pdf("x", img, ["a", "b"], np.array([3, 4]), 20.0, 0.7, "conf", ["r"], ["l"], "m")
    assert pdf[:5] == b"%PDF-"


def test_config_has_required_keys():
    import json
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "default.json").read_text())
    assert cfg["seeds"] == [0, 1, 2]
    assert 15 <= cfg["spatial_split"]["block"] <= 30
    assert cfg["spatial_split"]["buffer"] >= cfg["cnn"]["patch"] // 2   # buffer covers the CNN patch radius
    assert cfg["random_split"]["train_frac"] == 0.1


def test_cnn_receptive_field_equals_patch():
    import torch
    from src.crop_models import CNN3D, extract_patches
    net = CNN3D(30, 16, 7).eval()
    assert net(torch.zeros(2, 1, 30, 7, 7)).shape == (2, 16)
    assert net.features(torch.zeros(1, 1, 30, 7, 7)).shape[-2:] == (1, 1)   # 7x7 -> 1x1: no wider context
    Z = np.random.default_rng(0).random((10, 12, 30)).astype(np.float32)
    P = extract_patches(Z, np.array([0, 5]), np.array([0, 6]), 7)
    assert P.shape == (2, 1, 30, 7, 7) and np.allclose(P[1, 0, :, 3, 3], Z[5, 6])


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "models" / "indian_pines_spatial_svm.pkl").exists()
                    or not HAVE_DATA, reason="models or data missing")
def test_saved_models_load_and_full_scene_shapes():
    from src import inference
    from src.data_loader import load_scene
    s = load_scene("indian_pines")
    for m in ("svm", "cnn3d"):
        lab, P = inference.predict_crop(s, m)
        assert lab.shape == (145, 145) and P.shape == (145, 145, 16)
        assert np.allclose(P.sum(-1), 1, atol=1e-4) and lab.min() >= 0 and lab.max() < 16


def test_results_csv_schema():
    import csv
    p = Path(__file__).resolve().parents[1] / "results.csv"
    if not p.exists():
        pytest.skip("results.csv not generated yet")
    rows = list(csv.reader(open(p)))
    assert rows[0] == ["scene", "split", "model", "oa_mean", "oa_std", "aa_mean", "kappa_mean", "macro_f1_mean",
                       "ece_mean", "n_train_px", "n_test_px"]
    combos = {tuple(r[:3]) for r in rows[1:]}
    for c in [("indian_pines", "random", "svm"), ("indian_pines", "spatial", "svm"),
              ("indian_pines", "spatial", "cnn3d"), ("salinas", "spatial", "cnn3d")]:
        assert c in combos
    for r in rows[1:]:
        assert all(v != "" for v in r), r
        assert 0 <= float(r[3]) <= 1


def test_index_formula_uses_actual_band_centres():
    from src import indices
    centres = np.array([666.6, 696.5, 715.8, 744.7, 783.3, 792.9, 802.5])
    assert indices.describe("ndvi", centres) == "NDVI = (R802.5 - R666.6) / (R802.5 + R666.6)"
    assert indices.describe("ndre", centres) == "NDRE = (R792.9 - R715.8) / (R792.9 + R715.8)"
    cube = np.ones((2, 2, 7))
    cube[..., 1], cube[..., 3] = 0.2, 0.6            # R700 = 0.2, R740 = 0.6, Rre = (1 + 1) / 2 = 1 -> clipped
    cube[..., 0], cube[..., 4] = 0.2, 0.6            # Rre = 0.4 -> halfway between the two centres
    assert np.allclose(indices.red_edge_position(cube, centres), (696.5 + 744.7) / 2)

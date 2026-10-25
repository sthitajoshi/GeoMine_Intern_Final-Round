"""Spectral Earth Mini - Streamlit prototype.

Run:  python run_app.py      (downloads data if missing, then starts Streamlit)
  or  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from src import continuum_removal as crm  # noqa: E402
from src import exports, indices, inference, reporting  # noqa: E402
from src import visualization as vz  # noqa: E402
from src.data_loader import SCENES, load_scene, true_colour  # noqa: E402
from src.mineral_library import TARGETS  # noqa: E402

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # click-to-inspect falls back to number inputs
    streamlit_image_coordinates = None

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
st.set_page_config(page_title="Spectral Earth Mini", layout="wide")


# ----------------------------------------------------------------------------- cached loaders
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner="Loading scene ...")
def get_scene(name):
    return load_scene(name)


@st.cache_data(show_spinner="Running crop model inference on the full scene ...")
def crop_inference(name, model, _key):
    """_key = model file mtimes, so a retrained model invalidates the cache."""
    lab, P = inference.predict_crop(get_scene(name), model, "spatial")
    return lab, P.max(-1), P


@st.cache_data(show_spinner="Loading mineral maps ...")
def mineral_maps(_key):
    p = EXP / "mineral" / "cuprite_maps.npz"
    if not p.exists():
        raise FileNotFoundError("experiments/mineral/cuprite_maps.npz missing - run scripts/run_mineral_mapping.py")
    z = dict(np.load(p))
    lib = dict(np.load(EXP / "mineral" / "library.npz"))
    res = json.loads((EXP / "mineral" / "mineral_results.json").read_text())
    return z, lib, res


@st.cache_data(show_spinner="Re-running the synthetic-trained MLP on Cuprite (~1 min) ...")
def mineral_live(_key):
    return inference.predict_minerals(get_scene("cuprite"))


@st.cache_data
def results_table():
    p = ROOT / "outputs" / "reports" / "results_extended.json"
    return json.loads(p.read_text()) if p.exists() else []


def blend(base, layer, alpha, mask=None):
    m = np.ones(base.shape[:2], bool) if mask is None else mask
    out = base.copy()
    out[m] = (1 - alpha) * base[m] + alpha * layer[m]
    return out


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Spectral Earth Mini")
scene_name = st.sidebar.selectbox("Scene", SCENES, format_func=lambda s: s.replace("_", " ").title())
try:
    scene = get_scene(scene_name)
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()
H, W, B = scene.cube.shape
is_crop = scene_name != "cuprite"
st.sidebar.caption(f"{H} x {W} px, {B} bands. Wavelengths: {scene.wavelength_source}"
                   + ("" if scene.wavelength_exact else " (proxy - see README)"))

base = true_colour(scene)
layers = {}
legend = None
if is_crop:
    model = st.sidebar.radio("Model (trained on the spatial split)", ["cnn3d", "svm"],
                             format_func=lambda m: {"cnn3d": "3-D CNN (spectral-spatial)", "svm": "RBF-SVM (spectral)"}[m])
    paths = inference.model_paths(scene_name)
    try:
        lab, conf, P = crop_inference(scene_name, model, (_mtime(paths["svm"]), _mtime(paths["cnn3d"])))
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    pal = vz.class_palette(len(scene.class_names))
    legend = [(n, tuple(pal[i])) for i, n in enumerate(scene.class_names)]
    layers["Class map"] = vz.colourise(lab, pal)
    layers["Confidence"] = vz.scalar_rgb(conf, "magma", 0, 1)
    idx_name = st.sidebar.selectbox("Index layer", ["NDVI", "NDRE (red edge)", "Red-edge position"])
    fn = {"NDVI": indices.ndvi, "NDRE (red edge)": indices.ndre, "Red-edge position": indices.red_edge_position}[idx_name]
    vr = {"NDVI": (-0.1, 0.9), "NDRE (red edge)": (-0.05, 0.6), "Red-edge position": (700, 740)}[idx_name]
    idx_val = fn(scene.cube, scene.centres)
    crop_ids = [i for i, n in enumerate(scene.class_names) if n in scene.crop_classes]
    only_crop = st.sidebar.checkbox("Index on predicted crop classes only", True)
    iv = np.where(np.isin(lab, crop_ids), idx_val, np.nan) if only_crop else idx_val
    layers["Index"] = vz.scalar_rgb(iv, "RdYlGn" if idx_name != "Red-edge position" else "viridis", *vr)
    layers["Ground truth"] = vz.colourise(scene.gt - 1, pal)
else:
    z, lib, res = mineral_maps(_mtime(EXP / "mineral" / "cuprite_maps.npz"))
    method = st.sidebar.radio("Mineral method", ["SAM (physics)", "MLP (synthetic-trained)"])
    mlab = z["lab_sam"] if method.startswith("SAM") else z["lab_ml"]
    if method.startswith("MLP") and st.sidebar.button("Re-run MLP inference with saved model"):
        live, live_conf, _ = mineral_live(_mtime(inference.model_paths("cuprite")["mlp"]))
        st.sidebar.success(f"Live inference agrees with stored map on {(live == z['lab_ml']).mean():.1%} of pixels")
        mlab = live
    legend = [(m, vz.matplotlib.colors.to_rgb(vz.MINERAL_COLOURS[m])) for m in TARGETS] + [("unknown / none", (0.85,) * 3)]
    layers["Mineral map"] = vz.mineral_rgb(mlab, TARGETS)
    if method.startswith("SAM"):
        q = np.where(z["lab_sam"] >= 0, 1 - z["sam_angle"] / res["sam_threshold_rad"], np.nan)
        layers["Confidence"] = vz.scalar_rgb(q, "viridis", 0, 1)
    else:
        layers["Confidence"] = vz.scalar_rgb(z["ml_conf"], "magma", 0, 1)
    dm = st.sidebar.selectbox("Band-depth layer", TARGETS)
    layers["Index"] = vz.scalar_rgb(z[f"depth_{dm}"], "inferno", 0, 0.25)

st.sidebar.markdown("### Layers")
show_base = st.sidebar.checkbox("True-colour image", True)
order = list(layers)
on, alpha = {}, {}
for name in order:
    default = name in ("Class map", "Mineral map")
    on[name] = st.sidebar.checkbox(name, default, key=f"on_{name}")
    alpha[name] = st.sidebar.slider(f"{name} opacity", 0.0, 1.0, 0.7 if default else 0.6, 0.05, key=f"a_{name}")

img = base.copy() if show_base else np.zeros_like(base)
for name in order:
    if on[name]:
        mask = None
        if name == "Mineral map":
            mask = mlab >= 0 if show_base else None
        if name == "Ground truth":
            mask = scene.gt > 0
        img = blend(img, layers[name], alpha[name], mask)

# ----------------------------------------------------------------------------- main panel
st.title("Spectral Earth Mini")
if is_crop:
    st.info("Accuracy below comes from a spatially separated test set (blocks + buffer). Colours outside "
            "labelled fields are extrapolations of the model and have no ground truth.")
else:
    st.warning("Cuprite has no pixel ground truth: no accuracy is claimed. Agreement between SAM and the "
               "synthetic-trained MLP, fit residuals and the published USGS map are the validation evidence.")

col_map, col_spec = st.columns([3, 2])
scale = max(1, int(700 / max(H, W)))
disp = (np.clip(img, 0, 1) * 255).astype(np.uint8)
disp = np.kron(disp, np.ones((scale, scale, 1), np.uint8))
with col_map:
    st.caption(f"Click a pixel to inspect it (display x{scale}).")
    if streamlit_image_coordinates is not None:
        click = streamlit_image_coordinates(Image.fromarray(disp), key=f"click_{scene_name}")
    else:
        st.image(disp)
        click = None
    if legend:
        st.markdown(" ".join(f"<span style='background:rgb{tuple(int(255*c) for c in col)};padding:0 8px'>&nbsp;</span> {n}"
                             for n, col in legend), unsafe_allow_html=True)

if click:
    r, c = min(H - 1, click["y"] // scale), min(W - 1, click["x"] // scale)
else:
    r, c = H // 2, W // 2
with col_spec:
    r = st.number_input("row", 0, H - 1, int(r))
    c = st.number_input("col", 0, W - 1, int(c))
    spec = scene.cube[r, c]
    order_w = np.argsort(scene.centres)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(scene.centres[order_w], spec[order_w], "k-", lw=1, label="pixel")
    if is_crop:
        k = int(lab[r, c])
        truth = scene.gt[r, c]
        st.metric("Predicted class", scene.class_names[k], f"confidence {conf[r, c]:.2f}", delta_color="off")
        st.write(f"Ground truth: **{scene.class_names[truth - 1] if truth else 'unlabelled'}**")
        top = np.argsort(-P[r, c])[:3]
        st.write("Top-3: " + ", ".join(f"{scene.class_names[t]} ({P[r, c, t]:.2f})" for t in top))
        same = (lab == k) & (scene.gt == k + 1)
        if same.any():
            mean_spec = scene.cube[same].mean(0)
            ax.plot(scene.centres[order_w], mean_spec[order_w], "C1--", lw=1, label=f"mean labelled {scene.class_names[k]}")
        ax.set_ylabel("at-sensor value")
    else:
        lab_s, lab_m = int(z["lab_sam"][r, c]), int(z["lab_ml"][r, c])
        name = lambda v: TARGETS[v] if v >= 0 else ("unknown / none" if v == -1 else "masked")  # noqa: E731
        st.metric("SAM match", name(lab_s), f"angle {z['sam_angle'][r, c]:.3f} rad (thr {res['sam_threshold_rad']})"
                  if np.isfinite(z["sam_angle"][r, c]) else None, delta_color="off")
        st.metric("MLP prediction", name(lab_m), f"p = {z['ml_conf'][r, c]:.2f}" if np.isfinite(z["ml_conf"][r, c]) else None, delta_color="off")
        if np.isfinite(z["fit_rms"][r, c]):
            st.write(f"Feature-fit RMS residual {z['fit_rms'][r, c]:.4f}, R² {z['fit_r2'][r, c]:.2f}, "
                     f"relative depth {z['fit_scale'][r, c]:.2f}")
        ax.set_ylabel("reflectance")
        show = lab_s if lab_s >= 0 else lab_m
        if show >= 0:
            ref = lib[f"conv_{TARGETS[show]}"][0]
            ok = np.isfinite(ref)
            scale_ref = np.nanmedian(spec[ok]) / np.nanmedian(ref[ok])
            ax.plot(scene.centres[order_w], (ref * scale_ref)[order_w], "C3--", lw=1, label=f"library {TARGETS[show]} (scaled)")
        ax.set_xlim(1950, 2500)
        # continuum-removed view
        cr_pix, idx = crm.continuum_remove(spec[None], scene.centres, 2000, 2450, lib["band_ok"])
        fig2, ax2 = plt.subplots(figsize=(5, 2.5))
        ax2.plot(scene.centres[idx], cr_pix[0], "k-", label="pixel (CR)")
        if show >= 0:
            ax2.plot(z["cr_x"], lib[f"cr_{TARGETS[show]}"][0], "C3--", label="library (CR)")
        ax2.set_xlabel("nm")
        ax2.legend(fontsize=7)
    ax.set_xlabel("nm")
    ax.legend(fontsize=7)
    st.pyplot(fig)
    if not is_crop:
        st.pyplot(fig2)

# ----------------------------------------------------------------------------- summary + export
st.markdown("---")
st.subheader("Summary and export")
c1, c2 = st.columns(2)
if is_crop:
    counts = np.bincount(lab.ravel(), minlength=len(scene.class_names))
    names = scene.class_names
    mean_conf = float(conf.mean())
    conf_label = "max-probability confidence (temperature-scaled for CNN)"
    rows = [f"{e['split']:8s} {e['model']:6s} OA {e['oa_mean']:.3f}±{e['oa_std']:.3f}  AA {e['aa_mean']:.3f}  "
            f"kappa {e['kappa_mean']:.3f}  macroF1 {e['macro_f1_mean']:.3f}  ECE {e['ece_mean']:.3f}"
            for e in results_table() if e["scene"] == scene_name]
    lims = ["Spatial-split numbers are the honest estimate; random-split numbers are inflated by neighbouring-pixel leakage.",
            "Indian Pines labels are field-level and the scene is small: some classes occupy one or two fields.",
            "Values are at-sensor; vegetation indices are relative."]
    method = f"{model} trained on the spatial block split (seed 0)"
else:
    counts = np.array([(mlab == k).sum() for k in range(len(TARGETS))] + [(mlab == -1).sum()])
    names = TARGETS + ["unknown / none"]
    mean_conf = float(np.nanmean(z["ml_conf"][mlab >= 0])) if method.startswith("MLP") else \
        float(np.nanmean(1 - z["sam_angle"][mlab >= 0] / res["sam_threshold_rad"]))
    conf_label = "MLP probability over mapped pixels" if method.startswith("MLP") else "SAM match quality over mapped pixels"
    a = res["agreement"]
    rows = [f"SAM vs MLP pixel agreement {a['overall_agreement']:.3f}, Cohen kappa {a['cohen_kappa_sam_vs_ml']:.3f}",
            f"MLP on held-out SYNTHETIC spectra: OA {res['ml_synthetic']['oa']:.3f} (not a real-scene accuracy)"]
    lims = ["No pixel ground truth: agreement is not accuracy.",
            "1997 AVIRIS band centres approximated by the 1998 table (~1-2 nm uncertainty).",
            "Synthetic training set cannot represent every Cuprite mineral (e.g. dickite, opal mixtures)."]
with c1:
    st.write(f"Mean {conf_label}: **{mean_conf:.3f}**")
    st.dataframe({"class": names, "pixels": counts.tolist(),
                  "area (ha)": [round(float(v) * scene.gsd_m ** 2 / 1e4, 1) for v in counts]}, height=300)
with c2:
    for rrow in rows:
        st.text(rrow)
    title = f"{scene_name} - {'+'.join(n for n in order if on[n]) or 'true colour'}"
    st.download_button("Download current view (PNG + legend)", exports.png_bytes(img, title, legend),
                       f"{scene_name}_view.png", "image/png")
    main_layer = lab if is_crop else mlab
    st.download_button("Download class/mineral labels (TIFF)", exports.tiff_bytes(main_layer.astype(np.int16), f"{scene_name} labels; classes: {names}"),
                       f"{scene_name}_labels.tif", "image/tiff")
    st.download_button("Download one-page summary (PDF)",
                       reporting.summary_pdf(scene_name, img, names, counts, scene.gsd_m, mean_conf, conf_label, rows, lims,
                                             method if is_crop else method),
                       f"{scene_name}_summary.pdf", "application/pdf")

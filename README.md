# Spectral Earth Mini

A hyperspectral pipeline for the GeoMine intern final round. It has three parts:

* **A.** Crop and land-cover classification on AVIRIS Indian Pines and Salinas, scored on a leakage-free spatial split.
* **B.** Alteration-mineral mapping on AVIRIS Cuprite (1997) without ground-truth labels.
* **C.** A Streamlit app that serves both.

The headline numbers in this README come from the spatial split. Random-split numbers are reported only to measure the leakage gap.

| Command | What it does | Time (8-thread CPU laptop) |
|---|---|---|
| `pip install -r requirements.txt` | dependencies (CPU torch) | 2–4 min |
| `python run_app.py` | downloads data if missing (130 MB), starts the app at http://localhost:8501 | 1–3 min first time |
| `python scripts/reproduce_all.py` | every number, map and figure in the report, then the tests | ~90 min: main 3-seed run 32 min, band study 35 min (permutation importance), 30-px ablation 35 min, rest ~5 min |
| `python scripts/reproduce_all.py --quick` | seed 0 only, smoke test | ~12 min |
| `python -m pytest -q tests` | unit and leakage tests | ~30 s |

Trained models (`models/`, about 10 MB) and the Cuprite mapping results (`experiments/mineral/`) are committed. That way the app runs from a clean clone without retraining.

---

## 1. Repository layout

```
app.py                 Streamlit app (Part C)          run_app.py   one-command launcher
configs/default.json   all experiment settings (block size, buffer, grids, thresholds, seeds)
src/
  wavelengths.py       band-centre recovery (sources below)
  data_loader.py       .mat loading, shape / class-count checks, true colour
  preprocessing.py     band dropping + train-only z-score / PCA
  splits.py            random + spatial block split, buffer, formal leakage verifier
  crop_models.py       RBF-SVM, 3-D CNN, patch extraction, temperature scaling
  metrics.py           OA, AA, kappa, per-class P/R/F1, confusion, ECE, reliability bins
  band_selection.py    permutation importance, <=15-band selection, Sentinel-2 simulation
  indices.py           NDVI, NDRE, red-edge position
  mineral_library.py   splib07 loading, Gaussian convolution to AVIRIS bands
  continuum_removal.py convex-hull continuum removal, band depth / position
  sam.py               spectral angle mapper, feature-fit residuals
  synthetic_data.py    mixture / illumination / noise generator, image SNR estimate
  mineral_models.py    synthetic-trained MLP with unknown class + rejection
  inference.py         full-scene inference with saved models (used by app and outputs)
  visualization.py, exports.py, reporting.py, utils.py
scripts/
  download_data.py     fetch + checksum all data
  train_crop_models.py Part A main experiment (3 seeds)
  band_experiments.py  band importance, 15 bands, S2-like, block-size study
  run_mineral_mapping.py  Part B
  generate_outputs.py  results.csv, maps, figures
  make_report.py       technical report PDF
  reproduce_all.py     everything above in order
tests/test_core.py     unit tests (synthetic fixtures) + real-data and saved-split leakage tests
outputs/               class_maps/ confidence_maps/ mineral_maps/ index_maps/ figures/ reports/
results.csv            challenge results file
```

## 2. Data sources and how they were obtained

`python scripts/download_data.py` fetches everything into `data/raw/` (git-ignored) and verifies SHA-256 checksums for the `.mat` files.

| Data | Source | Notes |
|---|---|---|
| Indian Pines corrected + GT, Salinas corrected + GT, Cuprite `f970619t01p02_r02_sc03.a.rfl` | [EHU Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) | The EHU server returned HTTP 403 to scripted downloads from our network. The script falls back to the Internet Archive's byte-exact capture of the *same official URLs* (`web.archive.org/web/2020id_/…`). File sizes match the EHU listing (5.7 MB, 26.5 MB, 95.3 MB), and the published class counts are checked at load time. |
| USGS Spectral Library v7 | DOI [10.5066/F7RR1WDJ](https://doi.org/10.5066/F7RR1WDJ) | ScienceBase only offers a single 5.5 GB zip, behind a bot check. We fetch the individual records that the USGS Spectroscopy Lab served under `crustal.usgs.gov/speclab/data/` (directly, else from the Internet Archive capture of that USGS URL): 13 native-resolution `splib07a` records (spectrum + wavelength + bandpass tables) and the matching `splib07b_cvAVIRISc1998` convolved records. |
| AVIRIS 1992 channel calibration | `92AV3C.spc`, distributed by Purdue MultiSpec with the original Indian Pines image (mirror: spectralpython/sample-data) | 220 rows: centre, FWHM. |
| Published USGS Cuprite mineral map | [USGS image page](https://www.usgs.gov/media/images/mineral-map-derived-aviris-data-obtained-over-cuprite-nevada) | "Original" image from the USGS asset bucket. The page has no colour key and the map covers a different extent, so it supports a qualitative comparison only. |

### Wavelength centres (`src/wavelengths.py`)

The `.mat` files carry no wavelengths. Each scene's centres come from a different source:

* **Indian Pines (AVIRIS, June 1992).** Centres and FWHM come from `92AV3C.spc`. EHU's 200-band cube removed bands 104–108, 150–163 and 220 (1-based), so we drop the same rows. Range: 400–2499 nm.
* **Salinas (AVIRIS, 1998).** Centres come from the splib07 AVIRIS-Classic 1998 table. EHU removed bands 108–112, 154–167 and 224. FWHM is approximated by the channel spacing.
* **Cuprite (AVIRIS, 19 June 1997).** The splib07 1997 table is only inside the 5.5 GB zip, so we use the **1998 table as a documented proxy**. As a check, every channel that the 1997 reflectance product zero-fills (water vapour) falls at 1.36–1.39 µm and 1.83–1.93 µm in the 1998 table; this is asserted in the tests. Year-to-year drift is about 1–2 nm, against a ~10 nm FWHM.
* AVIRIS has four spectrometers. At the joins (~670, ~1260, ~1880 nm) the centre sequence steps backwards, so the code never assumes a sorted or linear wavelength axis.

## 3. Evaluation protocol (Part A)

* **Random split.** Stratified: 10% of each class trains, 90% tests. Tuning uses a stratified 20% hold-out of the *training* pixels. The final model refits on all training pixels.
* **Spatial split** (`src/splits.py`):
  * Squares of `block` px are assigned whole to train/val/test. Default block = **15 px**; the reasoning is under "Block size" below.
  * The search draws 1,500 seeded random assignments and keeps the one closest to 60/20/20 by labelled pixels.
  * Hard constraint: every class with ≥ 50 px appears in test.
  * **Buffer r = 3 px** (the CNN patch radius, 7×7 patch). Training pixels within r of any test or validation block, and validation pixels within r of a test block, are dropped. So no training input patch contains a test pixel, and validation is not optimistic either.
  * `verify_spatial()` re-checks block purity, the patch condition and class coverage from the final masks. It is asserted at run time and in `tests/` (including a test that injects a leak and expects failure).
  * All splits are saved in `experiments/splits/`.
* **Block size.** No value satisfies every goal on Indian Pines, because big fields span few blocks. At 20–30 px, keeping every ≥50-px class in test forces some classes out of train or val altogether (counts in `outputs/reports/band_summary.json`). 15 px is the smallest allowed size, keeps all classes in train for all three seeds, and the buffer handles boundary autocorrelation. The block-size study reruns the SVM at 15/20/30 px (30 = "doubled") on 3 seeds.
* **Models.** The SVM is an RBF-SVM on per-band z-scored spectra (all retained bands). C and γ are chosen by grid search on validation OA. Probabilities are Platt-scaled.
  * The deep model is a 3-D CNN (3 conv layers + 2 FC, 96k parameters) on 7×7×30 patches. The 30 PCA components are fitted on training pixels only.
  * Learning rate is chosen on validation. Checkpoint and early stopping use validation OA. Temperature scaling is fitted on validation.
  * Salinas SVM training is capped at 8,000 stratified pixels (RBF-SVM cost is quadratic in the number of training pixels).
* **Metrics.** OA, AA (mean recall over classes present in test), Cohen's κ, per-class P/R/F1 (`zero_division=0`), macro-F1, confusion matrices, and ECE (15 equal-width bins on max-probability) with reliability diagrams.
  * `results.csv` gives mean ± sd over seeds 0/1/2. Each seed redraws the split, so the sd includes split variance.
  * Raw ECE is reported. Temperature-scaled CNN ECE is in `outputs/reports/results_extended.json`.
  * `n_train_px` / `n_test_px` are means over the three seeds (spatial splits differ per seed). Per-seed counts are in `outputs/reports/results_per_seed.csv`.
* **Test-set discipline.** Test labels are used once per run, after tuning. Band selection uses validation pixels of the same seed (seed k's validation can overlap seed k′'s test, so importances are never shared across seeds).

## 4. Part B summary

* **Library.** Native splib07a spectra are convolved with Gaussian SRFs (FWHM = channel spacing) to the adopted band centres. The result is cross-checked against USGS's own AVIRIS-1998 convolution: RMS difference ≤ 0.006 reflectance in the SWIR (`outputs/figures/library_convolution_check.png`).
* **Continuum removal.** Upper convex hull over 2.00–2.45 µm on usable bands. Band depth and position are computed at each mineral's diagnostic feature.
* **SAM** runs on absorption vectors (1 − CR), with a minimum band depth of 0.02 (≈5× the estimated noise σ). The angle threshold comes from labelled *synthetic* spectra, never the real scene. Unknown/none is kept as a class. Feature-fit residuals give a relative-abundance proxy and a residual map.
* **ML.** An MLP is trained only on synthetic mixtures of library spectra:
  * linear plus Hapke-style intimate mixing;
  * illumination ×0.6–1.4 plus an offset;
  * per-band noise from the image's estimated SNR;
  * an explicit unknown class (non-targets, sub-detection fractions, featureless backgrounds) and probability rejection at p < 0.5.
* **No-label validation** (`experiments/mineral/mineral_results.json`):
  * SAM vs MLP agreement and κ;
  * example pixel spectra against the library;
  * fit residuals;
  * sensitivity to threshold and depth gate;
  * noise-robustness on synthetic data;
  * a side-by-side with the USGS map.
  * No accuracy is claimed for Cuprite.

## 5. Changing things (expected live changes)

* **Block size.** Edit `spatial_split.block` in `configs/default.json`, or `python scripts/train_crop_models.py --scenes indian_pines --block 30 --tag b30`.
* **Remove wavelength ranges.** `python scripts/train_crop_models.py --drop_ranges_nm "1300,1500" --tag no1300`, or set `preprocessing.drop_ranges_nm`. Dropped bands are logged in each run JSON under `preprocessing`.
* **Add a class.** Crop: labels come from the GT array; add the name to `IP_CLASSES` / `SALINAS_CLASSES` in `src/data_loader.py` (and its count). Mineral: add a record to `RECORDS` / `TARGETS` in `src/mineral_library.py`, add its record name to `NATIVE_RECORDS` in `scripts/download_data.py`, add its diagnostic feature to `FEATURES` in `src/continuum_removal.py`, and add a colour in `src/visualization.py`.
* **Replace SAM.** `src/sam.py:sam_classify` takes `(X, refs, threshold)`. Any matcher with that signature (e.g. spectral information divergence or a correlation matcher) plugs into `scripts/run_mineral_mapping.py:sam_on_cr`.

## 6. Results (3 seeds; regenerated by `scripts/generate_outputs.py`)

| scene | split | model | OA (mean ± sd) | AA | κ | macro-F1 | ECE |
|---|---|---|---|---|---|---|---|
| indian_pines | random | svm | 0.801 ± 0.005 | 0.656 | 0.771 | 0.670 | 0.109 |
| indian_pines | spatial | svm | 0.636 ± 0.042 | 0.682 | 0.578 | 0.620 | 0.120 |
| indian_pines | spatial | cnn3d | 0.616 ± 0.035 | 0.619 | 0.552 | 0.562 | 0.285 |
| salinas | spatial | cnn3d | 0.969 ± 0.024 | 0.977 | 0.965 | 0.975 | 0.021 |
| indian_pines | random | cnn3d | 0.934 ± 0.007 | 0.908 | 0.924 | 0.927 | 0.012 |
| salinas | random | cnn3d | 0.991 ± 0.001 | 0.995 | 0.990 | 0.995 | 0.002 |
| salinas | random | svm | 0.934 ± 0.002 | 0.965 | 0.927 | 0.967 | 0.023 |
| salinas | spatial | svm | 0.929 ± 0.015 | 0.949 | 0.921 | 0.944 | 0.015 |

**Key findings** (details and evidence in the report):

* On Indian Pines the spatial split costs 16.5 OA points for the SVM and 31.8 for the CNN.
  * Under the random split, 54% of test pixels touch a training pixel. Even inside that split, CNN accuracy falls from 0.97 (adjacent) to 0.80 (4–6 px away). See `outputs/figures/gap_evidence.png`.
  * Under the honest spatial split, the SVM and CNN are level.
* Doubling the block to 30 px drops Indian Pines SVM OA to 0.46, partly because whole classes leave the training set.
* 12 simulated Sentinel-2 bands do as well as all 200 under the spatial split. 15 selected bands do not beat 15 random bands: selection on a small, spatially correlated validation set overfits.
* Cuprite: where SAM maps a mineral, the synthetic-trained MLP agrees on 92% of pixels. The MLP maps about 3× more pixels than SAM, which is the sim-to-real gap. Likely errors are listed in the report (kaolinite vs alunite, buddingtonite false positives, extra calcite).

## 7. Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Repository + README | this repo |
| 2 | Technical report (5 pages + appendix) | `reports/technical_report.pdf` (built by `scripts/make_report.py`) |
| 3 | `results.csv` | repo root (exact challenge column order; 4 required rows first, then 4 extra rows) |
| 4 | Class / confidence / mineral maps with legends | `outputs/class_maps`, `outputs/confidence_maps`, `outputs/mineral_maps`, `outputs/index_maps` |
| 5 | Demo video (~3 min, captions, no voice) | recorded with `scripts/record_demo.py` to `outputs/demo/demo.webm` (git-ignored, 44 MB; upload separately); step screenshots in `outputs/demo/` |

## 8. Limitations and AI-assistance disclosure

See `results.csv`, `outputs/reports/` and the technical report `reports/technical_report.pdf`. The report covers results, the random-vs-spatial gap, band importance, mineral validation and limitations.

**AI assistance.** This repository was built with substantial help from an AI coding assistant (Claude, via Claude Code). It took part in:

* data-source discovery and the download fallbacks;
* code for every module, script, test and the app;
* running the experiments and drafting the report text.

The design decisions (split design, buffer, model choices, validation strategy) and every reported number come from code in this repository and its logs (`experiments/logs/`). The author must be able to explain and modify every line in the defence. No result was entered by hand.

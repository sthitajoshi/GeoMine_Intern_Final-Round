# Final audit: Spectral Earth Mini vs the GeoMine challenge brief

Audit date: 25 Oct 2026. Scope: source, config, saved runs/splits/models, logs, CSV, figures, report, app, install and reproduction.

**How the audit was done:**
* I read every training and evaluation path in the code.
* `scripts/audit_checks.py` independently re-derives the numbers from the stored artefacts.
* A reproducibility run happened in a separate fresh clone.
* The app was tested headlessly (Streamlit AppTest) and in a real browser (Playwright demo).

Statuses: **PASS** = the code exists *and* a run, test or artefact was checked. **FAIL** = not met. **BLOCKED** = needs something outside this environment.

## 1. Tests and checks executed

| What | Command | Result |
|---|---|---|
| Unit + leakage + data + app tests | `python -m pytest -q tests` | **28 passed** (107 s). 25 core tests plus 3 headless app tests (all scenes, both crop models, both mineral methods, every layer checkbox, every opacity slider, pixel inspector inputs, export buttons present) |
| Programmatic audit | `python scripts/audit_checks.py` | **All checks PASS** (`outputs/reports/audit_checks.json`) |
| Metric recomputation | inside `audit_checks.py` | OA, AA, κ, macro-F1 and ECE recomputed from all 42 stored confusion matrices (36 main + 6 ablation) and reliability bins. Max abs. difference 1.1e-16 |
| `results.csv` re-aggregation | inside `audit_checks.py` | All 8 rows equal mean (and sample sd, ddof=1) over exactly seeds 0, 1, 2 of the per-seed run files |
| Split re-verification | inside `audit_checks.py` | All 24 saved splits pass `verify_spatial`/`verify_random`, plus an independent brute-force (2r+1)² window check on **every** training pixel |
| Reproducibility | fresh `git clone` into `%TEMP%\semr`, same data, `python scripts/reproduce_all.py --quick` | Exit 0 in 12.9 min. All 8 seed-0 runs reproduce **bit-identically** (OA and ECE differences 0.0000); Cuprite pixel counts and κ identical |
| Clean install | fresh clone, new venv, `pip install -r requirements.txt`, `python run_app.py` | Install 5.4 min. Data download ~15 min on this network. App healthy 5 s after data is present. CNN page rendered in a browser. 25/25 tests passed in that clone at the time |
| Band-noise ablation (new) | `train_crop_models.py --splits spatial --models svm --drop_ranges_nm "1340,1450;1790,1960" --tag nowater` | IP SVM OA 0.648 (vs 0.636); Salinas 0.928 (vs 0.929). Both within seed sd. Default **not** changed (that would be selection after seeing test) |
| Demo video | `scripts/record_demo.py`, then ffmpeg | `outputs/demo/demo.mp4`, 1:49, 13 captioned steps |

## 2. Defects found and fixed during this audit

| # | Defect | Fix | Verified by |
|---|---|---|---|
| D1 | `results.csv` reported `n_train_px = 25,639` for the Salinas spatial SVM, but the SVM is fitted on at most 8,000 pixels (`max_train` cap) | `generate_outputs.py` now reports the pixels actually fitted (`n_svm_fit_px`); the split size is kept as `n_train_split_px` in `results_per_seed.csv` | CSV row now shows 8000; `audit_checks` PASS |
| D2 | App: undefined (NaN) pixels of the index layer (non-crop pixels) and the SAM match-quality layer (unmapped pixels) were blended as **black** over the image | `app.py` blends these layers only where values are finite | `tests/test_app.py` (all layers toggled), demo re-recorded |
| D3 | README claimed the 0.02 depth gate is "≈5× noise σ"; it is ~1–2× in CR units | README corrected; it now states the gate is weak and has no effect at the chosen angle | text |
| D4 | Band **position** was computed but never rendered | `outputs/mineral_maps/cuprite_band_position_maps.png` added (positions shown only where depth ≥ 0.03) | file present |
| D5 | Report did not disclose analysis choices made after seeing results | Report §8 now lists them (see risks R1–R3) | report text |
| D6 | Demo video existed only as a 44 MB git-ignored webm | Converted to a 2 MB `demo.mp4` and committed | file present |
| D7 (earlier) | Download script spent ~2 min per file retrying blocked hosts | Blocked hosts are skipped after the first refusal; exponential back-off for the archive | clean-clone download completed |

## 3. Requirement-by-requirement status

### Audit questions

| # | Question | Status | Evidence |
|---|---|---|---|
| 1 | No train/val/test leakage, incl. CNN patch overlap | **PASS** (spatial) | `splits.apply_buffer`: training pixels within r=3 (the CNN patch radius; 7×7 patch, three valid 3×3 convs → 1×1, test `test_cnn_receptive_field_equals_patch`) of any test **or** val block area are dropped, and val pixels within 3 px of a test block are dropped. `verify_spatial` asserts it. `audit_checks` brute-forces every training pixel of all saved splits. A test injects a leak and expects failure. Test patches *may* contain training pixels (inputs, not labels); this is allowed. The random split leaks by design: that is the experiment. |
| 2 | Spatial split requirements verified programmatically | **PASS** | Block purity, patch clearance, val/test clearance and class coverage (every class ≥50 px in test) are asserted at run time and in tests. Proportions are checked in `audit_checks` (max deviation from 60/20/20 = 0.029; IP seed 0 0.574/0.199/0.227; Salinas ≈0.60/0.20/0.20). **Caveat:** coverage of *train/val* is soft. On IP, val lacks some ≥50-px classes (seed 0: Wheat; seed 1: Corn, Stone-Steel-Towers; seed 2: Grass-trees, Stone-Steel-Towers), so tuning is blind to them. Small classes (<50 px) are sometimes in test but not train (seed 0: Alfalfa), and count as 0 recall in AA. |
| 3 | Preprocessing / PCA / normalisation / feature selection on training data only | **PASS** | `SpectralPreprocessor.fit(Xall[fit_idx])` for the z-score; CNN PCA fitted on `fit_idx` only. Band importance and forward selection use train (fit) + **validation** of the same seed. Importances are never shared across seeds (seed k's val can be seed k′'s test). S2 averaging is a fixed transform; its z-score is fitted on train. |
| 4 | Test labels did not influence selection or tuning | **PASS in code; disclosed exceptions in process** | `test_idx` appears only in final `svm_proba`/`extract_patches` + `evaluate` calls (grep in §4). SVM grid, CNN lr, checkpoint and temperature all use validation. Block size was chosen from class-coverage counts. **Process exceptions (disclosed, R1–R3):** the CNN checkpoint rule change happened after a seed-0 trial whose test OA was visible; forward selection was added after seeing the 15-band test result (both methods reported). |
| 5 | Metrics come from actual runs and are mathematically correct | **PASS** | 42 run JSONs. Independent recomputation from confusion matrices matches to 1e-16. Unit tests check known OA/AA/κ/ECE values. Logs in `experiments/logs/`. |
| 6 | Mean/sd use the actual three seeds | **PASS** | `audit_checks` confirms seeds {0,1,2} per row and recomputes mean and sample sd (ddof=1). Each seed redraws the split, so the sd includes split variance. |
| 7 | Wavelength metadata sourced, documented, consistent | **PASS (IP, Salinas); PASS-with-proxy (Cuprite)** | IP: 92AV3C.spc (220 rows; channel-number gaps at the 3 spectrometer joins). EHU's removed bands map to 1363–1402 / 1819–1933 / 2499 nm, exactly the water windows; same for Salinas with the AV98 table. Cuprite uses the 1998 table as a **proxy**: the 1997 product's zero-filled channels fall at 1365–1395 / 1833–1928 nm in it (asserted). Not an exact 1997 table, as documented. Near-water low-signal bands are retained; the ablation shows a negligible effect. |
| 8 | ≤15-band and Sentinel-2-like experiments | **PASS** | Three selections, each 15 bands, all evaluated on the spatial split over 3 seeds: importance-top-15, forward selection, and random-15 baseline. S2: average of AVIRIS bands with centres inside each S2A band (centre ± bw/2); 12 bands; B10 reported unavailable, not invented. Only the SVM was evaluated (no CNN), stated in the report. |
| 9 | Salinas transfer preserves the protocol | **PASS** | Same code and config: block 15, buffer 3, same grids, 3 seeds, tuning on Salinas validation only. One size-triggered change (SVM cap 8,000) is reported. Weakness: a 15-px block is 55 m at 3.7 m GSD, so the spatial split is weaker there (30-px ablation: SVM 0.890, CNN 0.959). |
| 10 | SAM, CR, resampling, synthetic training, unknown handling are defensible | **PASS with stated limits** | Own Gaussian convolution of native splib07a spectra; RMS vs USGS AV98 convolution ≤0.0056. Convex-hull CR is unit-tested on a known feature and handles non-monotonic axes and NaN. SAM threshold chosen on synthetic labels (95% knee). Synthetic generator (linear + Hapke-style intimate mixing, illumination, image-estimated noise, explicit unknown class + p<0.5 rejection) is unit-tested and deterministic. Limits: one library sample for most minerals; the depth gate is weak; MLP probabilities are uncalibrated. |
| 11 | Mineral validation separates independent evidence from ground truth | **PASS** | App banner, report §7 and summary PDF state that no accuracy is claimed. Synthetic metrics are labelled "synthetic only". Agreement is called agreement. The USGS comparison is explicitly qualitative (no colour key, different extent). |
| 12 | Confidence / probability / match quality / band depth / abundance labelled correctly | **PASS** | CNN: "temperature-scaled softmax"; SVM: "Platt-scaled"; SAM: "match quality = 1 − angle/threshold" on mapped pixels only; MLP: "max probability (NOT calibrated on real data)"; depth: "1 − CR"; abundance: "feature-fit scale … relative abundance proxy … not a calibrated abundance". `results.csv` ECE is **raw**; temperature-scaled ECE is in `results_extended.json`. |
| 13 | App inspector, layers, opacity, exports, summary work | **PASS** | AppTest drives every widget without exceptions. A browser session (demo) shows click inspection on all 3 scenes. Exports: unit test round-trips the PNG header, TIFF array equality and PDF header. TIFFs are explicitly *not georeferenced* (no geo metadata exists). |
| 14 | Report, CSV, maps, demo exist | **PASS** | `reports/technical_report.pdf` (5 pages + appendix), `results.csv` (exact header, 4 required rows + 4 extra), maps under `outputs/` with legends or colour bars, `outputs/demo/demo.mp4` (1:49). |
| 15 | Clean install and reproduction commands run | **PASS (commands) / FAIL (≤10 min on this network)** | Clean clone → install → app works; `reproduce_all.py --quick` exit 0 with bit-identical results. The full ≤10-min clean start was **not** achieved here: data download took ~15 min because EHU/USGS block scripted clients and the Internet Archive rate-limits. Full `reproduce_all.py` is ~90 min (not re-run end to end in this audit; its components ran individually and are logged). |

### Challenge brief checklist

| Brief item | Status | Where |
|---|---|---|
| A1 band centres recovered + source documented; band drops and normalisation justified | PASS | `src/wavelengths.py`, README §2, report §1 |
| A2 random split 10%/class | PASS | `splits.random_split`, verified |
| A2 spatial blocks 15–30 px, ~60/20/20, no training patch touches test, classes ≥50 in test | PASS | §3 Q1–2 |
| A3 classical + deep model, tuned on validation only | PASS (see R1) | `crop_models.py` |
| A4 OA, AA, κ, per-class F1, confusion, ECE + reliability; 3-seed mean ± sd | PASS | `results.csv`, `per_class_f1.csv`, figures |
| A5 random-vs-spatial gap quantified and explained with evidence | PASS | report §5, `gap_evidence.png` (distance-to-train, accuracy vs distance, autocorrelation) |
| A6 band ranking tied to physics; ≤15 bands; S2-like | PASS | report §6 |
| A7 Salinas transfer, what broke | PASS | report "Transfer" section |
| A8 class map, confidence map, NDVI / red-edge for crop classes | PASS | `outputs/class_maps`, `confidence_maps`, `index_maps` |
| B1 library resampled; own Gaussian convolution stated | PASS | report §7 |
| B2 CR 2.0–2.5 µm, band depth/position, SAM | PASS (window 2.00–2.45 µm; the last channels are noisy or outside ASD range, as stated) | `continuum_removal.py`, position + depth maps |
| B3 ML trained only on synthetic data, unknown output | PASS | `synthetic_data.py`, `mineral_models.py` |
| B4 no-label validation: agreement, example spectra, residuals, USGS side-by-side, likely errors | PASS | report §7, figures |
| B5 mineral class map, abundance/band-depth map, confidence map | PASS | `outputs/mineral_maps`, `confidence_maps` |
| C1–C4 app: scenes, inference, layers + opacity, pixel spectrum + confidence, PNG/TIFF + one-page summary | PASS | §3 Q13 |
| C5 clean clone ≤10 min, one command | **FAIL on this network** (works, but ~21 min end to end) | §3 Q15 |
| D1 Git repo + README (setup, app command, reproduce command, time) | PASS | pushed to `github.com/sthitajoshi/GeoMine_Intern_Final-Round` |
| D2 report ≤6 pages + appendix with the 9 sections | PASS | |
| D3 results.csv format | PASS | |
| D4 maps with legends in `outputs/` | PASS | |
| D5 demo video ≤5 min | PASS (file); **BLOCKED** for the submission link: you must upload it and send the link | `outputs/demo/demo.mp4` |

## 4. Code-path evidence for "test labels not used for tuning"

* `scripts/train_crop_models.py`: `test_idx` is used only at the `cm.svm_proba(model, Xs[test_idx])` and `extract_patches(... test_idx ...)` + `evaluate(yall[test_idx], ...)` lines, all after `tune_svm` / `train_cnn` / `fit_temperature`, which receive only `fit_idx` and `val_idx`.
* `scripts/band_experiments.py`: `test_idx` is used only inside `svm_eval` → `evaluate`. Importance and selection use `val_idx`.
* `scripts/run_mineral_mapping.py`: no labels exist. The threshold comes from synthetic labelled data.
* `scripts/gap_analysis.py` reads test labels **after** all models are frozen, for analysis only.

## 5. Remaining risks (understand these before the defence)

* **R1 – Checkpoint rule changed after seeing a test number.** The CNN originally checkpointed on validation loss, which peaked at epoch 2 (over-confidence). I switched to validation OA after a seed-0 trial run whose test OA (0.46 → 0.59) was visible. The motivation was the validation curve, but a strict reviewer may count this as test-informed. Disclosed in report §8.
* **R2 – SAM knee rule chosen after viewing real-scene class counts** (no labels involved). Disclosed.
* **R3 – Forward selection added after seeing the importance-based 15-band test result.** Both methods are reported, and neither beats random-15.
* **R4 – Validation sets miss some classes on Indian Pines**, so tuning ignores them; tiny classes (Alfalfa, Oats, Grass-pasture-mowed) can be in test without being in train. This is structural on a 145×145 scene.
* **R5 – Pixel-level, not field-level, separation.** A field cut by a block edge contributes to both train and test beyond the 3-px buffer. A field-level split is the real fix.
* **R6 – Salinas spatial numbers are optimistic** because 15 px = 55 m is small relative to its fields.
* **R7 – CNN is not better than the SVM on the honest split** (0.616 vs 0.636). Its raw ECE is 0.29; temperature scaling (T ≈ 2.7–3.2) brings it to ~0.08.
* **R8 – Cuprite wavelengths are a 1998 proxy** (expect ~1 channel of position uncertainty). The library has one sample for most minerals.
* **R9 – Mineral map errors:** kaolinite is under-mapped by SAM (alunite and kaolinite share the 2.165 µm band); the MLP's buddingtonite pixels are false positives (CO₂ window edge); the MLP's extra calcite likely includes other 2.33 µm absorbers; dickite is not a target. MLP probabilities are uncalibrated.
* **R10 – Data provenance:** EHU and USGS refused scripted downloads, so files come from the Internet Archive's byte-exact captures of the *official* URLs. SHA-256 checksums and published class counts are verified.
* **R11 – Clean-install time** depends on network speed; it was ~21 min here (download-bound).
* **R12 – `n_train_px` nuance:** random-split CNN rows count all 1,025 training pixels, of which 206 were used only for early stopping and temperature scaling (not gradient updates).
* **R13 – `SVC(probability=True)`** is deprecated in scikit-learn 1.9 (removal in 1.11). Pinned versions avoid breakage.
* **R14 – The demo video has captions but no narration.**

## 6. Commands

```bash
pip install -r requirements.txt
python run_app.py
python scripts/reproduce_all.py
python scripts/reproduce_all.py --quick
python scripts/audit_checks.py
python -m pytest -q tests
```

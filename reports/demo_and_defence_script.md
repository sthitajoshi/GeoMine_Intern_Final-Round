# Demo video, live defence and final self-audit: your script

There are three parts:

* **A.** A word-for-word script for recording the demo video (≤ 5 min).
* **B.** A run sheet for the 30-minute live defence: demo, questions, live change.
* **C.** The final audit you run yourself before you submit.

Every number below comes from `results.csv`, `outputs/reports/` or `reports/final_audit.md`. If you rerun anything, re-check them.

---

## A. Recording the demo video (target 4 min, hard limit 5)

**Setup (before you press record)**

1. `python run_app.py`. Wait for the app at http://localhost:8501. Open Indian Pines once so the inference cache is warm.
2. Close other windows. Set the browser to full screen, zoom 90–100%.
3. Have `outputs/figures/gap_evidence.png` and `outputs/figures/mineral_vs_usgs.png` open in another tab.
4. Recorder: Windows **Win+Alt+R** (Xbox Game Bar) or OBS. Microphone on. Speak slowly.

| Time | On screen (what you do) | What you say |
|---|---|---|
| 0:00–0:20 | App open on Indian Pines | "This is Spectral Earth Mini. It's one hyperspectral pipeline that classifies crops on Indian Pines and Salinas, maps minerals on Cuprite, and serves both in this app. Every accuracy number comes from a spatially separated test set." |
| 0:20–0:50 | Point at the blue banner, then the class map | "The model shown is a small 3-D CNN trained on the *spatial* split. The scene is cut into 15-pixel blocks, and whole blocks go to train, validation or test. A 3-pixel buffer, equal to the CNN's patch radius, guarantees no training patch touches a test pixel. Colours outside the labelled fields are extrapolation; there's no ground truth there." |
| 0:50–1:20 | Click a pixel inside a field | "Clicking a pixel shows its spectrum next to the mean spectrum of its predicted class, the confidence, and the top three classes. This pixel is predicted Soybean-mintill with confidence X, and the ground truth agrees." *(Read the real values off the screen.)* |
| 1:20–1:45 | Switch to RBF-SVM. Tick **Confidence**, move its opacity slider | "Both models load from saved files; nothing retrains. Confidence is low along field edges, where pixels mix. These are calibrated probabilities: temperature-scaled for the CNN, Platt-scaled for the SVM." |
| 1:45–2:05 | Tick **Index**, choose NDRE, keep "crop classes only" | "The index layer shows NDVI or the red-edge index, only on pixels predicted as crops. The values are at-sensor, so they're relative, not absolute." |
| 2:05–2:35 | Scroll to **Summary and export**, click the PDF and PNG buttons | "The summary gives class areas in hectares, mean confidence and the metrics. On the honest spatial split the SVM reaches 0.64 overall accuracy; on a random split it would be 0.80, and the CNN 0.93. That gap is leakage between neighbouring pixels, and it's the main result of the report. I can export the current view as a PNG with a legend, the labels as a TIFF, and a one-page PDF summary." |
| 2:35–2:55 | Switch scene to **Salinas**, click a pixel | "Salinas runs through the same pipeline, unchanged. Accuracy is much higher, 0.97. Part of that is because 15 pixels here is only 55 metres, smaller than the fields, so the spatial split is weaker. The report discusses this." |
| 2:55–3:40 | Switch to **Cuprite**. SAM. Click a red (alunite) pixel | "Cuprite has no pixel ground truth, so the app claims no accuracy; see the warning. The physics baseline removes the continuum between 2.0 and 2.45 microns and matches absorption shapes against USGS library spectra with the spectral angle mapper. This pixel matches alunite at an angle of X radians, below the 0.35 threshold. The plots show the pixel and the library spectrum, both raw and continuum-removed." |
| 3:40–4:10 | Switch to **MLP (synthetic-trained)**. Click **Re-run MLP inference** | "The learned model was trained only on synthetic mixtures of library spectra, with noise matched to the image, and it has an 'unknown' class. Re-running it from the saved model reproduces the stored map. Where SAM maps a mineral, the two methods agree on 92% of pixels. The ML model maps about three times more pixels than SAM, which is the sim-to-real gap I discuss in the report." |
| 4:10–4:30 | Show the `mineral_vs_usgs.png` tab | "As independent evidence, here is our map beside the published USGS map. That comparison is qualitative only. The report lists where I believe the map is wrong, for example kaolinite being absorbed into alunite. Thank you." |

**After recording:** trim the start and end. Check it's under 5:00. Upload as **YouTube unlisted** or **Google Drive (anyone with the link)**. Open the link in a private window to make sure it plays.

> A captioned, voice-less backup exists: `outputs/demo/demo.mp4` (1:49). Use it only if you can't record with your voice.

---

## B. Live defence run sheet (30 min)

### B1. Demo and walkthrough (10 min)

Do section A live (≈4 min), then spend ≈6 min on the report:

1. **Split design (1 min):** open `outputs/figures/split_maps.png`. Point at the white buffer rings. Say: "`verify_spatial` asserts this at run time, and a test injects a leak and checks it gets caught."
2. **The gap (2.5 min)**, the most important paragraph. Open `gap_evidence.png`:
   * "With a random split, 54% of test pixels touch a training pixel. With the spatial split, none do; the nearest is at least 4 pixels away."
   * "Even inside the random split, CNN accuracy falls from 0.97 for test pixels next to a training pixel to 0.80 at 4–6 pixels. Accuracy depends on distance to the training data, and a random split never measures the far end."
   * "Neighbouring pixels are near-duplicates: spectral angle within a class is 1.9° at 1 pixel apart and 4.3° far apart."
   * "So the honest numbers are SVM 0.64 ± 0.04 and CNN 0.62 ± 0.04, not 0.80 and 0.93."
3. **Wavelengths (1 min):** `band_importance.png`. Say: "The red edge at 700–740 nm dominates, then leaf water near 1158 nm and the 2.0–2.35 µm lignin/soil region. 12 Sentinel-2-like bands do as well as all 200 on the honest split; 15 selected bands do not beat 15 random bands."
4. **Minerals (1.5 min):** `mineral_example_spectra.png` and the agreement figure. Then list the four likely errors (kaolinite vs alunite, buddingtonite false positives, extra calcite, dickite missing).

### B2. Questions you should expect (10 min): short answers

| Question | Answer (keep it to this) |
|---|---|
| Why is spatial so much lower than random? | The numbers from B1.2: adjacency, autocorrelation, and accuracy vs distance. A random split measures interpolation within a field; the spatial split measures generalisation to new ground. |
| How do you know nothing leaks? | Buffer = patch radius. The CNN's receptive field is exactly the 7×7 patch (it's tested). `verify_spatial` checks every saved split, and `audit_checks.py` brute-forces every training pixel. |
| Did test labels influence any choice? | In code, no: test indices are only used to predict and score. **Disclose this one yourself:** "I changed the CNN checkpoint rule from validation loss to validation accuracy after a seed-0 trial run where I had seen the test accuracy. The reason was the validation curve, which peaked at epoch 2, but I've disclosed it in the report." |
| Why is the CNN not better than the SVM? | Its advantage is spatial context, which helps when test pixels sit inside training fields. On new fields it overfits field-specific texture, and with ~4.6k training pixels that's expected. Its raw probabilities are over-confident (ECE 0.29); temperature scaling brings that to 0.08. |
| Why 15-pixel blocks? | It's the smallest allowed size. At 20 and 30 pixels, keeping every class in test forces 1–4 classes out of training on Indian Pines. I tested 20 and 30 as well: 30 drops the SVM to 0.46, partly because of those missing classes. |
| Where do the wavelengths come from? | Indian Pines: the 92AV3C.spc calibration file. Salinas: the splib07 AVIRIS 1998 table. Cuprite: the 1998 table as a proxy, because the 1997 one wasn't retrievable. The proof: the removed and zeroed bands land exactly on the water windows. |
| Why the 3D CNN size you chose? | 96k parameters, 7×7×30 PCA patches. It trains in under a minute on Indian Pines on a CPU, and its receptive field equals the patch, which makes the buffer argument exact. |
| What does "confidence" mean for Cuprite? | SAM: match quality = 1 − angle/threshold. MLP: max probability, not calibrated on real data. Band depth = 1 − continuum-removed value. Abundance is a relative fit scale, not a calibrated abundance. |
| What would you do with two more weeks? | Field-level splits, a block size set in metres, more library samples (and dickite as a target), a Tetracorder-style multi-feature fit, and georegistering the USGS map for a quantitative comparison. |

### B3. Live change (10 min): rehearse each once

Each command below takes about 1–3 minutes on Indian Pines (SVM only is fastest). Say what you expect *before* you run it.

| Likely request | Command | What to expect / say |
|---|---|---|
| Double the block size | `python scripts/train_crop_models.py --scenes indian_pines --splits spatial --models svm --block 30 --tag b30` | Lower OA (we measured 0.46 ± 0.06). Larger blocks mean test pixels are farther from training, *and* more classes missing from training. |
| Remove 1.3–1.5 µm bands | `python scripts/train_crop_models.py --scenes indian_pines --splits spatial --models svm --drop_ranges_nm "1300,1500" --tag no1300` | Small change, within the seed sd. Dropped bands are listed in the run JSON under `preprocessing`. |
| Add a class (mineral) | Add the record to `RECORDS` / `TARGETS` (`src/mineral_library.py`), its feature to `FEATURES` (`src/continuum_removal.py`), and a colour (`src/visualization.py`); then `python scripts/run_mineral_mapping.py` | The record must already be downloaded (dickite and montmorillonite are, as distractors). Moving dickite from "other" to "target" is the easiest live demo. |
| Replace SAM with another matcher | Write e.g. `sid_classify(X, refs, threshold)` in `src/sam.py` with the same return format, and swap the call inside `sam_on_cr` in `scripts/run_mineral_mapping.py` | Re-pick the threshold the same way (synthetic data); compare agreement with the MLP. |

To read a result after a run: open `experiments/runs/<scene>_spatial_svm_s0_<tag>.json` → `metrics.oa`, or `grep OA= experiments/logs/train_crop_models_<tag>.log`.

---

## C. Final self-audit before you submit (≈ 10 minutes)

Run these in order from the repo folder. Each line says what "good" looks like.

| # | Command / action | Pass if |
|---|---|---|
| 1 | `git status` | "nothing to commit, working tree clean" |
| 2 | `git log origin/main -1 --oneline` and `git log -1 --oneline` | Both show the same commit, i.e. GitHub has your latest work. If not: `git push` |
| 3 | `python -m pytest -q tests` | `28 passed` |
| 4 | `python scripts/audit_checks.py` | Ends with `FAILED: none` |
| 5 | Open `results.csv` | Header exactly `scene,split,model,oa_mean,oa_std,aa_mean,kappa_mean,macro_f1_mean,ece_mean,n_train_px,n_test_px`; the 4 required rows are first; no blank cells |
| 6 | Open `reports/technical_report.pdf` | Body ≤ 6 pages (appendix starts on p. 6); all 9 sections present; figures readable |
| 7 | Check `outputs/class_maps/`, `confidence_maps/`, `mineral_maps/` | Maps have legends or colour bars |
| 8 | In a private browser window, open your GitHub repo URL | It's visible to the reviewers (public, or they're added as collaborators); the README renders |
| 9 | Optional clean-clone check: `git clone <repo-url> test_clone`, `cd test_clone`, `pip install -r requirements.txt`, `python run_app.py` | The app opens. (Data download speed depends on the network; the README says so.) |
| 10 | Open the video link in a private window | It plays, is ≤ 5 min, and shows all three scenes |
| 11 | Re-read `reports/final_audit.md` §5 "Remaining risks" | You can explain each one in a sentence |

**Email checklist:** the repo link, `technical_report.pdf` attached, and the video link. Send before 23:59 IST on day 5. Optional last line: "AI assistance is disclosed in the README; a requirement-by-requirement audit is in `reports/final_audit.md`."

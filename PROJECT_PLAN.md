# Project Plan — Temporal Bushfire Segmentation

UTS 31005 Machine Learning, Spring 2026 — Assignment 2 (Option 2: practical ML system).
Related to (but kept separate from) the TRM project `au-bushfire-boundary-detection`.

This file is the stable premise of the project.
- Open design choices, with their options and trade-offs, live in `DECISIONS.md`. Anything marked
  **TBD (Dn)** here points to decision `Dn` there.
- Decisions and changes over time go in `JOURNAL.md`.
- If anything here changes, update this file **and** log it in the journal.

---

## 1. Research questions

**RQ1 (primary).** On FLOGA (Greek wildfires, Sentinel-2), does adding a pre-fire image to a
post-fire image improve burnt-area segmentation compared with post-fire-only input, under
otherwise matched conditions (same backbone, bands, loss, optimiser, data, seeds)?

**RQ2 (secondary).** When models trained on FLOGA are applied zero-shot to held-out Australian
fires, does the bi-temporal advantage hold, grow, or shrink?

### Why this is an open question

The FLOGA paper (Sdraka et al., 2024) adopts bi-temporal input specifically to avoid false
positives on previously burnt land and spectrally similar surfaces (water, dark soil, shadows,
harvested fields). However, every deep learning model in its benchmark is bi-temporal, so the
paper never ablates the pre-fire image. Its Siamese vs. pseudo-Siamese comparison (Table VII)
varies weight sharing, not whether pre-fire information is present. Meanwhile, uni-temporal
post-fire segmentation is common in the literature (e.g. Knopp et al. 2020; Hu et al. 2021).
The size of the pre-fire benefit is assumed rather than measured.

### Hypotheses

- **H1.** Bi-temporal input improves event-level F1/IoU on FLOGA, driven mainly by higher
  **precision** (fewer false positives) rather than higher recall.
- **H2.** The gain concentrates on pixels that look burnt without being burnt by *this* fire:
  previously burnt areas, water, shadow, bare or dark soil, and cropland.
- **H3.** The gap is larger on Australian fires, where frequent reburning makes older fire
  scars more common in post-fire imagery.

---

## 2. Task definition (system I/O)

### Training

- **Input — uni-temporal:** post-fire Sentinel-2 L2A patch, tensor `C × 256 × 256`.
  - C = 9 bands: B02, B03, B04, B05, B06, B07, B8A, B11, B12.
  - All bands at 20 m; 10 m bands resampled to 20 m, matching FLOGA.
- **Input — bi-temporal:** the pre-fire and post-fire patches of the same location on an
  identical pixel grid.
- **Pixel values:** bottom-of-atmosphere **surface reflectance**, the fraction of sunlight
  reflected by the ground per band (physically ~0–1).
  - Stored by ESA as integers × 10,000, with a +1000 offset for scenes processed from
    baseline 04.00 (Jan 2022) onward.
  - Values are converted to physical reflectance, outliers clipped (**TBD D1**) and normalised
    (**TBD D2**), identically for FLOGA and AU.
- **Target:** per-pixel label, `256 × 256`, values:
  - `1` = burnt by this event;
  - `0` = not burnt by this event, **including previously burnt land**;
  - `255` = ignore (no data, cloud or smoke, uncertain), excluded from loss and metrics.

### Deployment / inference

- **Input:** a pre/post image pair, or a post-only image, for a fire of arbitrary size. It is
  tiled into 256 × 256 patches.
- **Output:** per-pixel logit z; the probability of "burnt by this event" is σ(z) = 1/(1+e^−z).
  - Thresholded (**TBD D21**) to a binary mask and stitched back to scene extent.
  - Burnt area (ha) = burnt pixel count × 0.04.

### Where input assumptions break

Most inputs satisfy the assumptions, but not all:
- residual haze or smoke not caught by cloud screening;
- long pre/post acquisition lags (FLOGA: 0–119 days, median 10), allowing regrowth or unrelated
  change;
- co-registration error between dates;
- the older-burnt vs this-event distinction depending on annotator judgement.

---

## 3. Models

| ID  | Name | Input | Notes |
| --- | --- | --- | --- |
| M1  | Uni-temporal | post (9 ch) | Baseline. |
| M2  | Early fusion | concat(pre, post) (18 ch) | Identical to M1 except the first layer; cleanest RQ1 comparison. |
| M3* | Siamese, shared encoder | pre and post through one encoder | Stretch goal. Fusion op **TBD D8**. |

- **Backbone, depth/width and normalisation layer:** **TBD (D5, D6, D7)**. Must be identical
  across models, explainable layer by layer, and trainable in Colab.
- Report parameter counts for every model.
- **Capacity control (stretch):** M1 widened to match M2/M3 parameter count.

**Theory note.** With early fusion, M1's hypothesis space is a subset of M2's: setting M2's
first-layer weights on the pre-fire channels to zero reproduces M1 exactly. Any failure of M2 to
beat M1 must therefore come from estimation or optimisation, not representational capacity.

---

## 4. Training protocol — TBD

All items are open; see `DECISIONS.md`.

- **D3:** augmentation
- **D9:** loss
- **D10:** pixel weighting
- **D11:** optimiser
- **D12:** learning rate and schedule
- **D13:** batch size, epochs, early stopping
- **D14:** hyperparameter fairness across models
- **D15:** training patch composition
- **D17:** checkpoint selection
- **D18:** seeds

The constraint that is already fixed: every model gets the same data, the same budget, and a
fairness policy decided up front under D14.

---

## 5. Data

### FLOGA (train / val / test)

- **Source:** FLOGA V2 GeoTIFF release (Hugging Face), ~1.1 TB locally including MODIS. MODIS
  is unused.
- **Split:** by event using the official `data_split.csv`, never random.
  - **Blocker:** confirm that V2 filename event IDs map to `data_split.csv`'s `event_id`.
- **Preprocessing** (run locally, `src/prepare_floga.py`):
  1. Select the 9 bands at 20 m.
  2. Tile into 256 × 256 non-overlapping patches.
  3. Drop patches with more than 90% water, following FLOGA.
  4. Store compact arrays (uint16 imagery, uint8 labels) plus a patch index CSV with event_id,
     split and burnt fraction.
- **Labels:**
  - Check FLOGA V2's label encoding, in particular how "other/older burnt areas" are marked
    (paper Fig. 15).
  - Map this event → 1, older burnt → 0, and keep a separate `prev_burnt` mask for stratified
    evaluation.
- **Auxiliary layers:** FLOGA's CLC land-cover and water masks are kept for stratified error
  analysis only. They are not model inputs.
- **Caveat:** V2 imagery was reprocessed by Copernicus, so its pixel values differ from V1.
  Compare against published BAM-CD numbers with that caveat.

### Australian test set (RQ2 only — frozen, never used for training or tuning)

- **Size:** at least 10 fires from the annotation pipeline in `au-bushfire-boundary-detection`.
- **Bands:** the download script must fetch the same 9 bands as FLOGA.
  - **To verify:** radiometric harmonisation (processing-baseline offset) between the Earth
    Search L2A scenes and FLOGA V2.
- **Label mapping:**
  - `burnt` → 1;
  - `burnt_previous` → 0, and also kept in the `prev_burnt` mask;
  - `water` → 0;
  - `cloud`, `cloud_shadow`, `smoke_dense`, `uncertain` → 255.
  - Polygon rasterisation rule: **TBD D4**.
- **Land cover for stratified analysis.** CORINE (CLC) is Europe-only.
  - The Australian equivalent is **DEA Land Cover** (Geoscience Australia): FAO LCCS taxonomy,
    25 m, annual.
  - **To check:** which years it covers relative to each fire, and pick the last pre-fire year.
  - **NVIS** (National Vegetation Information System) Major Vegetation Groups is a possible
    alternative, with finer vegetation types.
  - Neither matches CLC classes, so build a documented **crosswalk to shared coarse classes**:
    forest/woody, shrub, grassland/herbaceous, cropland, bare, water, urban.
  - This is used for evaluation stratification only; it is not needed for annotation.
- **Pre/post image selection rule, matching FLOGA's intent:**
  - pre = last clear image before ignition;
  - post = first clear image after the final extent is reached.
- **Limitation — acquisition lag.** Public Australian records do not consistently state when a
  fire was contained or extinguished, so "days since fire end" can't be measured reliably.
  - Possible proxy: the date of the **last satellite hotspot detection** (DEA Hotspots or NASA
    FIRMS) within the final extent.
  - Otherwise record the lag from ignition and state the limitation.
  - Long-running AU fires also mean early-burnt areas may already be regreening by the post
    image date.

---

## 6. Evaluation — TBD

All items are open; see `DECISIONS.md`:
- **D16:** evaluation population
- **D19:** metric set
- **D20:** aggregation
- **D21:** decision threshold
- **D22:** uncertainty and significance
- **Section 8:** loss vs. task objective

Initial leanings (not final):
- recall and F1 as the primary pair, reflecting that under-mapping is the costlier error;
- metrics computed per event;
- the target use case, which sets the cost direction, is to be confirmed in DECISIONS §8 step 1.

Analyses already committed to, independent of metric choice:
- **Stratified error analysis (H2):** errors by land-cover class, on water pixels and on
  `prev_burnt` pixels, M1 vs M2.
- **Pre-image ablation (inference only, no retraining):** evaluate trained M2/M3 with the
  pre-fire image replaced by (a) the post image and (b) a pre-fire image from a different event.
- **Qualitative:** best and worst events per model, plus the events where M1 and M2 disagree
  most.

---

## 7. Scope tiers

- **Core — main objectives:**
  - `prepare_floga.py`;
  - M1 and M2 × seeds;
  - per-event evaluation and significance testing;
  - stratified analysis;
  - pre-image ablation;
  - notebook on Colab;
  - report and journal.
- **Secondary:** AU zero-shot test (RQ2) on at least 10 fires.
- **Stretch / if time permits:**
  - M3 Siamese;
  - parameter-matched M1;
  - additional seeds;
  - loss / threshold strategy comparison (DECISIONS §8 step 3);
  - boundary metric.

---

## 8. Repo and working conventions

- The theory-bearing code lives in small `src/` modules: `models.py`, `losses.py`, `metrics.py`,
  `train.py`.
  - The notebook imports them and displays their source next to the corresponding maths.

- **Tests** (`tests/`, pytest, CPU-only):
  - metrics on hand-computed cases;
  - the masked loss ignores `255`;
  - no event appears in more than one split;
  - pre and post arrays share shape and grid;
  - forward-pass shapes for every model.
  - **AU label ↔ imagery alignment:** rasterised labels share CRS, transform (origin and 20 m
    pixel size) and shape with the image. Plus a signal check: mean dNBR inside burnt labels is
    clearly higher than outside, and it peaks at zero pixel shift (a shifted label drops the
    contrast). This catches misregistration, not just metadata mismatch.
  - **Preprocessing reproducibility:** the raw-sample demo output matches the published patches.
- **CI:** a GitHub Action runs pytest on push.
- **`JOURNAL.md`:** dated entries tagged `[DECISION]`, `[CHALLENGE]`, `[AI]` or `[GAP]`. These
  map directly to the required implementation log. Each `DECISIONS.md` choice gets a
  `[DECISION]` entry with its reasoning.

---

## 9. Core decisions and plan project work order

## Decisions and justifications
 
**Key constraint:** the friend's 16GB-VRAM PC may only be available for a day, or for limited hours of it. Fallback is the 4GB RTX 3050 laptop.
 
### Preprocessing and data
 
- **D1 (outlier clipping): open.** Decide later.
- **D2: per-band z-score.**
  - Statistics from training events only, one shared set for pre and post images, nodata (65535) excluded. AU uses FLOGA's statistics, which keeps RQ2 genuinely zero-shot.
  - *Why:* preserves absolute reflectance and the pre→post change. Per-image normalisation would erase the change signal the bi-temporal model relies on.
- **D3: random flips and 90° rotations.**
  - Applied identically to pre, post and label, with the same policy for every model.
  - *Why:* only ~334 positive training patches, so overfitting is a real risk. These transforms are label-preserving for overhead imagery, and an identical policy keeps the M1/M2 comparison fair.
- **D4: `all_touched=False` (pixel centre inside polygon).**
  - *Why:* `True` fattens AU labels by about one pixel around every perimeter, which would create an artificial recall drop against FLOGA's labels. It is also consistent with treating FN ≈ FP.
- **D15: balanced 1:1 (positive : negative patches).**
  - Negatives drawn randomly from all land patches, including label-2 (older burn) patches. Fixed seed, chosen patch IDs saved to the index so every model and seed trains on the same set.
  - *Why:* follows the natural data distribution with no hand-engineered sampling rule to justify, and matches FLOGA.
  - *Known limitation:* only ~14 old-burn negatives expected in training. Record the actual count, and mention it if M2 shows little advantage on old burns.
### Model
 
- **D5: U-Net, no attention.** Still to decide: plain vs residual connections, before building.
  - *Why:* explainable layer by layer, standard in the burnt-area literature, suited to small data. The research question is about input, not architecture.
- **D6: depth 4, base width 32.** Move to 64 only if the model underfits.
  - *Why:* ~334 positives and limited GPU time. Base 64 is about 4× the parameters and compute.
- **D7: GroupNorm.**
  - *Why:* works at any batch size, so the same model trains identically on the 4GB and 16GB machines. BatchNorm is unreliable at small batches.
- **D8: M3 (Siamese) parked** for this project.
### Training
 
- **D9: masked BCE, ignoring label 255.**
  - *Why:* post-fire mapping is not real-time, so false negatives are not necessarily costlier than false positives. BCE gives calibrated probabilities and was FLOGA's best-performing loss.
- **D10: uniform weighting for the core runs.** Event-based or class-based weighting is an optional ablation.
- **D11: Adam.**
  - *Why:* robust default.
- **D12: cosine annealing (leaning; constant is the alternative).**
  - *Why:* a fixed epoch budget pairs naturally with best-checkpoint selection.
- **D13: batch size and epochs TBD** from the smoke test.
- **D14: identical hyperparameters for all models.**
  - *Why:* the input is the only difference between models. Because M1's hypothesis space is a subset of M2's, any gap reflects estimation or optimisation, not capacity.
- **D17: select checkpoints by per-event F1** on balanced validation patches grouped by event. Full scenes are used only for the final test.
  - *Why:* aligns selection with per-event evaluation. Full-scene validation every epoch would cost too much of the limited GPU time.
- **D18: 3 seeds.** Within a seed, both models see identical batches (data loader seeded separately from model initialisation, deterministic cuDNN).
### Evaluation
 
- **D16: evaluate on both** balanced test patches (comparable to FLOGA) and full scenes (operational performance).
- **D19–D22 and DECISIONS §8: deferred to analysis.**
  - *Why deferral is safe:* only training needs the friend's PC. Evaluation reruns on the 4GB laptop from saved checkpoints.
  - The D9 reasoning already implies the §8 use case, so write it down when you get there.
---
 
## Process
 
1. **Decisions.** Done, apart from D1, D5 (plain vs residual) and D13.
2. **Preprocessing.**
   - Write the scripts: ignore irrelevant data, slim the dataset, then write `dataset.py`.
   - Run on ~5 events first, with preprocessing tests.
3. **Core model code:** `models.py`, `losses.py`, `train.py`.
4. **Supporting code:** `metrics.py`, `evaluate.py`, `visualisation.py`.
   - The notebook calls these scripts and holds the full process.
5. **Simple end-to-end test.**
6. **Remaining tests and CI.**
7. **Full preprocessing run** over the whole dataset.
8. **Smoke test on the 4GB laptop** to estimate compute, then make final adjustments.
   - Rehearse a fresh-clone setup, with the dataset path read from a config file.
9. **Train M1 and M2** across all seeds on the friend's PC.
   - Bring back checkpoints, logs and manifests on the SSD.
10. **Evaluation and analysis.**
11. **Tidy the notebook.**
12. **AU zero-shot**, ideally 10 fires.
13. **AU analysis.**
14. **Report and journal.**

---
## References

- Hu, Ban & Nascetti (2021). Uni-temporal multispectral imagery for burned area mapping with deep
  learning. *Remote Sensing*, 13(8).
- Knopp et al. (2020). A deep learning approach for burned area segmentation with Sentinel-2
  data. *Remote Sensing*, 12(15).
- Sdraka et al. (2024). FLOGA: A machine-learning-ready dataset, a benchmark, and a novel deep
  learning model for burnt area mapping with Sentinel-2. *IEEE JSTARS*, 17, 7801–7824.
- Further method references: see `DECISIONS.md`.

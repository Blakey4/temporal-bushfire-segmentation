# Data

## Source
**FLOGA V2** (Sdraka et al., 2024), GeoTIFF release on Hugging Face:
`orion-ai-lab/FLOGA-GeoTIFFs`, ~1.1 TB.

```
hf download orion-ai-lab/FLOGA-GeoTIFFs --repo-type dataset --local-dir <DATA_ROOT>
```

This gives `<DATA_ROOT>/sen2_20_mod_500/00000.tar … 00055.tar` (56 shards, ~20 GB each).
Annotation polygons and fire dates are in <https://github.com/Orion-AI-Lab/FLOGA-annotations>.

## Raw format (FLOGA V2)
- **461 rasters** covering 343 fires, 2017–2021. A *raster* is one fire seen in one Sentinel-2 tile.
  Large fires span 2–4 rasters.
- Tar members are named `…/sen2_20_mod_500/<year>/<fire_id>_<n>_<layer>.tiff`:
  - `fire_id` is the FLOGA-annotations `ID` field;
  - `n` numbers the (fire, tile) rasters within a year.
- One raster's files are scattered over 4–12 shards. They are read in place with GDAL
  `/vsisubfile/<offset>_<size>,<shard.tar>`, so nothing needs extracting.
- All layers of a raster share one grid:
  - EPSG:4326, pixel ~0.000206° (≈ 18 m × 23 m);
  - ~4600 × 6200 px (roughly a 1° × 1° area).

| Layer (file suffix) | Contents | Used |
|---|---|---|
| `SEN2_2a_20_before` / `_after` | Pre-/post-fire Sentinel-2 L2A, uint16, 10 bands `B01,B02,B03,B04,B05,B06,B07,B11,B12,B8A` (note that B8A is last) | yes |
| `SEN2_2a_20_label` | 0 = not burnt, 1 = burnt by this fire, 2 = burnt by other fires in the same year | yes |
| `SEN2_2a_cloud_mask_20_before` / `_after` | Sentinel-2 Scene Classification Layer (SCL), 0–11 | yes |
| `SEN2_2a_20_sea_mask_after` (and identical `_before`) | Land = {0, 2, 4}; water = everything else | yes |
| `SEN2_2a_20_clc_mask` | CORINE land cover, classes 1–44 (0 / 128 = none) | kept, analysis only |
| `MODIS_*` (5 files) | MODIS imagery, QC and label | no |

**Pixel values:**
- Nodata is **DN 0**. The files declare 65535, but that value is never used.
- Every year carries the **+1000 offset** (FLOGA V2 is reprocessed), so
  reflectance = (DN − 1000) / 10000. A few valid pixels (dark water) are slightly below 1000.

## Prepared dataset (`src/prepare_floga.py`)
Run from the notebook:
```python
from src.prepare_floga import prepare
from src.make_splits import make_splits
index = prepare(DATA_ROOT, OUT_DIR)      # fires=[784755, 637342] to test on a few fires
splits = make_splits(OUT_DIR, seed=0)
```

**Label rules.** 1 = this fire, 0 = not burnt, **255 = ignore**. Ignore covers:
- FLOGA label 2;
- no image data (SCL 0, or any band 0);
- cloud in either image (SCL 8, 9 or 10).

Dark-area (SCL 2) and cloud shadow (SCL 3) are **not** masked: 12.9% of burnt pixels are classed as SCL 2 in the post-fire image.

**Patches.**
- Non-overlapping 256 × 256 patches, with raster edges padded as nodata.
- A patch is dropped if it is all ignore, ≥ 90% water, or > 15% cloud.
- Per raster, all positive patches are kept, plus an equal number of random negatives (seed 0).

**Output (`OUT_DIR`):**
- `patches/<year>_<fire_id>_<n>_r<row>_c<col>.npz`, containing:
  - `pre`, `post`: 9 × 256 × 256 uint16 raw DN, bands `B02,B03,B04,B05,B06,B07,B8A,B11,B12`;
  - `label`: uint8, values 0/1/255;
  - `label_raw`: FLOGA's original 0/1/2 label;
  - `clc`: CORINE class.
- `patch_index.csv`: one row per patch, with fire, raster, position, `is_positive`, `burnt_px`,
  `ignore_frac`, `cloud_frac`, `water_frac` and `other_burnt_px`.
- `tar_index.csv`: cached location of every needed file inside the shards.
- `splits.csv`: fire → train/val/test, grouped by fire and stratified by year, 60/20/20.

Offset removal, clipping and normalisation happen in `dataset.py`, not here.

## Known quirks
- Some negative patches contain visible burn scars with no label (FLOGA label noise).
- Some label-1 areas lie under cloud in one of the images. They become 255, and the patch is dropped if cloud > 15%.

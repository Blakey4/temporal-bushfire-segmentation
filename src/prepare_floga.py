"""Cut the raw FLOGA V2 download into 256x256 training patches.

Input: the folder the Hugging Face download went into (it contains sen2_20_mod_500/*.tar).
Each tar shard holds GeoTIFFs named <fire_id>_<n>_<layer>.tiff, where one (fire_id, n) pair is
one fire seen in one Sentinel-2 tile (a "raster"). A raster's files are spread over several
shards, so files are read in place from inside the tars and nothing is extracted.

For every raster this script:
  1. builds the training label (1 = burnt by this fire, 0 = not burnt, 255 = ignore),
  2. cuts the scene into non-overlapping 256x256 patches,
  3. drops patches that are empty, mostly water or too cloudy,
  4. keeps every positive patch plus the same number of random negative patches (1:1),
  5. saves each kept patch as an .npz file and records it in patch_index.csv.

Called from the notebook:
    from src.prepare_floga import prepare
    index = prepare(DATA_ROOT, OUT_DIR)            # or fires=[784755, 637342] to test
"""
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

SHARD_DIR = "sen2_20_mod_500"

# Band order inside the FLOGA SEN2 files, and the 9 bands we keep (B01 dropped, B8A moved
# next to the other red-edge / NIR bands).
BANDS_IN = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B11", "B12", "B8A"]
BANDS_OUT = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
BAND_INDEXES = [BANDS_IN.index(b) + 1 for b in BANDS_OUT]  # rasterio bands are 1-based

PATCH = 256
IGNORE = 255
CLOUD_CLASSES = [8, 9, 10]  # Sentinel-2 SCL: cloud medium prob., cloud high prob., thin cirrus
LAND_CLASSES = [0, 2, 4]    # FLOGA sea mask values that are land; every other value is water
MAX_CLOUD = 0.15            # drop a patch if more than 15% of it is cloud (pre or post)
MAX_WATER = 0.9             # drop a patch if 90% or more of it is water (as FLOGA does)
SEED = 0

# Short layer name -> FLOGA V2 file suffix. Everything else in the tars (MODIS) is ignored.
LAYERS = {
    "pre": "SEN2_2a_20_before",
    "post": "SEN2_2a_20_after",
    "label": "SEN2_2a_20_label",
    "scl_pre": "SEN2_2a_cloud_mask_20_before",
    "scl_post": "SEN2_2a_cloud_mask_20_after",
    "sea": "SEN2_2a_20_sea_mask_after",
    "clc": "SEN2_2a_20_clc_mask",
}
SUFFIX_TO_LAYER = {suffix: name for name, suffix in LAYERS.items()}
MEMBER_NAME = re.compile(r"(\d{4})/(\d+)_(\d+)_(.+)\.tiff$")


def index_tars(data_root, cache=None):
    """List where each needed file sits inside the tar shards.

    Returns one row per file: year, fire_id, n, layer, shard, offset, size. Scanning the tar
    headers of the full download takes a few minutes, so the result is cached to `cache`.
    """
    if cache is not None and Path(cache).exists():
        return pd.read_csv(cache)

    rows = []
    for shard in sorted((Path(data_root) / SHARD_DIR).glob("*.tar")):
        with tarfile.open(shard) as tar:
            for member in tar:
                match = MEMBER_NAME.search(member.name)
                if match is None or match.group(4) not in SUFFIX_TO_LAYER:
                    continue
                year, fire_id, n, suffix = match.groups()
                rows.append(dict(year=int(year), fire_id=int(fire_id), n=int(n),
                                 layer=SUFFIX_TO_LAYER[suffix], shard=shard.name,
                                 offset=member.offset_data, size=member.size))
    files = pd.DataFrame(rows)
    if cache is not None:
        files.to_csv(cache, index=False)
    return files


def open_layer(data_root, file):
    """Open one GeoTIFF directly inside its tar shard (GDAL reads just that byte range)."""
    tar_path = (Path(data_root) / SHARD_DIR / file["shard"]).as_posix()
    return rasterio.open(f"/vsisubfile/{file['offset']}_{file['size']},{tar_path}")


def read_layer(data_root, file):
    """Read a whole single-band layer (label and masks are uint8, ~30 MB each)."""
    with open_layer(data_root, file) as src:
        return src.read(1)


def read_patch(src, row, col, bands):
    """Read a 256x256 window, padding with 0 (= nodata) where it runs past the image edge."""
    height = min(PATCH, src.height - row)
    width = min(PATCH, src.width - col)
    data = src.read(bands, window=Window(col, row, width, height))
    return np.pad(data, ((0, 0), (0, PATCH - height), (0, PATCH - width)))


def pad_to_grid(array, fill):
    """Pad the bottom/right edges so height and width are multiples of PATCH."""
    height, width = array.shape
    return np.pad(array, ((0, -height % PATCH), (0, -width % PATCH)), constant_values=fill)


def make_label(label_raw, nodata, scl_pre, scl_post):
    """Turn FLOGA's 0/1/2 label into our training label.

    1 = burnt by this fire, 0 = not burnt, 255 = ignore (left out of loss and metrics):
      - FLOGA label 2: burnt by another fire in the same year; it may or may not look burnt
        depending on the image dates, so it is neither a clean positive nor a clean negative;
      - no image data;
      - cloud in the pre-fire or post-fire image.
    """
    label = (label_raw == 1).astype(np.uint8)
    cloud = np.isin(scl_pre, CLOUD_CLASSES) | np.isin(scl_post, CLOUD_CLASSES)
    label[(label_raw == 2) | nodata | cloud] = IGNORE
    return label


def select_patches(label, cloud, water, rng):
    """Choose which grid patches of one raster to keep.

    A patch is dropped if every pixel is ignore, if it is >= MAX_WATER water, or if it is
    > MAX_CLOUD cloud. A patch is positive if it has at least one burnt pixel. All positives
    are kept, plus the same number of randomly chosen negatives.
    """
    candidates = []
    height, width = label.shape
    for row in range(0, height, PATCH):
        for col in range(0, width, PATCH):
            window = (slice(row, row + PATCH), slice(col, col + PATCH))
            if (label[window] == IGNORE).all():
                continue
            cloud_frac = float(cloud[window].mean())
            water_frac = float(water[window].mean())
            if water_frac >= MAX_WATER or cloud_frac > MAX_CLOUD:
                continue
            candidates.append(dict(row=row, col=col, is_positive=bool((label[window] == 1).any()),
                                   cloud_frac=cloud_frac, water_frac=water_frac))

    positives = [p for p in candidates if p["is_positive"]]
    negatives = [p for p in candidates if not p["is_positive"]]
    n_neg = min(len(positives), len(negatives))
    picked = sorted(rng.choice(len(negatives), size=n_neg, replace=False))
    return positives + [negatives[i] for i in picked]


def prepare_raster(data_root, out_dir, year, fire_id, n, files):
    """Build the label for one raster, choose its patches and save them. Returns index rows."""
    label_raw = read_layer(data_root, files.loc["label"])
    scl_pre = read_layer(data_root, files.loc["scl_pre"])
    scl_post = read_layer(data_root, files.loc["scl_post"])
    sea = read_layer(data_root, files.loc["sea"])

    # SCL class 0 marks pixels with no image data.
    nodata = (scl_pre == 0) | (scl_post == 0)
    label = pad_to_grid(make_label(label_raw, nodata, scl_pre, scl_post), IGNORE)
    label_raw = pad_to_grid(label_raw, 0)
    cloud = pad_to_grid(np.isin(scl_pre, CLOUD_CLASSES) | np.isin(scl_post, CLOUD_CLASSES), False)
    water = pad_to_grid(~np.isin(sea, LAND_CLASSES), False)

    rng = np.random.default_rng([SEED, year, fire_id, n])
    chosen = select_patches(label, cloud, water, rng)

    rows = []
    with open_layer(data_root, files.loc["pre"]) as pre_src, \
         open_layer(data_root, files.loc["post"]) as post_src, \
         open_layer(data_root, files.loc["clc"]) as clc_src:
        for patch in chosen:
            row, col = patch["row"], patch["col"]
            window = (slice(row, row + PATCH), slice(col, col + PATCH))
            pre = read_patch(pre_src, row, col, BAND_INDEXES)
            post = read_patch(post_src, row, col, BAND_INDEXES)
            clc = read_patch(clc_src, row, col, [1])[0]

            # Also ignore the odd pixel where a single band has no data.
            patch_label = label[window].copy()
            patch_label[(pre == 0).any(axis=0) | (post == 0).any(axis=0)] = IGNORE
            patch_raw = label_raw[window]

            patch_id = f"{year}_{fire_id}_{n}_r{row:05d}_c{col:05d}"
            np.savez(out_dir / "patches" / f"{patch_id}.npz", pre=pre, post=post,
                     label=patch_label, label_raw=patch_raw, clc=clc)
            rows.append(dict(patch_id=patch_id, year=year, fire_id=fire_id, raster=f"{fire_id}_{n}",
                             row=row, col=col, is_positive=patch["is_positive"],
                             burnt_px=int((patch_label == 1).sum()),
                             ignore_frac=float((patch_label == IGNORE).mean()),
                             cloud_frac=patch["cloud_frac"], water_frac=patch["water_frac"],
                             other_burnt_px=int((patch_raw == 2).sum())))
    return rows


def prepare(data_root, out_dir, fires=None):
    """Run the whole preprocessing. `fires` optionally limits it to a list of fire IDs."""
    data_root, out_dir = Path(data_root), Path(out_dir)
    (out_dir / "patches").mkdir(parents=True, exist_ok=True)

    files = index_tars(data_root, cache=out_dir / "tar_index.csv")
    if fires is not None:
        files = files[files["fire_id"].isin(fires)]

    rasters = files.groupby(["year", "fire_id", "n"])
    rows = []
    for i, ((year, fire_id, n), raster_files) in enumerate(rasters, start=1):
        raster_files = raster_files.set_index("layer")
        missing = set(LAYERS) - set(raster_files.index)
        if missing:
            print(f"[{i}/{len(rasters)}] {year} {fire_id}_{n}: skipped, missing {sorted(missing)}")
            continue
        new_rows = prepare_raster(data_root, out_dir, year, fire_id, n, raster_files)
        n_pos = sum(r["is_positive"] for r in new_rows)
        print(f"[{i}/{len(rasters)}] {year} {fire_id}_{n}: {n_pos} positive, "
              f"{len(new_rows) - n_pos} negative patches")
        rows += new_rows

    index = pd.DataFrame(rows)
    index.to_csv(out_dir / "patch_index.csv", index=False)
    if len(index):
        print(f"Saved {len(index)} patches ({int(index['is_positive'].sum())} positive) "
              f"from {index['fire_id'].nunique()} fires to {out_dir}")
    return index


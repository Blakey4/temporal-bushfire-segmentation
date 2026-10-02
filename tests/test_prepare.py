"""Tests for the FLOGA preprocessing (CPU only, no real data: a tiny fake FLOGA tar is built)."""
import tarfile

import numpy as np
import pandas as pd
import rasterio
from affine import Affine

from src import prepare_floga as pf
from src.make_splits import make_splits

P = pf.PATCH


def test_make_label_rules():
    raw = np.array([0, 1, 2, 1, 1, 0])
    nodata = np.array([0, 0, 0, 1, 0, 0], bool)
    scl_pre = np.array([4, 4, 4, 4, 9, 4])
    scl_post = np.array([4, 4, 4, 4, 4, 3])  # 3 = cloud shadow: not masked
    label = pf.make_label(raw, nodata, scl_pre, scl_post)
    assert label.tolist() == [0, 1, 255, 255, 255, 0]
    assert label.dtype == np.uint8


def test_pad_to_grid():
    padded = pf.pad_to_grid(np.zeros((300, 256), np.uint8), 255)
    assert padded.shape == (512, 256)
    assert (padded[300:] == 255).all() and (padded[:300] == 0).all()


def test_select_patches_filters_and_balances():
    # A 2 x 3 grid of patches:
    #   (0,0) burnt    (0,256) cloudy       (0,512) water
    #   (256,0) empty  (256,256) negative   (256,512) negative
    label = np.zeros((2 * P, 3 * P), np.uint8)
    cloud = np.zeros_like(label, bool)
    water = np.zeros_like(label, bool)
    label[10:20, 10:20] = 1
    cloud[:P, P:P + P // 2] = True
    water[:, 2 * P:] = True
    water[P:, 2 * P:] = False
    label[P:, :P] = pf.IGNORE

    chosen = pf.select_patches(label, cloud, water, np.random.default_rng(0))
    positions = [(p["row"], p["col"]) for p in chosen]
    assert positions[0] == (0, 0) and chosen[0]["is_positive"]
    assert len(chosen) == 2  # 1 positive + 1 of the 2 clean negatives
    assert positions[1] in [(P, P), (P, 2 * P)]


def write_tif(path, array, nodata=None):
    array = array if array.ndim == 3 else array[None]
    profile = dict(driver="GTiff", count=array.shape[0], height=array.shape[1], width=array.shape[2],
                   dtype=array.dtype, crs="EPSG:4326", transform=Affine(0.0002, 0, 20, 0, -0.0002, 40),
                   nodata=nodata)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)


def make_fake_floga(tmp_path, height=300, width=300):
    """One raster (fire 111, n=0, year 2020) whose layers are spread over two tar shards."""
    rng = np.random.default_rng(0)
    band_values = np.arange(1, 11, dtype=np.uint16)[:, None, None] * 100 + 1000  # band i -> 1000+100*i
    image = np.broadcast_to(band_values, (10, height, width)).copy()
    image[:, 250:260, 250:260] = 0                    # a nodata hole
    image[3, 5, 5] = 0                                # a single-band dropout
    label = np.zeros((height, width), np.uint8)
    label[20:60, 20:60] = 1                           # this fire
    label[100:110, 100:110] = 2                       # another fire
    scl = np.full((height, width), 4, np.uint8)
    scl[250:260, 250:260] = 0
    scl[30:32, 30:32] = 9                             # small cloud over the burn
    layers = dict(pre=image, post=image, label=label, scl_pre=scl, scl_post=scl,
                  sea=np.zeros((height, width), np.uint8),
                  clc=rng.integers(1, 45, (height, width), dtype=np.uint8))

    shard_dir = tmp_path / "raw" / pf.SHARD_DIR
    shard_dir.mkdir(parents=True)
    tif_dir = tmp_path / "tifs"
    tif_dir.mkdir()
    shards = [tarfile.open(shard_dir / f"0000{i}.tar", "w") for i in range(2)]
    for i, (name, array) in enumerate(layers.items()):
        filename = f"111_0_{pf.LAYERS[name]}.tiff"
        write_tif(tif_dir / filename, array)
        shards[i % 2].add(tif_dir / filename, arcname=f"cellar/{pf.SHARD_DIR}/2020/{filename}")
    for shard in shards:
        shard.close()
    return tmp_path / "raw"


def test_prepare_end_to_end(tmp_path):
    data_root = make_fake_floga(tmp_path)
    out_dir = tmp_path / "out"
    index = pf.prepare(data_root, out_dir)

    # 300x300 -> 2x2 grid of patches; 1 positive + 1 negative kept
    assert len(index) == 2 and index["is_positive"].sum() == 1
    assert (out_dir / "patch_index.csv").exists() and (out_dir / "tar_index.csv").exists()

    pos = index[index["is_positive"]].iloc[0]
    patch = np.load(out_dir / "patches" / f"{pos['patch_id']}.npz")
    assert patch["pre"].shape == (9, P, P) and patch["pre"].dtype == np.uint16
    assert set(np.unique(patch["label"])) <= {0, 1, pf.IGNORE}

    # Band reordering: output band k must hold the input band named BANDS_OUT[k]
    for k, band in enumerate(pf.BANDS_OUT):
        assert patch["pre"][k, 0, 0] == 1000 + 100 * (pf.BANDS_IN.index(band) + 1)

    label = patch["label"]
    assert label[40, 40] == 1                 # burnt
    assert label[30, 30] == pf.IGNORE         # cloud over burn
    assert label[105, 105] == pf.IGNORE       # other fire (label 2)
    assert patch["label_raw"][105, 105] == 2
    assert label[5, 5] == pf.IGNORE           # single-band dropout
    assert label[200, 200] == 0               # clean unburnt
    assert pos["burnt_px"] == 40 * 40 - 4


def test_make_splits_disjoint_and_stratified(tmp_path):
    rows = [dict(patch_id=f"{y}_{f}", year=y, fire_id=f, is_positive=True)
            for y in (2017, 2018) for f in range(y * 100, y * 100 + 10)]
    pd.DataFrame(rows).to_csv(tmp_path / "patch_index.csv", index=False)

    splits = make_splits(tmp_path, seed=0)
    assert splits["fire_id"].is_unique
    counts = splits.groupby(["year", "split"]).size()
    for year in (2017, 2018):
        assert counts[year].to_dict() == {"test": 2, "train": 6, "val": 2}

    again = make_splits(tmp_path, seed=0)
    assert again.equals(splits)

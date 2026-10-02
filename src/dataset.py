"""PyTorch dataset for the prepared FLOGA patches.

Each patch file (written by prepare_floga.py) holds the pre- and post-fire images as raw
Sentinel-2 digital numbers (DN) and the label (1 = burnt by this fire, 0 = not, 255 = ignore).
This module turns them into model inputs:

  1. DN -> surface reflectance: (DN - 1000) / 10000, clipped to the physical range [0, 1];
  2. per-band z-score with mean/std from the training fires only (one shared set for pre and post);
  3. uni mode returns the post image (9 channels), bi mode returns pre and post stacked (18 channels);
  4. training augmentation: a random flip and 90-degree rotation, applied identically to image and label.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DN_OFFSET = 1000   # FLOGA V2 (reprocessed Sentinel-2) adds +1000 to every band
DN_SCALE = 10000
N_BANDS = 9


def to_reflectance(dn):
    """Raw DN -> surface reflectance in [0, 1]."""
    return np.clip((dn.astype(np.float32) - DN_OFFSET) / DN_SCALE, 0.0, 1.0)


def nodata_mask(pre, post):
    """Pixels where any band of either image has no data (DN 0)."""
    return (pre == 0).any(axis=0) | (post == 0).any(axis=0)


def load_split(data_dir, split):
    """The patch_index.csv rows whose fire is in the given split (train / val / test)."""
    data_dir = Path(data_dir)
    index = pd.read_csv(data_dir / "patch_index.csv")
    splits = pd.read_csv(data_dir / "splits.csv")
    return index.merge(splits, on=["year", "fire_id"]).query("split == @split").reset_index(drop=True)


def compute_band_stats(data_dir):
    """Per-band mean and std of reflectance over the valid pixels of all training patches.

    Pre and post images are pooled into one set of statistics, so both dates are scaled the same
    way and the pre -> post change is preserved. Saved to band_stats.json.
    """
    total = np.zeros(N_BANDS)
    total_sq = np.zeros(N_BANDS)
    count = 0
    for patch_id in load_split(data_dir, "train")["patch_id"]:
        patch = np.load(Path(data_dir) / "patches" / f"{patch_id}.npz")
        for image in (patch["pre"], patch["post"]):
            valid = ~(image == 0).any(axis=0)
            pixels = to_reflectance(image[:, valid]).astype(np.float64)
            total += pixels.sum(axis=1)
            total_sq += (pixels ** 2).sum(axis=1)
            count += pixels.shape[1]
    mean = total / count
    std = np.sqrt(total_sq / count - mean ** 2)
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    with open(Path(data_dir) / "band_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def load_band_stats(data_dir):
    """Read band_stats.json, computing it first if it doesn't exist yet."""
    path = Path(data_dir) / "band_stats.json"
    if not path.exists():
        return compute_band_stats(data_dir)
    with open(path) as f:
        return json.load(f)


def normalise(dn, nodata, stats):
    """DN -> reflectance -> per-band z-score. No-data pixels become 0 (= the band mean)."""
    mean = np.array(stats["mean"], np.float32)[:, None, None]
    std = np.array(stats["std"], np.float32)[:, None, None]
    z = (to_reflectance(dn) - mean) / std
    z[:, nodata] = 0.0  # these pixels are ignored in the label anyway
    return z


def model_input(pre, post, mode, stats):
    """The model input for one image pair: post only (uni, 9 ch) or pre then post (bi, 18 ch)."""
    nodata = nodata_mask(pre, post)
    post_z = normalise(post, nodata, stats)
    if mode == "uni":
        return post_z
    return np.concatenate([normalise(pre, nodata, stats), post_z])


def augment(arrays, rng):
    """Apply one of the 8 flip/rotation combinations to every array (same one for all)."""
    k = rng.integers(4)
    flip = rng.integers(2)
    out = []
    for a in arrays:
        a = np.rot90(a, k, axes=(-2, -1))
        if flip:
            a = np.flip(a, axis=-1)
        out.append(np.ascontiguousarray(a))
    return out


class FlogaPatches(Dataset):
    """Returns (x, y, fire_id) for one patch.

    x: float32 [9, H, W] (mode "uni": post only) or [18, H, W] (mode "bi": pre then post).
    y: float32 [1, H, W] with values 0, 1 or 255 (ignore).
    """

    def __init__(self, data_dir, split, mode, stats, augment=False, seed=0):
        assert mode in ("uni", "bi")
        self.data_dir = Path(data_dir)
        self.patches = load_split(data_dir, split)
        self.mode = mode
        self.stats = stats
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        """Called once per epoch so each epoch gets new (but reproducible) augmentations."""
        self.epoch = epoch

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        row = self.patches.iloc[idx]
        patch = np.load(self.data_dir / "patches" / f"{row['patch_id']}.npz")
        x = model_input(patch["pre"], patch["post"], self.mode, self.stats)
        y = patch["label"][None].astype(np.float32)

        if self.augment:
            # Seeded by (seed, epoch, idx) only, so uni and bi models get identical augmentations.
            rng = np.random.default_rng([self.seed, self.epoch, idx])
            x, y = augment([x, y], rng)

        return torch.from_numpy(x), torch.from_numpy(y), int(row["fire_id"])

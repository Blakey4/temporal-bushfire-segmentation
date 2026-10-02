"""Assign whole fires to train / val / test.

Splitting by fire (not by patch) keeps every patch of a fire in the same set, including a fire
that spans several Sentinel-2 tiles, so neighbouring or overlapping patches can't leak between
sets. Fires are split separately within each year, so every set covers 2017-2021.

Called from the notebook:
    from src.make_splits import make_splits
    splits = make_splits(OUT_DIR, seed=0)
"""
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ["train", "val", "test"]


def make_splits(out_dir, ratios=(0.6, 0.2, 0.2), seed=0):
    """Read patch_index.csv, split its fires per year, and write splits.csv."""
    out_dir = Path(out_dir)
    index = pd.read_csv(out_dir / "patch_index.csv")
    fires = index[["year", "fire_id"]].drop_duplicates().sort_values(["year", "fire_id"])

    rng = np.random.default_rng(seed)
    parts = []
    for year, year_fires in fires.groupby("year"):
        fire_ids = rng.permutation(year_fires["fire_id"].to_numpy())
        n_train = round(ratios[0] * len(fire_ids))
        n_val = round(ratios[1] * len(fire_ids))
        split = ["train"] * n_train + ["val"] * n_val + ["test"] * (len(fire_ids) - n_train - n_val)
        parts.append(pd.DataFrame(dict(year=year, fire_id=fire_ids, split=split)))

    splits = pd.concat(parts, ignore_index=True)
    splits.to_csv(out_dir / "splits.csv", index=False)

    summary = index.merge(splits, on=["year", "fire_id"]).groupby("split").agg(
        fires=("fire_id", "nunique"), patches=("patch_id", "size"),
        positive=("is_positive", "sum"))
    print(summary.reindex(SPLITS).to_string())
    return splits


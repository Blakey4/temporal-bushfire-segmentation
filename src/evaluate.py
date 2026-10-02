"""Run trained models on the test set and on whole fires.

- load_model: rebuild a U-Net from a saved checkpoint.
- predict_counts: per-patch confusion counts on the test patches (the numbers behind every metric),
  plus counts per land-cover group and on other-fire (FLOGA label 2) pixels for the error analysis.
- predict_patch: one test patch's probability map, for visual examples.
- PreImageAblation: evaluate a bi-temporal model with its pre-fire image replaced (no retraining).
- predict_fire: predict a whole fire from the raw rasters, for visual comparison only.
- burn_index_counts: the traditional (non-ML) baseline - threshold a spectral burn index.

Predictions use the threshold probability > 0.5 (logit > 0) unless another threshold is given.
"""
import numpy as np
import pandas as pd
import torch
from rasterio.windows import Window

from src import prepare_floga as pf
from src.dataset import model_input, to_reflectance
from src.metrics import COUNTS, confusion
from src.models import UNet

# CORINE Land Cover classes (1-44, raster legend order) grouped into coarse classes that other
# land-cover products (e.g. DEA Land Cover for the Australian fires) can also be mapped to.
CLC_GROUPS = {
    **{c: "artificial" for c in range(1, 12)},
    **{c: "agriculture" for c in range(12, 23)},
    **{c: "forest" for c in range(23, 26)},
    **{c: "shrub/grass" for c in range(26, 30)},
    **{c: "bare/sparse" for c in range(30, 35)},   # includes CLC 33 "burnt areas" (older burns)
    **{c: "wetland" for c in range(35, 40)},
    **{c: "water" for c in range(40, 45)},
}


def load_model(checkpoint, device="cpu"):
    """Rebuild the model saved by train.py. Returns (model, mode)."""
    saved = torch.load(checkpoint, map_location=device)
    config = saved["config"]
    model = UNet(9 if config["mode"] == "uni" else 18, base=config.get("base", 32), norm=config["norm"],
                 block=config.get("block", "plain"))
    model.load_state_dict(saved["model"])
    return model.to(device).eval(), config["mode"]


@torch.no_grad()
def predict_counts(model, dataset, device="cpu", batch_size=16, threshold=0.5):
    """Confusion counts for every patch of `dataset` (a FlogaPatches, or an ablation wrapper).

    `threshold` is applied to the probability; vary it on validation for a threshold sweep.

    Returns two DataFrames:
      patches: one row per patch - patch_id, fire_id, raster, tp, fp, fn, tn, and other_burnt_px /
               other_burnt_fp (FLOGA label-2 pixels, and how many of them were predicted burnt);
      groups:  one row per patch and land-cover group - patch_id, fire_id, clc_group, tp, fp, fn, tn.
    """
    model.eval()
    patch_rows, group_rows = [], []
    for start in range(0, len(dataset), batch_size):
        idx = range(start, min(start + batch_size, len(dataset)))
        x = torch.stack([dataset[i][0] for i in idx]).to(device)
        pred = (torch.sigmoid(model(x)) > threshold)[:, 0].cpu().numpy()

        for i, p in zip(idx, pred):
            row = dataset.patches.iloc[i]
            patch = np.load(dataset.data_dir / "patches" / f"{row['patch_id']}.npz")
            label, other_burnt = patch["label"], patch["label_raw"] == 2
            ids = dict(patch_id=row["patch_id"], fire_id=row["fire_id"], raster=row["raster"])

            patch_rows.append(dict(ids, **dict(zip(COUNTS, confusion(p, label))),
                                   other_burnt_px=int(other_burnt.sum()),
                                   other_burnt_fp=int((p & other_burnt).sum())))
            groups = np.vectorize(CLC_GROUPS.get)(patch["clc"], "none")
            for group in np.unique(groups):
                in_group = groups == group
                counts = confusion(p[in_group], label[in_group])
                group_rows.append(dict(ids, clc_group=group, **dict(zip(COUNTS, counts))))
    return pd.DataFrame(patch_rows), pd.DataFrame(group_rows)


@torch.no_grad()
def predict_patch(model, dataset, patch_id, device="cpu"):
    """One patch's prediction, in the same format as predict_fire (for plotting).

    Returns a dict with pre and post (9-band DN), label (0/1/255) and prob.
    """
    idx = dataset.patches.index[dataset.patches["patch_id"] == patch_id][0]
    x = dataset[idx][0][None].to(device)
    prob = torch.sigmoid(model.eval()(x))[0, 0].cpu().numpy()
    patch = np.load(dataset.data_dir / "patches" / f"{patch_id}.npz")
    return dict(pre=patch["pre"], post=patch["post"], label=patch["label"], prob=prob)


class PreImageAblation:
    """Wraps a bi-temporal FlogaPatches dataset and replaces the pre-fire channels (0-8).

    how="post":       pre := post, so the model sees no change at all;
    how="other_fire": pre := the pre image of a patch from a different fire (fixed pairing),
                      so the pre image is real imagery but of the wrong place.
    If the model relies on the pre image, its scores should drop.
    """

    def __init__(self, dataset, how):
        assert dataset.mode == "bi" and how in ("post", "other_fire")
        if how == "other_fire" and dataset.patches["fire_id"].nunique() < 2:
            raise ValueError("how='other_fire' needs patches from at least two fires")
        self.dataset, self.how = dataset, how
        self.patches, self.data_dir = dataset.patches, dataset.data_dir

    def __len__(self):
        return len(self.dataset)

    def partner(self, idx):
        """A patch from a different fire: start half-way round the list, step until the fire differs."""
        fires = self.patches["fire_id"].to_numpy()
        j = (idx + len(fires) // 2) % len(fires)
        while fires[j] == fires[idx]:
            j = (j + 1) % len(fires)
        return j

    def __getitem__(self, idx):
        x, y, fire_id = self.dataset[idx]
        x = x.clone()
        if self.how == "post":
            x[:9] = x[9:]
        else:
            x[:9] = self.dataset[self.partner(idx)][0][:9]
        return x, y, fire_id


@torch.no_grad()
def predict_tiled(model, x, device="cpu", tile=256, overlap=64, batch_size=8):
    """Logits for an image of any size, predicted in overlapping tiles.

    Tiles of `tile` px are placed every (tile - overlap) px, and only the centre of each tile's
    prediction is kept, so every output pixel had at least overlap/2 px of context around it and
    there are no seam lines between tiles.
    """
    border = overlap // 2
    step = tile - overlap
    _, height, width = x.shape
    rows, cols = -(-height // step), -(-width // step)  # ceiling division
    padded = np.zeros((x.shape[0], rows * step + overlap, cols * step + overlap), np.float32)
    padded[:, border:border + height, border:border + width] = x

    out = np.zeros((rows * step, cols * step), np.float32)
    positions = [(r * step, c * step) for r in range(rows) for c in range(cols)]
    for start in range(0, len(positions), batch_size):
        batch = positions[start:start + batch_size]
        tiles = torch.from_numpy(np.stack([padded[:, r:r + tile, c:c + tile] for r, c in batch]))
        logits = model(tiles.to(device))[:, 0].cpu().numpy()
        for (r, c), logit in zip(batch, logits):
            out[r:r + step, c:c + step] = logit[border:border + step, border:border + step]
    return out[:height, :width]


def predict_fire(model, mode, data_root, tar_index, stats, fire_id, n, margin=256, device="cpu"):
    """Predict one whole fire (its burnt-area bounding box + margin) from the raw rasters.

    `tar_index` is the table from prepare_floga.index_tars. Returns a dict with pre and post
    (9-band DN), label (0/1/255) and prob (probability of "burnt by this fire").
    """
    files = tar_index[(tar_index["fire_id"] == fire_id) & (tar_index["n"] == n)].set_index("layer")
    label_raw = pf.read_layer(data_root, files.loc["label"])
    scl_pre = pf.read_layer(data_root, files.loc["scl_pre"])
    scl_post = pf.read_layer(data_root, files.loc["scl_post"])
    label = pf.make_label(label_raw, (scl_pre == 0) | (scl_post == 0), scl_pre, scl_post)

    rows, cols = np.nonzero(label_raw == 1)
    top, left = max(rows.min() - margin, 0), max(cols.min() - margin, 0)
    bottom = min(rows.max() + margin + 1, label.shape[0])
    right = min(cols.max() + margin + 1, label.shape[1])
    window = Window(left, top, right - left, bottom - top)

    with pf.open_layer(data_root, files.loc["pre"]) as src:
        pre = src.read(pf.BAND_INDEXES, window=window)
    with pf.open_layer(data_root, files.loc["post"]) as src:
        post = src.read(pf.BAND_INDEXES, window=window)

    x = model_input(pre, post, mode, stats)
    logits = predict_tiled(model, x, device)
    return dict(pre=pre, post=post, label=label[top:bottom, left:right],
                prob=1 / (1 + np.exp(-logits)))


# Traditional baseline: thresholding a spectral burn index


def nbr(dn):
    """Normalised Burn Ratio (B8A - B12) / (B8A + B12) on reflectance; B8A = band 6, B12 = band 8.

    Burning drops NIR (B8A) and raises SWIR (B12), so NBR falls after a fire.
    """
    nir, swir = to_reflectance(dn[6]), to_reflectance(dn[8])
    return (nir - swir) / (nir + swir + 1e-6)


def burn_index(pre, post, kind):
    """A score where higher = more likely burnt.

    kind="dnbr":     dNBR = NBR(pre) - NBR(post), the standard bi-temporal burn index;
    kind="nbr_post": -NBR(post), the uni-temporal equivalent (post image only).
    """
    if kind == "dnbr":
        return nbr(pre) - nbr(post)
    if kind == "nbr_post":
        return -nbr(post)
    raise ValueError(kind)


def burn_index_counts(dataset, kind, thresholds):
    """Confusion counts of the baseline "burnt if burn_index > threshold".

    One row per patch and threshold. Pick the threshold with the best per-event F1 on the
    validation patches, then apply only that threshold to the test patches: the same rule a
    trained model's checkpoint is chosen by.
    """
    rows = []
    for _, row in dataset.patches.iterrows():
        patch = np.load(dataset.data_dir / "patches" / f"{row['patch_id']}.npz")
        score = burn_index(patch["pre"], patch["post"], kind)
        for threshold in thresholds:
            counts = confusion(score > threshold, patch["label"])
            rows.append(dict(patch_id=row["patch_id"], fire_id=row["fire_id"], raster=row["raster"],
                             threshold=float(threshold), **dict(zip(COUNTS, counts))))
    return pd.DataFrame(rows)

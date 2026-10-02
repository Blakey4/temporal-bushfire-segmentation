"""Figures for the notebook. Every function returns a matplotlib Figure.

Error-map colours: TP = burnt and predicted burnt, FP = false alarm, FN = missed burn,
TN = correctly unburnt, ignore = label 255 (other fires, cloud, no data).
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.dataset import to_reflectance

ERROR_CLASSES = ["TN", "TP", "FP", "FN", "ignore"]
ERROR_COLOURS = ["#3a3a3a", "#f08c00", "#e03131", "#1c7ed6", "#dee2e6"]
ERROR_CMAP = ListedColormap(ERROR_COLOURS)
LABEL_CMAP = ListedColormap(["#3a3a3a", "#f08c00", "#dee2e6"])  # 0, 1, ignore


def highlight_compress(x, knee=0.92):
    """Map [0, 1] to [0, 1] linearly up to `knee`, then roll off smoothly instead of clipping.

    Bright areas keep some detail, as in Sentinel Hub's HighlightCompressVisualizer.
    """
    x = np.maximum(x, 0)
    top = 1 - knee
    return np.where(x <= knee, x, knee + top * (1 - np.exp(-(x - knee) / top)))


def swir_composite(dn, max_reflectance=0.4):
    """SWIR false-colour image, rendered like the Copernicus Browser burnt-area view:
    R = B12, G = B8A, B = B04 (SWIR2 / NIR / Red), one fixed reflectance scale for all bands
    with `max_reflectance` (0.4) = full brightness, plus highlight compression.

    Vegetation is green, fresh burn red-brown, bare soil and dry fields tan, water dark,
    snow cyan. No-data pixels are black. Band positions in our 9-band order
    (B02, B03, B04, B05, B06, B07, B8A, B11, B12): B12 = 8, B8A = 6, B04 = 2.
    """
    rgb = highlight_compress(to_reflectance(dn[[8, 6, 2]]) / max_reflectance)
    return np.moveaxis(rgb, 0, -1)


def error_map(pred, label):
    """Integer map indexing ERROR_CLASSES: 0 TN, 1 TP, 2 FP, 3 FN, 4 ignore."""
    burnt = label == 1
    out = np.select([pred & burnt, pred & ~burnt, ~pred & burnt], [1, 2, 3], default=0)
    out[label == 255] = 4
    return out


def _show(ax, image, title, **kwargs):
    ax.imshow(image, interpolation="nearest", **kwargs)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_patch_comparison(pre, post, label, probs, title=None):
    """pre SWIR | post SWIR | label | one error map per model.

    probs: {model name: probability map}, e.g. {"uni": ..., "bi": ...}.
    """
    fig, axes = plt.subplots(1, 3 + len(probs), figsize=(3.2 * (3 + len(probs)), 3.6))
    _show(axes[0], swir_composite(pre), "pre-fire (SWIR)")
    _show(axes[1], swir_composite(post), "post-fire (SWIR)")
    _show(axes[2], np.where(label == 255, 2, label), "label", cmap=LABEL_CMAP, vmin=0, vmax=2)
    for ax, (name, prob) in zip(axes[3:], probs.items()):
        _show(ax, error_map(prob > 0.5, label), name, cmap=ERROR_CMAP, vmin=0, vmax=4)
    fig.legend(handles=[Patch(color=c, label=n) for n, c in zip(ERROR_CLASSES, ERROR_COLOURS)],
               loc="lower center", ncol=5, frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def plot_prediction(predictions, title=None):
    """Plot a patch or a whole fire for one or more models.

    predictions: {model name: output of evaluate.predict_patch or evaluate.predict_fire}.
    """
    first = next(iter(predictions.values()))
    probs = {name: p["prob"] for name, p in predictions.items()}
    return plot_patch_comparison(first["pre"], first["post"], first["label"], probs, title)


def plot_training_curves(logs):
    """Loss and validation event F1 per epoch. logs: {run name: DataFrame from train()}."""
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4))
    for name, log in logs.items():
        line, = ax_loss.plot(log["epoch"], log["train_loss"], label=f"{name} train")
        ax_loss.plot(log["epoch"], log["val_loss"], "--", color=line.get_color(), label=f"{name} val")
        ax_f1.plot(log["epoch"], log["val_event_f1"], color=line.get_color(), label=name)
    ax_loss.set(xlabel="epoch", ylabel="masked BCE", title="Loss")
    ax_f1.set(xlabel="epoch", ylabel="per-event F1", title="Validation F1")
    ax_loss.legend(fontsize=8)
    ax_f1.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_by_group(table, metric, group="clc_group", model="model"):
    """Grouped bars of `metric` per land-cover group, one bar per model.

    table: one row per (group, model), with a `metric` column and optionally `<metric>_std`.
    """
    means = table.pivot(index=group, columns=model, values=metric)
    stds = None
    if f"{metric}_std" in table:
        stds = table.pivot(index=group, columns=model, values=f"{metric}_std")
    fig, ax = plt.subplots(figsize=(1.2 * len(means) + 2, 4))
    means.plot.bar(ax=ax, yerr=stds, capsize=3, rot=30)
    ax.set(xlabel="", ylabel=metric, title=f"{metric} by land cover")
    fig.tight_layout()
    return fig

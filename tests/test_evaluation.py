"""Tests for metrics.py, evaluate.py and visualisation.py (CPU, synthetic data)."""
import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

matplotlib.use("Agg")

from src.dataset import FlogaPatches, compute_band_stats
from src.evaluate import (PreImageAblation, burn_index, burn_index_counts, predict_counts,
                          predict_patch, predict_tiled)
from src.metrics import confusion, event_scores, scores, summarise
from src.models import UNet
from src import visualisation as vis
from tests.test_training import SIZE, make_fake_data


def test_confusion_ignores_255():
    pred = np.array([1, 1, 0, 0, 1, 0], bool)
    target = np.array([1, 0, 1, 0, 255, 255])
    assert confusion(pred, target) == (1, 1, 1, 1)


def test_scores_hand_computed():
    s = scores(tp=6, fp=2, fn=4)
    assert s["precision"] == pytest.approx(0.75)
    assert s["recall"] == pytest.approx(0.6)
    assert s["f1"] == pytest.approx(12 / 18)
    assert s["iou"] == pytest.approx(0.5)
    assert np.isnan(scores(0, 0, 0)["f1"])


def test_event_vs_pooled():
    # A big perfect fire and a small completely missed one:
    # per-event F1 = mean(1, 0) = 0.5; pooled F1 is dominated by the big fire.
    counts = pd.DataFrame([dict(fire_id=1, tp=90, fp=0, fn=0, tn=10),
                           dict(fire_id=2, tp=0, fp=0, fn=10, tn=90)])
    assert event_scores(counts)["f1"].tolist() == [1.0, 0.0]
    summary = summarise(counts)
    assert summary["event_f1"] == pytest.approx(0.5)
    assert summary["pooled_f1"] == pytest.approx(180 / 190)


class Echo(nn.Module):
    """Fake model whose 'logits' are its first input channel."""

    def forward(self, x):
        return x[:, :1]


def test_predict_tiled_has_no_seams():
    x = np.random.default_rng(0).normal(size=(3, 300, 437)).astype(np.float32)
    assert np.allclose(predict_tiled(Echo(), x), x[0])


def test_predict_counts_match_confusion(tmp_path):
    data = make_fake_data(tmp_path)
    stats = compute_band_stats(data)
    dataset = FlogaPatches(data, "test", "bi", stats)
    patches, groups = predict_counts(Echo(), dataset)
    assert len(patches) == len(dataset)

    x, y, _ = dataset[0]
    expected = confusion((x[0] > 0).numpy(), y[0].numpy())
    assert tuple(patches.iloc[0][["tp", "fp", "fn", "tn"]]) == expected
    # The land-cover breakdown adds up to the patch totals
    assert groups[["tp", "fp", "fn", "tn"]].sum().tolist() == patches[["tp", "fp", "fn", "tn"]].sum().tolist()


def test_pre_image_ablation(tmp_path):
    data = make_fake_data(tmp_path)
    dataset = FlogaPatches(data, "test", "bi", compute_band_stats(data))
    x, _, _ = dataset[0]

    same, _, _ = PreImageAblation(dataset, "post")[0]
    assert same.shape == x.shape and torch.equal(same[:9], x[9:])

    other = PreImageAblation(dataset, "other_fire")
    j = other.partner(0)
    swapped, _, _ = other[0]
    assert dataset.patches["fire_id"][j] != dataset.patches["fire_id"][0]
    assert torch.equal(swapped[:9], dataset[j][0][:9]) and torch.equal(swapped[9:], x[9:])

    one_fire = FlogaPatches(data, "test", "bi", compute_band_stats(data))
    one_fire.patches = one_fire.patches[one_fire.patches["fire_id"] == one_fire.patches["fire_id"][0]]
    with pytest.raises(ValueError):
        PreImageAblation(one_fire, "other_fire")


def test_figures_render():
    rng = np.random.default_rng(0)
    pre = rng.integers(1000, 3000, (9, SIZE, SIZE)).astype(np.uint16)
    label = rng.choice([0, 1, 255], (SIZE, SIZE))
    probs = {"uni": rng.random((SIZE, SIZE)), "bi": rng.random((SIZE, SIZE))}
    vis.plot_patch_comparison(pre, pre, label, probs)
    vis.plot_prediction({"uni": dict(pre=pre, post=pre, label=label, prob=probs["uni"])})
    assert vis.swir_composite(pre).shape == (SIZE, SIZE, 3)
    log = pd.DataFrame(dict(epoch=[1, 2], train_loss=[0.5, 0.4], val_loss=[0.6, 0.5], val_event_f1=[0.1, 0.2]))
    vis.plot_training_curves({"uni_seed0": log, "bi_seed0": log})
    table = pd.DataFrame(dict(clc_group=["forest", "forest", "water", "water"],
                              model=["uni", "bi", "uni", "bi"], f1=[0.5, 0.6, 0.2, 0.3]))
    vis.plot_by_group(table, "f1")


def test_predict_patch(tmp_path):
    data = make_fake_data(tmp_path)
    dataset = FlogaPatches(data, "test", "uni", compute_band_stats(data))
    patch_id = dataset.patches["patch_id"][1]
    out = predict_patch(UNet(9), dataset, patch_id)
    assert out["prob"].shape == out["label"].shape == (SIZE, SIZE)
    assert out["pre"].shape == (9, SIZE, SIZE) and 0 <= out["prob"].min() <= out["prob"].max() <= 1


def test_burn_index_direction():
    """Vegetation (high NIR, low SWIR) that burns (low NIR, high SWIR) must score high on both indices."""
    pre = np.full((9, 2, 2), 1000, np.uint16)
    post = pre.copy()
    pre[6], pre[8] = 4000, 1500           # healthy: B8A 0.30, B12 0.05
    post[6], post[8] = 4000, 1500
    post[6, 0, 0], post[8, 0, 0] = 1800, 3500   # pixel (0,0) burnt: B8A 0.08, B12 0.25
    dnbr = burn_index(pre, post, "dnbr")
    assert dnbr[0, 0] > 1.0 and abs(dnbr[1, 1]) < 1e-6
    assert burn_index(pre, post, "nbr_post")[0, 0] > burn_index(pre, post, "nbr_post")[1, 1]


def test_burn_index_counts(tmp_path):
    data = make_fake_data(tmp_path)
    dataset = FlogaPatches(data, "val", "bi", compute_band_stats(data))
    counts = burn_index_counts(dataset, "dnbr", [0.0, 0.1])
    assert len(counts) == 2 * len(dataset)
    assert set(counts["threshold"]) == {0.0, 0.1}
    assert (counts[["tp", "fp", "fn", "tn"]].sum(axis=1) > 0).all()

"""Tests for dataset.py, models.py, losses.py and train.py (CPU, synthetic patches)."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import FlogaPatches, augment, compute_band_stats, to_reflectance
from src.losses import make_loss, masked_bce, masked_dice
from src.models import UNet, count_parameters
from src.metrics import event_f1
from src.train import train

SIZE = 32  # small patches keep the tests fast; the U-Net only needs a multiple of 16


def make_fake_data(tmp_path, fires_per_split=2, patches_per_fire=2):
    """Write synthetic patches: the burnt square is darker in the post image than the pre image."""
    rng = np.random.default_rng(0)
    (tmp_path / "patches").mkdir()
    index, splits = [], []
    fire_id = 100
    for split in ("train", "val", "test"):
        for _ in range(fires_per_split):
            fire_id += 1
            splits.append(dict(year=2020, fire_id=fire_id, split=split))
            for k in range(patches_per_fire):
                pre = rng.integers(2000, 4000, (9, SIZE, SIZE)).astype(np.uint16)
                post = pre.copy()
                label = np.zeros((SIZE, SIZE), np.uint8)
                label[4:16, 4:16] = 1
                post[:, 4:16, 4:16] = 1100
                label[20:24, 20:24] = 255
                pre[:, 0, 0] = 0  # one nodata pixel
                label[0, 0] = 255
                patch_id = f"2020_{fire_id}_0_r{k}"
                np.savez(tmp_path / "patches" / f"{patch_id}.npz", pre=pre, post=post, label=label,
                         label_raw=label, clc=np.zeros((SIZE, SIZE), np.uint8))
                index.append(dict(patch_id=patch_id, year=2020, fire_id=fire_id, raster=f"{fire_id}_0",
                                  is_positive=True))
    pd.DataFrame(index).to_csv(tmp_path / "patch_index.csv", index=False)
    pd.DataFrame(splits).to_csv(tmp_path / "splits.csv", index=False)
    return tmp_path


@pytest.mark.parametrize("channels", [9, 18])
def test_unet_output_shape(channels):
    model = UNet(channels)
    out = model(torch.zeros(1, channels, 256, 256))
    assert out.shape == (1, 1, 256, 256)


def test_uni_and_bi_differ_only_in_first_conv():
    uni, bi = UNet(9), UNet(18)
    first_conv_extra = 9 * 32 * 3 * 3  # 9 extra input channels x 32 filters x 3x3 kernel
    assert count_parameters(bi) - count_parameters(uni) == first_conv_extra


def test_batchnorm_option():
    assert UNet(9, norm="batch")(torch.zeros(2, 9, SIZE, SIZE)).shape == (2, 1, SIZE, SIZE)


def test_masked_bce_ignores_255():
    target = torch.tensor([[0.0, 1.0, 255.0]])
    a = masked_bce(torch.tensor([[0.3, -0.2, 5.0]]), target)
    b = masked_bce(torch.tensor([[0.3, -0.2, -40.0]]), target)
    assert torch.isclose(a, b)
    assert masked_bce(torch.zeros(1, 2), torch.full((1, 2), 255.0)) == 0


def test_to_reflectance_offset_and_clip():
    assert to_reflectance(np.array([900, 1000, 3500, 20000])).tolist() == [0.0, 0.0, 0.25, 1.0]


def test_band_stats_exclude_nodata(tmp_path):
    data = make_fake_data(tmp_path)
    stats = compute_band_stats(data)
    assert len(stats["mean"]) == 9
    assert min(stats["mean"]) > 0.05  # zeros from the nodata pixel would drag the mean down
    assert (tmp_path / "band_stats.json").exists()


def test_dataset_channels_and_labels(tmp_path):
    data = make_fake_data(tmp_path)
    stats = compute_band_stats(data)
    x_uni, y, fire = FlogaPatches(data, "train", "uni", stats)[0]
    x_bi, _, _ = FlogaPatches(data, "train", "bi", stats)[0]
    assert x_uni.shape == (9, SIZE, SIZE) and x_bi.shape == (18, SIZE, SIZE)
    assert torch.equal(x_bi[9:], x_uni)  # bi = [pre, post]
    assert set(y.unique().tolist()) <= {0.0, 1.0, 255.0}
    assert x_bi[:, 0, 0].abs().sum() == 0  # nodata pixel set to the band mean (0 after z-score)


def test_augment_same_transform_for_image_and_label():
    image = np.arange(2 * 4 * 4).reshape(2, 4, 4)
    label = image[:1].copy()
    for seed in range(8):
        x, y = augment([image, label], np.random.default_rng(seed))
        assert np.array_equal(x[:1], y)


def test_uni_and_bi_get_identical_augmentation(tmp_path):
    data = make_fake_data(tmp_path)
    stats = compute_band_stats(data)
    uni = FlogaPatches(data, "train", "uni", stats, augment=True, seed=3)
    bi = FlogaPatches(data, "train", "bi", stats, augment=True, seed=3)
    for epoch in (1, 2):
        uni.set_epoch(epoch)
        bi.set_epoch(epoch)
        for i in range(len(uni)):
            x_uni, y_uni, _ = uni[i]
            x_bi, y_bi, _ = bi[i]
            assert torch.equal(y_uni, y_bi) and torch.equal(x_bi[9:], x_uni)


def test_event_f1():
    counts = {1: [10, 0, 0], 2: [0, 5, 5]}  # perfect fire, then a completely wrong one
    assert event_f1(counts) == pytest.approx(0.5)


def test_train_runs_and_writes_outputs(tmp_path):
    (tmp_path / "data").mkdir()
    data = make_fake_data(tmp_path / "data")
    log = train(data, tmp_path / "run", mode="bi", epochs=2, batch_size=2)
    assert list(log["epoch"]) == [1, 2] and log["train_loss"].notna().all()
    assert (tmp_path / "run" / "best.pt").exists() and (tmp_path / "run" / "config.json").exists()


def test_dice_and_combined_losses_ignore_255():
    target = torch.tensor([[0.0, 1.0, 255.0]])
    for loss in (masked_dice, make_loss("bce+dice"), make_loss("bce", pos_weight=3.0)):
        a = loss(torch.tensor([[0.3, -0.2, 5.0]]), target)
        b = loss(torch.tensor([[0.3, -0.2, -40.0]]), target)
        assert torch.isclose(a, b)
    perfect = masked_dice(torch.tensor([[-20.0, 20.0]]), torch.tensor([[0.0, 1.0]]))
    assert perfect < 1e-3


def test_pos_weight_increases_loss_on_missed_burn():
    missed = (torch.tensor([[-2.0]]), torch.tensor([[1.0]]))
    assert make_loss("bce", pos_weight=3.0)(*missed) > make_loss("bce")(*missed)


def test_residual_unet_shape():
    model = UNet(18, block="residual")
    assert model(torch.zeros(1, 18, SIZE, SIZE)).shape == (1, 1, SIZE, SIZE)


@pytest.mark.parametrize("changes", [dict(loss="bce+dice", optimizer="adamw", weight_decay=1e-4, schedule="constant"),
                                     dict(loss="dice", optimizer="sgd", lr=0.01, block="residual", norm="batch")])
def test_train_options_run(tmp_path, changes):
    (tmp_path / "data").mkdir()
    data = make_fake_data(tmp_path / "data")
    log = train(data, tmp_path / "run", mode="bi", epochs=1, batch_size=2, **changes)
    assert {"val_bce", "val_event_precision", "val_event_recall", "val_pooled_f1"} <= set(log.columns)
    from src.evaluate import load_model
    model, mode = load_model(tmp_path / "run" / "best.pt")
    assert mode == "bi"

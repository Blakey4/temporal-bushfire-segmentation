"""Train one U-Net (uni or bi) on the prepared FLOGA patches.

The uni and bi runs share everything except the input: same patches, same augmentations, same
batch order, same hyperparameters. The checkpoint with the best per-event F1 on the validation
fires is kept (best.pt).

Every design choice that the refinement experiments vary is an argument of train(), so a change is
one setting in the notebook and is recorded in each run's config.json:
    loss        "bce" | "dice" | "bce+dice"         pos_weight   weight on burnt pixels in BCE (None = 1)
    optimizer   "adam" | "adamw" | "sgd"             weight_decay L2 penalty (AdamW: decoupled)
    schedule    "cosine" | "constant"                lr, batch_size, epochs
    base        U-Net width (32 / 64)                norm "group" | "batch",  block "plain" | "residual"

Called from the notebook, e.g.:
    from src.train import train
    log = train(OUT_DIR, RUNS_DIR / "bi_seed0", mode="bi", seed=0, epochs=50, base=64)
"""
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import FlogaPatches, load_band_stats
from src.losses import IGNORE, make_loss, masked_bce
from src.metrics import summarise
from src.models import UNet, count_parameters

LOG_COLUMNS = ["epoch", "lr", "train_loss", "val_loss", "val_bce", "val_event_f1", "val_event_precision",
               "val_event_recall", "val_pooled_f1", "seconds"]


def make_optimizer(name, parameters, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name}")


def make_scheduler(name, optimiser, epochs):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimiser, lambda epoch: 1.0)
    raise ValueError(f"unknown schedule: {name}")


def train_one_epoch(model, loader, loss_fn, optimiser, scaler, device, amp):
    model.train()
    total = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
        loss = loss_fn(logits.float(), y)
        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        total += loss.item() * len(x)
    return total / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, loss_fn, device, amp):
    """Validation loss (the training loss), BCE (comparable across loss functions) and per-event
    precision / recall / F1 plus pooled F1 (threshold 0.5, i.e. logit > 0)."""
    model.eval()
    total_loss = total_bce = 0.0
    counts = defaultdict(lambda: [0, 0, 0])  # fire_id -> [TP, FP, FN]
    for x, y, fire_ids in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
        logits = logits.float()
        total_loss += loss_fn(logits, y).item() * len(x)
        total_bce += masked_bce(logits, y).item() * len(x)

        pred = logits > 0
        valid = y != IGNORE
        burnt = y == 1
        tp = (pred & burnt & valid).sum(dim=(1, 2, 3))
        fp = (pred & ~burnt & valid).sum(dim=(1, 2, 3))
        fn = (~pred & burnt & valid).sum(dim=(1, 2, 3))
        for i, fire in enumerate(fire_ids.tolist()):
            counts[fire][0] += tp[i].item()
            counts[fire][1] += fp[i].item()
            counts[fire][2] += fn[i].item()

    table = pd.DataFrame([dict(fire_id=k, tp=v[0], fp=v[1], fn=v[2], tn=0) for k, v in counts.items()])
    s = summarise(table)
    n = len(loader.dataset)
    return dict(val_loss=total_loss / n, val_bce=total_bce / n, val_event_f1=s["event_f1"],
                val_event_precision=s["event_precision"], val_event_recall=s["event_recall"],
                val_pooled_f1=s["pooled_f1"])


def train(data_dir, run_dir, mode="uni", seed=0, epochs=50, batch_size=8, lr=1e-3, base=32, norm="group",
          block="plain", loss="bce", pos_weight=None, optimizer="adam", weight_decay=0.0, schedule="cosine",
          num_workers=0, amp=False):
    """Train one model and return its per-epoch log (also saved as run_dir/log.csv)."""
    config = dict(data_dir=str(data_dir), run_dir=str(run_dir), mode=mode, seed=seed, epochs=epochs,
                  batch_size=batch_size, lr=lr, base=base, norm=norm, block=block, loss=loss,
                  pos_weight=pos_weight, optimizer=optimizer, weight_decay=weight_decay, schedule=schedule,
                  num_workers=num_workers, amp=amp)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = amp and device.type == "cuda"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Data: augmentation and shuffling have their own seeds, so they don't depend on the model.
    stats = load_band_stats(data_dir)
    train_set = FlogaPatches(data_dir, "train", mode, stats, augment=True, seed=seed)
    val_set = FlogaPatches(data_dir, "val", mode, stats)
    shuffle_rng = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size, shuffle=True, generator=shuffle_rng,
                              num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size, num_workers=num_workers)

    torch.manual_seed(seed)
    model = UNet(in_channels=9 if mode == "uni" else 18, base=base, norm=norm, block=block).to(device)
    loss_fn = make_loss(loss, pos_weight)
    optimiser = make_optimizer(optimizer, model.parameters(), lr, weight_decay)
    scheduler = make_scheduler(schedule, optimiser, epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    print(f"{mode} model: {count_parameters(model):,} parameters | {len(train_set)} train / "
          f"{len(val_set)} val patches | device {device}")

    best_f1 = -1.0
    with open(run_dir / "log.csv", "w", newline="") as f:
        log = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        log.writeheader()
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_set.set_epoch(epoch)
            lr_now = optimiser.param_groups[0]["lr"]
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimiser, scaler, device, amp)
            val = validate(model, val_loader, loss_fn, device, amp)
            scheduler.step()
            seconds = time.time() - start
            log.writerow(dict(epoch=epoch, lr=lr_now, train_loss=train_loss, seconds=round(seconds, 1), **val))
            f.flush()

            # A plain float, so the checkpoint loads with torch.load(weights_only=True)
            val_f1 = float(val["val_event_f1"])
            checkpoint = dict(model=model.state_dict(), epoch=epoch, val_event_f1=val_f1, config=config)
            torch.save(checkpoint, run_dir / "last.pt")
            marker = ""
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(checkpoint, run_dir / "best.pt")
                marker = "  *best*"
            print(f"epoch {epoch:3d} | lr {lr_now:.2e} | train {train_loss:.4f} | val {val['val_loss']:.4f} | "
                  f"event F1 {val_f1:.3f} (P {val['val_event_precision']:.2f} R {val['val_event_recall']:.2f}) | "
                  f"{seconds:.0f}s{marker}")
    return pd.read_csv(run_dir / "log.csv")

"""Segmentation metrics from pixel counts.

Everything is built from the confusion counts of the "burnt by this fire" class, ignoring
pixels labelled 255:
    TP = predicted burnt and burnt       FP = predicted burnt but not burnt
    FN = predicted not burnt but burnt   TN = predicted not burnt and not burnt

    precision = TP / (TP + FP)     how much of what we mapped as burnt really is burnt
    recall    = TP / (TP + FN)     how much of the real burn we found
    F1        = 2TP / (2TP + FP + FN)
    IoU       = TP / (TP + FP + FN)

Two ways to aggregate over the test set:
    per-event: sum the counts within each fire, score each fire, then average over fires
               (every fire counts equally, however big it is);
    pooled:    sum the counts over all test pixels, then score once (big fires dominate).
"""
import numpy as np
import pandas as pd

IGNORE = 255
COUNTS = ["tp", "fp", "fn", "tn"]


def confusion(pred, target, ignore=IGNORE):
    """(tp, fp, fn, tn) for a boolean prediction array and a 0/1/255 target (numpy or torch)."""
    valid = target != ignore
    burnt = target == 1
    tp = int((pred & burnt & valid).sum())
    fp = int((pred & ~burnt & valid).sum())
    fn = int((~pred & burnt & valid).sum())
    tn = int((~pred & ~burnt & valid).sum())
    return tp, fp, fn, tn


def scores(tp, fp, fn):
    """Precision, recall, F1 and IoU. Works on numbers or arrays/columns; 0/0 gives NaN."""
    tp, fp, fn = (np.asarray(v, dtype=float) for v in (tp, fp, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = dict(precision=tp / (tp + fp), recall=tp / (tp + fn),
                      f1=2 * tp / (2 * tp + fp + fn), iou=tp / (tp + fp + fn))
    return {k: (float(v) if v.ndim == 0 else v) for k, v in result.items()}


def event_scores(counts, by="fire_id"):
    """Per-fire table: sum each fire's counts, then score it."""
    table = counts.groupby(by)[COUNTS].sum()
    for name, values in scores(table["tp"], table["fp"], table["fn"]).items():
        table[name] = values
    return table.reset_index()


def summarise(counts):
    """One row of headline numbers: per-event mean and pooled precision/recall/F1/IoU.

    Fires with no burnt pixels and no burnt predictions have an undefined (NaN) score and are
    left out of the per-event mean.
    """
    per_event = event_scores(counts).mean(numeric_only=True)
    pooled = scores(*(counts[c].sum() for c in ("tp", "fp", "fn")))
    row = {f"event_{k}": per_event[k] for k in ("precision", "recall", "f1", "iou")}
    row.update({f"pooled_{k}": v for k, v in pooled.items()})
    return pd.Series(row)


def event_f1(counts):
    """Mean over fires of F1, from a {fire_id: [tp, fp, fn]} dict (used for checkpoint selection)."""
    f1 = [2 * tp / (2 * tp + fp + fn) for tp, fp, fn in counts.values() if (2 * tp + fp + fn) > 0]
    return float(np.mean(f1)) if f1 else float("nan")

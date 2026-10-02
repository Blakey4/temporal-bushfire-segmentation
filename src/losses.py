"""Loss functions. All of them skip pixels labelled 255 (other fires, cloud, no data)."""
import torch
import torch.nn.functional as F

IGNORE = 255


def masked_bce(logits, target, ignore=IGNORE, pos_weight=None):
    """Binary cross-entropy averaged over the pixels that aren't marked `ignore`.

    For a pixel with label y in {0, 1} and logit z, with p = sigmoid(z):
        loss = -[w y log p + (1 - y) log(1 - p)]
    where w = pos_weight (default 1) up-weights errors on burnt pixels.
    """
    valid = target != ignore
    if not valid.any():
        return logits.sum() * 0.0  # keeps the graph valid for an all-ignore batch
    weight = None if pos_weight is None else torch.tensor(pos_weight, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits[valid], target[valid], pos_weight=weight)


def masked_dice(logits, target, ignore=IGNORE, smooth=1.0):
    """Soft Dice loss over the non-ignored pixels of the whole batch:
        1 - (2 * sum(p * y) + s) / (sum(p) + sum(y) + s)
    It directly targets overlap (it is 1 - a soft F1), so it is less dominated by the many easy
    unburnt pixels than BCE, but it is not a calibrated probability loss.
    """
    valid = target != ignore
    p, y = torch.sigmoid(logits[valid]), target[valid]
    return 1 - (2 * (p * y).sum() + smooth) / (p.sum() + y.sum() + smooth)


def make_loss(name="bce", pos_weight=None):
    """The loss used for training: "bce", "dice" or "bce+dice" (their sum)."""
    if name == "bce":
        return lambda logits, target: masked_bce(logits, target, pos_weight=pos_weight)
    if name == "dice":
        return masked_dice
    if name == "bce+dice":
        return lambda logits, target: masked_bce(logits, target, pos_weight=pos_weight) + masked_dice(logits, target)
    raise ValueError(f"unknown loss: {name}")

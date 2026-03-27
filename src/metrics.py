"""
metrics.py — Binary segmentation metrics via TorchMetrics 1.x.

All public functions accept raw logits (B, 1, H, W) and long targets
(B, 1, H, W) with values in {0, 1}.  Use _probs(logits) to convert
logits to probabilities before passing to TorchMetrics objects.

Design
------
Every metric is MICRO-averaged — TP/FP/FN/TN counts are accumulated
globally across all pixels in all batches, then divided once at compute().
This gives equal weight to every pixel regardless of batch size or ordering,
and is provably correct under DDP (global counts, not scalar averages).

DDP correctness
---------------
Register the MetricCollection returned by make_metrics() as an nn.Module
attribute of the LightningModule.  TorchMetrics then all-reduces the raw
integer state tensors (TP, FP, FN, TN) across ranks via NCCL before
compute(), so the result is the exact global metric — not an approximation.

Empty-mask convention (zero_division=1.0)
-----------------------------------------
When both prediction and GT are all-zero (true-negative image — no defect,
model says nothing): TP=FP=FN=0 → denominator=0.  We set zero_division=1.0
so the score is 1.0, treating "correctly predicting nothing" as perfect.
"""
import torch

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)

from . import config

# Shared kwargs for every Binary* metric.
_METRIC_KW = dict(threshold=config.THRESHOLD, zero_division=1.0)


def make_metrics() -> MetricCollection:
    """
    Return a MetricCollection with all standard binary-segmentation metrics.

    Register as an nn.Module attribute so Lightning tracks the device and
    DDP syncs state tensors automatically::

        self.val_metrics = make_metrics()       # in __init__
        # validation_step:
        self.val_metrics.update(_probs(logits), targets.long())
        # on_validation_epoch_end:
        m = self.val_metrics.compute()
        self.val_metrics.reset()

    Keys returned by compute(): mIoU | dice | precision | recall | px_acc
    """
    return MetricCollection({
        "mIoU":      BinaryJaccardIndex(**_METRIC_KW),
        "dice":      BinaryF1Score(**_METRIC_KW),
        "precision": BinaryPrecision(**_METRIC_KW),
        "recall":    BinaryRecall(**_METRIC_KW),
        "px_acc":    BinaryAccuracy(threshold=config.THRESHOLD),
    })


def _probs(logits: torch.Tensor) -> torch.Tensor:
    """Raw logits → sigmoid probabilities.  Shape is preserved."""
    return torch.sigmoid(logits)


# ── Stateless scalar helpers ──────────────────────────────────────────────────
# Used by visualize_ckpt.py and offline evaluation.
# Each call creates a temporary TorchMetrics object — not for hot training loops.

def iou_score(
    logits:    torch.Tensor,
    targets:   torch.Tensor,
    threshold: float = config.THRESHOLD,
) -> float:
    """Micro-IoU = TP/(TP+FP+FN) over the batch."""
    m = BinaryJaccardIndex(threshold=threshold, zero_division=1.0).to(logits.device)
    return m(_probs(logits), targets.long()).item()


def dice_score(
    logits:    torch.Tensor,
    targets:   torch.Tensor,
    threshold: float = config.THRESHOLD,
) -> float:
    """Micro-Dice = 2TP/(2TP+FP+FN) over the batch (= F1)."""
    m = BinaryF1Score(threshold=threshold, zero_division=1.0).to(logits.device)
    return m(_probs(logits), targets.long()).item()


def precision_recall(
    logits:    torch.Tensor,
    targets:   torch.Tensor,
    threshold: float = config.THRESHOLD,
) -> tuple[float, float]:
    """(precision, recall) — micro-averaged over the batch."""
    dev   = logits.device
    probs = _probs(logits)
    tgt   = targets.long()
    p = BinaryPrecision(threshold=threshold, zero_division=1.0).to(dev)(probs, tgt).item()
    r = BinaryRecall(threshold=threshold, zero_division=1.0).to(dev)(probs, tgt).item()
    return p, r


def pixel_accuracy(
    logits:    torch.Tensor,
    targets:   torch.Tensor,
    threshold: float = config.THRESHOLD,
) -> float:
    """Fraction of correctly classified pixels over the batch."""
    m = BinaryAccuracy(threshold=threshold).to(logits.device)
    return m(_probs(logits), targets.long()).item()

"""
losses.py — Loss functions for binary segmentation.

CombinedLoss = w_bce * BCEWithLogitsLoss  +  w_dice * DiceLoss

The model outputs raw logits (not sigmoid'd), so both losses apply sigmoid
internally.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class DiceLoss(nn.Module):
    """
    Soft Dice Loss computed from raw logits.

        DiceLoss = 1 - (2 * |P ∩ T| + smooth) / (|P| + |T| + smooth)

    where P = sigmoid(logits), T = targets ∈ {0,1}.
    Smooth factor prevents division by zero on all-background samples.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 1, H, W)  raw model output.
            targets: (B, 1, H, W)  float in [0, 1].
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dims → (B, N)
        probs   = probs.view(probs.size(0),   -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union        = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Weighted sum of BCEWithLogitsLoss and DiceLoss.

    Args:
        bce_weight:  contribution of BCE  (default from config).
        dice_weight: contribution of Dice (default from config).
        pos_weight:  optional class-frequency balancing for BCE.
                     If None, uses uniform weight.
    """

    def __init__(
        self,
        bce_weight:  float = config.BCE_WEIGHT,
        dice_weight: float = config.DICE_WEIGHT,
        pos_weight:  torch.Tensor | None = None,
    ):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns:
            total_loss: scalar tensor (differentiable).
            components: dict with 'bce', 'dice', 'total' for logging.
        """
        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total     = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return total, {
            "bce":   bce_loss.item(),
            "dice":  dice_loss.item(),
            "total": total.item(),
        }

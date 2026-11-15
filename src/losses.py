"""
Compound Loss functions for Vesuvius Challenge 3D Segmentation.

Key Features:
- Handles ignore label (label == 2) seamlessly.
- Balanced combination of BCEWithLogits and Soft Dice Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Apply valid mask
        probs = probs[valid_mask]
        targets = targets[valid_mask].float()
        
        if len(probs) == 0:
            return (logits * 0.0).sum()

        intersection = torch.sum(probs * targets)
        denominator = torch.sum(probs) + torch.sum(targets)
        
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice


class VesuviusCompoundLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, pos_weight: float = 1.0, ignore_label: int = 2):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.ignore_label = ignore_label
        self.dice_loss = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        if logits.dim() == 5 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        valid_mask = (targets != self.ignore_label)
        
        if not torch.any(valid_mask):
            zero_loss = (logits * 0.0).sum()
            return {"loss": zero_loss, "bce": zero_loss, "dice": zero_loss}

        pos_w = torch.tensor([self.pos_weight], device=logits.device) if self.pos_weight != 1.0 else None
        bce = F.binary_cross_entropy_with_logits(
            logits[valid_mask],
            targets[valid_mask].float(),
            pos_weight=pos_w
        )
        
        dice = self.dice_loss(logits, targets, valid_mask)
        
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        
        return {
            "loss": total_loss,
            "bce": bce,
            "dice": dice
        }


import torch
import numpy as np
import pytest
from src.models.unet3d import ResEncUNet3D
from src.models.segresnet3d import PureSegResNet3D
from src.losses import VesuviusCompoundLoss


def test_model_forward_shape():
    # Mini channels to keep test fast
    model = ResEncUNet3D(in_channels=1, out_channels=1, channels=[8, 16, 32, 64, 96])
    model.eval()
    
    # Input batch of 1 patch of 64x64x64
    x = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        out = model(x)
        
    assert out.shape == (1, 1, 64, 64, 64), f"Expected (1, 1, 64, 64, 64), got {out.shape}"


def test_segresnet_forward_shape():
    model = PureSegResNet3D(
        in_channels=1, out_channels=1, init_filters=8,
        blocks_down=(1, 1, 1, 1), blocks_up=(1, 1, 1),
    )
    model.eval()

    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 1, 32, 32, 32), f"Expected (1, 1, 32, 32, 32), got {out.shape}"


def test_loss_with_ignore_label():
    criterion = VesuviusCompoundLoss(bce_weight=0.5, dice_weight=0.5, ignore_label=2)
    
    logits = torch.randn(2, 1, 32, 32, 32, requires_grad=True)
    # Targets with 0, 1, and 2 (ignore)
    targets = torch.randint(0, 3, (2, 32, 32, 32)).long()
    
    loss_dict = criterion(logits, targets)
    assert "loss" in loss_dict
    assert loss_dict["loss"].item() > 0
    
    # Check backward pass
    loss_dict["loss"].backward()
    assert logits.grad is not None

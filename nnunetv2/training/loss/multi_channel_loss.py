"""
multi_channel_loss.py
======================
Loss function for 3-channel binary segmentation.

The network outputs 6 logit channels: [bg0, fg0, bg1, fg1, bg2, fg2].
The target seg is shape (B, 3, D, H, W) with integer labels 0/1 per channel.

We apply DC_and_CE_loss independently to each 2-class block and average.
The result is fully compatible with DeepSupervisionWrapper.
"""

import torch
from torch import nn
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

NUM_SEG_CHANNELS  = 3
CLASSES_PER_CHANNEL = 2   # binary: background + foreground


def _make_single_channel_loss(batch_dice: bool, ddp: bool) -> DC_and_CE_loss:
    return DC_and_CE_loss(
        soft_dice_kwargs={
            "batch_dice": batch_dice,
            "smooth": 1e-5,
            "do_bg": False,
            "ddp": ddp,
        },
        ce_kwargs={},
        weight_ce=1,
        weight_dice=1,
        ignore_label=None,
        dice_class=MemoryEfficientSoftDiceLoss,
    )


class MultiChannelSegLoss(nn.Module):
    """
    Averages DC+CE loss over the 3 segmentation channels.

    net_output : (B, 6, D, H, W)   — raw logits from the network
    target     : (B, 3, D, H, W)   — integer labels (0 or 1) per channel
                 OR a list of such tensors when deep supervision is active
                 (DeepSupervisionWrapper passes a list of scaled targets).
    """

    def __init__(self, batch_dice: bool = True, ddp: bool = False):
        super().__init__()
        self.losses = nn.ModuleList(
            [_make_single_channel_loss(batch_dice, ddp) for _ in range(NUM_SEG_CHANNELS)]
        )

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        net_output : (B, 6, D, H, W)
        target     : (B, 3, D, H, W)  — each entry is 0 or 1
        """
        total = torch.tensor(0.0, device=net_output.device, requires_grad=True)

        for ch in range(NUM_SEG_CHANNELS):
            # Slice out the 2 logit channels for this seg channel
            logit_start = ch * CLASSES_PER_CHANNEL
            logit_end   = logit_start + CLASSES_PER_CHANNEL
            pred_ch = net_output[:, logit_start:logit_end]          # (B, 2, D, H, W)

            # Target for this channel — shape must be (B, 1, D, H, W) for DC_and_CE_loss
            tgt_ch = target[:, ch:ch+1].long()                       # (B, 1, D, H, W)

            total = total + self.losses[ch](pred_ch, tgt_ch)

        return total / NUM_SEG_CHANNELS

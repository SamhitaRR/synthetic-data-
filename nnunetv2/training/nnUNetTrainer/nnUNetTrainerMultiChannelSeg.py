"""
nnUNetTrainerMultiChannelSeg.py
================================
Custom trainer for 3-channel binary segmentation output.
Inherits directly from the default nnUNetTrainer — standard augmentations,
standard training loop, saves every 50 epochs.

Only three things are changed:
  1. LabelManager  → MultiChannelLabelManager (6 output heads)
  2. Loss          → MultiChannelSegLoss (independent DC+CE per channel)
  3. Export        → writes 3 output files per case at inference

Train:
    nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerMultiChannelSeg

Predict:
    nnUNetv2_predict -i INPUT -o OUTPUT -d <ID> -c 3d_fullres -tr nnUNetTrainerMultiChannelSeg -f 0

Data layout (nnUNet_raw/<DatasetXXX>/labelsTr/):
    img_000.tif        <- label channel 0
    img_000_seg1.tif   <- label channel 1
    img_000_seg2.tif   <- label channel 2
"""

import numpy as np
import torch
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.multi_channel_label_handling import MultiChannelLabelManager
from nnunetv2.training.loss.multi_channel_loss import MultiChannelSegLoss
from nnunetv2.inference.multi_channel_export import export_multichannel_prediction_from_logits
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

NUM_SEG_CHANNELS    = 3
CLASSES_PER_CHANNEL = 2


class nnUNetTrainerMultiChannelSeg(nnUNetTrainer):
    """
    Inherits from the default nnUNetTrainer (nnUNetTrainer.py).
    Standard augmentations and training loop are unchanged.
    """

    # ------------------------------------------------------------------ #
    #  Override label manager                                              #
    # ------------------------------------------------------------------ #

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.label_manager = MultiChannelLabelManager(
            dataset_json.get("labels", {}),
            regions_class_order=None,
            force_use_labels=True,
        )

    # ------------------------------------------------------------------ #
    #  Override loss                                                       #
    # ------------------------------------------------------------------ #

    def _build_loss(self):
        loss = MultiChannelSegLoss(
            batch_dice=self.configuration_manager.batch_dice,
            ddp=self.is_ddp,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    # ------------------------------------------------------------------ #
    #  Override validation step                                            #
    # ------------------------------------------------------------------ #

    def validation_step(self, batch: dict) -> dict:
        data   = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # Use highest resolution output only (if deep supervision enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # Compute tp/fp/fn across all 3 channels combined for the online DICE display.
        # We do argmax per channel pair and accumulate.
        axes = [0] + list(range(2, output.ndim))

        tp_hard_all = []
        fp_hard_all = []
        fn_hard_all = []

        for ch in range(NUM_SEG_CHANNELS):
            start = ch * CLASSES_PER_CHANNEL
            end   = start + CLASSES_PER_CHANNEL

            pred_ch   = output[:, start:end]                        # (B, 2, ...)
            target_ch = target[:, ch:ch+1].long()                   # (B, 1, ...)

            # argmax → one-hot
            output_seg = pred_ch.argmax(1)[:, None]                 # (B, 1, ...)
            pred_onehot = torch.zeros(pred_ch.shape, device=pred_ch.device, dtype=torch.float32)
            pred_onehot.scatter_(1, output_seg, 1)                  # (B, 2, ...)

            # one-hot encode target for get_tp_fp_fn_tn — must be bool for ~ operator
            target_onehot = torch.zeros(pred_ch.shape, device=pred_ch.device, dtype=torch.bool)
            target_onehot.scatter_(1, target_ch, 1)                 # (B, 2, ...)

            tp, fp, fn, _ = get_tp_fp_fn_tn(pred_onehot, target_onehot, axes=axes, mask=None)

            # [1:] to drop background class, keep only foreground
            tp_hard_all.append(tp.detach().cpu().numpy()[1:])
            fp_hard_all.append(fp.detach().cpu().numpy()[1:])
            fn_hard_all.append(fn.detach().cpu().numpy()[1:])

        # Concatenate across channels → shape (3,) one foreground value per channel
        tp_hard = np.concatenate(tp_hard_all)
        fp_hard = np.concatenate(fp_hard_all)
        fn_hard = np.concatenate(fn_hard_all)

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    # ------------------------------------------------------------------ #
    #  Override export so 3 files are written per case at inference       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def export_prediction(
        predicted_logits,
        properties_dict,
        configuration_manager,
        plans_manager,
        dataset_json,
        output_file_truncated,
        save_probabilities=False,
    ):
        export_multichannel_prediction_from_logits(
            predicted_logits,
            properties_dict,
            configuration_manager,
            plans_manager,
            dataset_json,
            output_file_truncated,
            save_probabilities=save_probabilities,
        )

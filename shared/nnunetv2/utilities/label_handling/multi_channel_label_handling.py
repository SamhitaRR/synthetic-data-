"""
multi_channel_label_handling.py
================================
Drop-in replacement LabelManager for 3-channel label output.

Each of the 3 label channels is a BINARY segmentation (foreground vs background).
The network will output 3 independent pairs of logits (background + foreground),
i.e. num_segmentation_heads = 6  (2 classes × 3 channels).

dataset.json must look like:
{
  "labels": {
    "background": 0,
    "label_ch0": 1,
    "label_ch1": 2,
    "label_ch2": 3
  },
  "num_label_channels": 3,           # <-- NEW KEY
  "file_ending": ".tif",
  ...
}

At preprocessing time each training case has its segmentation stored as a
3-channel array of shape (3, D, H, W) where each channel contains 0/1 values.
"""

from typing import List, Union
import numpy as np
import torch
from nnunetv2.utilities.label_handling.label_handling import LabelManager


NUM_SEG_CHANNELS = 3       # one per label type
CLASSES_PER_CHANNEL = 2    # background + foreground (binary)


class MultiChannelLabelManager(LabelManager):
    """
    Extends LabelManager for 3 independent binary segmentation channels.

    Key changes vs. base class
    --------------------------
    * num_segmentation_heads  ->  NUM_SEG_CHANNELS * CLASSES_PER_CHANNEL  (= 6)
    * apply_inference_nonlin  ->  applies softmax independently to each 2-class
                                  block so each channel sums to 1
    * convert_logits_to_segmentation  ->  returns (3, D, H, W) argmax per channel
    * convert_probabilities_to_segmentation  ->  same but from probs
    """

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def num_segmentation_heads(self) -> int:
        """
        The network's final conv produces this many output channels.
        6 = 3 label channels × 2 classes (bg + fg) each.
        """
        return NUM_SEG_CHANNELS * CLASSES_PER_CHANNEL

    @property
    def foreground_labels(self) -> List[int]:
        # One foreground label per channel (value = 1 in that channel's map)
        return [1] * NUM_SEG_CHANNELS

    @property
    def all_labels(self) -> List[int]:
        return [0, 1]   # background and foreground within each channel

    # ------------------------------------------------------------------ #
    #  Inference helpers                                                   #
    # ------------------------------------------------------------------ #

    def apply_inference_nonlin(
        self, predicted_logits: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Apply softmax independently to each 2-class block.

        Input shape:  (6, D, H, W)  — or batched (B, 6, D, H, W)
        Output shape: same
        """
        if isinstance(predicted_logits, np.ndarray):
            predicted_logits = torch.from_numpy(predicted_logits)

        out = torch.zeros_like(predicted_logits, dtype=torch.float32)
        for ch in range(NUM_SEG_CHANNELS):
            start = ch * CLASSES_PER_CHANNEL
            end   = start + CLASSES_PER_CHANNEL
            out[..., start:end, :, :, :] = torch.softmax(
                predicted_logits[..., start:end, :, :, :], dim=-4
            )
        return out

    def convert_logits_to_segmentation(
        self, predicted_logits: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        Argmax per channel block → integer label map per channel.

        Input shape:  (6, D, H, W)
        Output shape: (3, D, H, W)   — each slice is 0/1
        """
        if isinstance(predicted_logits, np.ndarray):
            predicted_logits = torch.from_numpy(predicted_logits)

        segs = []
        for ch in range(NUM_SEG_CHANNELS):
            start = ch * CLASSES_PER_CHANNEL
            end   = start + CLASSES_PER_CHANNEL
            seg_ch = torch.argmax(predicted_logits[start:end], dim=0)   # (D,H,W)
            segs.append(seg_ch.cpu().numpy().astype(np.uint8))

        return np.stack(segs, axis=0)   # (3, D, H, W)

    def convert_probabilities_to_segmentation(
        self, predicted_probabilities: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """Same as above but input is already softmax'd."""
        return self.convert_logits_to_segmentation(predicted_probabilities)

    def revert_cropping_on_probabilities(
        self,
        predicted_probabilities: torch.Tensor,
        bbox: list,
        original_shape: tuple,
    ) -> torch.Tensor:
        """
        Put the cropped probability map back into the original image shape.
        Works channel-by-channel (6 channels total).
        """
        from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
        out = torch.zeros(
            (self.num_segmentation_heads, *original_shape),
            dtype=predicted_probabilities.dtype,
        )
        for c in range(self.num_segmentation_heads):
            out[c] = torch.from_numpy(
                insert_crop_into_image(
                    out[c].numpy(),
                    predicted_probabilities[c].cpu().numpy(),
                    bbox,
                )
            )
        return out

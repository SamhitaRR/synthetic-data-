"""
multi_channel_preprocessor.py
==============================
Subclass of DefaultPreprocessor that loads 3 separate segmentation files
per training case and stacks them into a single (3, D, H, W) array.

Naming convention expected in  nnUNet_raw/<dataset>/labelsTr/:
    <stem>.tif        — channel 0  (the main file nnUNet expects natively)
    <stem>_seg1.tif   — channel 1
    <stem>_seg2.tif   — channel 2

Channel 0 uses the plain stem filename so that nnUNet's fingerprint extractor,
integrity checker, and other pipeline steps that look for the canonical label
file all work without modification.

(Replace .tif with whatever your dataset's file_ending is.)

Usage: set  "preprocessor_name": "MultiChannelPreprocessor"  in the
       relevant configuration block of your nnUNetPlans.json.
"""

from typing import List, Union, Tuple

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, join

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

NUM_SEG_CHANNELS = 3


class MultiChannelPreprocessor(DefaultPreprocessor):
    """
    Overrides run_case() so that it loads 3 label files instead of 1
    and stacks them along axis-0  →  shape (3, D, H, W).

    Everything else (cropping, resampling, normalisation) is inherited
    unchanged from DefaultPreprocessor.
    """

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run_case(
        self,
        image_files: List[str],
        seg_file: Union[str, None],       # ← we IGNORE this; derive from stem
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: Union[dict, str],
    ):
        """
        seg_file is the path nnUNet passes for the canonical label file.
        We use it directly as channel 0, then derive channels 1 and 2
        by appending _seg1 and _seg2 to the stem.

        Expected files on disk:
            …/labelsTr/<stem>.tif       ← channel 0 (seg_file itself)
            …/labelsTr/<stem>_seg1.tif  ← channel 1
            …/labelsTr/<stem>_seg2.tif  ← channel 2
        """
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()

        # Load image(s) as usual
        data, data_properties = rw.read_images(image_files)

        # Build multi-channel seg
        if seg_file is not None:
            seg = self._load_multichannel_seg(seg_file, dataset_json, rw)
        else:
            seg = None

        if self.verbose:
            print(f"[MultiChannelPreprocessor] seg shape: {seg.shape if seg is not None else None}")

        data, seg, data_properties = self.run_case_npy(
            data, seg, data_properties, plans_manager, configuration_manager, dataset_json
        )
        return data, seg, data_properties

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_multichannel_seg(
        seg_file: str,
        dataset_json: dict,
        rw,
    ) -> np.ndarray:
        """
        Channel 0 is seg_file itself  (<stem>.tif).
        Channels 1 and 2 are derived as <stem>_seg1.tif, <stem>_seg2.tif.

        Returns ndarray of shape (3, D, H, W) with integer labels.
        """
        file_ending = dataset_json.get("file_ending", ".tif")
        assert seg_file.endswith(file_ending), (
            f"seg_file '{seg_file}' does not end with expected file_ending '{file_ending}'"
        )
        base = seg_file[: -len(file_ending)]  # e.g.  …/labelsTr/img_000

        # Channel 0 = the canonical label file nnUNet already knows about
        # Channels 1+ = _seg1, _seg2, ... suffixed files
        ch_paths = [seg_file] + [
            f"{base}_seg{ch_idx}{file_ending}"
            for ch_idx in range(1, NUM_SEG_CHANNELS)
        ]

        channels = []
        for ch_path in ch_paths:
            ch_data, _ = rw.read_seg(ch_path)   # returns (1, D, H, W)
            channels.append(ch_data)

        # Stack along channel axis → (3, D, H, W)
        seg = np.concatenate(channels, axis=0)
        return seg

    # ------------------------------------------------------------------ #
    #  Override _sample_foreground_locations to work per-channel          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sample_foreground_locations(
        seg: np.ndarray,
        classes_or_regions,
        seed: int = 1234,
        verbose: bool = False,
    ):
        """
        For a multi-channel seg of shape (3, D, H, W) we collect foreground
        locations from ALL channels combined.  The class key used is just 1
        (foreground), pooled across channels so the dataloader can
        oversample patches that contain any foreground.
        """
        import math
        import pandas as pd

        num_samples = 10000
        min_percent_coverage = 0.01
        rndst = np.random.RandomState(seed)

        # Treat any channel being == 1 as foreground
        foreground_mask = (seg > 0).any(axis=0)          # (D, H, W)
        foreground_coords = np.argwhere(foreground_mask)  # (N, 3)

        if len(foreground_coords) == 0:
            return {1: []}

        # Cap
        if len(foreground_coords) > 1e7:
            take_every = math.floor(len(foreground_coords) / 1e7)
            foreground_coords = foreground_coords[::take_every]

        target_num_samples = min(num_samples, len(foreground_coords))
        target_num_samples = max(
            target_num_samples,
            int(np.ceil(len(foreground_coords) * min_percent_coverage)),
        )
        selected = foreground_coords[
            rndst.choice(len(foreground_coords), target_num_samples, replace=False)
        ]
        # Prepend a dummy dim=0 index so shape matches what the dataloader expects
        # (it adds 1 for the channel dimension when indexing)
        selected_with_ch = np.hstack([np.zeros((len(selected), 1), dtype=int), selected])
        return {1: selected_with_ch}

"""
multi_channel_export.py
========================
Replaces export_prediction_from_logits for the 3-channel case.

Instead of writing one segmentation file, it writes THREE:
    <output_file_truncated>_seg0<ending>
    <output_file_truncated>_seg1<ending>
    <output_file_truncated>_seg2<ending>

Each file is a binary mask (uint8, values 0/1).
"""

from typing import Union
import numpy as np
import torch

from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

# Import our custom label manager just to re-use constants
NUM_SEG_CHANNELS    = 3
CLASSES_PER_CHANNEL = 2


def export_multichannel_prediction_from_logits(
    predicted_logits: Union[np.ndarray, torch.Tensor],
    properties_dict: dict,
    configuration_manager: ConfigurationManager,
    plans_manager: PlansManager,
    dataset_json_dict_or_file: Union[dict, str],
    output_file_truncated: str,
    save_probabilities: bool = False,
    num_threads_torch: int = default_num_processes,
):
    """
    predicted_logits : (6, D, H, W)  — raw network output for one case
    Writes up to 3 binary tif files (one per label channel).
    """
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    file_ending = dataset_json_dict_or_file["file_ending"]
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(predicted_logits, np.ndarray):
        predicted_logits = torch.from_numpy(predicted_logits)

    # ── 1. Resample back to original spacing ──────────────────────────
    spacing_transposed = [properties_dict["spacing"][i] for i in plans_manager.transpose_forward]
    current_spacing = (
        configuration_manager.spacing
        if len(configuration_manager.spacing)
        == len(properties_dict["shape_after_cropping_and_before_resampling"])
        else [spacing_transposed[0], *configuration_manager.spacing]
    )
    target_spacing = [
        properties_dict["spacing"][i] for i in plans_manager.transpose_forward
    ]

    predicted_logits = configuration_manager.resampling_fn_probabilities(
        predicted_logits,
        properties_dict["shape_after_cropping_and_before_resampling"],
        current_spacing,
        target_spacing,
    )

    # ── 2. Argmax per channel block → (3, D, H, W) ───────────────────
    if isinstance(predicted_logits, np.ndarray):
        predicted_logits = torch.from_numpy(predicted_logits)

    seg_channels = []
    for ch in range(NUM_SEG_CHANNELS):
        start = ch * CLASSES_PER_CHANNEL
        end   = start + CLASSES_PER_CHANNEL
        seg_ch = torch.argmax(predicted_logits[start:end], dim=0).cpu().numpy().astype(np.uint8)
        seg_channels.append(seg_ch)

    # ── 3. Revert cropping for each channel ──────────────────────────
    shape_before_crop = properties_dict["shape_before_cropping"]
    bbox              = properties_dict["bbox_used_for_cropping"]

    reverted = []
    for seg_ch in seg_channels:
        canvas = np.zeros(shape_before_crop, dtype=np.uint8)
        canvas = insert_crop_into_image(canvas, seg_ch, bbox)
        reverted.append(canvas)

    # ── 4. Revert transpose for each channel ─────────────────────────
    transpose_back = plans_manager.transpose_backward
    reverted = [ch.transpose(transpose_back) for ch in reverted]

    # ── 5. Write one file per channel ────────────────────────────────
    rw = plans_manager.image_reader_writer_class()
    for ch_idx, seg_ch in enumerate(reverted):
        out_path = f"{output_file_truncated}_seg{ch_idx}{file_ending}"
        rw.write_seg(seg_ch, out_path, properties_dict)
        print(f"  Saved channel {ch_idx} → {out_path}")

    # Optionally save properties for downstream use
    save_pickle(properties_dict, output_file_truncated + ".pkl")

    torch.set_num_threads(old_threads)

import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform

class RandomDimInsideMaskTransform(AbstractTransform):
    """
    Randomly dims image intensities inside a selected segmentation mask channel.
    Works with both NumPy arrays and PyTorch tensors.
    """
    def __init__(self, min_factor=0.3, max_factor=1.0, p_per_sample=0.7,
                 mask_idx_in_seg=0, data_key="image", seg_key="segmentation"):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.p_per_sample = p_per_sample
        self.mask_idx_in_seg = mask_idx_in_seg
        self.data_key = data_key
        self.seg_key = seg_key

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        seg = data_dict[self.seg_key]

        # Ensure batch dimension
        squeeze = False
        if data.ndim == 4:  # single sample (C, Z, Y, X)
            data = data[None, ...]
            seg = seg[None, ...]
            squeeze = True

        B, C = data.shape[:2]

        for b in range(B):
            if np.random.rand() < self.p_per_sample:
                factor = np.random.uniform(self.min_factor, self.max_factor)
                mask = seg[b, self.mask_idx_in_seg] > 0

                if isinstance(data, torch.Tensor):
                    # Convert mask to tensor if needed, and move to same device
                    if not isinstance(mask, torch.Tensor):
                        mask = torch.from_numpy(mask)
                    mask = mask.to(data.device)
                    # In-place dimming inside mask
                    data[b, :, mask] *= factor
                else:
                    # NumPy array
                    data[b, :, mask] *= factor

        if squeeze:
            data = data[0]

        data_dict[self.data_key] = data
        return data_dict


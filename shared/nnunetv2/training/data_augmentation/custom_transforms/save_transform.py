import os
import nibabel as nib
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np


class DebugSaveTransform(AbstractTransform):
    """
    Saves first N images in a batch. Compatible with 4D/5D tensors.
    """
    def __init__(self, save_dir, tag, max_images=3):
        self.save_dir = save_dir
        self.tag = tag
        self.max_images = max_images
        os.makedirs(save_dir, exist_ok=True)
        self.counter = 0

    def __call__(self, **data_dict):
        data = data_dict["image"]  # tensor (B, C, Z, Y, X) or (C, Z, Y, X)
        squeeze = False
        if data.ndim == 4:
            data = data[None, ...]
            squeeze = True

        n_to_save = min(data.shape[0], self.max_images)
        for b in range(n_to_save):
            img = data[b, 0]  # first channel
            # Convert tensor to numpy float32
            if hasattr(img, "detach"):  # it's a PyTorch tensor
                img = img.detach().cpu().numpy().astype(np.float32)

            filename = os.path.join(self.save_dir, f"{self.counter:03d}_b{b}_{self.tag}.nii.gz")
            nib.save(nib.Nifti1Image(img, np.eye(4)), filename)

        self.counter += 1
        if squeeze:
            data = data[0]
        data_dict["image"] = data
        return data_dict


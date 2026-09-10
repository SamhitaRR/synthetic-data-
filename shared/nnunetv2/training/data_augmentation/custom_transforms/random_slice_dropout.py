import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform

class RandomSliceDropoutTransform(AbstractTransform):
    """
    Randomly sets entire slices to zero along a chosen axis.
    """
    def __init__(self, axis=2, p_per_sample=0.3, max_slices_to_drop=3, fill_value=0.0):
        self.axis = axis
        self.p_per_sample = p_per_sample
        self.max_slices_to_drop = max_slices_to_drop
        self.fill_value = fill_value

    def __call__(self, **data_dict):
        data = data_dict["image"]  # (B, C, Z, Y, X) or (C, Z, Y, X)
        squeeze = False
        if data.ndim == 4:
            data = data[None, ...]
            squeeze = True

        for b in range(data.shape[0]):
            if np.random.rand() < self.p_per_sample:
                n_slices = np.random.randint(1, self.max_slices_to_drop + 1)
                spatial_dim = data.shape[self.axis]
                slices = np.random.choice(spatial_dim, n_slices, replace=False)

                for s in slices:
                    if self.axis == 2:
                        data[b, :, s, :, :] = self.fill_value
                    elif self.axis == 3:
                        data[b, :, :, s, :] = self.fill_value
                    elif self.axis == 4:
                        data[b, :, :, :, s] = self.fill_value

        if squeeze:
            data = data[0]
        data_dict["image"] = data
        return data_dict


import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform

class RandomCuboidDropoutTransform(AbstractTransform):
    """
    Randomly zeros out cuboid regions in 3D images.
    """
    def __init__(self, p_per_sample=0.3, max_cuboids=2, max_size_ratio=0.3, fill_value=0.0):
        self.p_per_sample = p_per_sample
        self.max_cuboids = max_cuboids
        self.max_size_ratio = max_size_ratio
        self.fill_value = fill_value

    def __call__(self, **data_dict):
        data = data_dict["image"]
        squeeze = False
        if data.ndim == 4:
            data = data[None, ...]
            squeeze = True

        B, C, Z, Y, X = data.shape
        for b in range(B):
            if np.random.rand() < self.p_per_sample:
                n_cuboids = np.random.randint(1, self.max_cuboids + 1)
                for _ in range(n_cuboids):
                    dz = int(Z * np.random.uniform(0.05, self.max_size_ratio))
                    dy = int(Y * np.random.uniform(0.05, self.max_size_ratio))
                    dx = int(X * np.random.uniform(0.05, self.max_size_ratio))
                    z0 = np.random.randint(0, Z - dz)
                    y0 = np.random.randint(0, Y - dy)
                    x0 = np.random.randint(0, X - dx)
                    data[b, :, z0:z0+dz, y0:y0+dy, x0:x0+dx] = self.fill_value

        if squeeze:
            data = data[0]
        data_dict["image"] = data
        return data_dict


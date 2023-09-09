import numpy as np
import torch

def construct_grid(arrays):
    if isinstance(arrays[0], np.ndarray):
        arrays = [torch.from_numpy(a) for a in arrays]
    return torch.moveaxis(torch.stack(torch.meshgrid(*arrays, indexing='ij'), dim=0), 0, -1)
    
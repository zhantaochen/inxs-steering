import numpy as np
import torch

def convert_to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().clone().numpy()
    elif isinstance(data, list):
        data = np.asarray(data)
    return data

def convert_to_torch(data):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data.copy())
    elif isinstance(data, list):
        data = torch.from_numpy(np.ndarray(data))
    return data

def construct_grid(arrays):
    if isinstance(arrays[0], np.ndarray):
        arrays = [torch.from_numpy(a) for a in arrays]
    return torch.moveaxis(torch.stack(torch.meshgrid(*arrays, indexing='ij'), dim=0), 0, -1)

def scale_tensor(tensor, bounds_init, bounds_fnal=(-1., 1.)):
    min_init, max_init = bounds_init
    min_fnal, max_fnal = bounds_fnal
    return ((tensor - min_init) * (max_fnal - min_fnal) / (max_init - min_init)) + min_fnal

import pickle
import torch
import numpy as np
from tqdm import tqdm

try:
    import cupy as xp
    from cupyx.scipy.ndimage import map_coordinates
    using_cupy = True
except ImportError:
    import numpy as xp
    from scipy.ndimage import map_coordinates
    using_cupy = False

def downsample_4d_with_map_coordinates(mask, scale_factors, order=0):
    mask = xp.asarray(mask.astype(xp.float32))
    if isinstance(scale_factors, (int, float)):
        scale_factors = [scale_factors] * 4
    
    # Ensure the scale_factors is a tuple/list of length 4
    if not isinstance(scale_factors, (tuple, list)) or len(scale_factors) != 4:
        raise ValueError("scale_factors should be a tuple or list of length 4")

    # Calculate the new dimensions
    new_dims = [int(d * sf) for d, sf in zip(mask.shape, scale_factors)]

    # Create a grid of coordinates in the old mask
    coord_grid = xp.array(xp.meshgrid(
        xp.linspace(0, mask.shape[0] - 1, new_dims[0]),
        xp.linspace(0, mask.shape[1] - 1, new_dims[1]),
        xp.linspace(0, mask.shape[2] - 1, new_dims[2]),
        xp.linspace(0, mask.shape[3] - 1, new_dims[3]),
        indexing='ij'
    ))

    # Map the coordinates to the original mask
    if using_cupy:
        downsampled_mask = map_coordinates(mask, coord_grid, order=order, mode='nearest').get()
    else:
        downsampled_mask = map_coordinates(mask, coord_grid, order=order, mode='nearest')
    downsampled_mask = downsampled_mask > 0.5
    return downsampled_mask

def downsample_1d_with_map_coordinates(grid, scale_factor):
    grid = xp.asarray(grid.astype(xp.float32))
    
    if not isinstance(scale_factor, (int, float)):
        raise ValueError("scale_factor should be a number")
    
    new_dim = int(grid.shape[0] * scale_factor)
    
    coord_grid = xp.array(
        xp.meshgrid(
            xp.linspace(0, grid.shape[0] - 1, new_dim),
            indexing='ij'
    ))
    if using_cupy:
        downsampled_grid = map_coordinates(grid, coord_grid, order=1, mode='nearest').get()
    else:
        downsampled_grid = map_coordinates(grid, coord_grid, order=1, mode='nearest')
    return downsampled_grid

class PsiMask:
    """
    Written with help from OpenAI's ChatGPT (GPT-4.0).
    """
    def __init__(self, folder_path, scale_factor=None, device='cpu', preload=True):
        """
        :param folder_path: Folder containing the .pkl mask files
        :param device: Either 'cpu' or 'cuda' to determine where masks are stored
        :param preload: If True, all masks are preloaded into memory. If False, masks are loaded on the fly.
        """
        self.folder_path = folder_path
        self.device = device
        self.preload = preload

        self.scale_factor = scale_factor
        
        if self.preload:
            self.masks = {}
            for i in tqdm(range(361)):  # assuming masks are stored for every degree from 0 to 360
                with open(f'{self.folder_path}/{i}.pkl', 'rb') as f:
                    _mask = pickle.load(f)['coverage']
                    if scale_factor is not None:
                        _mask = downsample_4d_with_map_coordinates(_mask, scale_factor)
                    self.masks[i] = torch.tensor(_mask, dtype=torch.bool).to(device)
        else:
            self.masks = None
            
        with open(f'{self.folder_path}/metadata', 'rb') as f:
            _metadata = pickle.load(f)
        for _key in _metadata.keys():
            _grid = _metadata[_key]
            if scale_factor is not None:
                if using_cupy:
                    _grid = xp.asarray(_grid)
                    _grid = downsample_1d_with_map_coordinates(_grid, scale_factor)
                    _grid = xp.asnumpy(_grid)
                else:
                    _grid = downsample_1d_with_map_coordinates(_grid, scale_factor)
            setattr(self, _key, torch.from_numpy(_grid).to(device))
    
    def scale_mask(self, mask):
        if using_cupy:
            mask = xp.asarray(mask)
            mask = downsample_4d_with_map_coordinates(mask, self.scale_factor)
            mask = xp.asnumpy(mask)
        else:
            mask = downsample_4d_with_map_coordinates(mask, self.scale_factor)
        return mask

    def get_mask(self, degree):
        """
        Get mask for a given degree.
        """
        lower_degree = int(np.floor(degree)) % 360
        upper_degree = int(np.ceil(degree)) % 360

        if not self.preload:
            with open(f'{self.folder_path}/{lower_degree}.pkl', 'rb') as f:
                lower_mask = self.scale_mask(pickle.load(f)['coverage'])
                lower_mask = torch.tensor(lower_mask, dtype=torch.bool).to(self.device)
            if lower_degree == upper_degree:
                upper_mask = lower_mask
            else:
                with open(f'{self.folder_path}/{upper_degree}.pkl', 'rb') as f:
                    upper_mask = self.scale_mask(pickle.load(f)['coverage'])
                    upper_mask = torch.tensor(upper_mask, dtype=torch.bool).to(self.device)
        else:
            lower_mask = self.masks[lower_degree]
            if lower_degree == upper_degree:
                upper_mask = lower_mask
            else:
                upper_mask = self.masks[upper_degree]

        # Linearly interpolate between the two masks
        if lower_degree == upper_degree:
            mask = lower_mask
        else:
            alpha = degree - lower_degree
            mask = (1 - alpha) * lower_mask + alpha * upper_mask
        
        mask = mask > 0.5
        
#         if (self.scale_factor is not None) and (not self.preload):
#             if using_cupy:
#                 mask = xp.asarray(mask)
#                 mask = downsample_4d_with_map_coordinates(mask, self.scale_factor)
#                 mask = xp.asnumpy(mask)
#             else:
#                 mask = downsample_4d_with_map_coordinates(mask, self.scale_factor)

        mask = torch.tensor(mask, dtype=torch.bool).to(self.device)
        return mask

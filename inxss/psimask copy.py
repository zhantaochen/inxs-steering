import pickle
import torch
import numpy as np
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator

try:
    import cupy as xp
    from cupyx.scipy.ndimage import map_coordinates
    using_cupy = True
except ImportError:
    import numpy as xp
    from scipy.ndimage import map_coordinates
    using_cupy = False

def downsample_4d_with_map_coordinates(mask, scale_factor, shrink_factor=0., order=0):
    if isinstance(shrink_factor, (int, float)):
        shrink_factor = [shrink_factor,] * 4
    if isinstance(scale_factor, (int, float)):
        scale_factor = [scale_factor] * 4
    
    mask = xp.asarray(mask.astype(xp.float32))
    # Ensure the scale_factor is a tuple/list of length 4
    if not isinstance(scale_factor, (tuple, list)) or len(scale_factor) != 4:
        raise ValueError("scale_factor should be a tuple or list of length 4")

    # Calculate the new dimensions
    new_dims = [int(d * sf) for d, sf in zip(mask.shape, scale_factor)]
    
    # Create a grid of coordinates in the old mask
    coord_grid = xp.array(xp.meshgrid(
        xp.linspace(shrink_factor[0]/2 * mask.shape[0], (1 - shrink_factor[0]/2) * mask.shape[0] - 1, new_dims[0]),
        xp.linspace(shrink_factor[1]/2 * mask.shape[1], (1 - shrink_factor[1]/2) * mask.shape[1] - 1, new_dims[1]),
        xp.linspace(shrink_factor[2]/2 * mask.shape[2], (1 - shrink_factor[2]/2) * mask.shape[2] - 1, new_dims[2]),
        xp.linspace(shrink_factor[3]/2 * mask.shape[3], (1 - shrink_factor[3]/2) * mask.shape[3] - 1, new_dims[3]),
        indexing='ij'
    ))

    # Map the coordinates to the original mask
    if using_cupy:
        downsampled_mask = map_coordinates(mask, coord_grid, order=order, mode='nearest').get()
    else:
        downsampled_mask = map_coordinates(mask, coord_grid, order=order, mode='nearest')
    downsampled_mask = downsampled_mask > 0.5
    return downsampled_mask

def downsample_1d_with_map_coordinates(grid, scale_factor, shrink_factor=0.):
    grid = xp.asarray(grid.astype(xp.float32))
    
    if not isinstance(scale_factor, (int, float)):
        raise ValueError("scale_factor should be a number")
    if not isinstance(shrink_factor, (int, float)):
        raise ValueError("shrink_factor should be a number")
    
    new_dim = int(grid.shape[0] * scale_factor)
    
    coord_grid = xp.array(
        xp.meshgrid(
            xp.linspace(shrink_factor/2 * grid.shape[0], (1 - shrink_factor/2) * grid.shape[0] - 1, new_dim),
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
    def __init__(self, folder_path, scale_factor=None, shrink_factor=0., device='cpu', preload=True):
        """
        :param folder_path: Folder containing the .pkl mask files
        :param device: Either 'cpu' or 'cuda' to determine where masks are stored
        :param preload: If True, all masks are preloaded into memory. If False, masks are loaded on the fly.
        """
        self.folder_path = folder_path
        self.device = device
        self.preload = preload

        
        if isinstance(scale_factor, (int, float)):
            self.scale_factor = [scale_factor,] * 4
        else:
            self.scale_factor = scale_factor
        if isinstance(shrink_factor, (int, float)):
            self.shrink_factor = [shrink_factor,] * 4
        else:
            self.shrink_factor = shrink_factor
        
        if self.preload:
            self.masks = {}
            for i in tqdm(range(361)):  # assuming masks are stored for every degree from 0 to 360
                with open(f'{self.folder_path}/{i}.pkl', 'rb') as f:
                    _mask = pickle.load(f)['coverage']
                    if self.scale_factor is not None:
                        _mask = downsample_4d_with_map_coordinates(_mask, self.scale_factor, shrink_factor=self.shrink_factor)
                    self.masks[i] = torch.tensor(_mask, dtype=torch.bool).to(device)
        else:
            self.masks = None
            
        with open(f'{self.folder_path}/metadata', 'rb') as f:
            _metadata = pickle.load(f)
        for _key in _metadata.keys():
            _tmp_indices = {'h_grid': 0, 'k_grid': 1, 'l_grid': 2, 'w_grid': 3}
            _grid = _metadata[_key]
            if self.scale_factor is not None:
                if using_cupy:
                    _grid = xp.asarray(_grid)
                    _grid = downsample_1d_with_map_coordinates(_grid, self.scale_factor[_tmp_indices[_key]], shrink_factor=self.shrink_factor[_tmp_indices[_key]])
                    _grid = xp.asnumpy(_grid)
                else:
                    _grid = downsample_1d_with_map_coordinates(_grid, self.scale_factor[_tmp_indices[_key]], shrink_factor=self.shrink_factor[_tmp_indices[_key]])
            # print(_key)
            setattr(self, _key, torch.from_numpy(_grid).to(device))
        
        self.psi_grid = torch.arange(360).to(self.h_grid)
        # masks = []
        # for _angle in tqdm(range(360)):
        #     masks.append(self.get_mask(_angle))
        # masks = torch.stack(masks)
        
        # self.psi_mask_func = RegularGridInterpolator(
        #     [self.psi_grid.cpu().numpy(), self.h_grid.cpu().numpy(), 
        #      self.k_grid.cpu().numpy(), self.l_grid.cpu().numpy(), self.w_grid.cpu().numpy()
        #     ], masks.numpy(), bounds_error=False, fill_value=0.)
        
        self.hklw_grid = torch.moveaxis(
            torch.stack(torch.meshgrid(self.h_grid, self.k_grid, self.l_grid, self.w_grid, indexing='ij'), dim=0), 0, -1)
        self.hkw_grid = torch.moveaxis(
            torch.stack(torch.meshgrid(self.h_grid, self.k_grid, self.w_grid, indexing='ij'), dim=0), 0, -1)
        
    def scale_mask(self, mask):
        if self.scale_factor is None:
            return mask
        else:
            if using_cupy:
                mask = xp.asarray(mask)
                mask = downsample_4d_with_map_coordinates(mask, self.scale_factor, shrink_factor=self.shrink_factor)
                mask = xp.asnumpy(mask)
            else:
                mask = downsample_4d_with_map_coordinates(mask, self.scale_factor, shrink_factor=self.shrink_factor)
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
        return mask
    
    def get_model_input(self, param, grid='hkw'):
        if grid == 'hklw':
            param = param.squeeze()[None, None, None, None, :].expand(self.hklw_grid.shape[:-1]+(-1,))
            coords = torch.cat([self.hklw_grid.to(param)[...,[0,1,3]], param], dim=-1)
            return coords, self.hklw_grid.to(param)[...,[2]]
        elif grid == 'hkw':
            param = param.squeeze()[None, None, None, :].expand(self.hkw_grid.shape[:-1]+(-1,))
            coords = torch.cat([self.hkw_grid.to(param), param], dim=-1)
            return coords
        
    
#     def __call__(self, coords):
#         if isinstance(coords, torch.Tensor):
#             coords = coords.detach().cpu().numpy()
#         elif isinstance(coords, list):
#             coords = np.array(list)
          
#         coords[...,0] = coords[...,0] % 360
#         return self.psi_mask_func(coords)
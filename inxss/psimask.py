import pickle, os
import torch
import numpy as np
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator, interp1d

try:
    import cupy as xp
    from cupyx.scipy.ndimage import map_coordinates
    using_cupy = True
except ImportError:
    import numpy as xp
    from scipy.ndimage import map_coordinates
    using_cupy = False

def downsample_4d_with_map_coordinates(src_grid, tar_grid, mask, order=0):
    
    coords = []
    for _src, _tar in zip(src_grid, tar_grid):
        _func_interp = interp1d(_src, np.arange(len(_src)), bounds_error=True)
        coords.append(_func_interp(_tar))
    
    mask = xp.asarray(mask.astype(xp.float32))
    
    # Create a grid of coordinates in the old mask
    coord_grid = xp.array(xp.meshgrid(*coords, indexing='ij'))

    # Map the coordinates to the original mask
    if using_cupy:
        downsampled_mask = map_coordinates(mask, coord_grid, order=order, mode='nearest').get()
    else:
        downsampled_mask = map_coordinates(mask, coord_grid, order=order, mode='nearest')
    downsampled_mask = downsampled_mask > 0.5
    return downsampled_mask

class PsiMask:
    """
    Written with help from OpenAI's ChatGPT (GPT-4.0).
    """
    def __init__(self, raw_mask_path, memmap_mask_path=None, grid_info=None, device='cpu', preload=True, build_from_scratch_if_no_memmap=False):
        """
        :param raw_mask_path: Folder containing the .pkl mask files
        :param device: Either 'cpu' or 'cuda' to determine where masks are stored
        :param preload: If True, all masks are preloaded into memory. If False, masks are loaded on the fly.
        """
        self.raw_mask_path = raw_mask_path
        self.device = device
        self.preload = preload

        with open(f'{self.raw_mask_path}/metadata', 'rb') as f:
            _metadata = pickle.load(f)
            
        if grid_info is None:
            self.need_scale = False
            self.grid_info = {key:[] for key in ['h_grid', 'k_grid', 'l_grid', 'w_grid']}
        else:
            self.need_scale = True
            self.grid_info = grid_info
            
        for _key in _metadata.keys():
            _grid = _metadata[_key]
            setattr(self, f'{_key}_src', torch.from_numpy(_grid).to(device))
            if grid_info is None:
                setattr(self, _key, torch.from_numpy(_grid).to(device))
            else:
                setattr(self, _key, torch.linspace(*self.grid_info[_key]).to(device))
        
        if 'psi_grid' not in self.grid_info.keys():
            self.psi_grid = torch.arange(360).to(self.h_grid)
        else:
            self.psi_grid = torch.linspace(*self.grid_info['psi_grid']).to(device)
        
        self.hklw_grid = torch.moveaxis(
            torch.stack(torch.meshgrid(self.h_grid, self.k_grid, self.l_grid, self.w_grid, indexing='ij'), dim=0), 0, -1)
        self.hkw_grid = torch.moveaxis(
            torch.stack(torch.meshgrid(self.h_grid, self.k_grid, self.w_grid, indexing='ij'), dim=0), 0, -1)
        
        if self.preload:
            self.masks = {}
            for i in tqdm(range(361)):  # assuming masks are stored for every degree from 0 to 360
                with open(f'{self.raw_mask_path}/{i}.pkl', 'rb') as f:
                    _mask = pickle.load(f)['coverage']
                    if self.scale_factor is not None:
                        _mask = downsample_4d_with_map_coordinates(
                            [self.h_grid_src, self.k_grid_src, self.l_grid_src, self.w_grid_src],
                            [self.h_grid, self.k_grid, self.l_grid, self.w_grid],
                            _mask
                        )
                    self.masks[i] = torch.tensor(_mask, dtype=torch.bool).to(device)
        else:
            self.masks = None

        self.memmap_mask_path = memmap_mask_path
        if self.memmap_mask_path is not None:
            try:
                self.get_memmap_mask_fname()
                self.mask_memmap = np.load(os.path.join(self.raw_mask_path, self.filename), mmap_mode='r')
            except FileNotFoundError:
                if build_from_scratch_if_no_memmap:
                    print("mask memmap not found, building from scratch (typically ~10 mins)...")
                    self.build_memmap_mask_from_scratch(self.memmap_mask_path)
                    self.mask_memmap = np.load(os.path.join(self.raw_mask_path, self.filename), mmap_mode='r')
                else:
                    print("mask memmap not found, you might want to build from scratch (typically ~10 mins)...")
            
        
    def scale_mask(self, mask):
        if not self.need_scale:
            return mask
        else:
            if using_cupy:
                mask = xp.asarray(mask)
                mask = downsample_4d_with_map_coordinates(
                            [self.h_grid_src, self.k_grid_src, self.l_grid_src, self.w_grid_src],
                            [self.h_grid, self.k_grid, self.l_grid, self.w_grid],
                            mask
                        )
                mask = xp.asnumpy(mask)
            else:
                mask = downsample_4d_with_map_coordinates(
                            [self.h_grid_src, self.k_grid_src, self.l_grid_src, self.w_grid_src],
                            [self.h_grid, self.k_grid, self.l_grid, self.w_grid],
                            mask
                        )
            return mask

    def load_mask(self, degree):
        """
        Get mask for a given degree.
        """
        lower_degree = int(np.floor(degree)) % 360
        upper_degree = int(np.ceil(degree)) % 360

        if not self.preload:
            with open(f'{self.raw_mask_path}/{lower_degree}.pkl', 'rb') as f:
                lower_mask = self.scale_mask(pickle.load(f)['coverage'])
                lower_mask = torch.tensor(lower_mask, dtype=torch.bool).to(self.device)
            if lower_degree == upper_degree:
                upper_mask = lower_mask
            else:
                with open(f'{self.raw_mask_path}/{upper_degree}.pkl', 'rb') as f:
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
    
    def get_memmap_mask_fname(self, grid_info=None):
        if grid_info is None:
            grid_info = self.grid_info
        filename_parts = []
        for key, values in grid_info.items():
            prefix = key.split('_')[0]  # Take the first part of the key (like "h" from "h_grid")
            
            # Format values: if it's a float, format to 1 decimal point; else, convert to string
            values_str = "_".join([f"{v:.1f}" if isinstance(v, float) else str(v) for v in values])
            
            filename_parts.append(f"{prefix}_{values_str}")

        self.filename = "mask_" + "_".join(filename_parts) + ".npy"
        print("obtained memmap mask name as:", self.filename)

    def build_memmap_mask_from_scratch(self, save_path):
        mask_complete = np.zeros(
            [360,] + [self.grid_info[key][-1] for key in ['h_grid', 'k_grid', 'l_grid', 'w_grid']], dtype=bool)
        for i in tqdm(self.psi_grid):
            mask_complete[i] = self.load_mask(i)
        dst_fname = os.path.join(save_path, self.filename)
        np.save(dst_fname, mask_complete)
        print("saved memmap mask to:", dst_fname)

    
#     def __call__(self, coords):
#         if isinstance(coords, torch.Tensor):
#             coords = coords.detach().cpu().numpy()
#         elif isinstance(coords, list):
#             coords = np.array(list)
          
#         coords[...,0] = coords[...,0] % 360
#         return self.psi_mask_func(coords)
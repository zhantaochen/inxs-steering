import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .utils_spectrum import calc_Sqw_from_Syy_Szz

class SimulatedExperiment:
    
    def __init__(self, w_grid, q_grid, Syy_grid, Szz_grid):
        h_grid = np.sort(np.unique(q_grid[0].numpy()))
        k_grid = np.sort(np.unique(q_grid[1].numpy()))
        self.Syy_func = RegularGridInterpolator(
            [w_grid.numpy(), h_grid, k_grid],
            Syy_grid.numpy().reshape((len(w_grid), len(h_grid), len(k_grid)))
        )
        self.Szz_func = RegularGridInterpolator(
            [w_grid.numpy(), h_grid, k_grid],
            Szz_grid.numpy().reshape((len(w_grid), len(h_grid), len(k_grid)))
        )
        self.w_grid = w_grid
        self.h_grid = torch.from_numpy(h_grid)
        self.k_grid = torch.from_numpy(k_grid)
        
        self.full_grid = torch.moveaxis(torch.stack(torch.meshgrid(self.w_grid, self.h_grid, self.k_grid, indexing='ij'), dim=0), 0, -1)
        
    def get_S(self, wQ):
        S_out = calc_Sqw_from_Syy_Szz(wQ, self.Syy_func, self.Szz_func)
        return S_out
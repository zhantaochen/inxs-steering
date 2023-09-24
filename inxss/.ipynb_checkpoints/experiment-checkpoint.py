import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .utils_spectrum import calc_Sqw_from_Syy_Szz

class SimulatedExperiment:
    
    def __init__(self, q_grid, w_grid, Syy_grid, Szz_grid, neutron_flux=1e6):
        h_grid = np.sort(np.unique(q_grid[0].numpy()))
        k_grid = np.sort(np.unique(q_grid[1].numpy()))
        self.neutron_flux = neutron_flux
        self.Syy_func = RegularGridInterpolator(
            [h_grid, k_grid, w_grid.numpy()],
            self.neutron_flux * Syy_grid.reshape((len(w_grid), len(h_grid), len(k_grid))).permute(1,2,0).numpy(),
            bounds_error=False, fill_value=0, method='linear'
        )
        self.Szz_func = RegularGridInterpolator(
            [h_grid, k_grid, w_grid.numpy()],
            self.neutron_flux * Szz_grid.reshape((len(w_grid), len(h_grid), len(k_grid))).permute(1,2,0).numpy(),
            bounds_error=False, fill_value=0, method='linear'
        )
        self.h_grid = torch.from_numpy(h_grid)
        self.k_grid = torch.from_numpy(k_grid)
        self.w_grid = w_grid
        
        
    def get_measurements_on_coords(self, coords, poisson=True):
        S_out = calc_Sqw_from_Syy_Szz(coords, self.Syy_func, self.Szz_func)
        if poisson:
            S_out = torch.poisson(S_out)
        return S_out
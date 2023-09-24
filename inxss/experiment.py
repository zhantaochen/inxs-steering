import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .utils_spectrum import calc_Sqw_from_Syy_Szz

class SimulatedExperiment:
    
    def __init__(self, q_grid, w_grid, Syy_grid, Szz_grid, neutron_flux=1e3):
        h_grid_src = np.sort(np.unique(q_grid[0].numpy()))
        k_grid_src = np.sort(np.unique(q_grid[1].numpy()))
        self.neutron_flux = neutron_flux
        self.Syy_func = RegularGridInterpolator(
            [h_grid_src, k_grid_src, w_grid.numpy()],
            self.neutron_flux * Syy_grid.reshape((len(w_grid), len(h_grid_src), len(k_grid_src))).permute(1,2,0).numpy(),
            bounds_error=False, fill_value=0, method='linear'
        )
        self.Szz_func = RegularGridInterpolator(
            [h_grid_src, k_grid_src, w_grid.numpy()],
            self.neutron_flux * Szz_grid.reshape((len(w_grid), len(h_grid_src), len(k_grid_src))).permute(1,2,0).numpy(),
            bounds_error=False, fill_value=0, method='linear'
        )
        self.h_grid_src = torch.from_numpy(h_grid_src)
        self.k_grid_src = torch.from_numpy(k_grid_src)
        self.w_grid_src = w_grid
    
    def prepare_experiment(self, coords):
        self.Sqw = self.get_measurements_on_coords(coords, poisson=False)
    
    def get_measurements_by_mask(self, mask, poisson=True):
        S_out = self.Sqw[mask]
        if poisson:
            S_out = torch.poisson(S_out)
        return S_out
        
    def get_measurements_on_coords(self, coords, poisson=True):
        S_out = calc_Sqw_from_Syy_Szz(coords, self.Syy_func, self.Szz_func)
        if poisson:
            S_out = torch.poisson(S_out)
        return S_out
import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .utils_spectrum import calc_Sqw_from_Syy_Szz
from .utils_grid import convert_to_numpy, convert_to_torch

class SimulatedExperiment:
    
    def __init__(self, q_grid, w_grid, Syy_grid, Szz_grid, neutron_flux=1e3):
        """
        q_grid of shape (3, num_q_grid)
        w_grid of shape (num_w_grid)
        """
        q_grid = convert_to_numpy(q_grid)
        w_grid = convert_to_numpy(w_grid)
        Syy_grid = convert_to_numpy(Syy_grid)
        Szz_grid = convert_to_numpy(Szz_grid)
        
        h_grid_src = np.sort(np.unique(q_grid[0]))
        k_grid_src = np.sort(np.unique(q_grid[1]))
        self.neutron_flux = neutron_flux
        self.Syy_func = RegularGridInterpolator(
            [h_grid_src, k_grid_src, w_grid],
            self.neutron_flux * np.transpose(Syy_grid.reshape((len(w_grid), len(h_grid_src), len(k_grid_src))), (1,2,0)),
            bounds_error=False, fill_value=0, method='linear'
        )
        self.Szz_func = RegularGridInterpolator(
            [h_grid_src, k_grid_src, w_grid],
            self.neutron_flux * np.transpose(Szz_grid.reshape((len(w_grid), len(h_grid_src), len(k_grid_src))), (1,2,0)),
            bounds_error=False, fill_value=0, method='linear'
        )
        self.h_grid_src = torch.from_numpy(h_grid_src)
        self.k_grid_src = torch.from_numpy(k_grid_src)
        self.w_grid_src = w_grid
    
    def prepare_experiment(self, coords):
        self.Sqw = self.get_measurements_on_coords(coords, poisson=False)
        self.Sqw = self.Sqw.clamp_min(0.0)
    
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

class NeutronExperiment:

    def __init__(self, q_grid, w_grid, S_grid, S_scale_factor=1.):
        """
        q_grid: tuple of (h_grid, k_grid, l_grid), each of shape (num_qi) for i = h,k,l
        w_grid: array of shape (num_w,)
        S_grid: array of shape (num_h, num_k, num_l, num_w)
        """
        self.h_grid = convert_to_torch(q_grid[0])
        self.k_grid = convert_to_torch(q_grid[1])
        self.l_grid = convert_to_torch(q_grid[2])
        self.w_grid = convert_to_torch(w_grid)

        self.S_scale_factor = S_scale_factor

        self.S_func = RegularGridInterpolator(
            [convert_to_numpy(_) for _ in [q_grid[0], q_grid[1], q_grid[2], w_grid]],
            self.S_scale_factor * convert_to_numpy(S_grid),
            bounds_error=False, fill_value=0, method='linear'
        )
    
    def prepare_experiment(self, coords):
        self.Sqw = torch.from_numpy(self.get_measurements_on_coords(coords))
        self.Sqw = self.Sqw.clamp_min(0.0)
    
    def get_measurements_by_mask(self, mask):
        S_out = self.Sqw[mask]
        return S_out
    
    def get_measurements_on_coords(self, coords):
        S_out = self.S_func(coords)
        return S_out
    
class Background:
    
    def __init__(self, q_grid, w_grid, bkg_grid):
        """
        q_grid: tuple of (h_grid, k_grid, l_grid), each of shape (num_qi) for i = h,k,l
        w_grid: array of shape (num_w,)
        bkg_grid: array of shape (num_h, num_k, num_l, num_w)
        """
        self.h_grid = convert_to_torch(q_grid[0])
        self.k_grid = convert_to_torch(q_grid[1])
        self.l_grid = convert_to_torch(q_grid[2])
        self.w_grid = convert_to_torch(w_grid)

        self.bkg_func = RegularGridInterpolator(
            [convert_to_numpy(_) for _ in [q_grid[0], q_grid[1], q_grid[2], w_grid]],
            convert_to_numpy(bkg_grid),
            bounds_error=False, fill_value=0, method='linear'
        )
    
    def prepare_experiment(self, coords):
        self.bkg_qw = torch.from_numpy(self.get_background_on_coords(coords))
        self.bkg_qw = self.bkg_qw.clamp_min(0.0)
    
    def get_background_by_mask(self, mask):
        S_out = self.bkg_qw[mask]
        return S_out
    
    def get_background_on_coords(self, coords):
        S_out = self.bkg_func(coords)
        return S_out
    
import numpy as np
import torch


from tqdm import tqdm
from inxss import NeutronExperiment, Particle, PsiMask, OnlineVariance
from inxss.experiment import Background

class NeutronExperimentSteerer:
    def __init__(
        self, 
        model, 
        particle_filter_config,
        mask_config,
        experiment_config,
        background_config=None,
        utility_sf_sigma=90.0,
        tqdm_pbar=True,
        lkhd_dict=None,
        dtype=torch.float32,
        device='cpu'
        ):

        torch.set_default_dtype(dtype)
        
        if lkhd_dict is None:
            self.lkhd_dict = {
                'type': 'gaussian',
                'scale': True,
                'std': 0.1,
                'sample_ratio': 0.25
            }
        else:
            self.lkhd_dict = lkhd_dict
        
        self.psi_mask = PsiMask(**mask_config)
        self.particle_filter = Particle(**particle_filter_config).to(device).to(dtype)
        self.experiment = NeutronExperiment(**experiment_config)
        
        self.particle_filter.grad_off()
        self.experiment.prepare_experiment(self.psi_mask.hklw_grid)
        
        self.background = Background(**background_config) if background_config is not None else None
        if self.background is not None:
            self.background.prepare_experiment(self.psi_mask.hklw_grid)
        
        self.utility_sf_sigma = utility_sf_sigma
        
        self.dtype = dtype
        self.device = device
        self.model = model.to(self.device)
        
        self.tqdm_pbar = tqdm_pbar
        
        self.get_mask_on_full_psi_grid()
        
        self.utility_history = []
        self.lkhd_history = []
        self.measured_angles_history = []
        self.sig_bkg_factors_history = []
    
    def reset(self):
        self.particle_filter.reset()
        self.utility_history = []
        self.lkhd_history = []
        self.measured_angles_history = []
        self.sig_bkg_factors_history = []
    
    def _progress_bar(self, iterable, **kwargs):
        if self.tqdm_pbar:
            return tqdm(iterable, **kwargs)
        return iterable
        
    def get_mask_on_full_psi_grid(self,):
        mask = []
        for _angle in self._progress_bar(self.psi_mask.psi_grid, desc="Computing mask on full psi grid"):
            _mask = self.psi_mask.load_memmap_mask(_angle)
            mask.append(_mask.sum(dim=-2))
        self.mask_on_full_psi_grid = torch.stack(mask)
        self.attainable_mask_on_full_psi_grid = (self.mask_on_full_psi_grid.sum(dim=0) > 0).view(-1)
    
    def compute_utility_scaling_factor(self, current_angle):
        angles = self.psi_mask.psi_grid - current_angle
        angles[angles > 180] -= 360
        angles[angles <-180] += 360
        sf = torch.exp(-0.5 * angles.pow(2) / self.utility_sf_sigma**2)
        return sf
    
    def _compute_utility(self,):
        # std = self.compute_prediction_std_over_parameters()
        std = self.compute_prediction_std_over_sampled_parameters()
        utility = torch.einsum("ijkl, jkl -> i", self.mask_on_full_psi_grid.to(std), std)
        return utility
    
    def compute_utility(self,):
        utility = self._compute_utility()
        # utility_sf = torch.ones_like(utility)
        if len(self.measured_angles_history) > 0:
            utility_sf = self.compute_utility_scaling_factor(self.measured_angles_history[-1])
        else:
            utility_sf = torch.ones_like(utility)
        return utility * utility_sf.to(utility)
        
    def compute_prediction_std_over_all_parameters(self,):
        
        x_input = self.psi_mask.get_model_input(torch.zeros(2), grid='hkw').view(-1, 5)
        x_input = x_input.unsqueeze(0).repeat(self.particle_filter.num_particles, 1, 1)
        for i_param, param in enumerate(self.particle_filter.positions.T):
            x_input[i_param,...,-2:] = param
        # x_input shape of [num_params, num_grid, 5]
        x_input_chunks = torch.chunk(x_input, x_input.shape[0] // 5, dim=0)
        output_list = []
        with torch.no_grad():
            for _input in self._progress_bar(x_input_chunks, total=len(x_input_chunks), desc="Computing pred std over params"):
                _output = torch.zeros(_input.shape[:-1]).to(_input)
                _output[:,self.attainable_mask_on_full_psi_grid] = self.model(_input[:,self.attainable_mask_on_full_psi_grid].to(self.device)).detach().cpu()
                output_list.append(_output)
        output_list = torch.cat(output_list, dim=0)
        weighted_mean = torch.sum((output_list * self.particle_filter.weights[:,None]) / self.particle_filter.weights.sum(), dim=0)
        weighted_std = torch.sum(((output_list - weighted_mean[None,:]).pow(2) * self.particle_filter.weights[:,None]) / self.particle_filter.weights.sum(), dim=0).sqrt()
        return weighted_std.view(self.psi_mask.hkw_grid.shape[:-1])
    
    def compute_prediction_std_over_sampled_parameters(self,):
        
        sampled_particles, sampled_particle_weights = self.particle_filter.random_subset(proportion=0.1)
        sampled_particle_weights = torch.ones_like(sampled_particle_weights) / len(sampled_particle_weights)
        
        x_input = self.psi_mask.get_model_input(torch.zeros(2), grid='hkw').view(-1, 5)
        x_input = x_input.unsqueeze(0).repeat(sampled_particles.shape[1], 1, 1)
        for i_param, param in enumerate(sampled_particles.T):
            x_input[i_param,...,-2:] = param
        # x_input shape of [num_params, num_grid, 5]
        x_input_chunks = torch.chunk(x_input, x_input.shape[0] // 5, dim=0)
        
        output_list = []
        with torch.no_grad():
            for _input in self._progress_bar(x_input_chunks, total=len(x_input_chunks), desc="Computing pred std over params"):
                _output = torch.zeros(_input.shape[:-1]).to(_input)
                _output[:,self.attainable_mask_on_full_psi_grid] = self.model(_input[:,self.attainable_mask_on_full_psi_grid].to(self.device)).detach().cpu()
                output_list.append(_output)
        output_list = torch.cat(output_list, dim=0).to(sampled_particle_weights)
        weighted_mean = torch.sum((output_list * sampled_particle_weights[:,None]) / sampled_particle_weights.sum(), dim=0)
        weighted_std = torch.sum(((output_list - weighted_mean[None,:]).pow(2) * sampled_particle_weights[:,None]) / sampled_particle_weights.sum(), dim=0).sqrt()
        return weighted_std.view(self.psi_mask.hkw_grid.shape[:-1])
    
    def update_parameter_distribution(self,):
        pass
        
    def get_optimal_angle(self,):
        utility = self.compute_utility()
        optimal_psi = self.psi_mask.psi_grid[utility.argmax()]
        
        self.utility_history.append(utility.cpu().numpy().squeeze())
        self.measured_angles_history.append(optimal_psi.cpu().numpy())
        return optimal_psi
    
    def get_good_angle(self,):
        utility = self.compute_utility()
        p = torch.nn.functional.softmax(utility / (self.experiment.S_scale_factor / 5), dim=0).detach().cpu().numpy()
        idx = np.random.choice(np.arange(utility.shape[0]), 1, p=p)
        
        good_psi = self.psi_mask.psi_grid[idx]
        
        self.utility_history.append(utility.cpu().numpy().squeeze())
        self.measured_angles_history.append(good_psi.cpu().numpy())
        return good_psi
    
    def get_unique_optimal_angle(self,):
        utility = self.compute_utility()
        # Sort the indices of the utilities in descending order
        sorted_indices = utility.argsort(descending=True)

        # Iterate over the sorted utility indices to get the highest utility angle that hasn't been measured
        for index in sorted_indices:
            psi_candidate = self.psi_mask.psi_grid[index]
            if psi_candidate.cpu().numpy() not in self.measured_angles_history:
                
                self.utility_history.append(utility.cpu().numpy().squeeze())
                self.measured_angles_history.append(psi_candidate.cpu().numpy())  # Store the newly measured angle
                return psi_candidate

        # If all angles have been measured, return None or handle this case appropriately
        # return None
        print("All angles have been measured. Returning the best angle in the list.")
        return self.get_good_angle()
    
    def solve_background_signal_factors(self, measurement, background, signal):
        A = torch.cat([background.squeeze()[:,None], signal.squeeze()[:,None]], dim=-1)
        x = torch.einsum('ij, kj, k -> i', torch.linalg.inv(torch.einsum('ji, jk -> ik', A, A)), A, measurement)
        return x.clamp(0.0)
        
    def compute_likelihood(self, next_measurement, next_mask, likelihood_mask):
        x_input, l_input = self.psi_mask.get_model_input(torch.zeros(2), grid='hklw')
        x_input = x_input[next_mask][likelihood_mask]
        l_input = l_input[next_mask][likelihood_mask]
        
        exist_signal_in_exp = (self.experiment.Sqw > 1e-10)[next_mask][likelihood_mask].unsqueeze(0).repeat(self.particle_filter.num_particles, 1)

        x_input = x_input.unsqueeze(0).repeat(self.particle_filter.num_particles, 1, 1)
        l_input = l_input.unsqueeze(0).repeat(self.particle_filter.num_particles, 1, 1)

        with torch.no_grad():
            for i_param, param in enumerate(self.particle_filter.positions.T):
                x_input[i_param,...,-2:] = param
                
        with torch.no_grad():
            predictions = self.model(x_input.to(self.device), l_input.to(self.device)).detach().cpu()
            predictions = exist_signal_in_exp * predictions
        # pred_measurement = self.experiment.S_scale_afctor * predictions.clip(0.)
        
        if self.background is not None:
            _background = self.background.get_background_by_mask(next_mask)[likelihood_mask].clone()
            factors = self.solve_background_signal_factors(next_measurement, _background, predictions.mean(dim=0))
            self.sig_bkg_factors_history.append(factors)
            
            pred_measurement = _background * factors[0] + predictions * factors[1]
        else:
            pred_measurement = predictions / predictions.mean() * next_measurement.mean()
            # pred_measurement = 1000. * predictions.clip(0.)
        
        # pred_measurement = torch.log(1 + pred_measurement)
        # next_measurement = torch.log(1 + next_measurement)
        
        if self.lkhd_dict['type'] == 'gaussian':
            likelihood = torch.exp(-0.5 * ((pred_measurement.clamp_min(0.) - next_measurement[None]) / self.lkhd_dict['std']).pow(2)).mean(dim=-1)
        elif self.lkhd_dict['type'] == 'poisson':
            nll = torch.nn.functional.poisson_nll_loss(pred_measurement.clamp_min(0), next_measurement[None], 
                                                       log_input=False, full=True, reduction='none')
            likelihood = torch.exp(-nll).mean(dim=-1)
        
        torch.nan_to_num(likelihood, nan=0., posinf=0., neginf=0., out=likelihood)
        return likelihood
    
    @torch.no_grad()
    def step_steer(self, mode='unique_optimal'):
        
        if mode == 'unique_optimal':
            next_angle = self.get_unique_optimal_angle()
        elif mode == 'optimal':
            next_angle = self.get_optimal_angle()
        elif mode == 'good':
            next_angle = self.get_good_angle()
        else:
            raise ValueError("mode must be one of 'unique_optimal', 'optimal', 'good'")
            
        if self.tqdm_pbar:
            print("next angle:", next_angle)
        next_mask = self.psi_mask.load_memmap_mask(next_angle).bool()
        
        next_measurement = self.experiment.get_measurements_by_mask(next_mask)
        
        likelihood_mask = torch.zeros_like(next_measurement, dtype=torch.bool)
        likelihood_mask[np.random.choice(np.arange(likelihood_mask.shape[0]), min(150, int(likelihood_mask.shape[0] * self.lkhd_dict['sample_ratio'])), replace=False)] = True
        
        next_measurement = next_measurement[likelihood_mask]
        
        likelihood = self.compute_likelihood(next_measurement, next_mask, likelihood_mask)
        
        if self.lkhd_dict['scale']:
            # this is very dirty, need to look into later
            if likelihood.max() == likelihood.min():
                likelihood_normed = torch.rand_like(likelihood)
            else:
                likelihood_normed = (likelihood - likelihood.min()) / (likelihood.max() - likelihood.min())
        else:
            likelihood_normed = likelihood
        self.lkhd_history.append(likelihood_normed.detach().cpu().numpy().squeeze())
        
        self.particle_filter.bayesian_update(likelihood_normed)
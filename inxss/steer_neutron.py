import numpy as np
import torch

# def is_notebook():
#     try:
#         from IPython import get_ipython
#         if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
#             return False
#     except ImportError:
#         return False
#     return True

# if is_notebook():
#     from tqdm.notebook import tqdm
# else:
#     from tqdm import tqdm

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
        likelihood_mask_threshold=0.1,
        tqdm_pbar=True,
        dtype=torch.float32,
        device='cpu'
        ):

        torch.set_default_dtype(dtype)
        
        self.psi_mask = PsiMask(**mask_config)
        self.particle_filter = Particle(**particle_filter_config).to(device).to(dtype)
        self.experiment = NeutronExperiment(**experiment_config)
        self.variance_tracker = OnlineVariance((np.prod(self.psi_mask.hkw_grid.shape[:-1]),1))
        
        self.particle_filter.grad_off()
        self.experiment.prepare_experiment(self.psi_mask.hklw_grid)
        
        self.background = Background(**background_config) if background_config is not None else None
        if self.background is not None:
            self.background.prepare_experiment(self.psi_mask.hklw_grid)
        
        self.dtype = dtype
        self.device = device
        self.model = model.to(self.device)
        
        self.tqdm_pbar = tqdm_pbar
        self.likelihood_mask_threshold = likelihood_mask_threshold
        
        self.get_mask_on_full_psi_grid()
        
        self.measured_angles = []
    
    def reset(self):
        self.particle_filter.reset()
        self.measured_angles = []
    
    def _progress_bar(self, iterable, **kwargs):
        if self.tqdm_pbar:
            return tqdm(iterable, **kwargs)
        return iterable
    
    # def compute_utility(self,):
    #     std = self.compute_prediction_std_over_parameters()
    #     utility = []
    #     for _angle in self._progress_bar(self.psi_mask.psi_grid):
    #         _mask = self.psi_mask.load_memmap_mask(_angle)
    #         utility.append((_mask.sum(dim=2) * std).mean())
    #     utility = torch.stack(utility)
    #     return utility
    
    def get_mask_on_full_psi_grid(self,):
        mask = []
        for _angle in self._progress_bar(self.psi_mask.psi_grid, desc="Computing mask on full psi grid"):
            _mask = self.psi_mask.load_memmap_mask(_angle)
            mask.append(_mask.sum(dim=-2))
        self.mask_on_full_psi_grid = torch.stack(mask)
        self.attainable_mask_on_full_psi_grid = (self.mask_on_full_psi_grid.sum(dim=0) > 0).view(-1)
    
    def compute_utility(self,):
        # std = self.compute_prediction_std_over_parameters()
        std = self.compute_prediction_std_over_sampled_parameters()
        utility = torch.einsum("ijkl, jkl -> i", self.mask_on_full_psi_grid.to(std), std)        
        return utility
        
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
                # output_list.append(self.model(_input.to(self.device)).detach().cpu())
        output_list = torch.cat(output_list, dim=0)
        weighted_mean = torch.sum((output_list * self.particle_filter.weights[:,None]) / self.particle_filter.weights.sum(), dim=0)
        weighted_std = torch.sum(((output_list - weighted_mean[None,:]).pow(2) * self.particle_filter.weights[:,None]) / self.particle_filter.weights.sum(), dim=0).sqrt()
        return weighted_std.view(self.psi_mask.hkw_grid.shape[:-1])
        
    def compute_prediction_std_over_sampled_parameters(self,):
        
        sampled_particles, sampled_particle_weights = self.particle_filter.random_subset(proportion=0.1)
        sampled_particle_weights = torch.ones_like(sampled_particle_weights) / len(sampled_particle_weights)
        
        x_input = self.psi_mask.get_model_input(torch.zeros(2), grid='hkw').view(-1, 5)
        x_input = x_input.unsqueeze(0).repeat(sampled_particles.shape[1], 1, 1)
        # print(x_input.shape)
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
                # output_list.append(self.model(_input.to(self.device)).detach().cpu())
        output_list = torch.cat(output_list, dim=0)
        weighted_mean = torch.sum((output_list * sampled_particle_weights[:,None]) / sampled_particle_weights.sum(), dim=0)
        weighted_std = torch.sum(((output_list - weighted_mean[None,:]).pow(2) * sampled_particle_weights[:,None]) / sampled_particle_weights.sum(), dim=0).sqrt()
        return weighted_std.view(self.psi_mask.hkw_grid.shape[:-1])
    
    def update_parameter_distribution(self,):
        pass
        
    def get_optimal_angle(self,):
        utility = self.compute_utility()
        return self.psi_mask.psi_grid[utility.argmax()]
    
    def get_good_angle(self,):
        utility = self.compute_utility()
        p = torch.nn.functional.softmax(utility / (self.experiment.S_scale_factor / 5), dim=0).detach().cpu().numpy()
        idx = np.random.choice(np.arange(utility.shape[0]), 1, p=p)
        return self.psi_mask.psi_grid[idx]
    
    def get_unique_optimal_angle(self,):
        utility = self.compute_utility()
        # Sort the indices of the utilities in descending order
        sorted_indices = utility.argsort(descending=True)

        # Iterate over the sorted utility indices to get the highest utility angle that hasn't been measured
        for index in sorted_indices:
            psi_candidate = self.psi_mask.psi_grid[index]
            if psi_candidate not in self.measured_angles:
                self.measured_angles.append(psi_candidate)  # Store the newly measured angle
                return psi_candidate

        # If all angles have been measured, return None or handle this case appropriately
        # return None
        return self.get_good_angle()
        
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
        # pred_measurement = self.experiment.S_scale_factor * predictions.clip(0.)
        
        if self.background is not None:
            # bkg = self.background.get_background_by_mask(next_mask)[likelihood_mask]
            _mask_inner =  x_input[...,:2].norm(dim=-1)  <= self.background.scale_separator['inner_mid']
            _mask_mid   = (x_input[...,:2].norm(dim=-1)  >  self.background.scale_separator['inner_mid']) * (x_input[...,:2].norm(dim=-1) <= self.background.scale_separator['mid_outer'])
            _mask_outer =  x_input[...,:2].norm(dim=-1)  >  self.background.scale_separator['mid_outer']
            
            _mask_inner = _mask_inner * exist_signal_in_exp
            _mask_mid   = _mask_mid   * exist_signal_in_exp
            _mask_outer = _mask_outer * exist_signal_in_exp
            
            pred_measurement = self.background.get_background_by_mask(next_mask)[likelihood_mask].clone().unsqueeze(0).repeat(self.particle_filter.num_particles, 1)
            # print(x_input.shape, _mask_inner.shape, predictions.shape, pred_measurement.shape)
            pred_measurement[_mask_inner] += predictions[_mask_inner] * self.background.scale['inner']['scale_factor']
            pred_measurement[_mask_mid]   += predictions[_mask_mid]   * self.background.scale['mid']['scale_factor']
            pred_measurement[_mask_outer] += predictions[_mask_outer] * self.background.scale['outer']['scale_factor']
        else:
            pred_measurement = pred_measurement / pred_measurement.mean() * next_measurement.mean()
        
        
        # poisson_negative_log_likelihood = torch.nn.functional.poisson_nll_loss(
        #     pred_measurement, next_measurement[None], 
        #     log_input=False, reduction="none").mean(dim=-1)
        # # TODO: check if this is correct
        # likelihood = torch.exp(-poisson_negative_log_likelihood / (self.experiment.S_scale_factor / 10))
        
        pred_measurement = torch.log(1 + pred_measurement)
        next_measurement = torch.log(1 + next_measurement)
        
        likelihood = torch.exp(-0.5 * ((pred_measurement - next_measurement[None]) / next_measurement[None].sqrt()).pow(2)).mean(dim=-1)
        # print(likelihood.shape)
        torch.nan_to_num(likelihood, nan=0., posinf=0., neginf=0., out=likelihood)
        return likelihood
    
    @torch.no_grad()
    def step_steer(self, ):
        
        # next_angle = self.get_good_angle()
        # next_angle = self.get_optimal_angle()
        next_angle = self.get_unique_optimal_angle()
        if self.tqdm_pbar:
            print("next angle:", next_angle)
        next_mask = self.psi_mask.load_memmap_mask(next_angle)
        
        next_measurement = self.experiment.get_measurements_by_mask(next_mask)
        # likelihood_mask = next_measurement > next_measurement.max() * self.likelihood_mask_threshold
        
        likelihood_mask = torch.zeros_like(next_measurement, dtype=torch.bool)
        likelihood_mask[np.random.choice(np.arange(likelihood_mask.shape[0]), min(150, int(likelihood_mask.shape[0] * self.likelihood_mask_threshold)), replace=False)] = True
        
        next_measurement = next_measurement[likelihood_mask]
        
        likelihood = self.compute_likelihood(next_measurement, next_mask, likelihood_mask)
        # this is very dirty, need to look into later
        likelihood_normed = (likelihood - likelihood.min()) / (likelihood.max() - likelihood.min())
        # likelihood_normed = likelihood
        
        self.particle_filter.bayesian_update(likelihood_normed)
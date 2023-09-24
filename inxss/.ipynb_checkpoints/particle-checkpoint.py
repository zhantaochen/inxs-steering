'''
Adapted from optbayesexpt/particlepdf.py
https://github.com/usnistgov/optbayesexpt/blob/master/optbayesexpt/particlepdf.py
'''

import numpy as np

import torch
import torch.nn as nn

import warnings

class Particle(nn.Module):
    
    def __init__(self, num_particles, dim_particles, prior_configs=None, resample_configs=None):
        super().__init__()
        self.num_particles = num_particles
        self.dim_particles = dim_particles
        self.rng = np.random.default_rng()
        
        if prior_configs is not None:
            self.prior_configs = prior_configs
        else:
            self.prior_configs = {
                'types': ['normal'] * self.dim_particles,
                'args': [{'loc': 0.0, 'scale': 1.0, 'size': (1, self.num_particles)}] * self.dim_particles
            }
        self._parse_prior_configs(self.prior_configs)

        if resample_configs is not None:
            self.resample_configs = resample_configs
        else:
            self.resample_configs = {
                'auto_resample': True,
                'threshold': 0.5,
                'scale': True,
                'a_param': 0.98
            }

    def _parse_prior_configs(self, prior_configs):
        
        if 'types' in prior_configs.keys() and 'args' in prior_configs.keys():
            positions = np.zeros((self.dim_particles, self.num_particles))
            self.register_parameter('weights', nn.Parameter(torch.ones(self.num_particles) / self.num_particles))
            for _dim in range(self.dim_particles):
                if 'size' not in prior_configs['args'][_dim].keys():
                    prior_configs['args'][_dim]['size'] = (1, self.num_particles)
                positions[_dim, :] = eval(f"self.rng.{prior_configs['types'][_dim]}(**prior_configs['args'][{_dim}])")
            self.register_parameter('positions', nn.Parameter(torch.from_numpy(positions)))
        elif 'positions' in prior_configs.keys():
            positions = prior_configs['positions'] if isinstance(prior_configs['positions'], torch.Tensor) else torch.from_numpy(prior_configs['positions'])
            self.register_parameter('positions', nn.Parameter(positions))
            if 'weights' in prior_configs.keys():
                weights = prior_configs['weights'] if isinstance(prior_configs['weights'], torch.Tensor) else torch.from_numpy(prior_configs['weights'])
                self.register_parameter('weights', nn.Parameter(weights))
            else:
                self.register_parameter('weights', nn.Parameter(torch.ones(self.num_particles) / self.num_particles))
        else:
            raise ValueError("Invalid prior_configs. Must contain 'positions' and optionally 'weights' or 'types' and 'args'.")
    
    def grad_off(self,):
        for p in self.parameters():
            p.requires_grad = False
    
    def grad_on(self,):
        for p in self.parameters():
            p.requires_grad = True
        
    def set_particles(self, positions=None, weights=None):
        if positions is None:
            self._parse_prior_configs(self.prior_configs)
        else:
            self.positions.data = positions.to(self.positions) if isinstance(positions, torch.Tensor) else torch.from_numpy(positions).to(self.positions)
            if weights is not None:
                self.weights.data = weights.to(self.weights) if isinstance(weights, torch.Tensor) else torch.from_numpy(weights).to(self.weights)
            else:
                self.weights.data = torch.ones(self.num_particles).to(self.weights) / self.num_particles
                
    def _normalize_weights(self, ):
        """
        Normalizes the weights of the particles to sum up to 1.
        """
        if not torch.allclose(torch.sum(self.weights), torch.tensor(1.0)):
            total_weight = torch.sum(self.weights)
            self.weights.data = self.weights / total_weight

    def mean(self, ):
        self._normalize_weights()
        return torch.sum(self.positions * self.weights.unsqueeze(0), dim=1)
    
    def cov(self, ):
        self._normalize_weights()
        cov = torch.cov(self.positions, aweights=self.weights)
        if self.dim_particles == 1:
            return cov.view(1, 1)
        else:
            return cov
    
    def std(self, return_mean=False):
        mean = self.mean()
        weights_sq_sum = torch.sum(self.weights**2)
        normalization_factor = 1 / (1 - weights_sq_sum + torch.finfo(self.weights.dtype).eps)
        var = normalization_factor * torch.sum(self.weights * (self.positions - mean.unsqueeze(1))**2, dim=1)
        
        if return_mean:
            return mean, torch.sqrt(var)
        else:
            return torch.sqrt(var)
        
    def bayesian_update(self, likelihood):
        self.weights = self._normalized_product(self.weights, likelihood)
        if self.resample_configs['auto_resample']:
            self._resample_test()

    def _resample_test(self, ):
        weights_sq = torch.nan_to_num(self.weights.pow(2))
        n_eff = 1 / weights_sq.sum()
        if n_eff < 0.1 * self.num_particles:
            warnings.warn("\nParticle filter rejected > 90 % of particles. "
                          f"N_eff = {n_eff:.2f}. "
                          "Particle impoverishment may lead to errors.",
                          RuntimeWarning)
            self.resample()
        elif n_eff < self.resample_configs['threshold'] * self.num_particles:
            self.resample()

    def resample(self, ):
        old_pos = self.random_draw(self.num_particles)
        origin = torch.zeros(self.dim_particles).to(self.positions)
        
        old_mu = self.mean().reshape(-1, 1)
        old_cov = self.cov()
        
        new_cov = (1 - self.resample_configs['a_param'] ** 2) * old_cov
        nudged_pos = old_pos + torch.from_numpy(
            self.rng.multivariate_normal(
                origin.detach().cpu().numpy(), new_cov.detach().cpu().numpy(), size=self.num_particles
                ).T
            ).to(self.positions)
        
        if self.resample_configs['scale']:
            self.positions.data = nudged_pos * self.resample_configs['a_param'] + old_mu * (1 - self.resample_configs['a_param'])
        else:
            self.positions.data = nudged_pos
        
        self.weights.data = torch.ones(self.num_particles).to(self.weights) / self.num_particles
        
    def random_draw(self, num_draws=1):
        self._normalize_weights()
        indices = self.rng.choice(self.num_particles, size=num_draws, p=self.weights.detach().cpu().numpy(), replace=True)
        return self.positions[:, indices]

    def _normalized_product(self, p, q):
        o = torch.nan_to_num(p * q)
        if torch.allclose(o.sum(), torch.tensor(0.0)):
            o = o / (o.sum() + torch.finfo(o.dtype).eps)
        else:
            o = o / o.sum()
        return o
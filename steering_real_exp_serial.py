from inxss.utils_spectrum import calc_Sqw_from_Syy_Szz
from inxss.experiment import SimulatedExperiment

import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from inxss import SpectrumDataset, SpecNeuralRepr, Particle, PsiMask, OnlineVariance, linspace_2D_equidistant
from inxss.utils_visualization import arc_arrow, rad_arrow

import matplotlib.pyplot as plt

from tqdm import tqdm 

import os
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

from inxss.experiment import Background
from inxss.steer_neutron import NeutronExperimentSteerer


import hydra

@hydra.main(version_base=None, config_path="conf/final")
def main(cfg : DictConfig):
    num_steps = cfg['general']['num_steps']

    scale_likelihood = cfg['likelihood']['scale']
    likelihood_type = cfg['likelihood']['type']


    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if cfg['likelihood']['type'] == 'gaussian':
        output_path = os.path.join(
            cfg['paths']['output_path'],
            f"EXP_lkhd_{likelihood_type}_std_{cfg['likelihood']['std']}_scaled_{scale_likelihood}_steps_{num_steps}_{time_stamp}_{cfg['general']['name']}"
        )
    else:
        output_path = os.path.join(
            cfg['paths']['output_path'],
            f"EXP_lkhd_{likelihood_type}_scaled_{scale_likelihood}_steps_{num_steps}_{time_stamp}_{cfg['general']['name']}"
        )
    if 'steer' in cfg:
        output_path = output_path + f"_steer_{cfg['steer']['mode']}_{cfg['steer']['start']}_{cfg['steer']['end']}_{num_steps}"
        
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('output_path:', output_path)

    OmegaConf.save(cfg, os.path.join(output_path, "config.yaml"))


    model_path = cfg['paths']['model_path']
    data = torch.load(cfg['paths']['data_path'])
    print(data.keys())

    background = Background(
        tuple([data['grid'][_grid] for _grid in ['h_grid', 'k_grid', 'l_grid']]), 
        data['grid']['w_grid'], 
        data['background']
    )

    if 'particle_filter' in cfg:
        particle_filter_config = OmegaConf.to_container(cfg['particle_filter'], resolve=True)
        print('loading particle filter from config file...')
        print(particle_filter_config)
    else:
        particle_filter_config = {
            "num_particles": 1000,
            "dim_particles": 2,
            "prior_configs": {'types': ['uniform', 'uniform'], 'args': [{'low': 20, 'high': 40}, {'low': -5, 'high': 5}]}
        }

    grid_info = {
        k: [v.min().item(), v.max().item(), len(v)] for k,v in data['grid'].items()
    }

    mask_config = {
        "raw_mask_path": cfg['paths']['raw_mask_path'],
        "memmap_mask_path": cfg['paths']['memmap_mask_path'],
        "grid_info": grid_info,
        "preload": False,
        "build_from_scratch_if_no_memmap": True,
        "global_mask": data['S']>0
    }

    # experiment_config = {
    #     "q_grid": tuple([data['grid'][_grid] for _grid in ['h_grid', 'k_grid', 'l_grid']]),
    #     "w_grid": data['grid']['w_grid'],
    #     "S_grid": data['S'],
    #     "S_scale_factor": 1.
    # }
    
    if hasattr(cfg, 'experiment_config'):
        _experiment_config = OmegaConf.to_container(cfg['experiment_config'], resolve=True)
        print('loading experiment config from config file...')
        print(_experiment_config)
        experiment_config = {
            "q_grid": tuple([data['grid'][_grid] for _grid in ['h_grid', 'k_grid', 'l_grid']]),
            "w_grid": data['grid']['w_grid'],
            "S_grid": data['S'],
            "S_scale_factor": _experiment_config["S_scale_factor"],
            "poisson": _experiment_config["poisson"]
        }
    else:
        experiment_config = {
            "q_grid": tuple([data['grid'][_grid] for _grid in ['h_grid', 'k_grid', 'l_grid']]),
            "w_grid": data['grid']['w_grid'],
            "S_grid": data['S'],
            "S_scale_factor": 1.,
            "poisson": False
        }

    background_config = {
        "q_grid": tuple([data['grid'][_grid] for _grid in ['h_grid', 'k_grid', 'l_grid']]),
        "w_grid": data['grid']['w_grid'],
        "bkg_grid": data['background']
    }

    model = SpecNeuralRepr.load_from_checkpoint(model_path).to(device)

    steer = NeutronExperimentSteerer(
        model, particle_filter_config=particle_filter_config, 
        mask_config=mask_config, experiment_config=experiment_config, background_config=background_config,
        use_utility_sf=cfg['utility']['use_utility_sf'], utility_sf_sigma=cfg['utility']['utility_sf_sigma'],
        tqdm_pbar=False, lkhd_dict=cfg['likelihood'], device='cuda')



    for i_run in range(20):

        steer.reset()

        mean_list = [steer.particle_filter.mean().detach().cpu().clone()]
        std_list = [steer.particle_filter.std().detach().cpu().clone()]

        posisition_list = [steer.particle_filter.positions.data.T[None].cpu().clone()]
        weights_list = [steer.particle_filter.weights.data[None].cpu().clone()]
        
        if not 'steer' in cfg:
            steer_mode = 'unique_optimal'
            angles = [None,] * num_steps
        else:
            if cfg['steer']['mode'] == 'sequential':
                steer_mode = 'custom'
                angles = np.linspace(cfg['steer']['start'], cfg['steer']['end'], num_steps, endpoint=cfg['steer']['endpoint'])
            elif cfg['steer']['mode'] == 'random':
                steer_mode = 'custom'
                angles = np.linspace(cfg['steer']['start'], cfg['steer']['end'], num_steps, endpoint=cfg['steer']['endpoint'])
                np.random.shuffle(angles)
            else:
                steer_mode = cfg['steer']['mode']
                angles = [None,] * num_steps

        print("steer mode: ", steer_mode)
        print("angles: \n", angles)

        with torch.no_grad():
            progress_bar = tqdm(range(num_steps))
            for i in progress_bar:
                steer.step_steer(mode=steer_mode, next_angle=angles[i])
                current_mean = steer.particle_filter.mean().detach().cpu()
                current_std = steer.particle_filter.std().detach().cpu()
                progress_bar.set_description(
                    f'means: [{current_mean[0]:.3f}, {current_mean[1]:.3f}] '
                    f' stds: [{current_std[0]:.3f}, {current_std[1]:.3f}]'
                )
                mean_list.append(current_mean.clone())
                std_list.append(current_std.clone())

                posisition_list.append(steer.particle_filter.positions.data.T[None].cpu().clone())
                weights_list.append(steer.particle_filter.weights.data[None].cpu().clone())

        sub_result_dict = {
            'means': torch.vstack(mean_list).double(),
            'positions': torch.vstack(posisition_list).double(),
            'weights': torch.vstack(weights_list).double(),
            'measured_angles': torch.from_numpy(np.vstack(steer.measured_angles_history).squeeze()).double(),
            'background_signal_factors': torch.stack(steer.sig_bkg_factors_history).double(),
            'utility': torch.from_numpy(np.vstack(steer.utility_history).squeeze()).double(),
            'likelihood': torch.from_numpy(np.vstack(steer.lkhd_history).squeeze()).double(),
            'true_params': torch.tensor([29.0, 1.68]).double(),
        }

        torch.save(sub_result_dict, os.path.join(output_path, f'{i_run:02d}.pt'))


if __name__ == '__main__':
    main()
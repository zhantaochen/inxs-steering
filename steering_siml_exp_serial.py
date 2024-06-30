from inxss.experiment import SimulatedExperiment

import torch
import numpy as np

from inxss import SpectrumDataset, SpecNeuralRepr, Particle, PsiMask, OnlineVariance, linspace_2D_equidistant

from tqdm import tqdm 
from inxss.experiment import Background, SimulatedExperiment
from inxss.steer_neutron import NeutronExperimentSteerer
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter

import os
from datetime import datetime

torch.set_default_dtype(torch.float32)

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf/final")
def main(cfg : DictConfig):
    spinw_data = torch.load(cfg['paths']['spinw_data_path'])

    train_idx, val_test_idx = train_test_split(np.arange(spinw_data['Syy'].shape[0]), test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, random_state=42)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")


    num_steps = cfg['general']['num_steps']

    scale_likelihood = cfg['likelihood']['scale']
    likelihood_type = cfg['likelihood']['type']


    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if cfg['likelihood']['type'] == 'gaussian':
        output_path = os.path.join(
            cfg['paths']['output_path'],
            f"lkhd_{likelihood_type}_std_{cfg['likelihood']['std']}_scaled_{scale_likelihood}_steps_{num_steps}_{time_stamp}_{cfg['general']['name']}"
        )
    else:
        output_path = os.path.join(
            cfg['paths']['output_path'],
            f"lkhd_{likelihood_type}_scaled_{scale_likelihood}_steps_{num_steps}_{time_stamp}_{cfg['general']['name']}"
        )
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('output_path:', output_path)

    OmegaConf.save(cfg, os.path.join(output_path, "config.yaml"))

    model_path = cfg['paths']['model_path']
    data = torch.load(cfg['paths']['data_path'])
    global_mask = (data['S']>0).bool()

    if 'particle_filter' in cfg:
        particle_filter_config = cfg['particle_filter']
        print('loading particle filter from config file...')
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
        "global_mask": None
    }
    psi_mask = PsiMask(**mask_config)


    for idx_sample in tqdm(test_idx):
        # sim_experiment = SimulatedExperiment(
        #     spinw_data['q_grid'], spinw_data['w_grid'], 
        #     spinw_data['Syy'][idx_sample], spinw_data['Szz'][idx_sample],
        #     neutron_flux=300
        # )
        # sim_experiment.prepare_experiment(psi_mask.hklw_grid)
        # experiment_config = {
        #     "q_grid": tuple([data['grid'][_grid] for _grid in ['h_grid', 'k_grid', 'l_grid']]),
        #     "w_grid": data['grid']['w_grid'],
        #     "S_grid": torch.from_numpy(data['background']) + \
        #         global_mask * sim_experiment.Sqw,
        #     "S_scale_factor": 1.
        # }

        # background_config = {
        #     "q_grid": tuple([data['grid'][_grid] for _grid in ['h_grid', 'k_grid', 'l_grid']]),
        #     "w_grid": data['grid']['w_grid'],
        #     "bkg_grid": data['background']
        # }

        # model = SpecNeuralRepr.load_from_checkpoint(model_path).to(device)

        # steer = NeutronExperimentSteerer(
        #     model, particle_filter_config=particle_filter_config,
        #     mask_config=mask_config, experiment_config=experiment_config, background_config=background_config,
        #     tqdm_pbar=False, lkhd_dict=cfg['likelihood'], device=device)
        
        sim_experiment = SimulatedExperiment(
            spinw_data['q_grid'], spinw_data['w_grid'], 
            spinw_data['Syy'][idx_sample], spinw_data['Szz'][idx_sample],
            neutron_flux=1
        )

        sim_experiment.prepare_experiment(psi_mask.hklw_grid)


        hklw_grid_norm = psi_mask.hklw_grid[...,:2].norm(dim=-1)
        q_inn_mid = 1.1441
        q_mid_out = 1.8512

        mask_inn = (hklw_grid_norm <= q_inn_mid).numpy()
        mask_mid = (hklw_grid_norm >  q_inn_mid).numpy() * (hklw_grid_norm <=  q_mid_out).numpy()
        mask_out = (hklw_grid_norm >  q_mid_out).numpy()

        sigmas = []
        scales = []
        mask_exp = data['S'] > 1e-10

        for reg in ['inn', 'mid', 'out']:
            idx_sigma, idx_scale = np.where(data['background_dict'][f'{reg}_ddv'] == data['background_dict'][f'{reg}_ddv'].min())
            _sigma = data['background_dict']['sigmas'][idx_sigma]
            _scale = data['background_dict']['scales'][idx_scale]
            print('sigma: ', _sigma, '   scale: ', _scale)
            scales.append(_scale[0])
            sigmas.append(_sigma[0])
        centers = np.array([0.5 * np.sqrt(2), np.sqrt(1.5 ** 2 + 0.5 ** 2), 1.5 * np.sqrt(2)])
        scales = np.array(scales).squeeze()

        s_pred_masked_sm_inn = gaussian_filter(sim_experiment.Sqw, sigmas[0]) * mask_exp.numpy()
        s_pred_masked_sm_mid = gaussian_filter(sim_experiment.Sqw, sigmas[1]) * mask_exp.numpy()
        s_pred_masked_sm_out = gaussian_filter(sim_experiment.Sqw, sigmas[2]) * mask_exp.numpy()
        interped_scales = np.interp(hklw_grid_norm.numpy(), centers, scales)
        Sqw_syn = np.zeros_like(sim_experiment.Sqw)
        Sqw_syn[mask_inn] = s_pred_masked_sm_inn[mask_inn] * interped_scales[mask_inn]
        Sqw_syn[mask_mid] = s_pred_masked_sm_mid[mask_mid] * interped_scales[mask_mid]
        Sqw_syn[mask_out] = s_pred_masked_sm_out[mask_out] * interped_scales[mask_out]
        Sqw_syn += data['background']

        # experiment_config = {
        #     "q_grid": tuple([data['grid'][_grid] for _grid in ['h_grid', 'k_grid', 'l_grid']]),
        #     "w_grid": data['grid']['w_grid'],
        #     "S_grid": torch.from_numpy(Sqw_syn),
        #     "S_scale_factor": 1.
        # }
        
        if hasattr(cfg, 'experiment_config'):
            _experiment_config = OmegaConf.to_container(cfg['experiment_config'], resolve=True)
            print('loading experiment config from config file...')
            print(_experiment_config)
            experiment_config = {
                "q_grid": tuple([data['grid'][_grid] for _grid in ['h_grid', 'k_grid', 'l_grid']]),
                "w_grid": data['grid']['w_grid'],
                "S_grid": torch.from_numpy(Sqw_syn),
                "S_scale_factor": _experiment_config["S_scale_factor"],
                "poisson": _experiment_config["poisson"]
            }
        else:
            # experiment_config = {
            #     "q_grid": tuple([data['grid'][_grid] for _grid in ['h_grid', 'k_grid', 'l_grid']]),
            #     "w_grid": data['grid']['w_grid'],
            #     "S_grid": torch.from_numpy(Sqw_syn),
            #     "S_scale_factor": 1.
            # }
            experiment_config = {
                "q_grid": tuple([data['grid'][_grid] for _grid in ['h_grid', 'k_grid', 'l_grid']]),
                "w_grid": data['grid']['w_grid'],
                "S_grid": torch.from_numpy(Sqw_syn),
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
            
        mean_list = [steer.particle_filter.mean().detach().cpu().clone()]
        std_list = [steer.particle_filter.std().detach().cpu().clone()]

        posisition_list = [steer.particle_filter.positions.data.T[None].cpu().clone()]
        weights_list = [steer.particle_filter.weights.data[None].cpu().clone()]

        true_params = spinw_data['params'][idx_sample].numpy()

        print('true params: ', true_params)
        with torch.no_grad():
            progress_bar = tqdm(range(num_steps))
            for i in progress_bar:
                steer.step_steer(mode='unique_optimal')
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
            'true_params': torch.from_numpy(true_params).double(),
        }
        
        torch.save(sub_result_dict, os.path.join(output_path, f'{idx_sample}.pt'))

if __name__ == '__main__':
    main()
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from .utils_grid import convert_to_numpy, convert_to_torch, scale_tensor
import glob, os

class SpectrumDataset(Dataset):
    def __init__(self, data_path, num_wq=100):
        self.data_dict = torch.load(data_path)
        self.num_wq = num_wq

    def __len__(self):
        return self.data_dict['Syy'].size(0)
    
    def __getitem__(self, index):
        # Choose nw indices randomly along 2nd dimension
        nw_indices = np.random.choice(self.data_dict['Syy'].shape[1], self.num_wq, replace=True)
        # Choose nq indices randomly along 3rd dimension
        nq_indices = np.random.choice(self.data_dict['Syy'].shape[2], self.num_wq, replace=True)
        
        w = self.data_dict['w_grid'][None,nw_indices]
        q = self.data_dict['q_grid'][:2,nq_indices]
        p = self.data_dict['params'][index, None].T.repeat(1, self.num_wq)
        x = torch.transpose(torch.cat((q, w, p), dim=0), 1, 0)
        # Index into the tensor to get the random nw by nq sample
        Syy = self.data_dict['Syy'][index, nw_indices, nq_indices, None]
        Szz = self.data_dict['Szz'][index, nw_indices, nq_indices, None]
        
        return x, (Syy, Szz)


class BackgroundDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.data_dict['S'] = convert_to_torch(self.data_dict['S'].reshape(-1, 1))
        self.data_dict['x'] = convert_to_torch(self.data_dict['x'].reshape(-1, 4))
        
    def __len__(self):
        return self.data_dict['S'].size(0)
    
    def __getitem__(self, idx):
        return self.data_dict['x'][idx], self.data_dict['S'][idx]



class FullSpectrumDataset(Dataset):
    
    def __init__(self, data_file, grid_file, num_coords_per_sample=100):
        self.data_file = data_file
        
        self.grid_metadata = torch.load(grid_file)
        self.num_coords_per_sample = num_coords_per_sample
        
        for key, val in self.grid_metadata.items():
            setattr(self, f'{key}', np.linspace(*val))

        self.hklw_grid = torch.from_numpy(
            np.moveaxis(np.stack(np.meshgrid(self.h_grid, self.k_grid, self.l_grid, self.w_grid, indexing='ij'), axis=0), 0, -1))
    
    def __len__(self,):
        return len(self.data_file)
    
    def __getitem__(self, index):
        loaded_data = torch.load(self.data_file[index])
        S_data = loaded_data['S']

        param_data = loaded_data['param']
        # random_indices = np.random.choice(self.hklw_grid.numel() // 4, self.num_coords_per_sample, replace=False)
        random_indices = random.sample(range(self.hklw_grid.numel() // 4), self.num_coords_per_sample)

        # Use advanced indexing for batched data
        x = torch.cat([self.hklw_grid.view(-1,4)[random_indices], param_data.reshape(-1,2).repeat(self.num_coords_per_sample, 1)], dim=1)
        y = S_data.view(-1)[random_indices].view(-1,1)
        
        return x, y
    
    # def __getitem__(self, index):
    #     S_data = torch.from_numpy(self.S_memmap_array[index].copy())
    #     param_data = self.param[index]
    #     random_indices = np.random.choice(self.hklw_grid.numel() // 4, self.num_coords_per_sample)
    #     x = torch.cat([self.hklw_grid.view(-1,4)[random_indices], param_data.reshape(-1,2).repeat(self.num_coords_per_sample, 1)], dim=1)
    #     y = S_data.view(-1)[random_indices].view(-1,1)
    #     return x, y


# class FullSpectrumDataset(Dataset):
    
#     def __init__(self, S_data_file, param_data_file, grid_file, num_coords_per_sample=100):
#         self.S_shape = torch.load(os.path.join(os.path.dirname(S_data_file), 'S_shape.pt'))['S_shape']
#         self.S_memmap_array = np.memmap(S_data_file, dtype=np.float32, mode='r', shape=self.S_shape)
        
#         self.param = torch.load(param_data_file)
        
#         self.grid_metadata = torch.load(grid_file)
#         self.num_coords_per_sample = num_coords_per_sample
        
#         for key, val in self.grid_metadata.items():
#             setattr(self, f'{key}', np.linspace(*val))

#         self.hklw_grid = torch.from_numpy(
#             np.moveaxis(np.stack(np.meshgrid(self.h_grid, self.k_grid, self.l_grid, self.w_grid, indexing='ij'), axis=0), 0, -1))
    
#     def __len__(self,):
#         return self.S_shape[0]
    
#     # @staticmethod
#     # def collate_fn(batch):
#     #     indices = [item[0] for item in batch]
#     #     x, y = self[indices]  # use self to refer to the dataset instance
        
#     #     return x, y
    
#     def __getitem__(self, index):
#         if isinstance(index, (list, tuple)):
#             S_data = torch.from_numpy(self.S_memmap_array[index].copy()).view(len(index), -1)
#         else:
#             S_data = torch.from_numpy(self.S_memmap_array[index].copy()).view(1, -1)  # add an extra batch dimension

#         param_data = self.param[index].view(S_data.size(0), -1, 2)
#         # random_indices = np.random.choice(self.hklw_grid.numel() // 4, self.num_coords_per_sample, replace=False)
#         random_indices = random.sample(range(self.hklw_grid.numel() // 4), self.num_coords_per_sample)

#         # Use advanced indexing for batched data
#         x = torch.cat([self.hklw_grid.view(-1, 4)[random_indices].unsqueeze(0).repeat(S_data.size(0),1,1), param_data.repeat(1, self.num_coords_per_sample, 1)], dim=-1)
#         y = S_data[:, random_indices].view(-1, self.num_coords_per_sample, 1)
        
#         return x, y
    
#     # def __getitem__(self, index):
#     #     S_data = torch.from_numpy(self.S_memmap_array[index].copy())
#     #     param_data = self.param[index]
#     #     random_indices = np.random.choice(self.hklw_grid.numel() // 4, self.num_coords_per_sample)
#     #     x = torch.cat([self.hklw_grid.view(-1,4)[random_indices], param_data.reshape(-1,2).repeat(self.num_coords_per_sample, 1)], dim=1)
#     #     y = S_data.view(-1)[random_indices].view(-1,1)
#     #     return x, y
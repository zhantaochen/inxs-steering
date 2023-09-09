import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class SpectrumDataset(Dataset):
    def __init__(self, data_path, num_wq=100):
        self.data_dict = torch.load(data_path)
        # self.data_dict['Syy'] = self.data_dict['Syy'] / (
        #     self.data_dict['Syy'].amax(dim=(1,2), keepdims=True) + 1e-15)
        # self.data_dict['Szz'] = self.data_dict['Szz'] / (
        #     self.data_dict['Szz'].amax(dim=(1,2), keepdims=True) + 1e-15)
        self.num_wq = num_wq
        self.data_dict['params'][:,0] = self.scale_tensor(self.data_dict['params'][:,0], (20, 40), (0.0, 0.5))
        self.data_dict['params'][:,1] = self.scale_tensor(self.data_dict['params'][:,1], (-5, 5), (0.0, 0.5))
        self.data_dict['w_grid'] = self.scale_tensor(self.data_dict['w_grid'], (0, 150), (0.0, 0.5))
        
    def __len__(self):
        return self.data_dict['Syy'].size(0)
    
    def __getitem__(self, index):
        # Choose nw indices randomly along 2nd dimension
        nw_indices = np.random.choice(self.data_dict['Syy'].shape[1], self.num_wq, replace=True)
        # Choose nq indices randomly along 3rd dimension
        nq_indices = np.random.choice(self.data_dict['Syy'].shape[2], self.num_wq, replace=True)
        
        w = self.data_dict['w_grid'][None,nw_indices]
        q = self.data_dict['q_grid'][:2,nq_indices]
        p = self.data_dict['params'][index,None].T.repeat(1, self.num_wq)
        x = torch.transpose(torch.cat((q, w, p), dim=0), 1, 0)
        # Index into the tensor to get the random nw by nq sample
        Syy = self.data_dict['Syy'][index, nw_indices, nq_indices, None]
        Szz = self.data_dict['Szz'][index, nw_indices, nq_indices, None]
        
        return x, (Syy, Szz)
    
    def scale_tensor(self, tensor, bounds_init, bounds_fnal=(-1., 1.)):
        min_init, max_init = bounds_init
        min_fnal, max_fnal = bounds_fnal
        return ((tensor - min_init) * (max_fnal - min_fnal) / (max_init - min_init)) + min_fnal

        
    
# class SpectrumDataset(Dataset):
#     def __init__(self, data_path, num_wq=100):
#         self.data_dict = torch.load(data_path)
#         self.num_wq = num_wq
        
#         mean_intens = self.data_dict['Syy'].mean(dim=0) + self.data_dict['Szz'].mean(dim=0)
#         self.pr = (mean_intens.exp() / (mean_intens.exp().sum() + 1e-15)).reshape(-1)
#         self.wq_grid = torch.cat([
#             self.data_dict['w_grid'].view(self.data_dict['w_grid'].shape[0], -1).unsqueeze(1).expand(-1, self.data_dict['q_grid'].shape[1], -1), 
#             self.data_dict['q_grid'][:2,:].T.unsqueeze(0).expand(self.data_dict['w_grid'].shape[0], -1, -1)], dim=2).reshape(-1,3)
    
#     def __len__(self):
#         return self.data_dict['Syy'].size(0)
    
#     def __getitem__(self, index):
        
#         wq_index = np.random.choice(self.pr.numel(), self.num_wq, p=self.pr)
        
#         x = torch.zeros((self.num_wq, 5))
#         x[:,:3] = self.wq_grid[wq_index]
#         x[:,-2:] = self.data_dict['params'][index]
#         Syy = self.data_dict['Syy'][0].reshape(-1)[wq_index, None]
#         Szz = self.data_dict['Szz'][0].reshape(-1)[wq_index, None]
        
#         return x, (Syy, Szz)
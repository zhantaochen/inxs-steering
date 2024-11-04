from collections import namedtuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import lightning as L

from .siren import SirenNet
from .formfact import get_ff_params
from .utils_grid import scale_tensor

class SpecNeuralReprDropout(L.LightningModule):
    def __init__(
        self, 
        scale_dict={
            'J' : [(20, 40), (0, 0.5)], 
            'Jp': [(-5,  5), (0, 0.5)], 
            'w' : [(0, 150), (0, 0.5)]},
        dropout=0.2
    ):
        super().__init__()
        self.save_hyperparameters()
        # lattice constants
        self.latt_const = namedtuple('latt_const', ['a', 'c'])(3.89, 12.55)
        # form factor parameters
        self.ff = get_ff_params()
        self.dropout = dropout
        # networks
        self.Syy_net = torch.nn.Sequential(
            SirenNet(
                dim_in = 5,
                dim_hidden = 256,
                dim_out = 256,
                num_layers = 3,
                w0_initial = 30.,
                final_activation = torch.nn.ReLU(),
                dropout=self.dropout
            ),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(256, 1)
        )
        self.Szz_net = torch.nn.Sequential(
            SirenNet(
                dim_in = 5,
                dim_hidden = 256,
                dim_out = 256,
                num_layers = 3,
                w0_initial = 30.,
                final_activation = torch.nn.ReLU(),
                dropout=self.dropout
            ),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(256, 1)
        )
        self.scale_dict = scale_dict
 
    def prepare_input(self, x):
        shape = x.shape[:-1]
        x = x.view(-1, x.size(-1)).to(self.dtype)
        for key in self.scale_dict.keys():
            if key == 'w':
                i = 2
            elif key == 'J':
                i = 3
            elif key == 'Jp':
                i = 4
            x[:,i] = scale_tensor(x[:,i], *self.scale_dict[key])
        return x.view(shape+(x.shape[-1],))
        
    def forward(self, x_raw, l=None, Syy=None, Szz=None):
        """
        x_raw: (..., 5)
        the 1st and 2nd are the reciprocal lattice vectors (h,k)
        the 3rd dimension is the energy transfer w
        the 4th and 5th dimensions are the query parameters
        """
        # avoid inplace change of input x_raw
        x = self.prepare_input(x_raw.clone())
        shape = x.shape[:-1]
        x = x.view(-1, x.size(-1)).to(self.dtype)
        if l is None:
            l = torch.zeros_like(x[:,[0]]).to(self.dtype)
        else:
            l = l.view(-1, 1).to(self.dtype)
        # Q can reside in higher Brillouin zones
        # Q = torch.cat((x[:,:2], torch.zeros_like(x[:,1])[:,None]), dim=1)
        # Reduced reciprocal lattice vectors projected into the first quadrant of the Brillouin zone
        # since the models are trained on the first quadrant only
        Q = torch.cat((x[:,:2], l), dim=1)
        x[:,:2] = torch.abs(x[:,:2] - torch.round(x[:,:2]))
        if Syy is None:
            Syy = self.Syy_net(x).squeeze(-1)
        else:
            Syy = Syy.view(-1).to(self.dtype)
        if Szz is None:
            Szz = self.Szz_net(x).squeeze(-1)
        else:
            Szz = Szz.view(-1).to(self.dtype)
        S = self.calculate_Sqw(Q, Syy, Szz)
        
        return S.view(shape)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """Assuming reciprocal lattice vectors are in the first quadrant of the Brillouin zone
           Thus no need to project them into the first quadrant
        """
        x, (Syy, Szz) = train_batch
        x = self.prepare_input(x)
        x = x.view(-1, x.size(-1)).to(self.dtype)
        Syy = Syy.view(-1, Syy.size(-1)).to(self.dtype)
        Szz = Szz.view(-1, Szz.size(-1)).to(self.dtype)
        # Syy = torch.log(1. + Syy.view(-1, Syy.size(-1)).to(self.dtype))
        # Szz = torch.log(1. + Szz.view(-1, Szz.size(-1)).to(self.dtype))
        
        Syy_pred = self.Syy_net(x)
        Szz_pred = self.Szz_net(x)
        
        loss_Syy = F.mse_loss(Syy_pred, Syy)
        loss_Szz = F.mse_loss(Szz_pred, Szz)
        loss = loss_Syy + loss_Szz
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, (Syy, Szz) = val_batch
        x = self.prepare_input(x)
        x = x.view(-1, x.size(-1)).to(self.dtype)
        Syy = Syy.view(-1, Syy.size(-1)).to(self.dtype)
        Szz = Szz.view(-1, Szz.size(-1)).to(self.dtype)
        # Syy = torch.log(1. + Syy.view(-1, Syy.size(-1)).to(self.dtype))
        # Szz = torch.log(1. + Szz.view(-1, Szz.size(-1)).to(self.dtype))
        
        Syy_pred = self.Syy_net(x)
        Szz_pred = self.Szz_net(x)
        
        loss_Syy = F.mse_loss(Syy_pred, Syy)
        loss_Szz = F.mse_loss(Szz_pred, Szz)
        loss = loss_Syy + loss_Szz
        self.log('val_loss', loss)

    def formfact(self, Q):
        H, K, L = Q[:,0], Q[:,1], Q[:,2]
        a, c = self.latt_const.a, self.latt_const.c
        
        q = 2 * np.pi * torch.sqrt((H**2 + K**2) / a**2 + L**2 / c**2) # Scattering vector in Angstroem^-1
        s2 = (q/4/np.pi)**2
        j0 = (self.ff.A0 * torch.exp(-self.ff.a0*s2) + 
              self.ff.B0 * torch.exp(-self.ff.b0*s2) + 
              self.ff.C0 * torch.exp(-self.ff.c0*s2) + self.ff.D0)
        j4 = (self.ff.A4 * torch.exp(-self.ff.a4*s2) + 
              self.ff.B4 * torch.exp(-self.ff.b4*s2) + 
              self.ff.C4 * torch.exp(-self.ff.c4*s2) + self.ff.D4) * s2
        ff_q = j0 + j4 * 3/2 * (
                H**4 + K**4 + L**4 * (a/c)**4 - 
                3 * (H**2 * K**2 + H**2 * L**2 * (a/c)**2 + K**2 * L**2 * (a/c)**2)
            ) / (H**2 + K**2 + L**2 * (a/c)**2 + 1e-15) ** 2
        return ff_q
    
    def calculate_Sqw(self, Q, Syy, Szz):
        H, K, L = Q[:,0], Q[:,1], Q[:,2]
        a, c = self.latt_const.a, self.latt_const.c
        ql = 2 * np.pi * L / c # Out of plane component of the scattering vector
        q = 2 * np.pi * torch.sqrt((H**2 + K**2) / a**2 + L**2 / c**2) # Scattering vector in Angstroem^-1
        
        ff_q = self.formfact(Q).detach()
        S = (torch.abs(ff_q)**2) * (
            (1 + (ql/(q+1e-15))**2) / 2 * Syy + (1 - (ql/(q+1e-15))**2) * Szz
        )
        return S
    
    

class FullSpectrumNetwork(L.LightningModule):
    def __init__(
        self, 
        scale_dict={
            'h' : [(0, 1), (0, 10)],
            'k' : [(0, 1), (0, 10)],
            'w' : [(0, 150), (0, 15)]}
        ):
        super().__init__()
        self.save_hyperparameters()
        # lattice constants
        self.latt_const = namedtuple('latt_const', ['a', 'c'])(3.89, 12.55)
        # form factor parameters
        self.ff = get_ff_params()
        # networks
        self.net = torch.nn.Sequential(
            SirenNet(
                dim_in = 6,
                dim_hidden = 256,
                dim_out = 256,
                num_layers = 3,
                w0_initial = 30.,
                final_activation = torch.nn.ReLU()
            ),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.ReLU()
        )
        self.scale_dict = scale_dict
        self.scale_indices = {
            'h' : 0, 'k' : 1, 'l' : 2, 'w' : 3, 'J' : 4, 'Jp' : 5
        }
        
        self.loss_fn = torch.nn.MSELoss()
    

    def prepare_input(self, x):
        shape = x.shape[:-1]
        x = x.view(-1, x.size(-1)).to(self.dtype)
        for key in self.scale_dict.keys():
            i = self.scale_indices[key]
            x[:,i] = scale_tensor(x[:,i], *self.scale_dict[key])
        return x.view(shape+(x.shape[-1],))
            
    def forward(self, x_raw):
        x = self.prepare_input(x_raw.clone())
        output = self.net(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, S = batch
        
        S_pred = self.forward(x)
        loss = self.loss_fn(S_pred, S)
        
        self.log('train_loss', loss.item())
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, S = batch
        
        S_pred = self.forward(x)
        loss = self.loss_fn(S_pred, S)
        
        self.log('val_loss', loss.item())

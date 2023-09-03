from collections import namedtuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from .siren import SirenNet
from .formfact import get_ff_params

class SpecNeuralRepr(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # lattice constants
        self.latt_const = namedtuple('latt_const', ['a', 'c'])(3.89, 12.55)
        # form factor parameters
        self.ff = get_ff_params()
        # networks
        self.Syy_net = torch.nn.Sequential(
            SirenNet(
                dim_in = 5,
                dim_hidden = 256,
                dim_out = 256,
                num_layers = 3,
                w0_initial = 30.,
                final_activation = torch.nn.ReLU()
            ),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.Szz_net = torch.nn.Sequential(
            SirenNet(
                dim_in = 5,
                dim_hidden = 256,
                dim_out = 256,
                num_layers = 3,
                w0_initial = 30.,
                final_activation = torch.nn.ReLU()
            ),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        # self.Syy_net = torch.nn.Sequential(
        #     torch.nn.Linear(5, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 1)
        # )
        # self.Szz_net = torch.nn.Sequential(
        #     torch.nn.Linear(5, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 1)
        # )

    def forward(self, x):
        if x.ndim == 3:
            x = x.reshape(-1, x.size(-1))
        x = x.to(self.dtype)
        # Q can reside in higher Brillouin zones
        Q = torch.cat((x[:,1:3], torch.zeros_like(x[:,1])[:,None]), dim=1)
        # Reduced reciprocal lattice vectors projected into the first quadrant of the Brillouin zone
        # since the models are trained on the first quadrant only
        x[:, 1:3] = torch.abs(x[:, 1:3] - torch.round(x[:, 1:3]))
        Syy = self.Syy_net(x)
        Szz = self.Szz_net(x)
        
        S = self.calculate_Sqw(Q, Syy, Szz)
        
        return S

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """Assuming reciprocal lattice vectors are in the first quadrant of the Brillouin zone
           Thus no need to project them into the first quadrant
        """
        x, (Syy, Szz) = train_batch
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
        S = (torch.abs(ff_q)**2)[:,None] * (
            (1 + (ql/(q+1e-15))**2)[:,None] / 2 * Syy + (1 - (ql/(q+1e-15))**2)[:,None] * Szz
        )
        return S
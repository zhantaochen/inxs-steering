# %%
import numpy as np
import torch
from inxss import SirenNet

import glob, os

from sklearn.model_selection import train_test_split

from lightning.pytorch.loggers import TensorBoardLogger

# %%
from inxss import SpectrumDataset, SpecNeuralRepr
from inxss.dataset import FullSpectrumDataset
from inxss.specnet import FullSpectrumNetwork
from torch.utils.data import DataLoader

# %%
mmapped_data_filename = '/pscratch/sd/z/zhantao/inxs_steering/smoothed_SpinW_data/memmap_datasets/S_sigma_0.75.dat'
S_shape = torch.load('/pscratch/sd/z/zhantao/inxs_steering/smoothed_SpinW_data/memmap_datasets/S_shape.pt')['S_shape']
mmapped_data = np.memmap(mmapped_data_filename, dtype=np.float32, mode='r', shape=S_shape)

# %%
data_folder = '/pscratch/sd/z/zhantao/inxs_steering/smoothed_SpinW_data/sigma_0.75'

data_file = sorted(glob.glob(os.path.join(data_folder, '*.pt')))

train_data_file, val_test_data_file = train_test_split(data_file, test_size=1/6, random_state=42)
val_data_file,   test_data_file     = train_test_split(val_test_data_file, test_size=0.5, random_state=42)

train_dataset = FullSpectrumDataset(
    data_file = train_data_file,
    grid_file = '/pscratch/sd/z/zhantao/inxs_steering/smoothed_SpinW_data/sigma_0.75/grid_metadata',
    num_coords_per_sample = 10000
)
val_dataset = FullSpectrumDataset(
    data_file = val_data_file,
    grid_file = '/pscratch/sd/z/zhantao/inxs_steering/smoothed_SpinW_data/sigma_0.75/grid_metadata',
    num_coords_per_sample = 10000
)
test_dataset = FullSpectrumDataset(
    data_file = test_data_file,
    grid_file = '/pscratch/sd/z/zhantao/inxs_steering/smoothed_SpinW_data/sigma_0.75/grid_metadata',
    num_coords_per_sample = 10000
)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=10)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=10)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=10)

# %%
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

model = FullSpectrumNetwork(
    scale_dict={
        'h' : [(0, 1), (0, 10)],
        'k' : [(0, 1), (0, 10)],
        'w' : [(0, 150), (0, 15)]
        }
    )

# %%
checkpoint_callback = ModelCheckpoint(
    save_on_train_epoch_end=False, save_last=True, save_top_k=1, monitor="val_loss"
)

logger = TensorBoardLogger(save_dir='/pscratch/sd/z/zhantao/inxs_steering')

trainer = L.Trainer(
    max_epochs=1000, accelerator="gpu",
    callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback],
    logger=logger, log_every_n_steps=1, devices=1,
    enable_checkpointing=True,
    default_root_dir='/pscratch/sd/z/zhantao/inxs_steering'
)

trainer.fit(model, train_loader, val_loader)

# %%




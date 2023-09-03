import numpy as np
import torch
from inxss import SirenNet

from sklearn.model_selection import train_test_split

from inxss import SpectrumDataset, SpecNeuralRepr
from torch.utils.data import DataLoader

spec_dataset = SpectrumDataset(
    '/pscratch/sd/z/zhantao/inxs_steering/SpinW_data/summarized_AFM_data_2023Aug01.pt',
    num_wq=40000
    )

train_idx, val_test_idx = train_test_split(np.arange(len(spec_dataset)), test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, random_state=42)

train_loader = DataLoader([spec_dataset[i] for i in train_idx], batch_size=10, shuffle=True)
val_loader = DataLoader([spec_dataset[i] for i in val_idx], batch_size=10, shuffle=False)
test_loader = DataLoader([spec_dataset[i] for i in test_idx], batch_size=10, shuffle=False)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

model = SpecNeuralRepr()

checkpoint_callback = ModelCheckpoint(
    save_on_train_epoch_end=False, save_last=True, save_top_k=1, monitor="val_loss"
)

trainer = pl.Trainer(
    max_epochs=10000, accelerator="gpu",
    callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback],
    log_every_n_steps=1, devices=1,
    enable_checkpointing=True,
    default_root_dir='/pscratch/sd/z/zhantao/inxs_steering'
)

trainer.fit(model, train_loader, val_loader)
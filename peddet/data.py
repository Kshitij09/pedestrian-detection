import pandas as pd
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from .dataset import PennFudanDataset
from .transforms import TrainTransforms, ValTransforms
from .utils.utils import collate_fn
from hydra.utils import to_absolute_path, get_original_cwd, HydraConfig
import ast
import os

class PennFudanDataModule(pl.LightningDataModule):
    "Reusable DataModule for PennFudan Dataset"
    def __init__(self, cfg: DictConfig, train_transforms=None, val_transforms=None):
        super().__init__(train_transforms=train_transforms,
        val_transforms=val_transforms,
        test_transforms=val_transforms)
        
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.cfg = cfg

        if HydraConfig.initialized():
            # Current directory changed!
            # Update paths with original_cwd
            self.data_csv_path = to_absolute_path(cfg.data.csv_path)
            self.root_dir = to_absolute_path(cfg.data.root_dir)
        else:
            # Keep relative paths
            self.data_csv_path = cfg.data.csv_path
            self.root_dir = cfg.data.root_dir
    
    def setup(self, stage=None):
        super().setup(stage=stage)
        cfg = self.cfg
        df = pd.read_csv(self.data_csv_path)
        df[['x','y','x1','y1']] = pd.DataFrame(
            np.stack(df['box'].apply(ast.literal_eval)).astype(np.float32)
        )
        train_df = df.loc[df['fold'] != cfg.data.valid_fold].copy()
        valid_df = df.loc[df['fold'] == cfg.data.valid_fold].copy()
        self.train_dataset = PennFudanDataset(train_df,root_dir=self.root_dir,transforms=self.train_transforms,mode='train')
        self.val_dataset = PennFudanDataset(valid_df,root_dir=self.root_dir,transforms=self.val_transforms,mode='val')
    
    def train_dataloader(self):
        data = self.cfg.data
        return DataLoader(self.train_dataset,
                          num_workers=data.num_workers,
                          collate_fn=collate_fn,
                          shuffle=True,
                          batch_size=data.batch_size)

    def val_dataloader(self):
        data = self.cfg.data
        return DataLoader(self.val_dataset,
                          num_workers=data.num_workers,
                          collate_fn=collate_fn,
                          shuffle=False,
                          batch_size=data.batch_size)

    def test_dataloader(self):
        return self.val_dataloader()
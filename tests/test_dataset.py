from omegaconf.dictconfig import DictConfig
from peddet.transforms import TrainTransforms, ValTransforms
from peddet.dataset import PennFudanDataset
import pandas as pd
import numpy as np
import ast
import torch


def test_train_dataset(sample_config: DictConfig):
    "Assert dataset types and required keys in the target"
    cfg = sample_config
    train_transforms = TrainTransforms(cfg.aug)
    df = pd.read_csv(cfg.data.csv_path)
    df[["x", "y", "x1", "y1"]] = pd.DataFrame(
        np.stack(df["box"].apply(ast.literal_eval)).astype(np.float32)
    )
    train_df = df.loc[df["fold"] != cfg.data.valid_fold].copy()
    train_transforms = TrainTransforms(cfg.aug)
    train_dataset = PennFudanDataset(
        train_df,
        root_dir=cfg.data.root_dir,
        transforms=train_transforms,
        mode="train",
    )

    image, target = train_dataset[0]
    required_keys = ["boxes", "labels", "image_id", "area", "iscrowd"]
    assert isinstance(image, torch.Tensor)
    assert list(target.keys()) == required_keys


def test_val_dataset(sample_config: DictConfig):
    "Assert dataset types and required keys in the target"
    cfg = sample_config
    val_transforms = ValTransforms(cfg.aug)
    df = pd.read_csv(cfg.data.csv_path)
    df[["x", "y", "x1", "y1"]] = pd.DataFrame(
        np.stack(df["box"].apply(ast.literal_eval)).astype(np.float32)
    )
    valid_df = df.loc[df["fold"] == cfg.data.valid_fold].copy()
    train_transforms = TrainTransforms(cfg.aug)
    valid_dataset = PennFudanDataset(
        valid_df,
        root_dir=cfg.data.root_dir,
        transforms=train_transforms,
        mode="val",
    )

    image, target = valid_dataset[0]
    required_keys = ["boxes", "labels", "image_id", "area", "iscrowd"]
    assert isinstance(image, torch.Tensor)
    assert list(target.keys()) == required_keys
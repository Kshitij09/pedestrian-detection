from peddet.transforms import TrainTransforms, ValTransforms
from peddet.dataset import PennFudanDataset
import pandas as pd
import numpy as np
import ast
import torch
from hydra.experimental import initialize, compose


def test_train_dataset():
    "Assert dataset types and required keys in the target"
    with initialize(config_path="../peddet/conf"):
        cfg = compose(config_name="config", overrides=["data=sample"])
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

        image, target, image_id = train_dataset[0]
        required_keys = ["boxes", "labels", "image_id", "area", "iscrowd"]
        assert isinstance(image, torch.Tensor)
        assert list(target.keys()) == required_keys


def test_val_dataset():
    "Assert dataset types and required keys in the target"
    with initialize(config_path="../peddet/conf"):
        cfg = compose(config_name="config", overrides=["data=sample"])
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

        image, target, image_id = valid_dataset[0]
        required_keys = ["boxes", "labels", "image_id", "area", "iscrowd"]
        assert isinstance(image, torch.Tensor)
        assert list(target.keys()) == required_keys
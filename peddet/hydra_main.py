from os import name
from sys import stderr
import sys
import os
from pathlib import Path
from typing import Dict, List
from pytorch_lightning.callbacks import early_stopping
from pytorch_lightning.callbacks import model_checkpoint
from pytorch_lightning.core import datamodule
from peddet.models.faster_rcnn import FasterRCNNModule, get_default_fasterrcnn_model
from peddet.utils.coc_eval import CocoEvaluator
from peddet.utils.utils import IouTypes
from peddet.data import PennFudanDataModule
from peddet.transforms import TrainTransforms, ValTransforms
from peddet.utils.coco_utils import get_coco_api_from_dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import hydra
import torch
import importlib
import logging
from torch.utils.data import Dataset
from omegaconf import DictConfig, OmegaConf


def configure_trainer(cfg: DictConfig) -> pl.Trainer:
    # Default values from trainer
    early_stopping = False
    checkpoint_callback = True
    logger = True

    if cfg.training.early_stopping:
        early_stopping = EarlyStopping(**cfg.callbacks.early_stopping)

    if cfg.training.model_checkpoint:
        checkpoint_dir = hydra.utils.to_absolute_path(cfg.training.checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        file_path = os.path.join(
            cfg.training.checkpoint_dir, "{epoch:02}-{main_score:.04f}"
        )
        checkpoint_callback = ModelCheckpoint(
            filepath=file_path, **cfg.callbacks.model_checkpoint
        )

    if cfg.training.wandb_logger:
        try:
            wandb = importlib.import_module("wandb")
            SECRET = os.environ["WANDB_API_KEY"]
            wandb.login(key=SECRET)
            del SECRET
            logger = WandbLogger(**cfg.loggers.wandb)
        except ModuleNotFoundError:
            print(
                """
                You need to install 'wandb' to use this logger.
                To disable, set training.wandb_logger=False
                """
            )
            sys.exit(1)
        except KeyError:
            print("Make sure you've set 'WANDB_API_KEY' environment variable")
            sys.exit(1)

    trainer = pl.Trainer(
        logger=logger,
        early_stop_callback=early_stopping,
        checkpoint_callback=checkpoint_callback,
        **cfg.trainer
    )
    return trainer


def get_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
    train_transforms = TrainTransforms(cfg.aug)
    val_transforms = ValTransforms(cfg.aug)
    datamodule = PennFudanDataModule(cfg, train_transforms, val_transforms)
    return datamodule


def get_coco_evaluator(
    dataset: Dataset, iou_types: List[IouTypes] = [IouTypes.BBOX]
) -> CocoEvaluator:
    coco = get_coco_api_from_dataset(dataset)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    return coco_evaluator


def get_train_module(cfg, coco_evaluator: CocoEvaluator) -> pl.LightningModule:
    model = get_default_fasterrcnn_model()
    train_module = FasterRCNNModule(cfg, model, coco_evaluator)
    return train_module


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    datamodule = get_datamodule(cfg)
    datamodule.setup(stage="fit")
    coco_evaluator = get_coco_evaluator(datamodule.val_dataset)
    train_module = get_train_module(cfg, coco_evaluator)
    # trial run
    if cfg.training.debug:
        trainer = pl.Trainer(gpus=1, fast_dev_run=True)
        trainer.fit(train_module, datamodule=datamodule)

    # trainer = configure_trainer(cfg)
    # trainer.fit(model=train_module, datamodule=datamodule)


if __name__ == "__main__":
    sys.exit(main())

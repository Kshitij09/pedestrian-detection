import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from peddet.utils.utils import flatten_omegaconf
from peddet.utils.coc_eval import CocoEvaluator
from omegaconf.dictconfig import DictConfig
from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pytorch_lightning as pl


def get_default_fasterrcnn_model():
    num_classes = 3  # 2 classes + background
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class FasterRCNNModule(pl.LightningModule):
    def __init__(
        self, cfg: DictConfig, model: nn.Module, coco_evaluator: CocoEvaluator
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.hparams = flatten_omegaconf(cfg)
        self.model = model
        self.coco_evaluator = coco_evaluator

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, list(targets))
        total_loss = sum(loss for loss in loss_dict.values())
        loss_dict["total_loss"] = total_loss
        return {"loss": total_loss, "log": loss_dict, "progress_bar": loss_dict}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = list(targets)
        outputs = self.model(images, targets)
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }
        self.coco_evaluator.update(res)
        return {}

    def validation_epoch_end(self, outputs):
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()
        metric = self.coco_evaluator.coco_eval["bbox"].stats[0]
        metric = torch.as_tensor(metric)
        logs = {"main_score": metric}
        return {**logs, "log": logs, "progress_bar": logs}

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), **self.cfg.optim.adamw)
        schedulers: list = self._configure_schedulers(optimizer)
        return [optimizer], schedulers

    def _configure_schedulers(self, optimizer):
        schedulers = self.cfg.sched
        lr_plateau = ReduceLROnPlateau(optimizer, **schedulers.lr_plateau.params)
        lr_plateau_config = {
            "scheduler": lr_plateau,
            "interval": schedulers.lr_plateau.step,
            "monitor": schedulers.lr_plateau.monitor,
        }

        cosine_anneal = CosineAnnealingLR(optimizer, **schedulers.cosine_anneal.params)
        cosine_anneal_config = {
            "scheduler": cosine_anneal,
            "interval": schedulers.cosine_anneal.step,
            "monitor": schedulers.cosine_anneal.monitor,
        }

        return [lr_plateau_config, cosine_anneal_config]

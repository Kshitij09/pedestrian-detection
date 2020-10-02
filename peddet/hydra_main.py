from peddet.models.faster_rcnn import FasterRCNNModule, get_default_fasterrcnn_model
from peddet.utils.coc_eval import CocoEvaluator
from peddet.utils.utils import IOU_TYPES
from peddet.data import PennFudanDataModule
from peddet.transforms import TrainTransforms, ValTransforms
from peddet.utils.coco_utils import get_coco_api_from_dataset
import sys
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    train_transforms = TrainTransforms(cfg.aug)
    val_transforms = ValTransforms(cfg.aug)
    datamodule = PennFudanDataModule(cfg, train_transforms, val_transforms)
    datamodule.setup(stage="fit")
    val_dataset = datamodule.val_dataset
    coco = get_coco_api_from_dataset(val_dataset)
    # Considering only bounding boxes as of now
    iou_types = [IOU_TYPES.BBOX]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    model = get_default_fasterrcnn_model()
    train_module = FasterRCNNModule(cfg, model, coco_evaluator)
    trainer = pl.Trainer(gpus=1, fast_dev_run=True)
    trainer.fit(train_module, datamodule=datamodule)


if __name__ == "__main__":
    sys.exit(main())
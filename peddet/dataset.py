import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.core.composition import Compose
from typing import Tuple, List, Dict
import os
from pathlib import Path
import cv2
import pdb


class PennFudanDataset(Dataset):
    """
    Prepare PennFudan dataset

    Args:
    dataframe: dataframe with image_ids and bboxes
    mode: train/valid/test
    root_dir: path to root directory for dataset
    transforms: 'albumentations' transforms
    """

    def __init__(
        self,
        dataframe: pd.DataFrame = None,
        root_dir: str = "",
        transforms: Compose = None,
        mode: str = "train",
    ):
        self.df = dataframe
        self.root_dir = root_dir
        self.mode = mode
        self.transforms = transforms
        if self.df is not None:
            self.image_ids = self.df["id"].unique()
            # DataFrame should have 'x','y','x1','y1' and 'area' keys
            _rec = self.df.iloc[0]
            assert {"x", "y", "x1", "y1", "area", "label"} <= set(
                _rec.to_dict()
            ), "DataFrame should have 'x','y','x1','y1','label' and 'area' keys"

            classes, class_to_idx = self._find_classes(dataframe)
            self.classes = classes
            self.class_to_idx = class_to_idx
        else:
            self.image_ids = os.listdir(root_dir)

    def _find_classes(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, int]]:
        "Extract classes and class_to_idx mapping from dataframe"
        classes = df.label.unique()
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def get_image(self, image_path: str):
        "Read and preprocess image using cv2"
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        return image

    def coco_prepare(self, image_data: pd.core.series.Series, index: int):
        """Handle `image_data` and prepare target dict for
           coco evaluation
        Arguments:
            image_data (pandas.core.series.Series): Pandas series
            index (Int): label index
        Returns:
            target (Dict): target dictionary with keys
            ['boxes','labels','image_id','area','iscrowd']
        """

        target = {}

        bboxes = image_data[["x", "y", "x1", "y1"]].values

        areas = image_data["area"].values
        areas = torch.as_tensor(areas, dtype=torch.float32)

        # get labels
        labels = image_data["label"].values
        labels = list(map(self.class_to_idx.__getitem__, labels))
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # iscrowd=1 records are ignored while evaluation
        iscrowd = torch.zeros((image_data.shape[0],), dtype=torch.int64)

        # target dict as-per coco requirements
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([index])
        target["area"] = areas
        target["iscrowd"] = iscrowd
        return target

    def __getitem__(self, index: int):
        """
        Arguments:
            index (int): item index
        Returns:
            image (Tensor): preprocessed image tensor
            target (Dict): target dict compatible for coco evaluation
            image_id (str): image-id for logging purposes
        """

        image_id = Path(self.image_ids[index]).stem

        # For test-dataset
        target = {
            "labels": torch.as_tensor([[0]], dtype=torch.int64),
            "boxes": torch.as_tensor([[0, 0, 0, 0]], dtype=torch.float32),
        }

        if self.mode != "test":
            image_data = self.df.loc[self.df["id"] == image_id]
            path = os.path.join(self.root_dir, image_data.filename.min())
            image = self.get_image(path)
            target = self.coco_prepare(image_data, index)
            if self.transforms:
                image_dict = {
                    "image": image,
                    "bboxes": target["boxes"],
                    "labels": target["labels"],
                }
                image_dict = self.transforms(**image_dict)
                image = image_dict["image"]
                target["boxes"] = torch.as_tensor(
                    image_dict["bboxes"], dtype=torch.float32
                )
        else:
            path = os.path.join(self.root_dir, self.image_ids[index])
            image = self.get_image(path)
            image_dict = {
                "image": image,
                "bboxes": target["boxes"],
                "labels": target["labels"],
            }
            image = self.transforms(**image_dict)["image"]

        return image, target

    def __len__(self) -> int:
        return len(self.image_ids)

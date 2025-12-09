# src/data/dataset_detection.py
"""
COCO-style dataset loader for torchvision detection models.
Expects:
  - images/  (all images)
  - annotations.json (COCO-format)
Each item returns (image_tensor, target) where target is dict:
  boxes: FloatTensor[N,4] (x1,y1,x2,y2)
  labels: Int64Tensor[N]
  image_id: IntTensor[1]
  area: Tensor[N]
  iscrowd: Tensor[N]
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import json
import numpy as np
from pycocotools.coco import COCO

class CocoDetectionDataset(Dataset):
    def __init__(self, images_dir, ann_file, transforms=None):
        self.images_dir = Path(images_dir)
        self.ann_file = ann_file
        self.coco = COCO(ann_file)
        # list of image ids
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms or T.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = self.images_dir / img_info['file_name']
        img = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for a in anns:
            # COCO bbox: [x,y,w,h]
            x, y, w, h = a['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(a['category_id'])
            areas.append(a.get('area', w*h))
            iscrowd.append(a.get('iscrowd', 0))

        if len(boxes) == 0:
            # torchvision detection models expect at least empty tensors
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img = self.transforms(img)  # expect transforms -> Tensor

        return img, target

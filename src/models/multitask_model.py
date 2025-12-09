# src/models/multitask_model.py
import torch.nn as nn
import torchvision.models as models
from .heads import ClassificationHead, SegmentationHead

_BACKBONES = {
    "resnet18": {"fn": models.resnet18, "feat_channels": 512, "weights": models.ResNet18_Weights.IMAGENET1K_V1},
    "resnet34": {"fn": models.resnet34, "feat_channels": 512, "weights": models.ResNet34_Weights.IMAGENET1K_V1},
    "resnet50": {"fn": models.resnet50, "feat_channels": 2048, "weights": models.ResNet50_Weights.IMAGENET1K_V2},
}

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes, seg_classes=1, backbone="resnet18", pretrained=True):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"backbone must be one of {_BACKBONES.keys()}")
        info = _BACKBONES[backbone]
        # use weights only if pretrained True
        weights = info["weights"] if pretrained else None
        backbone_fn = info["fn"]
        backbone_model = backbone_fn(weights=weights)
        # remove avgpool + fc
        self.stem = nn.Sequential(*list(backbone_model.children())[:-2])
        feat_channels = info["feat_channels"]
        self.class_head = ClassificationHead(in_channels=feat_channels, num_classes=num_classes)
        self.seg_head = SegmentationHead(in_channels=feat_channels, num_classes=seg_classes)

    def forward(self, x, run_seg=True):
        feats = self.stem(x)
        cls_logits = self.class_head(feats)
        seg_logits = self.seg_head(feats) if run_seg else None
        return {'classification': cls_logits, 'segmentation': seg_logits}

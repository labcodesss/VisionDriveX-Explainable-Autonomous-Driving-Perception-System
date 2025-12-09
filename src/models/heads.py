# src/models/heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
    def forward(self, feats):
        x = self.pool(feats).view(feats.size(0), -1)
        return self.fc(x)

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv_out = nn.Conv2d(256, num_classes, 1)
    def forward(self, feats):
        x = F.interpolate(feats, scale_factor=32, mode='bilinear', align_corners=False)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv_out(x)

# src/data/dataset_classification.py
from torch.utils.data import Dataset
from PIL import Image
import os

class GTSRBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.items = []
        # expects structure: root_dir/<class_id>/*.png or .ppm
        for cls in sorted(os.listdir(root_dir)):
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path): continue
            for fn in os.listdir(cls_path):
                if fn.lower().endswith(('.png','.jpg','.jpeg','.ppm')):
                    self.items.append((os.path.join(cls_path, fn), int(cls)))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label

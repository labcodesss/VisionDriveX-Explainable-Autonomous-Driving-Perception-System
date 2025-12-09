# src/data/dataset_segmentation.py
from torch.utils.data import Dataset
from PIL import Image
import os


class SimpleSegDataset(Dataset):
    """
    Expects:
      images in images/ and masks in masks/ with same filename.
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.images_dir, fname)).convert('RGB')
        mask = Image.open(os.path.join(self.masks_dir, fname)).convert('L')
        if self.transform:
            # assume transform returns dict with image and mask if using albumentations
            out = self.transform(image=np.array(img), mask=np.array(mask))
            img = out['image']
            mask = out['mask']
        else:
            import torchvision.transforms as T
            img = T.ToTensor()(img)
            from torchvision.transforms import functional as F
            mask = F.to_tensor(mask)
        return img, mask.long().squeeze(0)

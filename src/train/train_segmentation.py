# src/train/train_segmentation.py
"""
Train script for lane segmentation using segmentation_models_pytorch (UNet/DeepLab).
Usage example (PowerShell):
py -m src.train.train_segmentation --images data/tusimple/images --masks data/tusimple/masks --epochs 10 --batch_size 4 --device cpu --checkpoint_dir checkpoints/seg --num_workers 0
"""
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

# optional libraries
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    A = None
    ToTensorV2 = None

try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None

# -------------------------
# Dataset
# -------------------------
class SimpleSegDataset(Dataset):
    """
    Expects:
      images_dir/*.png and masks_dir/*.png with matching filenames.
      Masks are single-channel (0 background, 255 foreground) or 0/1.
    Supports albumentations transforms if provided (transform expects image, mask).
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.files = sorted([p.name for p in self.images_dir.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg')])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = np.array(Image.open(self.images_dir / name).convert('RGB'))
        mask = np.array(Image.open(self.masks_dir / name).convert('L'))
        # convert mask to 0/1
        mask = (mask > 127).astype('float32')

        if self.transform is not None and A is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            # ensure mask is float tensor shape [H,W]
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float()
        else:
            # basic to-tensor fallback
            img = TF.to_tensor(Image.fromarray(img))
            mask = torch.from_numpy(mask).unsqueeze(0).float().squeeze(0)

        return img, mask

# -------------------------
# Losses & metrics
# -------------------------
def dice_loss(pred, target, smooth=1.0):
    # pred: logits (before sigmoid) or probabilities; we'll apply sigmoid outside if needed
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(1)
    denom = pred_flat.sum(1) + target_flat.sum(1)
    loss = 1 - ((2.0 * intersection + smooth) / (denom + smooth))
    return loss.mean()

def bce_dice_loss(logits, masks):
    bce = nn.BCEWithLogitsLoss()(logits, masks)
    dloss = dice_loss(logits, masks)
    return bce + dloss

def iou_score(pred_mask, true_mask, thresh=0.5, eps=1e-7):
    pred_bin = (pred_mask > thresh).float()
    inter = (pred_bin * true_mask).sum(dim=[1,2])
    union = ((pred_bin + true_mask) > 0).float().sum(dim=[1,2])
    return ((inter + eps) / (union + eps)).mean().item()

# -------------------------
# Training function
# -------------------------
def train(args):
    if smp is None:
        raise RuntimeError("segmentation_models_pytorch (smp) not found. Install it with: py -m pip install git+https://github.com/qubvel/segmentation_models.pytorch")
    device = torch.device(args.device if args.device in ("cpu","cuda") else ("cuda" if torch.cuda.is_available() else "cpu"))

    # transforms
    if A is not None:
        train_transform = A.Compose([
            A.Resize(args.height, args.width),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
    else:
        train_transform = None

    ds = SimpleSegDataset(args.images, args.masks, transform=train_transform)
    if len(ds) == 0:
        raise RuntimeError(f"No images found in {args.images}. Check path and files.")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # model (UNet) - binary segmentation (classes=1)
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=1, activation=None)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_iou = 0.0
        t0 = time.time()
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)  # shape [B,1,H,W]
            loss = bce_dice_loss(logits, masks.unsqueeze(1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += float(loss.item())
            preds = torch.sigmoid(logits).detach().cpu().squeeze(1)
            epoch_iou += iou_score(preds, masks.cpu())
        n = len(loader)
        avg_loss = epoch_loss / n
        avg_iou = epoch_iou / n
        epoch_time = time.time() - t0
        print(f"[Epoch {epoch}/{args.epochs}] loss {avg_loss:.4f} iou {avg_iou:.4f} time {epoch_time:.1f}s")
        ckpt = Path(args.checkpoint_dir) / f"seg_epoch{epoch}.pth"
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, ckpt)
        print("Saved:", ckpt)

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="path to images dir")
    p.add_argument("--masks", required=True, help="path to masks dir (same filenames)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--checkpoint_dir", default="checkpoints/seg")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--width", type=int, default=512, help="training width (resize)")
    p.add_argument("--height", type=int, default=256, help="training height (resize)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)

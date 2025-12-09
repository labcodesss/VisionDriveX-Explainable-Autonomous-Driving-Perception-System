# src/train/evaluate_classifier.py
"""
Evaluate a saved classifier checkpoint on a directory-structured dataset:
data/gtsrb/val/<class_id>/*.png

Outputs:
 - prints overall accuracy
 - saves confusion matrix as CSV to outputs/confusion_matrix.csv
Usage (PowerShell):
py -m src.train.evaluate_classifier --data data/gtsrb/val --checkpoint checkpoints\classifier_resnet18_epoch3.pth --batch_size 8 --img_size 128 --backbone resnet18 --device cpu
"""
import argparse
import torch
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
from PIL import Image

class FolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.items = []
        for cls in sorted([p.name for p in self.root.iterdir() if p.is_dir()]):
            cls_path = self.root/cls
            for fn in cls_path.iterdir():
                if fn.suffix.lower() in ('.png','.jpg','.jpeg','.ppm'):
                    self.items.append((str(fn), int(cls)))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        path,label = self.items[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, path

def load_checkpoint(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    # handle state dict wrapped in dict
    if "model_state" in sd:
        model.load_state_dict(sd["model_state"])
    else:
        model.load_state_dict(sd)
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = FolderDataset(args.data, transform=transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # import model lazily to avoid overhead if not installed
    from src.models.multitask_model import MultiTaskModel
    device = torch.device(args.device if args.device in ("cpu","cuda") else "cpu")
    model = MultiTaskModel(num_classes=43, backbone=args.backbone, pretrained=False)
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    paths = []

    with torch.no_grad():
        for imgs, labels, fpaths in loader:
            imgs = imgs.to(device)
            feats = model.stem(imgs)
            logits = model.class_head(feats)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            paths.extend(list(fpaths))

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    overall_acc = (y_true == y_pred).mean()
    print(f"Overall accuracy: {overall_acc:.4f}  ({len(y_true)} samples)")

    # confusion matrix
    num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t,p in zip(y_true, y_pred):
        cm[t,p] += 1

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    cm_df = pd.DataFrame(cm)
    cm_csv = out_dir/"confusion_matrix.csv"
    cm_df.to_csv(cm_csv, index=True, header=True)
    print("Saved confusion matrix to", cm_csv)

    # per-class accuracy
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-12)
    per_df = pd.DataFrame({"class": np.arange(len(per_class_acc)), "accuracy": per_class_acc})
    per_df.to_csv(out_dir/"per_class_accuracy.csv", index=False)
    print("Saved per-class accuracy to outputs/per_class_accuracy.csv")

if __name__ == "__main__":
    main()

# src/train/train_single_task.py
"""
Updated train script (select backbone, limit CPU threads, small defaults for CPU).
Usage examples (PowerShell):

# CPU, safe defaults
py -m src.train.train_single_task --task classification --data data/gtsrb/train --num_classes 43 --epochs 3 --batch_size 1 --img_size 128 --backbone resnet18 --device cpu

# If you have GPU:
py -m src.train.train_single_task --task classification --data data/gtsrb/train --num_classes 43 --epochs 5 --batch_size 8 --img_size 224 --backbone resnet50 --device cuda
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import time
import os

from src.models.multitask_model import MultiTaskModel
from src.data.dataset_classification import GTSRBDataset

def train_classification(args):
    # limit CPU threads for reproducible speed (helps Windows)
    if args.device == "cpu":
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = GTSRBDataset(args.data, transform=transform)
    if len(ds) == 0:
        raise RuntimeError(f"Dataset at {args.data} has 0 images. Check path and contents.")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = MultiTaskModel(num_classes=args.num_classes, backbone=args.backbone, pretrained=args.pretrained)
    device = torch.device(args.device if args.device in ("cpu","cuda") else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        t0 = time.time()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # Only run classification to save memory (multitask later)
            feats = model.stem(imgs)
            out = model.class_head(feats)

            loss = criterion(out, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            loss_sum += float(loss.item())
            preds = out.argmax(dim=1)
            total += labels.size(0)
            correct += int((preds == labels).sum().item())
        epoch_time = time.time() - t0
        acc = correct/total if total>0 else 0.0
        print(f"[Epoch {epoch}/{args.epochs}] loss {loss_sum/len(loader):.4f} acc {acc:.4f} time {epoch_time:.1f}s")
        ckpt_path = Path(args.checkpoint_dir)/f"classifier_{args.backbone}_epoch{epoch}.pth"
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, ckpt_path)
        print("Saved checkpoint:", ckpt_path)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="classification", choices=["classification","segmentation"])
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--num_classes", type=int, default=43)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=1)   # safer default for CPU
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","resnet34","resnet50"])
    p.add_argument("--pretrained", action="store_true", help="load imagenet weights for backbone")
    return p.parse_args()

def main():
    args = parse_args()
    if args.task == "classification":
        train_classification(args)
    else:
        raise NotImplementedError("Segmentation training not in this script.")

if __name__ == "__main__":
    main()

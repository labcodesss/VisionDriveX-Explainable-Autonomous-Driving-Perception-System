# src/train/train_detection_coco.py
"""
Train Faster R-CNN on COCO-format dataset (stop sign subset).
Usage example (PowerShell):
py -m src.train.train_detection_coco --images data/custom_stop/images --ann data/custom_stop/annotations.json --num_classes 2 --epochs 5 --batch_size 2 --device cpu --checkpoint_dir checkpoints/det
"""
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from pathlib import Path
from src.data.dataset_detection import CocoDetectionDataset

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes, pretrained_backbone=True):
    # Faster R-CNN with ResNet50-FPN backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
    # If you want to load pretrained backbone weights only (optional), you can change above.
    return model

def train(args):
    device = torch.device(args.device if args.device in ("cpu","cuda") else ("cuda" if torch.cuda.is_available() else "cpu"))

    # simple transforms: resize to shorter side 800 via torchvision transforms
    transforms = T.Compose([T.ToTensor()])

    dataset = CocoDetectionDataset(args.images, args.ann, transforms=transforms)
    if len(dataset) == 0:
        raise RuntimeError("Dataset appears empty. Check images/ and annotations.json")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    model = get_model(args.num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(v for v in loss_dict.values())
            optimizer.zero_grad(); losses.backward(); optimizer.step()
            epoch_loss += float(losses.item())
        avg_loss = epoch_loss / len(loader)
        print(f"[Epoch {epoch}/{args.epochs}] loss {avg_loss:.4f}")
        ckpt = Path(args.checkpoint_dir)/f"fasterrcnn_epoch{epoch}.pth"
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, ckpt)
        print("Saved:", ckpt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="images dir")
    parser.add_argument("--ann", required=True, help="COCO annotations json")
    parser.add_argument("--num_classes", type=int, default=2)  # background + stop
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--checkpoint_dir", default="checkpoints/det")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    train(args)

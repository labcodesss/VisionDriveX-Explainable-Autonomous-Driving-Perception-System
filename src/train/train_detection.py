# src/train/train_detection.py
"""
Minimal training loop for torchvision Faster R-CNN.
Assumes dataset implements __getitem__ returning (image, target) where target is a dict:
{'boxes': Tensor[N,4], 'labels': Tensor[N]}
"""
import torch
import torchvision
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

# User should implement CustomCOCODataset that returns image, target
# For brevity, user will plug their dataset here.
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=10):
        self.n = n
    def __len__(self): return self.n
    def __getitem__(self, idx):
        import torchvision.transforms as T
        from PIL import Image
        img = Image.new('RGB', (640,480), color=(128,128,128))
        img = T.ToTensor()(img)
        target = {'boxes': torch.tensor([[50,50,200,200]], dtype=torch.float32),
                  'labels': torch.tensor([1], dtype=torch.int64)}
        return img, target

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=args.num_classes)
    model.to(device)
    dataset = DummyDataset(50)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    model.train()
    for epoch in range(args.epochs):
        for images, targets in loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad(); losses.backward(); optimizer.step()
        print(f"Epoch {epoch} done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=2)  # background + stop sign label
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train(args)

# scripts/eval_classifier_cm.py
import torch, json, os
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import itertools

from src.models.multitask_model import MultiTaskModel

def load_image(p, img_size=224):
    img = Image.open(p).convert("RGB")
    tf = transforms.Compose([transforms.Resize((img_size,img_size)),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return tf(img).unsqueeze(0)

def main(checkpoint, data_root, labels_json="data/gtsrb/labels.json", img_size=224, device="cpu"):
    device = torch.device(device)
    ck = torch.load(checkpoint, map_location=device)
    model = MultiTaskModel(num_classes=43, backbone="resnet18", pretrained=False)
    sd = ck.get("model_state", ck)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    labels_map = json.load(open(labels_json)) if Path(labels_json).exists() else None

    # assume val folder contains subfolders per class
    root = Path(data_root)
    y_true, y_pred = [], []
    for cls_idx, cls_name in enumerate(sorted([p.name for p in root.iterdir() if p.is_dir()])):
        cls_dir = root / cls_name
        for imgf in cls_dir.iterdir():
            if imgf.suffix.lower() not in (".png",".jpg",".jpeg"): continue
            x = load_image(imgf, img_size=img_size).to(device)
            with torch.no_grad():
                logits = model(x, run_seg=False)["classification"]
                pred = int(torch.softmax(logits, dim=1)[0].argmax().item())
            y_true.append(cls_idx)
            y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("outputs/confusion_matrix.png")
    print("Saved outputs/confusion_matrix.png")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)  # e.g. data/gtsrb/val
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    main(args.checkpoint, args.data_root, img_size=args.img_size, device=args.device)

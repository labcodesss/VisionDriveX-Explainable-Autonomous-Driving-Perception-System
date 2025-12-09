# src/inference/run_demo.py
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import argparse
from src.models.multitask_model import MultiTaskModel
import matplotlib.pyplot as plt
import numpy as np

def visualize(img_tensor, cls_logits, seg_logits, out_path):
    img = np.transpose(img_tensor.cpu().numpy(), (1,2,0))
    img = (img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
    img = np.clip(img, 0, 1)
    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1); plt.imshow(img); plt.title("Image"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(torch.softmax(cls_logits, dim=0).cpu().numpy()); plt.title("Class softmax"); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(torch.sigmoid(seg_logits[0]).detach().cpu().numpy()); plt.title("Seg mask"); plt.axis('off')
    plt.savefig(out_path); print("Saved demo to", out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/demo.png")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(num_classes=43)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device); model.eval()
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    img = Image.open(args.image).convert('RGB')
    t = transform(img).unsqueeze(0).to(device)
    out = model(t)
    cls_logits = out['classification'][0].detach()
    seg_logits = out['segmentation'][0].detach()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    visualize(t[0], cls_logits, seg_logits, args.out)

if __name__ == "__main__":
    main()

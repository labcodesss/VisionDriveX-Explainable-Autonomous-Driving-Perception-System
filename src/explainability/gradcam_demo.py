# src/explainability/gradcam_demo.py
"""
Grad-CAM demo using Captum for the classifier.
Usage (PowerShell):
py src\explainability\gradcam_demo.py --image data/gtsrb/train/0/00_001.png --weights checkpoints\classifier_resnet18_epoch3.pth --out outputs/gradcam.png --img_size 224 --backbone resnet18 --device cpu
"""
import argparse
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
from pathlib import Path
from src.models.multitask_model import MultiTaskModel

# Small wrapper so Captum sees a model that returns logits tensor
class ClassifierWrapper(nn.Module):
    def __init__(self, multitask_model):
        super().__init__()
        self.model = multitask_model
    def forward(self, x):
        # return classification logits [B, num_classes]
        return self.model(x, run_seg=False)['classification']

def load_checkpoint_into_model(model, ckpt_path, map_location='cpu'):
    sd = torch.load(ckpt_path, map_location=map_location)
    if "model_state" in sd:
        model.load_state_dict(sd["model_state"])
    else:
        model.load_state_dict(sd)
    return model

def preprocess_image(img_path, img_size):
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tensor = transform(img).unsqueeze(0)
    return img, tensor

def overlay_attr_on_image(orig_pil, attr_np, colormap=plt.cm.jet, alpha=0.5):
    """
    orig_pil: PIL.Image (RGB)
    attr_np: numpy array, can be (H,W) or (C,H,W) or (1,H,W). We collapse to (H,W).
    Returns blended PIL image.
    """
    # if multi-channel attribution (C,H,W), collapse channels to single map
    if attr_np.ndim == 3:
        # common case: (C,H,W) or (1,H,W)
        # if single-channel first dim, squeeze; else average channels
        if attr_np.shape[0] == 1:
            attr_np = attr_np.squeeze(0)
        else:
            attr_np = attr_np.mean(axis=0)

    # now attr_np is (H,W)
    # ensure float and normalize to [0,1]
    attr = attr_np.astype('float32')
    attr -= attr.min()
    if attr.max() > 0:
        attr /= attr.max()
    else:
        attr = attr * 0.0

    # apply colormap -> returns H x W x 4 (RGBA floats 0..1)
    heatmap = colormap(attr)
    heatmap_rgb = (heatmap[:, :, :3] * 255).astype('uint8')  # H x W x 3 uint8

    # create PIL images
    heat_pil = Image.fromarray(heatmap_rgb).convert("RGBA")
    base = orig_pil.convert("RGBA")

    # resize heatmap to match original if needed
    if heat_pil.size != base.size:
        heat_pil = heat_pil.resize(base.size, resample=Image.BILINEAR)

    # blend and return
    blended = Image.blend(base, heat_pil, alpha=alpha)
    return blended


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default="outputs/gradcam.png")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--num_classes", type=int, default=43)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    device = torch.device(args.device if args.device in ("cpu","cuda") else "cpu")

    # build model (same API as training)
    model = MultiTaskModel(num_classes=args.num_classes, backbone=args.backbone, pretrained=False)
    model = load_checkpoint_into_model(model, args.weights, map_location=device)
    model.eval().to(device)

    # wrapper for Captum that returns classification logits
    wrapper = ClassifierWrapper(model).to(device)
    wrapper.eval()

    orig_pil, tensor = preprocess_image(args.image, args.img_size)
    tensor = tensor.to(device)

    # forward: predict class
    with torch.no_grad():
        logits = wrapper(tensor)
        probs = torch.softmax(logits, dim=1)
        pred = int(probs[0].argmax().item())
    print(f"Predicted class: {pred}  (top prob {float(probs[0,pred]):.4f})")

    # Choose a target layer: find last Conv2d inside model.stem
    target_layer = None
    for m in reversed(list(model.stem.named_modules())):
        if isinstance(m[1], nn.Conv2d):
            target_layer = m[1]
            break
    if target_layer is None:
        # fallback: last module
        target_layer = model.stem[-1]
    print("Using target layer:", target_layer)

    # Captum LayerGradCam
    lgc = LayerGradCam(wrapper, target_layer)

    # attribute for predicted class
    attr = lgc.attribute(tensor, target=pred)
    # upsample to input size
    up_attr = LayerAttribution.interpolate(attr, (args.img_size, args.img_size))[0]
    up_attr = up_attr.detach().cpu().numpy()
    # normalize to [0,1]
    up_attr = up_attr - up_attr.min()
    if up_attr.max() > 0:
        up_attr = up_attr / up_attr.max()
    else:
        up_attr = np.zeros_like(up_attr)

    # overlay and save
    blended = overlay_attr_on_image(orig_pil, up_attr)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    blended.save(args.out)
    print("Saved Grad-CAM overlay to", args.out)

if __name__ == "__main__":
    main()

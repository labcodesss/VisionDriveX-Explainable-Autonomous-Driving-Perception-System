# src/inference/infer_det_safe.py
"""
Safe detection inference loader:
Loads only matching keys from checkpoint into the defined model so we don't crash
when backbone/head shapes differ between training & inference builds.

Usage (PowerShell):
py src\inference\infer_det_safe.py --image data/custom_stop/images/img_002.png --weights checkpoints/det/fasterrcnn_epoch1.pth --out outputs/det_safe.png --num_classes 2
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import torchvision

def get_model_mobilenet(num_classes):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
    # replace classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def filter_and_load(model, ckpt_path, map_location='cpu'):
    ck = torch.load(ckpt_path, map_location=map_location)
    sd = ck.get("model_state", ck)  # support both formats

    model_sd = model.state_dict()
    load_sd = {}
    skipped = []
    for k, v in sd.items():
        if k in model_sd and v.size() == model_sd[k].size():
            load_sd[k] = v
        else:
            skipped.append((k, v.size() if isinstance(v, torch.Tensor) else None))

    # update model state dict then load
    model_sd.update(load_sd)
    model.load_state_dict(model_sd)
    print(f"Loaded {len(load_sd)} params into model; skipped {len(skipped)} params (mismatched or missing).")
    if len(skipped) > 0:
        print("Example skipped keys (first 10):")
        for t in skipped[:10]:
            print("  ", t[0], "->", t[1])
    return model

def visualize_and_save(img_pil, pred, out_path, score_thresh=0.3):
    draw = ImageDraw.Draw(img_pil)
    for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = [int(i) for i in box.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1, y1-10), f"{int(label.item())}:{score:.2f}", fill="yellow")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img_pil.save(out_path)
    print("Saved:", out_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default="outputs/det_safe.png")
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--score_thresh", type=float, default=0.3)
    args = p.parse_args()

    device = torch.device("cpu")
    img = Image.open(args.image).convert("RGB")
    t = transforms.ToTensor()(img)

    # build the model you want to run inference with (here: mobilenet backbone)
    model = get_model_mobilenet(num_classes=args.num_classes)
    model = filter_and_load(model, args.weights, map_location='cpu')
    model.eval()

    with torch.no_grad():
        preds = model([t])[0]

    visualize_and_save(img, preds, args.out, score_thresh=args.score_thresh)

if __name__ == "__main__":
    main()

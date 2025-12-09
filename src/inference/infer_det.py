import torch, torchvision
from PIL import Image, ImageDraw
from torchvision import transforms
from pathlib import Path
import argparse

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, weights_backbone=None
    )
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_channels=model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=num_classes
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", default="outputs/det_out.png")
    args = parser.parse_args()

    # Load image
    img = Image.open(args.image).convert("RGB")
    transform = transforms.ToTensor()
    t = transform(img).unsqueeze(0)

    # Load model
    model = get_model(num_classes=2)
    sd = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(sd["model_state"])
    model.eval()

    # Inference
    with torch.no_grad():
        preds = model([t[0]])[0]

    draw = ImageDraw.Draw(img)
    for box, score, label in zip(preds['boxes'], preds['scores'], preds['labels']):
        if score < 0.3:
            continue
        x1, y1, x2, y2 = [int(i) for i in box.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1, y1 - 10), f"{label}:{score:.2f}", fill="yellow")

    Path("outputs").mkdir(exist_ok=True)
    img.save(args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()

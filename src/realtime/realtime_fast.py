# src/realtime/realtime_fast.py
"""
Fast realtime demo (simple, robust).
- Shows webcam, runs classifier + optional detector.
- Loads human-readable labels from data/gtsrb/labels.json when present.
- Press ESC to exit cleanly, or CTRL+C in terminal.
Usage:
py -m src.realtime.realtime_fast --device cpu --cam 0 --cls_weights checkpoints/classifier_resnet18_epoch3.pth --det_weights checkpoints/det/fasterrcnn_epoch3.pth --img_size 128 --num_classes 43
"""
import time
import json
from pathlib import Path
import sys

import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision

# import multitask model from your repo
from src.models.multitask_model import MultiTaskModel

# path to labels.json (human readable mapping)
LABELS_PATH = Path("data/gtsrb/labels.json")


def load_labels(path=LABELS_PATH):
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in d.items()}
    except Exception as e:
        print("Failed to load labels.json:", e)
        return None


def build_detector_mobilenet(num_classes, device):
    """
    Build a lightweight Faster R-CNN with MobileNetV3 backbone.
    """
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, weights_backbone=None
    )
    # replace box predictor for correct num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    model.to(device).eval()
    return model


def load_classifier(weights_path, device, backbone="resnet18", num_classes=43):
    model = MultiTaskModel(num_classes=num_classes, backbone=backbone, pretrained=False)
    ck = torch.load(weights_path, map_location=device)
    sd = ck.get("model_state", ck)
    # partial load to avoid errors when some keys mismatch
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def safe_load_detector(model, weights_path, device):
    try:
        ck = torch.load(weights_path, map_location=device)
        sd = ck.get("model_state", ck)
        model.load_state_dict(sd, strict=False)
        print("Detector weights loaded (partial allowed).")
    except Exception as e:
        print("Detector load warning:", e)
    return model


def preprocess_frame_for_classifier(frame_bgr, img_size, device):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    t = transform(pil).unsqueeze(0).to(device)
    return t, pil


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--cls_weights", type=str, required=True)
    p.add_argument("--det_weights", type=str, default=None)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=43)
    p.add_argument("--max_time", type=int, default=180, help="auto-stop after N seconds")
    args = p.parse_args()

    device = torch.device(args.device if args.device in ("cpu", "cuda") else "cpu")
    print("Device:", device)

    # load labels map (human-readable)
    labels_map = load_labels()
    if labels_map:
        print("Loaded labels mapping (sample):")
        for k in sorted(list(labels_map.keys())[:10]):
            print(k, "->", labels_map[k])
    else:
        print("No labels.json found at", LABELS_PATH, "- classifier will show numeric IDs.")

    # load classifier
    print("Loading classifier...")
    cls_model = load_classifier(args.cls_weights, device, backbone="resnet18", num_classes=args.num_classes)

    # load detector (optional)
    det_model = None
    if args.det_weights:
        print("Building detector (mobilenet backbone)...")
        det_model = build_detector_mobilenet(num_classes=2, device=device)
        det_model = safe_load_detector(det_model, args.det_weights, device)

    # open webcam
    cap = cv2.VideoCapture(args.cam)
    if not cap or not cap.isOpened():
        print("ERROR: Cannot open camera index", args.cam)
        return

    print("Realtime demo started. Press ESC in the window to exit, or CTRL+C in terminal.")
    start = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed, exiting.")
                break
            frame_count += 1

            # classification
            inp, pil = preprocess_frame_for_classifier(frame, args.img_size, device)
            with torch.no_grad():
                feats = cls_model.stem(inp)
                logits = cls_model.class_head(feats)  # logits shape [1, C]
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred = int(probs.argmax())
                conf = float(probs[pred])

            # overlay classification text: human label if available
            if labels_map and pred in labels_map:
                main_label = labels_map[pred]
                txt = f"{main_label} ({pred}) {conf:.2f}"
            else:
                txt = f"CLS: {pred} ({conf:.2f})"
            cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            # show top-3 predictions under main text
            try:
                topk = np.argsort(-probs)[:3]
                y0 = 55
                for i, idx in enumerate(topk):
                    name = labels_map.get(int(idx)) if labels_map else str(int(idx))
                    s = f"{i+1}. {name} {probs[int(idx)]:.2f}"
                    cv2.putText(frame, s, (10, y0 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            except Exception:
                pass

            # detector (if available)
            if det_model is not None:
                try:
                    with torch.no_grad():
                        det_in = transforms.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)
                        outputs = det_model([det_in])[0]
                    boxes = outputs.get("boxes", [])
                    scores = outputs.get("scores", torch.tensor([])).cpu().numpy() if "scores" in outputs else []
                    labels = outputs.get("labels", torch.tensor([])).cpu().numpy() if "labels" in outputs else []
                    # draw top boxes above threshold
                    for i, box in enumerate(boxes):
                        if i >= len(scores):
                            break
                        if scores[i] < 0.35:
                            continue
                        # box might be on CPU or CUDA; convert safely
                        box_arr = box.cpu().numpy() if isinstance(box, torch.Tensor) else np.array(box)
                        x1, y1, x2, y2 = [int(x) for x in box_arr]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label_text = "stop" if len(labels) > i else "obj"
                        cv2.putText(frame, f"{label_text} {scores[i]:.2f}", (x1, max(0, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                except Exception as e:
                    # do not crash on detector error
                    print("Detector inference error:", e)

            # compute fps
            elapsed = max(time.time() - start, 1e-6)
            fps = frame_count / elapsed
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # show image
            cv2.imshow("Fast AV Perception Demo", frame)

            # ESC to exit (27)
            key = cv2.waitKey(1)
            if key == 27:
                print("ESC pressed. Exiting.")
                break

            # auto-stop
            if (time.time() - start) > args.max_time:
                print("Auto-stop reached. Exiting.")
                break

    except KeyboardInterrupt:
        print("KeyboardInterrupt received, exiting gracefully...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Demo finished.")


if __name__ == "__main__":
    main()

# src/realtime/realtime_demo_headless.py
"""
Headless realtime demo: process webcam and save annotated frames instead of showing a window.
Usage:
py -m src.realtime.realtime_demo_headless --device cpu --cam 0 --cls_weights checkpoints/classifier_resnet18_epoch3.pth --det_weights checkpoints/det/fasterrcnn_epoch3.pth --seg_weights checkpoints/seg/seg_epoch3.pth --img_size 128 --num_classes 43 --save_every 10
"""
import argparse, time
from pathlib import Path
import cv2, torch, torchvision
from PIL import Image
import torchvision.transforms as T
from src.models.multitask_model import MultiTaskModel

# minimal helpers reused from realtime_demo
def load_classifier(weights_path, device, backbone="resnet18", num_classes=43):
    model = MultiTaskModel(num_classes=num_classes, backbone=backbone, pretrained=False)
    sd = torch.load(weights_path, map_location=device)
    sd = sd.get("model_state", sd)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model

def build_detector(num_classes, device):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.to(device).eval()
    return model

def preprocess_for_classifier(frame_bgr, img_size=224, device="cpu"):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    transform = T.Compose([T.Resize((img_size,img_size)), T.ToTensor(),
                           T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    t = transform(pil).unsqueeze(0).to(device)
    return t, pil

def overlay_mask(frame, mask_prob, alpha=0.4):
    heat = (mask_prob*255).astype('uint8')
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 1-alpha, heat_color, alpha, 0)
    return overlay

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--cls_weights", required=True)
    p.add_argument("--det_weights", default=None)
    p.add_argument("--seg_weights", default=None)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=43)
    p.add_argument("--save_every", type=int, default=20, help="save annotated frame every N frames")
    args = p.parse_args()

    device = torch.device(args.device)
    print("Device:", device)

    cls_model = load_classifier(args.cls_weights, device, backbone="resnet18", num_classes=args.num_classes)
    det_model = None
    if args.det_weights:
        det_model = build_detector(num_classes=2, device=device)
        try:
            sd = torch.load(args.det_weights, map_location=device)
            det_model.load_state_dict(sd.get("model_state", sd), strict=False)
        except Exception as e:
            print("Detector load warning:", e)
    seg_model = None
    try:
        import segmentation_models_pytorch as smp
        if args.seg_weights:
            seg_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=1, activation=None)
            sd = torch.load(args.seg_weights, map_location=device)
            seg_model.load_state_dict(sd.get("model_state", sd), strict=False)
            seg_model.to(device).eval()
    except Exception:
        seg_model = None

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera", args.cam)

    out_dir = Path("outputs/realtime_headless")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        vis = frame.copy()

        # classifier
        inp, pil = preprocess_for_classifier(frame, img_size=args.img_size, device=device)
        with torch.no_grad():
            feats = cls_model.stem(inp)
            logits = cls_model.class_head(feats)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(probs.argmax())
        cv2.putText(vis, f"CLS:{pred} {probs[pred]:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # detector
        if det_model is not None:
            with torch.no_grad():
                d_inp = T.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)
                pred_det = det_model([d_inp])[0]
            boxes = pred_det.get("boxes", [])
            scores = pred_det.get("scores", [])
            for i,box in enumerate(boxes):
                if scores[i] < 0.3: continue
                x1,y1,x2,y2 = [int(v) for v in box]
                cv2.rectangle(vis, (x1,y1),(x2,y2),(0,255,0),2)

        # segmentation
        if seg_model is not None:
            with torch.no_grad():
                seg_in = T.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                logits = seg_model(seg_in)
                prob = torch.sigmoid(logits)[0,0].cpu().numpy()
                prob = cv2.resize(prob, (vis.shape[1], vis.shape[0]))
                vis = overlay_mask(vis, prob, alpha=0.4)

        # save every N frames
        if frame_idx % args.save_every == 0:
            out_path = out_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(out_path), vis)
            print("Saved", out_path)
        # stop after many frames (or Ctrl+C)
        if frame_idx > 10000:
            break

    cap.release()
    print("Done. Saved frames to", out_dir)
    print("Elapsed:", time.time()-start)

if __name__ == "__main__":
    main()

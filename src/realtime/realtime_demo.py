# src/realtime/realtime_demo.py
"""
Real-time demo: webcam -> classifier + detector + segmentation -> overlays.
Controls:
  q        : quit
  g        : compute & save Grad-CAM for current frame (saves outputs/gradcam_live.png)
  p        : pause/unpause processing
  c        : capture/save current frame to outputs/capture_<n>.png
Usage (PowerShell):
py -m src.realtime.realtime_demo --device cpu --cam 0 --cls_weights checkpoints/classifier_resnet18_epoch3.pth --det_weights checkpoints/det/fasterrcnn_epoch3.pth --seg_weights checkpoints/seg/seg_epoch3.pth
Notes:
 - Grad-CAM is computed on-demand (press 'g') because it's relatively expensive.
 - If you run on GPU, add --device cuda
"""
import argparse, time, os
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
import torchvision
from src.models.multitask_model import MultiTaskModel

# Optional modules (segmentation_models_pytorch)
try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None

# Captum (for Grad-CAM)
try:
    from captum.attr import LayerGradCam, LayerAttribution
    import torch.nn as nn
except Exception:
    LayerGradCam = None

# -------------------------
# Utilities
# -------------------------
def load_classifier(weights_path, device, backbone="resnet18", num_classes=43):
    model = MultiTaskModel(num_classes=num_classes, backbone=backbone, pretrained=False)
    sd = torch.load(weights_path, map_location=device)
    sd = sd.get("model_state", sd)
    # try partial load
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model

def build_detector(num_classes, device):
    # lightweight detector used in training script was mobilenet-based
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
    # replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.to(device).eval()
    return model

def load_segmentation(weights_path, device):
    if smp is None:
        return None
    # we used smp.Unet(resnet34) in training script
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=1, activation=None)
    ck = torch.load(weights_path, map_location=device)
    model.load_state_dict(ck.get("model_state", ck), strict=False)
    model.to(device).eval()
    return model

def preprocess_for_classifier(frame_bgr, img_size=224, device="cpu"):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    transform = T.Compose([T.Resize((img_size,img_size)), T.ToTensor(),
                           T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    t = transform(pil).unsqueeze(0).to(device)
    return t, pil

def overlay_boxes(frame, boxes, labels=None, scores=None, color=(0,255,0)):
    for i,box in enumerate(boxes):
        x1,y1,x2,y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        caption = ""
        if labels is not None:
            caption += str(int(labels[i].item()))
        if scores is not None:
            caption += f" {scores[i]:.2f}"
        if caption:
            cv2.putText(frame, caption, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    return frame

def overlay_mask(frame, mask_prob, alpha=0.4, color=(0,128,255)):
    # mask_prob: HxW numpy float 0..1, same size as frame
    heat = (mask_prob*255).astype('uint8')
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 1-alpha, heat_color, alpha, 0)
    return overlay

# -------------------------
# Grad-CAM helper: compute for a single frame
# -------------------------
def compute_and_save_gradcam(classifier_model, target_layer, input_tensor, orig_pil, out_path):
    if LayerGradCam is None:
        print("Captum not installed; cannot compute Grad-CAM.")
        return False
    wrapper = torch.nn.Sequential(classifier_model)  # classifier_model returns dict when using run_seg=False; adjust wrapper
    # our MultiTaskModel.forward accepts run_seg arg; create lightweight wrapper:
    class W(torch.nn.Module):
        def __init__(self, m): 
            super().__init__(); self.m = m
        def forward(self, x): 
            return self.m(x, run_seg=False)["classification"]
    w = W(classifier_model).eval()
    lgc = LayerGradCam(w, target_layer)
    attr = lgc.attribute(input_tensor, target=int(torch.argmax(w(input_tensor), dim=1).item()))
    up = LayerAttribution.interpolate(attr, (input_tensor.shape[-2], input_tensor.shape[-1]))[0].detach().cpu().numpy()
    # collapse channels if necessary
    if up.ndim == 3:
        if up.shape[0] == 1: up = up.squeeze(0)
        else: up = up.mean(axis=0)
    # normalize and overlay using PIL
    up = (up - up.min())
    up = up / (up.max()+1e-8)
    heatmap = (up*255).astype('uint8')
    heat_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    heat_pil = Image.fromarray(heat_color).resize(orig_pil.size)
    blended = Image.blend(orig_pil.convert("RGBA"), heat_pil.convert("RGBA"), alpha=0.5)
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    blended.save(out_path)
    return True

# -------------------------
# Main loop
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--cls_weights", type=str, required=True)
    p.add_argument("--det_weights", type=str, required=False, default=None)
    p.add_argument("--seg_weights", type=str, required=False, default=None)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_classes", type=int, default=43)
    args = p.parse_args()

    device = torch.device(args.device)
    print("Device:", device)

    # load models (best-effort, partial loads allowed)
    print("Loading classifier...")
    cls_model = load_classifier(args.cls_weights, device, backbone="resnet18", num_classes=args.num_classes)

    det_model = None
    if args.det_weights:
        print("Loading detector (mobilenet backbone) ...")
        det_model = build_detector(num_classes=2, device=device)
        try:
            sd = torch.load(args.det_weights, map_location=device)
            det_model.load_state_dict(sd.get("model_state", sd), strict=False)
        except Exception as e:
            print("Warning: detector weights load:", e)

    seg_model = None
    if args.seg_weights:
        print("Loading segmentation model...")
        seg_model = load_segmentation(args.seg_weights, device)

    # find a conv layer for Grad-CAM in classifier (last Conv2d in stem)
    target_layer = None
    for name, m in reversed(list(cls_model.stem.named_modules())):
        if isinstance(m, torch.nn.Conv2d):
            target_layer = m
            break

    print("Target layer for Grad-CAM:", target_layer)

    # open webcam
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera", args.cam)

    paused = False
    frame_id = 0
    capture_cnt = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed")
            break
        frame_id += 1
        vis = frame.copy()

        if not paused:
            # classifier
            inp, pil = preprocess_for_classifier(frame, img_size=args.img_size, device=device)
            with torch.no_grad():
                feats = cls_model.stem(inp)
                logits = cls_model.class_head(feats)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred = int(probs.argmax())

            # show predicted class on frame
            cv2.putText(vis, f"CLS: {pred} ({probs[pred]:.2f})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            # detector (if loaded)
            if det_model is not None:
                with torch.no_grad():
                    d_inp = T.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)
                    pred_det = det_model([d_inp])[0]
                boxes = pred_det.get("boxes", []).cpu().detach().numpy() if len(pred_det.get("boxes", []))>0 else []
                scores = pred_det.get("scores", torch.tensor([])).cpu().numpy() if "scores" in pred_det else []
                labels = pred_det.get("labels", torch.tensor([])).cpu().numpy() if "labels" in pred_det else []
                # overlay top detections
                for i,box in enumerate(boxes):
                    if i>=10: break
                    if scores[i] < 0.3: continue
                    x1,y1,x2,y2 = [int(v) for v in box]
                    cv2.rectangle(vis, (x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(vis, f"{scores[i]:.2f}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

            # segmentation (if loaded)
            if seg_model is not None:
                with torch.no_grad():
                    seg_in = T.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                    logits = seg_model(seg_in)
                    prob = torch.sigmoid(logits)[0,0].cpu().numpy()
                    prob = cv2.resize(prob, (vis.shape[1], vis.shape[0]))
                    vis = overlay_mask(vis, prob, alpha=0.45)

        # compute FPS
        fps = frame_id / (time.time()-t0 + 1e-6)
        cv2.putText(vis, f"FPS:{fps:.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.imshow("AV Perception Demo", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            paused = not paused
        if key == ord('c'):
            capture_cnt += 1
            Path("outputs").mkdir(exist_ok=True)
            cv2.imwrite(f"outputs/capture_{capture_cnt:03d}.png", frame)
            print("Saved capture", capture_cnt)
        if key == ord('g'):
            # compute grad-cam for current frame and save
            print("Computing Grad-CAM (this may be slow)...")
            inp_for_gc, pil_for_gc = preprocess_for_classifier(frame, img_size=args.img_size, device=device)
            if target_layer is not None and LayerGradCam is not None:
                out_path = f"outputs/gradcam_live_{int(time.time())}.png"
                success = compute_and_save_gradcam(cls_model, target_layer, inp_for_gc, pil_for_gc, out_path)
                if success:
                    print("Saved Grad-CAM:", out_path)
            else:
                print("Grad-CAM unavailable (captum missing or target layer missing).")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    main()

VisionDriveX — Explainable Autonomous Driving Perception System 🚗🧠

VisionDriveX is a multi-task autonomous driving perception system that performs traffic-sign classification, stop-sign detection, and lane segmentation, with explainable AI (Grad-CAM) to visualize model attention.
Designed for research, demo presentations, and real-time AV perception prototypes.

🔥 Features

Traffic Sign Classification (GTSRB — 43 classes)
Stop-Sign Detection using Faster R-CNN
Lane Segmentation (binary lane mask)
Real-Time Pipeline (webcam input, <code>ESC</code> to exit)
Explainability using Grad-CAM overlays
Confusion Matrix + Evaluation Scripts
Modular folder structure for training, inference, and deployment

⚙️ Installation
1️⃣ Create Virtual Environment
py -m venv .venv
.venv\Scripts\Activate.ps1

2️⃣ Install Packages
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

🧠 GTSRB Human Labels File

Generate class-name mapping:
py scripts\make_labels.py

Creates:
data/gtsrb/labels.json

🎥 Real-Time Demo (Webcam)

Run the fast perception pipeline:

py -m src.realtime.realtime_fast --device cpu --cam 0 ^
  --cls_weights checkpoints/classifier_resnet18_epoch3.pth ^
  --det_weights checkpoints/det/fasterrcnn_epoch3.pth ^
  --seg_weights checkpoints/seg/seg_epoch3.pth ^
  --img_size 128 --num_classes 43

Real-time outputs include:
CLS: 25 (0.92) → class index + confidence
Green bounding boxes → Stop-sign detection
Lane mask overlay
FPS counter

Press ESC to exit.

🧪 Model Training
1️⃣ Train Traffic-Sign Classifier
py -m src.train.train_single_task --task classification ^
  --data data/gtsrb/train --num_classes 43 --epochs 30 ^
  --batch_size 16 --img_size 224 --backbone resnet18 ^
  --checkpoint_dir checkpoints --device cpu

2️⃣ Train Lane Segmentation
py -m src.train.train_segmentation --images data/tusimple/images ^
  --masks data/tusimple/masks --epochs 20 --batch_size 4 ^
  --checkpoint_dir checkpoints/seg --device cpu

3️⃣ Train Stop-Sign Detector (COCO format)
py -m src.train.train_detection_coco --images data/custom_stop/images ^
  --ann data/custom_stop/annotations.json --num_classes 2 ^
  --epochs 10 --checkpoint_dir checkpoints/det --device cpu

📊 Evaluation
Confusion Matrix + Classification Report
py scripts\eval_classifier_cm.py ^
  --checkpoint checkpoints/classifier_resnet18_epoch3.pth ^
  --data_root data/gtsrb/val --img_size 224 --device cpu


Outputs saved to:

outputs/confusion_matrix.png
outputs/per_class_accuracy.csv

🔍 Explainability (Grad-CAM)

Generate Grad-CAM for a single input image:

py src/explainability/gradcam_demo.py ^
  --image data/gtsrb/val/14/00001.png ^
  --weights checkpoints/classifier_resnet18_epoch3.pth ^
  --out outputs/gradcam.png ^
  --img_size 224 --device cpu

  🌐 Optional Streamlit Demo

Run a simple web app:
streamlit run streamlit_app.py

🛣️ Demonstration Ideas for Presentations

Show webcam feed with:
Phone screen showing traffic-sign PNGs
STOP-sign printed image
Drawn white lines for lane segmentation
Side-by-side Grad-CAM overlay (what the model looks at)

Explain:
Multi-task perception
Real-time inference
Explainability and safety relevance

📄 License
MIT License
© 2025 Mouna C

👤 Author
VisionDriveX — Explainable Autonomous Driving Perception System
Created by: Mouna C
GitHub: https://github.com/labcodesss


import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from src.models.multitask_model import MultiTaskModel
import json
import numpy as np

DEVICE = "cpu"

# Load label mapping
labels = json.load(open("data/gtsrb/labels.json"))

# Load classifier
model = MultiTaskModel(num_classes=43, backbone="resnet18", pretrained=False)
ck = torch.load("checkpoints/classifier_resnet18_epoch3.pth", map_location=DEVICE)
model.load_state_dict(ck["model_state"], strict=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

st.title("🚦 Explainable Traffic Sign Classifier")
st.write("Upload a traffic sign image to classify and view Grad-CAM.")

uploaded_img = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded_img:
    img = Image.open(uploaded_img).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    inp = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = model.stem(inp)
        logits = model.class_head(feat)
        probs = torch.softmax(logits, dim=1)[0].numpy()

    pred = int(np.argmax(probs))
    conf = float(probs[pred])

    st.subheader(f"Prediction: **{labels[str(pred)]}**")
    st.text(f"Confidence: {conf:.2f}")

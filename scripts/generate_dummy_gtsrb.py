# scripts/generate_dummy_gtsrb.py
# Generates a dummy GTSRB-style dataset: data/gtsrb/train/<class_id>/*.png
from PIL import Image, ImageDraw, ImageFont
import os

ROOT = "data/gtsrb/train"
NUM_CLASSES = 43
IMAGES_PER_CLASS = 6   # adjust as you like
IMG_SIZE = (128, 128)

os.makedirs(ROOT, exist_ok=True)

# Try to load a truetype font, fallback to default
try:
    font = ImageFont.truetype("arial.ttf", 18)
except Exception:
    font = ImageFont.load_default()

for cls in range(NUM_CLASSES):
    cls_dir = os.path.join(ROOT, str(cls))
    os.makedirs(cls_dir, exist_ok=True)
    for i in range(IMAGES_PER_CLASS):
        img = Image.new("RGB", IMG_SIZE, (int((cls*5) % 255), int((cls*11) % 255), int((cls*17) % 255)))
        draw = ImageDraw.Draw(img)
        text = f"class {cls}\nimg {i+1}"
        draw.multiline_text((8, 8), text, fill=(255,255,255), font=font)
        fname = os.path.join(cls_dir, f"{cls:02d}_{i+1:03d}.png")
        img.save(fname)
print(f"Created dummy dataset at {ROOT} with {NUM_CLASSES} classes and {IMAGES_PER_CLASS} images per class.")

# scripts/generate_dummy_coco.py
"""
Creates a tiny COCO-format dataset under data/custom_stop:
 - data/custom_stop/images/*.png
 - data/custom_stop/annotations.json
Category ids: 1 -> stop sign
"""
import os, json
from PIL import Image, ImageDraw
ROOT = "data/custom_stop"
IMG_DIR = os.path.join(ROOT, "images")
os.makedirs(IMG_DIR, exist_ok=True)

images = []
annotations = []
categories = [{"id": 1, "name": "stop"}]

N = 20
w,h = 640,480
ann_id = 1
for i in range(1, N+1):
    fname = f"img_{i:03d}.png"
    path = os.path.join(IMG_DIR, fname)
    img = Image.new("RGB", (w,h), (100,100,120))
    draw = ImageDraw.Draw(img)
    # random-ish rectangle (a simple "stop sign like" box)
    x1 = 50 + (i*7)%200
    y1 = 60 + (i*5)%200
    x2 = x1 + 80
    y2 = y1 + 80
    draw.rectangle([x1,y1,x2,y2], fill=(200,0,0))
    img.save(path)
    images.append({"id": i, "width": w, "height": h, "file_name": fname})
    # create annotation for half the images only
    if i % 2 == 0:
        bbox = [x1, y1, x2-x1, y2-y1]
        area = bbox[2]*bbox[3]
        annotations.append({
            "id": ann_id,
            "image_id": i,
            "category_id": 1,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        })
        ann_id += 1

coco = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(os.path.join(ROOT, "annotations.json"), "w") as f:
    json.dump(coco, f)
print("Wrote dummy COCO to", ROOT)
print("Images:", len(images), "Annotations:", len(annotations))

# scripts/generate_dummy_seg.py
from PIL import Image, ImageDraw
import os
ROOT_IMG = "data/tusimple/images"
ROOT_MASK = "data/tusimple/masks"
os.makedirs(ROOT_IMG, exist_ok=True)
os.makedirs(ROOT_MASK, exist_ok=True)

N = 40
W, H = 640, 360

for i in range(N):
    img = Image.new("RGB", (W, H), (100, 120, 140))
    draw = ImageDraw.Draw(img)
    x = 50 + (i % 5) * 10
    draw.rectangle([x, int(H*0.3), x+10, H], fill=(200,200,50))
    name = f"img_{i:03d}.png"
    img.save(os.path.join(ROOT_IMG, name))

    mask = Image.new("L", (W, H), 0)
    drawm = ImageDraw.Draw(mask)
    drawm.rectangle([x, int(H*0.3), x+10, H], fill=255)
    mask.save(os.path.join(ROOT_MASK, name))

print("Created dummy seg dataset:", ROOT_IMG, ROOT_MASK)

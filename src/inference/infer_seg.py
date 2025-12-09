# quick inference snippet (not a full file)
import torch, numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=1)
ckpt = torch.load("checkpoints/seg/seg_epoch1.pth", map_location="cpu")
model.load_state_dict(ckpt['model_state'])
model.eval()
img = Image.open("data/tusimple/images/img_000.png").convert('RGB').resize((512,256))
import torchvision.transforms.functional as TF
x = TF.to_tensor(img).unsqueeze(0)
with torch.no_grad():
    logits = model(x)
    mask = torch.sigmoid(logits)[0,0].numpy()
# save mask
import matplotlib.pyplot as plt
plt.imsave("outputs/seg_mask.png", mask, cmap='gray')

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Load model
from app import UNet
model = UNet()
model.load_state_dict(torch.load('best_segmentation_model.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Test with a defective image
img = Image.open(r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase\metal_nut\test\bent\000.png").convert('RGB')
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    raw_output = model(img_tensor)

raw = raw_output.squeeze().numpy()

print(f"Min: {raw.min():.4f}")
print(f"Max: {raw.max():.4f}")
print(f"Mean: {raw.mean():.4f}")

# Show raw output (before thresholding)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img.resize((256, 256)))
axes[0].set_title("Input")
axes[1].imshow(raw, cmap='hot')
axes[1].set_title(f"Raw Output (max={raw.max():.3f})")
axes[2].imshow((raw > 0.3), cmap='gray')
axes[2].set_title("Threshold 0.3")
plt.savefig("seg_debug.png")
plt.show()
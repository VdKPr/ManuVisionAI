import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

dataset_path = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase\metal_nut"

# Ground truth masks exist for each defect type
gt_path = os.path.join(dataset_path, "ground_truth")
defect_types = os.listdir(gt_path)
print("Defect types with masks:", defect_types)

# Show image + mask side by side for each defect type
fig, axes = plt.subplots(len(defect_types), 3, figsize=(12, 4*len(defect_types)))

for i, defect in enumerate(defect_types):
    # Get first image of this defect
    test_img_path = os.path.join(dataset_path, "test", defect)
    mask_path = os.path.join(gt_path, defect)
    img_name = sorted(os.listdir(test_img_path))[0]
    mask_name = sorted(os.listdir(mask_path))[0]
    
    img = np.array(Image.open(os.path.join(test_img_path, img_name)))
    mask = np.array(Image.open(os.path.join(mask_path, mask_name)))
    
    # Show: original | mask | overlay
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"{defect} — Original")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(mask, cmap='gray')
    axes[i, 1].set_title(f"{defect} — Mask")
    axes[i, 1].axis('off')
    
    # Overlay mask on image
    overlay = img.copy()
    if len(mask.shape) == 2:
        overlay[mask > 127] = [255, 0, 0]  # Red where defect is
    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title(f"{defect} — Overlay")
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig("mask_samples.png")
plt.show()
print("Saved as mask_samples.png")
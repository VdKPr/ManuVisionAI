import os
from PIL import Image
import matplotlib.pyplot as plt

# Change this to your MVTec AD path
dataset_path = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase"

# See all product categories
categories = os.listdir(dataset_path)
print("Categories:", categories)
# Should show: bottle, cable, capsule, carpet, grid, hazelnut, 
#              leather, metal_nut, pill, screw, tile, toothbrush,
#              transistor, wood, zipper

# Pick one category to start — metal_nut is great for manufacturing
category = "metal_nut"
cat_path = os.path.join(dataset_path, category)

# See folder structure
print("\nFolders:", os.listdir(cat_path))
# Should show: train, test, ground_truth

# Training images (all GOOD — no defects)
train_path = os.path.join(cat_path, "train", "good")
train_images = os.listdir(train_path)
print(f"\nGood training images: {len(train_images)}")

# Test images (mix of good + different defect types)
test_path = os.path.join(cat_path, "test")
defect_types = os.listdir(test_path)
print(f"Defect types: {defect_types}")
# Should show: good, bent, color, flip, scratch

# Show some examples
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# Show 1 good image
good_img = Image.open(os.path.join(train_path, train_images[0]))
axes[0].imshow(good_img)
axes[0].set_title("GOOD")
axes[0].axis('off')

# Show 1 image from each defect type
for i, defect in enumerate(defect_types[:4]):
    defect_path = os.path.join(test_path, defect)
    img_name = os.listdir(defect_path)[0]
    img = Image.open(os.path.join(defect_path, img_name))
    axes[i+1].imshow(img)
    axes[i+1].set_title(defect.upper())
    axes[i+1].axis('off')

plt.suptitle(f"MVTec AD: {category}", fontsize=16)
plt.tight_layout()
plt.savefig("mvtec_samples.png")
plt.show()
print("\nSaved sample image as mvtec_samples.png")
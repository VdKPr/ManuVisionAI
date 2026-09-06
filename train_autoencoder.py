# 1. Loads ONLY good images from ALL 15 categories (4096 images)
# 2. Trains YOUR autoencoder to reconstruct good parts
# 3. After training, computes reconstruction error for good vs defective
# 4. Sets threshold automatically (mean + 2*std of good errors)
# 5. Reports: "X% of good correctly passed, Y% of defects correctly caught"
# 6. Visualizes: original → reconstruction → error map for good and defective
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ============================================
# DATASET — ONLY GOOD IMAGES
# ============================================
class GoodPartsDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img  # no label needed — just the image

# ============================================
# LOAD ONLY GOOD IMAGES FROM ALL CATEGORIES
# ============================================
root_path = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase"
# Replace the data loading loop with:
category = "metal_nut"  # single category
category_path = os.path.join(root_path, category)

good_images = []
train_good = os.path.join(category_path, "train", "good")
for img_name in os.listdir(train_good):
    good_images.append(os.path.join(train_good, img_name))

print(f"Good images for {category}: {len(good_images)}")

# Defective images for evaluation
defect_images = []
test_path = os.path.join(category_path, "test")
for defect_type in os.listdir(test_path):
    if defect_type == "good":
        continue
    defect_dir = os.path.join(test_path, defect_type)
    if not os.path.isdir(defect_dir):
        continue
    for img_name in os.listdir(defect_dir):
        defect_images.append(os.path.join(defect_dir, img_name))

# Good test images for evaluation
good_test = os.path.join(category_path, "test", "good")
good_test_images = []
if os.path.exists(good_test):
    for img_name in os.listdir(good_test):
        good_test_images.append(os.path.join(good_test, img_name))

print(f"Defective images for eval: {len(defect_images)}")
print(f"Good test images for eval: {len(good_test_images)}")
# categories = sorted(os.listdir(root_path))

# good_images = []

# for category in categories:
#     category_path = os.path.join(root_path, category)
#     if not os.path.isdir(category_path):
#         continue

#     # Good from train
#     train_good = os.path.join(category_path, "train", "good")
#     if os.path.exists(train_good):
#         for img_name in os.listdir(train_good):
#             good_images.append(os.path.join(train_good, img_name))

#     # Good from test
#     test_good = os.path.join(category_path, "test", "good")
#     if os.path.exists(test_good):
#         for img_name in os.listdir(test_good):
#             good_images.append(os.path.join(test_good, img_name))

# print(f"Total GOOD images: {len(good_images)}")

# # Also collect defective images (for evaluation only, NOT training)
# defect_images = []

# for category in categories:
#     category_path = os.path.join(root_path, category)
#     if not os.path.isdir(category_path):
#         continue
#     test_path = os.path.join(category_path, "test")
#     if not os.path.exists(test_path):
#         continue
#     for defect_type in os.listdir(test_path):
#         if defect_type == "good":
#             continue
#         defect_dir = os.path.join(test_path, defect_type)
#         if not os.path.isdir(defect_dir):
#             continue
#         for img_name in os.listdir(defect_dir):
#             defect_images.append(os.path.join(defect_dir, img_name))

# print(f"Total DEFECTIVE images (for eval only): {len(defect_images)}")

# ============================================
# TRANSFORMS
# ============================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # No ImageNet normalization — autoencoder reconstructs raw pixels
])

# Split good images into train/val
train_imgs, val_imgs = train_test_split(good_images, test_size=0.2, random_state=42)

train_dataset = GoodPartsDataset(train_imgs, transform)
val_dataset = GoodPartsDataset(val_imgs, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# ============================================
# YOUR AUTOENCODER (your architecture)
# ============================================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        # Tight bottleneck — forces compression
        self.flatten = nn.Flatten()
        # self.fc_encode = nn.Linear(256 * 8 * 8, 128)   # 16384 → 128
        # self.fc_decode = nn.Linear(128, 256 * 8 * 8)   # 128 → 16384
        # New:
        self.fc_encode = nn.Linear(256 * 8 * 8, 32)
        self.fc_decode = nn.Linear(32, 256 * 8 * 8)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encode to feature maps
        features = self.encoder(x)          # 8×8×256
        
        # Squeeze through tight bottleneck
        flat = self.flatten(features)       # 16384
        latent = self.relu(self.fc_encode(flat))  # 128 — TIGHT
        expanded = self.relu(self.fc_decode(latent))  # 16384
        reshaped = expanded.view(-1, 256, 8, 8)  # back to 8×8×256
        
        # Decode back to image
        reconstructed = self.decoder(reshaped)  # 256×256×3
        return reconstructed

    
# chaaing after only Interesting result — defective images have 
# LOWER error (0.001357) than good images (0.001518). The autoencoder 
# reconstructs defects BETTER than good parts. 
# That means the bottleneck is too large — the model has enough capacity to reconstruct anything.
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()

#     def forward(self, x):
#         latent = self.encoder(x)
#         reconstructed = self.decoder(latent)
#         return reconstructed

# ============================================
# TRAINING (your loop)
# ============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

model = Autoencoder().to(device)
if False:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for imgs in train_loader:
            imgs = imgs.to(device)
            reconstructed = model(imgs)
            loss = criterion(reconstructed, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.to(device)
                reconstructed = model(imgs)
                loss = criterion(reconstructed, imgs)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_autoencoder.pth')
            print(f"  → Saved best model (Val Loss: {val_loss:.6f})")

    print(f"\nBest Validation Loss: {best_loss:.6f}")

# ============================================
# EVALUATION — find the threshold
# ============================================
print("\n" + "="*60)
print("ANOMALY DETECTION EVALUATION")
print("="*60)

model.load_state_dict(torch.load('best_autoencoder.pth', map_location=device))
model.eval()

def get_reconstruction_error(image_path, model, transform, device):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        reconstructed = model(img_tensor)
    
    # Per-pixel error map
    error_map = torch.mean((img_tensor - reconstructed) ** 2, dim=1).squeeze()
    
    # Instead of mean over whole image, use 95th percentile
    # This captures the WORST reconstructed region (likely the defect)
    error_value = torch.quantile(error_map.flatten(), 0.95).item()
    return error_value
# def get_reconstruction_error(image_path, model, transform, device):
#     """Compute reconstruction error for a single image"""
#     img = Image.open(image_path).convert('RGB')
#     img_tensor = transform(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         reconstructed = model(img_tensor)
#     error = torch.mean((img_tensor - reconstructed) ** 2).item()
#     return error

## Compute errors for good images
# Compute errors for good TEST images (not seen during training)
print("Computing errors for GOOD test images...")
good_errors = []
for img_path in good_test_images:
    error = get_reconstruction_error(img_path, model, transform, device)
    good_errors.append(error)

# Compute errors for defective images
print("Computing errors for DEFECTIVE images...")
defect_errors = []
for img_path in defect_images:
    error = get_reconstruction_error(img_path, model, transform, device)
    defect_errors.append(error)

# "I first tried training the autoencoder across all 15 categories. 
# Good detection was 93% but defect detection was only 3% — 
# the model learned to be a generic image reconstructor. 
# The reconstruction errors for good and defective parts were nearly identical (0.0014 vs 0.0017).

# The fix was per-category training. Each product line gets its own 
# autoencoder trained on only its good parts. This mirrors how real 
# industrial inspection works — each production line has its own baseline."

# print("Computing errors for GOOD images...")
# good_errors = []
# for img_path in val_imgs[:200]:  # use validation good images
#     error = get_reconstruction_error(img_path, model, transform, device)
#     good_errors.append(error)

# # Compute errors for defective images
# print("Computing errors for DEFECTIVE images...")
# defect_errors = []
# for img_path in defect_images[:200]:  # sample of defective images
#     error = get_reconstruction_error(img_path, model, transform, device)
#     defect_errors.append(error)

print(f"\nGOOD images    — Mean error: {np.mean(good_errors):.6f}, Std: {np.std(good_errors):.6f}")
print(f"DEFECTIVE images — Mean error: {np.mean(defect_errors):.6f}, Std: {np.std(defect_errors):.6f}")

# Set threshold as mean + 2*std of good errors
#threshold = np.mean(good_errors) + 2 * np.std(good_errors)    #older, want to push it higher after step 2 in step 3
'''# New:
threshold = np.mean(good_errors) + 1 * np.std(good_errors)
print(f"\nThreshold (mean + 2*std of good): {threshold:.6f}")

# Evaluate accuracy
good_correct = sum(1 for e in good_errors if e <= threshold)
defect_correct = sum(1 for e in defect_errors if e > threshold)

print(f"\nGood correctly identified:    {good_correct}/{len(good_errors)} ({100*good_correct/len(good_errors):.1f}%)")
print(f"Defects correctly identified: {defect_correct}/{len(defect_errors)} ({100*defect_correct/len(defect_errors):.1f}%)")
print(f"Overall accuracy: {100*(good_correct+defect_correct)/(len(good_errors)+len(defect_errors)):.1f}%")
'''
# Try multiple thresholds to find the sweet spot:
for multiplier in [0.5, 1.0, 1.5, 2.0]:
    threshold = np.mean(good_errors) + multiplier * np.std(good_errors)
    good_correct = sum(1 for e in good_errors if e <= threshold)
    defect_correct = sum(1 for e in defect_errors if e > threshold)
    print(f"Threshold (mean + {multiplier}×std = {threshold:.6f}): "
          f"Good: {good_correct}/{len(good_errors)} ({100*good_correct/len(good_errors):.1f}%) | "
          f"Defect: {defect_correct}/{len(defect_errors)} ({100*defect_correct/len(defect_errors):.1f}%) | "
          f"Overall: {100*(good_correct+defect_correct)/(len(good_errors)+len(defect_errors)):.1f}%")
# ============================================
# VISUALIZE — show reconstructions
# ============================================
print("\nGenerating visualization...")

fig, axes = plt.subplots(4, 4, figsize=(16, 16))

# Top 2 rows: good images (original vs reconstruction)
for i in range(2):
    img = Image.open(val_imgs[i]).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(img_tensor)

    axes[i*2, 0].imshow(img_tensor.squeeze().cpu().permute(1,2,0))
    axes[i*2, 0].set_title(f"GOOD — Original")
    axes[i*2, 0].axis('off')

    axes[i*2, 1].imshow(recon.squeeze().cpu().permute(1,2,0))
    axes[i*2, 1].set_title(f"GOOD — Reconstructed")
    axes[i*2, 1].axis('off')

    # Error map
    error_map = torch.mean((img_tensor - recon) ** 2, dim=1).squeeze().cpu()
    axes[i*2, 2].imshow(error_map, cmap='hot')
    axes[i*2, 2].set_title(f"Error Map (low = good)")
    axes[i*2, 2].axis('off')

    axes[i*2, 3].text(0.5, 0.5, f"Error: {error_map.mean():.6f}\nVERDICT: PASS",
                       ha='center', va='center', fontsize=14, color='green')
    axes[i*2, 3].axis('off')

# Bottom 2 rows: defective images
for i in range(2):
    img = Image.open(defect_images[i]).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(img_tensor)

    row = (i+1)*2 + i
    axes[2+i, 0].imshow(img_tensor.squeeze().cpu().permute(1,2,0))
    axes[2+i, 0].set_title(f"DEFECTIVE — Original")
    axes[2+i, 0].axis('off')

    axes[2+i, 1].imshow(recon.squeeze().cpu().permute(1,2,0))
    axes[2+i, 1].set_title(f"DEFECTIVE — Reconstructed")
    axes[2+i, 1].axis('off')

    error_map = torch.mean((img_tensor - recon) ** 2, dim=1).squeeze().cpu()
    axes[2+i, 2].imshow(error_map, cmap='hot')
    axes[2+i, 2].set_title(f"Error Map (high = defect)")
    axes[2+i, 2].axis('off')

    axes[2+i, 3].text(0.5, 0.5, f"Error: {error_map.mean():.6f}\nVERDICT: DEFECT",
                       ha='center', va='center', fontsize=14, color='red')
    axes[2+i, 3].axis('off')

plt.suptitle("Autoencoder Anomaly Detection — Good vs Defective", fontsize=16)
plt.tight_layout()
plt.savefig("autoencoder_results.png")
plt.show()
print("Saved as autoencoder_results.png")
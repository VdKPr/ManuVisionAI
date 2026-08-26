import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================
# U-NET ARCHITECTURE (same concept as your BraTS project)
# ============================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (downsampling)
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder (upsampling)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 1024 because of skip connection
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Final output: 1 channel (defect mask)
        self.final = nn.Conv2d(64, 1, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return torch.sigmoid(self.final(d1))

# ============================================
# DATASET: image + mask pairs
# ============================================
class DefectSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=256):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # grayscale
        
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = (mask > 0.3).float()  # binary: 0 or 1
        
        return img, mask

# ============================================
# LOAD DATA: pair each defect image with its mask
# ============================================
dataset_path = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase\metal_nut"
test_path = os.path.join(dataset_path, "test")
gt_path = os.path.join(dataset_path, "ground_truth")
images = []
masks = []

for defect_type in os.listdir(gt_path):
    img_dir = os.path.join(test_path, defect_type)
    mask_dir = os.path.join(gt_path, defect_type)
    
    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    for img_f, mask_f in zip(img_files, mask_files):
        images.append(os.path.join(img_dir, img_f))
        masks.append(os.path.join(mask_dir, mask_f))

print(f"Total image-mask pairs: {len(images)}")

# Also add GOOD images with blank masks (all zeros)
good_path = os.path.join(test_path, "good")
if os.path.exists(good_path):
    for img_f in os.listdir(good_path):
        images.append(os.path.join(good_path, img_f))
        masks.append(None)  # no mask for good images

# Split
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

# Custom dataset that handles None masks (good images)
class DefectSegDatasetWithGood(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=256):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.img_transform(img)
        
        if self.mask_paths[idx] is not None:
            mask = Image.open(self.mask_paths[idx]).convert('L')
            mask = self.mask_transform(mask)
            mask = (mask > 0.3).float()
        else:
            mask = torch.zeros(1, self.img_size, self.img_size)
        
        return img, mask

train_dataset = DefectSegDatasetWithGood(train_imgs, train_masks)
val_dataset = DefectSegDatasetWithGood(val_imgs, val_masks)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ============================================
# DICE LOSS (same as your BraTS project!)
# ============================================
class DiceLoss(nn.Module):
    def forward(self, pred, target, smooth=1.0):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()
    
    def forward(self, pred, target):
        return self.dice(pred, target) + self.bce(pred, target)

# ============================================
# TRAIN
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

model = UNet().to(device)
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 50 #30
best_dice = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    
    for imgs, msks in train_loader:
        imgs, msks = imgs.to(device), msks.to(device)
        preds = model(imgs)
        loss = criterion(preds, msks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation Dice
    model.eval()
    val_dice = 0
    with torch.no_grad():
        for imgs, msks in val_loader:
            imgs, msks = imgs.to(device), msks.to(device)
            preds = model(imgs)
            preds_bin = (preds > 0.5).float()
            
            intersection = (preds_bin * msks).sum()
            dice = (2. * intersection) / (preds_bin.sum() + msks.sum() + 1e-8)
            val_dice += dice.item()
    
    val_dice /= len(val_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Dice: {val_dice:.4f}")
    
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), 'best_segmentation_model.pth')
        print(f"  → Saved (Dice: {val_dice:.4f})")

print(f"\nBest Dice: {best_dice:.4f}")
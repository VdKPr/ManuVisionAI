import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============================================
# STEP 1: Create Dataset
# ============================================
class DefectDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ============================================
# STEP 2: Load all images with labels
# =====
root_path = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase"
categories = sorted(os.listdir(root_path))

images = []
labels = []
class_names = ["good", "defective"]  # just 2 classes

for category in categories:
    category_path = os.path.join(root_path, category)
    if not os.path.isdir(category_path):
        continue
    
    # Good images → label 0
    train_good = os.path.join(category_path, "train", "good")
    if os.path.exists(train_good):
        for img_name in os.listdir(train_good):
            images.append(os.path.join(train_good, img_name))
            labels.append(0)
    
    test_good = os.path.join(category_path, "test", "good")
    if os.path.exists(test_good):
        for img_name in os.listdir(test_good):
            images.append(os.path.join(test_good, img_name))
            labels.append(0)
    
    # ALL defect types → label 1
    test_path = os.path.join(category_path, "test")
    for defect_type in sorted(os.listdir(test_path)):
        if defect_type == "good":
            continue
        defect_dir = os.path.join(test_path, defect_type)
        if not os.path.isdir(defect_dir):
            continue
        for img_name in os.listdir(defect_dir):
            images.append(os.path.join(defect_dir, img_name))
            labels.append(1)


good_count = labels.count(0)
defect_count = labels.count(1)
print(f"Good: {good_count}, Defective: {defect_count}, Total: {len(labels)}")

#=======================================
#added as in fix 1, to compute class weights for imbalanced data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")
# ============================================
# FIX 1: Compute class weights for imbalanced data
# DEVLOG: Model was achieving 80% accuracy but macro F1 was only 0.20
# Problem: 823 "good" samples vs 2-5 per defect type → model learned to predict "good"
# Fix: Weighted CrossEntropyLoss — rare classes get higher loss penalty
# Expected: Lower overall accuracy but much higher defect detection (recall)
# ============================================
class_counts = [labels.count(i) for i in range(len(class_names))]
print(f"\nClass distribution:")
for name, count in zip(class_names, class_counts):
    print(f"  {name}: {count}")
# ============================================
# remove empty classes
# ============================================
valid_indices = [i for i, count in enumerate(class_counts) if count > 0]
class_names = [class_names[i] for i in valid_indices]
class_counts = [c for c in class_counts if c > 0]

label_map = {old: new for new, old in enumerate(valid_indices)}
labels = [label_map.get(l, 0) for l in labels]

print(f"\nAfter removing empty classes:")
print(f"Total classes: {len(class_names)}")
print(f"Total images: {len(labels)}")

# Weight = 1/count (rare classes get higher weight)
class_weights = [1.0 / max(count, 1) for count in class_counts]  # max(count,1) avoids division by zero

# Normalize weights so they sum to num_classes
total = sum(class_weights)
class_weights = [w * len(class_names) / total for w in class_weights]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print(f"\nTop 5 highest weighted classes (rarest defects):")
sorted_weights = sorted(zip(class_names, class_weights), key=lambda x: x[1], reverse=True)
for name, w in sorted_weights[:5]:
    print(f"  {name}: weight={w:.2f}")

# ============================================
# STEP 3: Split and Transform
# ============================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

train_dataset = DefectDataset(train_imgs, train_labels, transform)
val_dataset = DefectDataset(val_imgs, val_labels, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# ============================================
# STEP 4: Model — ResNet18 (pretrained, fine-tuned)
# ============================================


model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

#criterion = nn.CrossEntropyLoss()
weight = torch.tensor([1.0, good_count/defect_count]).to(device)
criterion = nn.CrossEntropyLoss(weight=weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# ============================================
# STEP 5: Train
# ============================================
num_epochs = 20
best_acc = 0
#if False:  # SKIP — already trained
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += lbls.size(0)
        correct += predicted.eq(lbls).sum().item()

    train_acc = 100. * correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            val_total += lbls.size(0)
            val_correct += predicted.eq(lbls).sum().item()

    val_acc = 100. * val_correct / val_total
    
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_defect_model.pth')
        print(f"  → Saved best model (Val Acc: {val_acc:.1f}%)")
    scheduler.step()
print(f"\nBest Validation Accuracy: {best_acc:.1f}%")
print("Model saved as best_defect_model.pth")

# ============================================
# EVALUATION — Per-class metrics
# ============================================
from sklearn.metrics import classification_report

model.load_state_dict(torch.load('best_defect_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, lbls in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(lbls.numpy())

unique_labels = sorted(set(all_labels + all_preds))
used_names = [class_names[i] for i in unique_labels]
print(classification_report(all_labels, all_preds, labels=unique_labels, target_names=used_names))
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
# ============================================
root_path = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase"
categories = os.listdir(root_path)  # ['bottle', 'cable', 'capsule', ...]

images = []
labels = []
class_names = []
label_id = 0  # FIX 2: ONE counter, starts before loop, NEVER resets

# dataset_path = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase\metal_nut"

# images = []
# labels = []
# class_names = []

# Good images from train (label = 0)
for category in categories:
    category_path = os.path.join(root_path, category)

    if not os.path.isdir(category_path):
        continue
    
    train_good = os.path.join(category_path, "train", "good")
    if os.path.exists(train_good):
        for img_name in os.listdir(train_good):
            images.append(os.path.join(train_good, img_name))
            labels.append(0)

    # Also good images from test
    test_good = os.path.join(category_path, "test", "good")
    if os.path.exists(test_good):
        for img_name in os.listdir(test_good):
            images.append(os.path.join(test_good, img_name))
            labels.append(0)

    class_names.append("good")

    # Defect images from test (label = 1, 2, 3...)
    test_path = os.path.join(category_path, "test")
    
    for defect_type in sorted(os.listdir(test_path)):
        if defect_type == "good":
            continue
        
        defect_dir = os.path.join(test_path, defect_type)
        if not os.path.isdir(defect_dir):
            continue
        class_names.append(f"{category}_{defect_type}")
        for img_name in os.listdir(defect_dir):
            images.append(os.path.join(defect_dir, img_name))
            labels.append(label_id)
        label_id += 1
print(f"Total categories: {len(categories)}")
print(f"Total classes: {len(class_names)}")
print(f"Total images: {len(images)}")
print(f"\nFirst 10 classes: {class_names[:10]}")
print(f"Last 10 classes: {class_names[-10:]}")

    # print(f"Total images: {len(images)}")
    # print(f"Classes: {class_names}")
    # print(f"Class distribution: {[labels.count(i) for i in range(len(class_names))]}")

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# ============================================
# STEP 5: Train
# ============================================
num_epochs = 40
best_acc = 0
if False:  # SKIP — already trained
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
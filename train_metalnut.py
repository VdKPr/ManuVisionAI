import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

# Metal nut only — 5 classes
dataset_path = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase\metal_nut"

images = []
labels = []
class_names = ["good"]

# Good images
for split in ["train", "test"]:
    good_path = os.path.join(dataset_path, split, "good")
    if os.path.exists(good_path):
        for img_name in os.listdir(good_path):
            images.append(os.path.join(good_path, img_name))
            labels.append(0)

# Defect images
label_id = 1
test_path = os.path.join(dataset_path, "test")
for defect_type in sorted(os.listdir(test_path)):
    if defect_type == "good":
        continue
    class_names.append(defect_type)
    defect_dir = os.path.join(test_path, defect_type)
    for img_name in os.listdir(defect_dir):
        images.append(os.path.join(defect_dir, img_name))
        labels.append(label_id)
    label_id += 1

print(f"Classes: {class_names}")
print(f"Total images: {len(images)}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

train_loader = DataLoader(DefectDataset(train_imgs, train_labels, transform), batch_size=16, shuffle=True)
val_loader = DataLoader(DefectDataset(val_imgs, val_labels, transform), batch_size=16, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_acc = 0

for epoch in range(20):
    model.train()
    correct = total = 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
        total += lbls.size(0)
        correct += predicted.eq(lbls).sum().item()
    
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            val_total += lbls.size(0)
            val_correct += predicted.eq(lbls).sum().item()
    
    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch+1}/20 | Train: {100*correct/total:.1f}% | Val: {val_acc:.1f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_defect_model_metalnut.pth')
        print(f"  → Saved best model ({val_acc:.1f}%)")

print(f"\nBest Validation Accuracy: {best_acc:.1f}%")
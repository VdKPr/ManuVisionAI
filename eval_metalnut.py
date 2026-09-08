"""
eval_metalnut.py — regenerate the REAL per-class results for Table 2.

Why this exists
---------------
train_metalnut.py imports classification_report but never calls it, so the
per-class numbers currently in the paper (Table 2) did not come from this
experiment. Its `good` row reports support=823, which is impossible: metal_nut
has only 242 good images total (220 train + 22 test), so a stratified 80/20
split leaves ~48 in validation. 823 is 20% of the 4,116 good images in the
ALL-CATEGORY run — a different experiment.

This script rebuilds the image list and the split EXACTLY as train_metalnut.py
does (same order, same seed), loads the saved checkpoint, and prints the true
classification report plus a ready-to-paste LaTeX table.

Run
---
    D:\\envs\\VSCODE_AI_Bootcamp\\My_Projects\\ManuVision AI\\.venv_ManuVisionAI\\Scripts\\python.exe eval_metalnut.py

Nothing is trained and nothing is overwritten — this is read-only.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATASET = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase\metal_nut"
CKPT    = "best_defect_model_metalnut.pth"


class DefectDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths, self.labels, self.transform = image_paths, labels, transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ---- rebuild the dataset in the SAME order as train_metalnut.py ----------
images, labels, class_names = [], [], ["good"]

for split in ["train", "test"]:
    good_path = os.path.join(DATASET, split, "good")
    if os.path.exists(good_path):
        for img_name in os.listdir(good_path):
            images.append(os.path.join(good_path, img_name))
            labels.append(0)

label_id = 1
test_path = os.path.join(DATASET, "test")
for defect_type in sorted(os.listdir(test_path)):
    if defect_type == "good":
        continue
    class_names.append(defect_type)
    for img_name in os.listdir(os.path.join(test_path, defect_type)):
        images.append(os.path.join(test_path, defect_type, img_name))
        labels.append(label_id)
    label_id += 1

print(f"Classes      : {class_names}")
print(f"Total images : {len(images)}")
print("Distribution : " + ", ".join(
    f"{n}={labels.count(i)}" for i, n in enumerate(class_names)))

# ---- identical split: test_size=0.2, random_state=42, stratified ---------
train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"\nTrain: {len(train_imgs)}   Val: {len(val_imgs)}")
print("Val distribution : " + ", ".join(
    f"{n}={val_labels.count(i)}" for i, n in enumerate(class_names)))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_loader = DataLoader(DefectDataset(val_imgs, val_labels, transform),
                        batch_size=16, shuffle=False)

# ---- load checkpoint and run inference -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(CKPT, map_location=device))
model = model.to(device).eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, lbls in val_loader:
        preds = model(imgs.to(device)).max(1)[1]
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.numpy())

acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
print(f"\nValidation accuracy: {acc*100:.1f}%   "
      f"(paper reports 89.6% — these should agree)")

print("\n" + "=" * 68)
print("CLASSIFICATION REPORT")
print("=" * 68)
print(classification_report(all_labels, all_preds,
                            labels=list(range(len(class_names))),
                            target_names=class_names, digits=2, zero_division=0))

print("CONFUSION MATRIX  (rows = true, cols = predicted)")
cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
print(" " * 9 + "".join(f"{n:>9}" for n in class_names))
for n, row in zip(class_names, cm):
    print(f"{n:>9}" + "".join(f"{v:>9}" for v in row))

# ---- emit the LaTeX table ------------------------------------------------
rep = classification_report(all_labels, all_preds,
                            labels=list(range(len(class_names))),
                            target_names=class_names,
                            output_dict=True, zero_division=0)
w = rep["weighted avg"]

print("\n" + "=" * 68)
print("PASTE THIS INTO THE PAPER (replaces Table 2)")
print("=" * 68)
print(r"""\begin{table}[h]
\centering
\caption{Per-class classification results on metal\_nut (validation split,
%d images)}
\label{tab:classification}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Support} \\
\hline""" % len(val_labels))
for n in class_names:
    r = rep[n]
    print(f"{n.replace('_', chr(92)+'_'):<7} & {r['precision']:.2f} & "
          f"{r['recall']:.2f} & {r['f1-score']:.2f} & {int(r['support'])} \\\\")
print(r"\hline")
print(f"\\textbf{{Weighted Avg}} & \\textbf{{{w['precision']:.2f}}} & "
      f"\\textbf{{{w['recall']:.2f}}} & \\textbf{{{w['f1-score']:.2f}}} & "
      f"\\textbf{{{int(w['support'])}}} \\\\")
print(r"""\hline
\end{tabular}
\end{table}""")

print("\n" + "=" * 68)
print(f"Macro F1: {rep['macro avg']['f1-score']:.3f}   "
      f"Weighted F1: {w['f1-score']:.3f}")
low = [n for n in class_names if rep[n]["support"] < 10]
if low:
    print("NOTE: these classes have <10 validation samples: " + ", ".join(low))
    print("      Their per-class metrics are dominated by one or two images.")
    print("      Report them, but say so in the text — and this is exactly why")
    print("      the multi-seed run (audit item M8) is worth doing.")

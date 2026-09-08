"""
eval_segmentation.py — quantitative segmentation results + post-processing ablation.
Closes audit items M1 (no segmentation metrics anywhere in the paper) and
M2 (post-processing presented as a contribution with no evidence).

Why this is needed
------------------
Segmentation is in the paper's title and abstract, but no Dice, IoU or pixel
metric appears anywhere. train_segmentation.py DOES track val Dice each epoch,
but (a) the number was never recorded, and (b) it is computed at threshold 0.5,
which DEVLOG says produced "mask completely black" — while the deployed system
(app.py / api.py / agent.py) uses threshold 0.25 plus morphological
post-processing. So the tracked number describes a configuration you do not ship.

This script evaluates the SHIPPED configuration and every intermediate step,
so one table answers both "how good is the segmentation" and "does the
post-processing actually help".

Two methodology notes, both reported explicitly:
  * The training split mixes defect images (real masks) with good images
    (all-zero masks). Dice is degenerate on an empty ground truth, so defect
    and good images are scored separately: Dice/IoU on defect images, and a
    false-positive rate on good images.
  * train_segmentation.py accumulates Dice per BATCH, which is more forgiving
    than the per-image mean. Both are reported here.

Run
---
    .venv_ManuVisionAI\\Scripts\\python.exe eval_segmentation.py

Read-only: nothing is trained, nothing is overwritten.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from scipy.ndimage import binary_opening, binary_closing, label

ROOT     = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase\metal_nut"
CKPT     = "best_segmentation_model.pth"
IMG_SIZE = 256
GT_BIN   = 0.19       # matches DefectSegDatasetWithGood
MIN_SIZE = 50         # matches app.py / api.py / agent.py


# ---------------- architecture (identical to train_segmentation.py) --------
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2 = DoubleConv(3, 64), DoubleConv(64, 128)
        self.enc3, self.enc4 = DoubleConv(128, 256), DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4, self.dec4 = nn.ConvTranspose2d(1024, 512, 2, 2), DoubleConv(1024, 512)
        self.up3, self.dec3 = nn.ConvTranspose2d(512, 256, 2, 2), DoubleConv(512, 256)
        self.up2, self.dec2 = nn.ConvTranspose2d(256, 128, 2, 2), DoubleConv(256, 128)
        self.up1, self.dec1 = nn.ConvTranspose2d(128, 64, 2, 2), DoubleConv(128, 64)
        self.final, self.pool = nn.Conv2d(64, 1, 1), nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.final(d1))


# ---------------- rebuild the identical split ------------------------------
test_path, gt_path = os.path.join(ROOT, "test"), os.path.join(ROOT, "ground_truth")
images, masks = [], []

for defect_type in os.listdir(gt_path):
    img_dir, mask_dir = os.path.join(test_path, defect_type), os.path.join(gt_path, defect_type)
    for img_f, mask_f in zip(sorted(os.listdir(img_dir)), sorted(os.listdir(mask_dir))):
        images.append(os.path.join(img_dir, img_f))
        masks.append(os.path.join(mask_dir, mask_f))

good_path = os.path.join(test_path, "good")
if os.path.exists(good_path):
    for img_f in os.listdir(good_path):
        images.append(os.path.join(good_path, img_f))
        masks.append(None)

_, val_imgs, _, val_masks = train_test_split(images, masks, test_size=0.2, random_state=42)

n_def = sum(m is not None for m in val_masks)
print(f"Total pairs      : {len(images)}  ({sum(m is not None for m in masks)} defect + "
      f"{sum(m is None for m in masks)} good)")
print(f"Validation split : {len(val_imgs)}  ({n_def} defect + {len(val_imgs)-n_def} good)")

# ---------------- inference: cache probability maps ------------------------
tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

probs, gts = [], []
with torch.no_grad():
    for ip, mp in zip(val_imgs, val_masks):
        x = tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(device)
        probs.append(model(x).squeeze().cpu().numpy())
        if mp is None:
            gts.append(np.zeros((IMG_SIZE, IMG_SIZE), np.uint8))
        else:
            m = Image.open(mp).convert("L").resize((IMG_SIZE, IMG_SIZE))
            gts.append((np.array(m) / 255.0 > GT_BIN).astype(np.uint8))
print(f"Inference done on {len(probs)} images (device: {device})\n")


# ---------------- post-processing stages -----------------------------------
def post(mask, opening=False, closing=False, minsize=False):
    if opening:
        mask = binary_opening(mask, structure=np.ones((5, 5))).astype(np.uint8)
    if closing:
        mask = binary_closing(mask, structure=np.ones((3, 3))).astype(np.uint8)
    if minsize:
        lab, n = label(mask)
        for i in range(1, n + 1):
            if (lab == i).sum() < MIN_SIZE:
                mask[lab == i] = 0
    return mask


CONFIGS = [
    ("thr 0.50, raw",                        0.50, dict()),
    ("thr 0.25, raw",                        0.25, dict()),
    ("thr 0.25 + opening",                   0.25, dict(opening=True)),
    ("thr 0.25 + opening + closing",         0.25, dict(opening=True, closing=True)),
    ("thr 0.25 + open + close + min-size",   0.25, dict(opening=True, closing=True, minsize=True)),
]

rows = []
for name, thr, kw in CONFIGS:
    dices, ious = [], []
    TP = FP = FN = 0
    good_hit, good_fp_px = 0, []

    for p, g in zip(probs, gts):
        pred = post((p > thr).astype(np.uint8), **kw)
        if g.sum() == 0:                                  # good image: empty GT
            good_hit += int(pred.sum() > 0)
            good_fp_px.append(int(pred.sum()))
            continue
        inter = int((pred & g).sum())
        dices.append(2 * inter / (pred.sum() + g.sum() + 1e-8))
        ious.append(inter / (((pred | g).sum()) + 1e-8))
        TP += inter
        FP += int((pred & (1 - g)).sum())
        FN += int(((1 - pred) & g).sum())

    agg = 2 * TP / (2 * TP + FP + FN + 1e-8)
    prec = TP / (TP + FP + 1e-8)
    rec = TP / (TP + FN + 1e-8)
    n_good = len(good_fp_px)
    rows.append(dict(name=name, dice=float(np.mean(dices)), iou=float(np.mean(ious)),
                     agg=agg, prec=prec, rec=rec, n=len(dices),
                     good_fp=good_hit / max(n_good, 1),
                     good_px=float(np.mean(good_fp_px)) if n_good else 0.0))

# ---------------- report ---------------------------------------------------
print("=" * 96)
print("SEGMENTATION RESULTS + POST-PROCESSING ABLATION")
print(f"Defect images: {rows[0]['n']}   Good images (empty GT): "
      f"{len(probs) - rows[0]['n']}")
print("=" * 96)
hdr = f"{'Configuration':<36}{'Dice':>8}{'IoU':>8}{'AggDice':>9}{'Prec':>8}{'Rec':>8}{'GoodFP%':>9}{'FPpx':>9}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['name']:<36}{r['dice']:>8.3f}{r['iou']:>8.3f}{r['agg']:>9.3f}"
          f"{r['prec']:>8.3f}{r['rec']:>8.3f}{r['good_fp']*100:>8.0f}%{r['good_px']:>9.0f}")

print("""
Dice / IoU : per-image mean over defect images (the number to report)
AggDice    : dataset-level Dice from pooled TP/FP/FN (less noisy, less standard)
Prec / Rec : pixel-level precision and recall on defect images
GoodFP%    : share of GOOD images with at least one predicted defect pixel
             (a false alarm -> a conforming part sent for rework)
FPpx       : mean falsely predicted pixels per good image""")

base, dep = rows[1], rows[-1]
print("\n" + "=" * 96)
print("WHAT THE ABLATION SHOWS")
print("=" * 96)
print(f"threshold 0.5 vs 0.25   : Dice {rows[0]['dice']:.3f} -> {rows[1]['dice']:.3f}")
print(f"post-processing effect  : Dice {base['dice']:.3f} -> {dep['dice']:.3f} "
      f"({dep['dice']-base['dice']:+.3f})")
print(f"                          precision {base['prec']:.3f} -> {dep['prec']:.3f} "
      f"({dep['prec']-base['prec']:+.3f})")
print(f"                          recall    {base['rec']:.3f} -> {dep['rec']:.3f} "
      f"({dep['rec']-base['rec']:+.3f})")
print(f"                          false alarms on good parts "
      f"{base['good_fp']*100:.0f}% -> {dep['good_fp']*100:.0f}%")

# ---------------- LaTeX ----------------------------------------------------
print("\n" + "=" * 96)
print("PASTE THIS INTO THE PAPER (new table in Section 4)")
print("=" * 96)
print(r"""\begin{table}[h]
\centering
\small
\caption{Segmentation performance and post-processing ablation on the
metal\_nut validation split (%d defect images with pixel-level ground truth,
%d defect-free images). Dice and IoU are per-image means over defect images;
precision and recall are pixel-level. The final row is the deployed
configuration.}
\label{tab:segmentation}
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Configuration} & \textbf{Dice} & \textbf{IoU} & \textbf{Precision} & \textbf{Recall} & \textbf{FP rate} \\
\hline""" % (rows[0]['n'], len(probs) - rows[0]['n']))
for i, r in enumerate(rows):
    nm = r['name'].replace('+', '$+$')
    bold = r"\textbf{%s}" if i == len(rows) - 1 else "%s"
    print(f"{bold % nm} & {r['dice']:.3f} & {r['iou']:.3f} & "
          f"{r['prec']:.3f} & {r['rec']:.3f} & {r['good_fp']*100:.0f}\\% \\\\")
print(r"""\hline
\end{tabular}
\end{table}""")

if rows[0]['n'] < 25:
    print(f"\nNOTE: only {rows[0]['n']} defect images carry ground truth in this split.")
    print("      Report the count in the caption (done above) and treat these")
    print("      as indicative rather than tight estimates.")

"""
multiproduct_eval.py — corrected re-run of Experiment 3a (multi-class, 73 defect
types across all 15 MVTec AD categories). Closes audit item M6.

The bug in train_detector.py
----------------------------
    label_id = 0                      # <-- starts at 0
    for category in categories:
        ...  labels.append(0)         # good images also get 0
        class_names.append("good")    # appended once PER CATEGORY (15 times)
        for defect_type in ...:
            class_names.append(f"{category}_{defect_type}")
            for img: labels.append(label_id)
            label_id += 1

Three consequences, all verified against the dataset:

  1. label_id starts at 0, the same value used for `good`, so the FIRST defect
     type is merged into the good class: 20 bottle_broken_large images are
     labelled `good`. The "4,116 good samples" figure in the paper is this
     contaminated count; the true number of good images is 4,096.
  2. class_names accumulates "good" once per category, giving 88 entries for
     73 defect types + 1 good class. model.fc is therefore built with 88
     outputs, 15 of which can never be trained.
  3. Because class_names has 88 entries but labels only run 0..72, every
     label -> name lookup in the classification report is misaligned.

So the reported 80.1% accuracy / 0.20 macro F1 describe a model trained on a
corrupted target. The qualitative conclusion (majority-class collapse under
extreme imbalance) is probably still right, but the numbers cannot be quoted.

This script rebuilds the labels correctly (good = 0, defect types 1..73) and
re-runs under the same protocol as Experiment 1: stratified 60/20/20,
checkpoint selected on validation, scored once on test, lr = 1e-4.

Also corrects a second reporting error: the paper states defect classes contain
"2-30 samples". Measured over the dataset the true range is 8-30, median 17.

Cost
----
5,354 images. Roughly 20-30 minutes per seed on an RTX 4060. SEEDS is set to a
single split by default; set SEEDS = [42, 43, 44] for mean +/- std consistent
with Experiment 1, at roughly an hour of compute.

Run
---
    .venv_ManuVisionAI\\Scripts\\python.exe multiproduct_eval.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

ROOT   = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase"
SEEDS  = [42]          # -> [42, 43, 44] for mean +/- std (~1 hour)
EPOCHS = 20
LR     = 1e-4          # matches Experiment 1


class DS(Dataset):
    def __init__(self, p, l, tf):
        self.p, self.l, self.tf = p, l, tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), self.l[i]


# ---- CORRECT label construction: good = 0, defect types 1..73 --------------
cats = sorted(c for c in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, c)))
images, labels, class_names = [], [], ["good"]
lid = 1
for cat in cats:
    tp = os.path.join(ROOT, cat, "test")
    if not os.path.isdir(tp):
        continue
    for sp in ("train", "test"):
        g = os.path.join(ROOT, cat, sp, "good")
        if os.path.exists(g):
            for f in os.listdir(g):
                images.append(os.path.join(g, f)); labels.append(0)
    for dt in sorted(os.listdir(tp)):
        d = os.path.join(tp, dt)
        if dt == "good" or not os.path.isdir(d):
            continue
        class_names.append(f"{cat}_{dt}")
        for f in os.listdir(d):
            images.append(os.path.join(d, f)); labels.append(lid)
        lid += 1

n_def = len(class_names) - 1
cnt = np.bincount(labels, minlength=len(class_names))
dc = cnt[1:]
print(f"categories        : {len(cats)}")
print(f"classes           : {len(class_names)}  (1 good + {n_def} defect types)")
print(f"images            : {len(images)}")
print(f"good images       : {cnt[0]}  ({100*cnt[0]/len(images):.1f}%)")
print(f"defect class size : min {dc.min()}, median {int(np.median(dc))}, max {dc.max()}")
print(f"                    (paper states '2-30 samples' -- true range is "
      f"{dc.min()}-{dc.max()})")
print(f"imbalance ratio   : {cnt[0]/dc.min():.0f}:1 (good vs rarest defect)\n")

tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

accs, macros, weis, npred, goodfrac = [], [], [], [], []

for seed in SEEDS:
    trx, tmpx, tryy, tmpy = train_test_split(
        images, labels, test_size=0.4, random_state=seed, stratify=labels)
    vax, tex, vay, tey = train_test_split(
        tmpx, tmpy, test_size=0.5, random_state=seed, stratify=tmpy)
    mk = lambda x, y, s: DataLoader(DS(x, y, tf), batch_size=16, shuffle=s,
                                    num_workers=0)
    trl, val, tel = mk(trx, tryy, True), mk(vax, vay, False), mk(tex, tey, False)
    print(f"--- seed {seed} | train {len(trx)}  val {len(vax)}  test {len(tex)} ---")

    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, len(class_names))
    m = m.to(device)
    crit, opt = nn.CrossEntropyLoss(), torch.optim.Adam(m.parameters(), lr=LR)

    best, state = -1.0, None
    for ep in range(EPOCHS):
        m.train()
        for x, y in trl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(m(x), y).backward(); opt.step()
        m.eval(); P, G = [], []
        with torch.no_grad():
            for x, y in val:
                P.extend(m(x.to(device)).max(1)[1].cpu().numpy()); G.extend(y.numpy())
        f1 = f1_score(G, P, average="macro", zero_division=0)
        if f1 > best:
            best = f1
            state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        print(f"  epoch {ep+1:>2}/{EPOCHS}  val acc {accuracy_score(G,P)*100:5.1f}%  "
              f"val macro F1 {f1:.3f}" + ("  <- best" if f1 == best else ""))

    m.load_state_dict(state); m.eval()
    torch.save(state, f"best_multiproduct_seed{seed}.pth")
    P, G = [], []
    with torch.no_grad():
        for x, y in tel:
            P.extend(m(x.to(device)).max(1)[1].cpu().numpy()); G.extend(y.numpy())
    P, G = np.array(P), np.array(G)
    accs.append(accuracy_score(G, P))
    macros.append(f1_score(G, P, average="macro", zero_division=0))
    weis.append(f1_score(G, P, average="weighted", zero_division=0))
    npred.append(len(set(P.tolist())))
    goodfrac.append(float((P == 0).mean()))
    print(f"  TEST acc {accs[-1]*100:.1f}%  macro F1 {macros[-1]:.3f}  "
          f"predicted {npred[-1]}/{len(class_names)} classes\n")

A, M, W = np.array(accs)*100, np.array(macros), np.array(weis)
sd = lambda v: v.std(ddof=1) if len(v) > 1 else 0.0
print("=" * 70)
print("CORRECTED EXPERIMENT 3a — held-out test")
print("=" * 70)
print(f"accuracy            : {A.mean():.1f}%" + (f" +/- {sd(A):.1f}" if len(A)>1 else ""))
print(f"macro F1            : {M.mean():.3f}" + (f" +/- {sd(M):.3f}" if len(M)>1 else ""))
print(f"weighted F1         : {W.mean():.3f}" + (f" +/- {sd(W):.3f}" if len(W)>1 else ""))
print(f"classes ever predicted : {np.mean(npred):.0f} of {len(class_names)}")
print(f"predictions that are 'good' : {100*np.mean(goodfrac):.1f}%"
      f"   (good is {100*cnt[0]/len(images):.1f}% of the data)")
print(f"\npaper currently reports 80.1% accuracy / 0.20 macro F1 "
      f"(from the corrupted labels)")

print("\n" + "=" * 70)
print("LATEX — replacement sentences for Experiment 3a")
print("=" * 70)
print(f"""\\textbf{{Result:}} {A.mean():.1f}\\% overall accuracy but \\textbf{{macro F1 of
only {M.mean():.2f}}}. The class distribution is severely imbalanced: the
\\textit{{good}} class holds {cnt[0]} of {len(images)} images
({100*cnt[0]/len(images):.0f}\\%), while individual defect classes contain
{dc.min()}--{dc.max()} samples (median {int(np.median(dc))}), a
{cnt[0]/dc.min():.0f}:1 ratio between the majority class and the rarest defect.
The model predicts only {np.mean(npred):.0f} of the {len(class_names)} classes
on the test split, and {100*np.mean(goodfrac):.0f}\\% of its predictions are
\\textit{{good}}: it attains high aggregate accuracy through majority-class bias
rather than genuine defect discrimination.""")

"""
binary_eval.py — corrected re-run of Experiment 3b (binary good vs defective
across all 15 MVTec AD categories).

Why this is needed
------------------
Experiment 3b currently reports "76% accuracy but 0% defect recall -- the model
again predicts 'good' for all inputs", and the paper concludes from 3a + 3b that
multi-product generalisation with a single supervised classifier is impractical.

Two reasons to doubt that:

  1. The corrected re-run of 3a (multiproduct_eval.py) reached macro F1 0.623
     across 74 classes and predicted 64 of them -- nothing like the collapse the
     paper describes. The original 3a number came from a corrupted label map.

  2. train_detector_multiproduct.py uses lr=1e-3. The learning-rate study
     (lr_study.py) showed that value destabilises fine-tuning badly enough to
     collapse the model onto a single class -- exactly the "predicts good for
     everything" signature 3b reports. So 3b's result may be an optimisation
     artifact rather than a property of the task.

This re-runs 3b under the same protocol as Experiments 1 and 3a: stratified
60/20/20, lr = 1e-4, checkpoint selected on validation BALANCED accuracy (not
raw accuracy -- with 76.5% good, accuracy alone rewards predicting the majority
class and would reproduce the original failure by construction).

Both the weighted loss used in the original script and an unweighted baseline
are run, since the paper specifically credits weighted cross-entropy with
failing to fix the imbalance.

Cost
----
5,354 images, two configurations, roughly 40-50 minutes total on an RTX 4060.
Set CONFIGS to just one entry to halve it.

Run
---
    .venv_ManuVisionAI\\Scripts\\python.exe binary_eval.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, recall_score,
                             precision_score)

ROOT    = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase"
SEED    = 42
EPOCHS  = 20
LR      = 1e-4
CONFIGS = [("weighted", True), ("unweighted", False)]


class DS(Dataset):
    def __init__(self, p, l, tf):
        self.p, self.l, self.tf = p, l, tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), self.l[i]


# ---- good = 0, any defect = 1 ---------------------------------------------
images, labels = [], []
for cat in sorted(c for c in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, c))):
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
        for f in os.listdir(d):
            images.append(os.path.join(d, f)); labels.append(1)

ngood, ndef = labels.count(0), labels.count(1)
print(f"images   : {len(images)}   good {ngood} ({100*ngood/len(images):.1f}%)  "
      f"defective {ndef} ({100*ndef/len(images):.1f}%)")
print(f"a 'predict good always' baseline scores {100*ngood/len(images):.1f}% accuracy, "
      f"0% defect recall\n")

tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trx, tmpx, tryy, tmpy = train_test_split(
    images, labels, test_size=0.4, random_state=SEED, stratify=labels)
vax, tex, vay, tey = train_test_split(
    tmpx, tmpy, test_size=0.5, random_state=SEED, stratify=tmpy)
mk = lambda x, y, s: DataLoader(DS(x, y, tf), batch_size=16, shuffle=s)
trl, val, tel = mk(trx, tryy, True), mk(vax, vay, False), mk(tex, tey, False)
print(f"train {len(trx)}  val {len(vax)}  test {len(tex)}\n")

results = {}
for name, weighted in CONFIGS:
    print(f"--- {name} cross-entropy, lr={LR:g} ---")
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m = m.to(device)
    if weighted:
        w = torch.tensor([1.0, ngood / ndef], dtype=torch.float32).to(device)
        crit = nn.CrossEntropyLoss(weight=w)
        print(f"  class weights: [1.00, {ngood/ndef:.2f}]")
    else:
        crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), lr=LR)

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
        ba = balanced_accuracy_score(G, P)
        if ba > best:
            best = ba
            state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        print(f"  epoch {ep+1:>2}/{EPOCHS}  val acc {accuracy_score(G,P)*100:5.1f}%  "
              f"balanced {ba*100:5.1f}%  defect recall "
              f"{recall_score(G,P,pos_label=1,zero_division=0)*100:5.1f}%"
              + ("  <- best" if ba == best else ""))

    m.load_state_dict(state); m.eval()
    torch.save(state, f"best_binary_{name}.pth")
    P, G = [], []
    with torch.no_grad():
        for x, y in tel:
            P.extend(m(x.to(device)).max(1)[1].cpu().numpy()); G.extend(y.numpy())
    P, G = np.array(P), np.array(G)
    results[name] = dict(
        acc=accuracy_score(G, P), bal=balanced_accuracy_score(G, P),
        rec=recall_score(G, P, pos_label=1, zero_division=0),
        pre=precision_score(G, P, pos_label=1, zero_division=0),
        f1=f1_score(G, P, average="macro", zero_division=0),
        cm=confusion_matrix(G, P, labels=[0, 1]))
    r = results[name]
    print(f"  TEST acc {r['acc']*100:.1f}%  balanced {r['bal']*100:.1f}%  "
          f"defect recall {r['rec']*100:.1f}%  macro F1 {r['f1']:.3f}\n")

print("=" * 72)
print("CORRECTED EXPERIMENT 3b — held-out test")
print("=" * 72)
print(f"{'Loss':<14}{'Accuracy':>11}{'Balanced':>11}{'Def.recall':>12}"
      f"{'Def.prec':>11}{'Macro F1':>11}")
for name, _ in CONFIGS:
    r = results[name]
    print(f"{name:<14}{r['acc']*100:>10.1f}%{r['bal']*100:>10.1f}%"
          f"{r['rec']*100:>11.1f}%{r['pre']*100:>10.1f}%{r['f1']:>11.3f}")

for name, _ in CONFIGS:
    cm = results[name]["cm"]
    print(f"\nconfusion matrix — {name}  (rows true, cols predicted)")
    print(f"{'':>12}{'good':>8}{'defect':>8}")
    print(f"{'good':>12}{cm[0,0]:>8}{cm[0,1]:>8}")
    print(f"{'defect':>12}{cm[1,0]:>8}{cm[1,1]:>8}")

print(f"\npaper currently reports: 76% accuracy, 0% defect recall")
best = max(results, key=lambda k: results[k]["bal"])
r = results[best]
if r["rec"] > 0.5:
    print(f"\n>>> The original 0% defect recall does NOT reproduce at lr=1e-4.")
    print(f">>> Best configuration ({best}) reaches {r['rec']*100:.1f}% defect recall.")
    print(f">>> Experiment 3b's conclusion needs to be rewritten, and with it the")
    print(f">>> paper's claim that multi-product generalisation is impractical.")
elif r["rec"] < 0.1:
    print(f"\n>>> The collapse reproduces at lr=1e-4 ({r['rec']*100:.1f}% defect recall),")
    print(f">>> so it is a property of the task, not an optimisation artifact.")
    print(f">>> Experiment 3b's conclusion stands as written.")
else:
    print(f"\n>>> Partial: {r['rec']*100:.1f}% defect recall — neither collapse nor success.")
    print(f">>> Experiment 3b needs rewriting with the measured numbers.")

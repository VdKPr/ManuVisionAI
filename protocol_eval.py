"""
protocol_eval.py — honest held-out evaluation of the metal_nut classifier.
Closes audit item M3 (no held-out test set) and M8 (single seed, no variance).

The problem
-----------
train_metalnut.py splits 80/20 into train/validation, selects the best epoch
by validation accuracy, and then reports that same validation accuracy as the
result (89.6%). Selecting a checkpoint on a set and reporting on it is
optimistically biased -- there is no held-out test set anywhere in the paper,
so the reported number cannot be defended as a generalisation estimate.

What this does instead
----------------------
  * three-way stratified split: 60% train / 20% validation / 20% test
  * the checkpoint is selected on VALIDATION, exactly as before
  * accuracy is reported on TEST, which is never seen during training or
    model selection
  * repeated over 3 seeds, reported as mean +/- std
  * confusion matrices are POOLED across seeds (3 x 67 = ~201 predictions),
    which gives far better per-class estimates than any single 67-image split

Note on the deployed model
--------------------------
This script writes its checkpoints as best_metalnut_seed<N>.pth and does NOT
touch best_defect_model_metalnut.pth. The existing deployed checkpoint (trained
on the larger 80/20 split) stays in place for the app, the API and Experiments
5 and 6. The paper should report the held-out number here as the performance
estimate, and note that the shipped model is trained on more data.

Run
---
    .venv_ManuVisionAI\\Scripts\\python.exe protocol_eval.py

Expect roughly 5 minutes total on an RTX 4060.
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
SEEDS   = [42, 43, 44]
EPOCHS  = 20


class DefectDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


# ---- dataset, built exactly as train_metalnut.py does ---------------------
images, labels, class_names = [], [], ["good"]
for split in ["train", "test"]:
    p = os.path.join(DATASET, split, "good")
    if os.path.exists(p):
        for f in os.listdir(p):
            images.append(os.path.join(p, f)); labels.append(0)

lid = 1
for dt in sorted(os.listdir(os.path.join(DATASET, "test"))):
    if dt == "good":
        continue
    class_names.append(dt)
    d = os.path.join(DATASET, "test", dt)
    for f in os.listdir(d):
        images.append(os.path.join(d, f)); labels.append(lid)
    lid += 1

print(f"Classes: {class_names}")
print(f"Images : {len(images)}  (" +
      ", ".join(f"{n}={labels.count(i)}" for i, n in enumerate(class_names)) + ")\n")

train_tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

accs, reports, cms = [], [], []

for seed in SEEDS:
    # 60 / 20 / 20, stratified, nested so proportions are exact
    tr_x, tmp_x, tr_y, tmp_y = train_test_split(
        images, labels, test_size=0.4, random_state=seed, stratify=labels)
    va_x, te_x, va_y, te_y = train_test_split(
        tmp_x, tmp_y, test_size=0.5, random_state=seed, stratify=tmp_y)

    dl = lambda x, y, s: DataLoader(DefectDataset(x, y, train_tf), batch_size=16, shuffle=s)
    tr_l, va_l, te_l = dl(tr_x, tr_y, True), dl(va_x, va_y, False), dl(te_x, te_y, False)

    print(f"--- seed {seed} | train {len(tr_x)}  val {len(va_x)}  test {len(te_x)} ---")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_va, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        for x, y in tr_l:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()

        model.eval(); c = t = 0
        with torch.no_grad():
            for x, y in va_l:
                p = model(x.to(device)).max(1)[1].cpu()
                c += (p == y).sum().item(); t += y.size(0)
        va = c / t
        if va > best_va:
            best_va = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep+1:>2}/{EPOCHS}  val {va*100:.1f}%" +
              ("  <- best" if va == best_va else ""))

    # restore the checkpoint chosen on VALIDATION, then score TEST once
    model.load_state_dict(best_state); model.eval()
    torch.save(best_state, f"best_metalnut_seed{seed}.pth")

    preds, gts = [], []
    with torch.no_grad():
        for x, y in te_l:
            preds.extend(model(x.to(device)).max(1)[1].cpu().numpy()); gts.extend(y.numpy())
    acc = float(np.mean(np.array(preds) == np.array(gts)))
    accs.append(acc)
    reports.append(classification_report(gts, preds, labels=list(range(len(class_names))),
                                         target_names=class_names, output_dict=True,
                                         zero_division=0))
    cms.append(confusion_matrix(gts, preds, labels=list(range(len(class_names)))))
    print(f"  selected val {best_va*100:.1f}%  ->  HELD-OUT TEST {acc*100:.1f}%\n")

# ---- aggregate ------------------------------------------------------------
A = np.array(accs) * 100
print("=" * 72)
print("HELD-OUT TEST RESULTS")
print("=" * 72)
for s, a in zip(SEEDS, A):
    print(f"  seed {s}: {a:.1f}%")
print(f"\n  mean {A.mean():.1f}%   std {A.std(ddof=1):.1f}%   "
      f"(paper currently reports 89.6% validation accuracy)")

def agg(cls, key):
    v = np.array([r[cls][key] for r in reports])
    return v.mean(), v.std(ddof=1)

print("\nPER-CLASS (mean +/- std over seeds, held-out test)")
print(f"{'Class':<10}{'Precision':>16}{'Recall':>16}{'F1':>16}{'Support':>9}")
for c in class_names:
    sup = np.mean([r[c]["support"] for r in reports])
    print(f"{c:<10}" + "".join(f"{m:>10.2f} +/-{s:<4.2f}"
          for m, s in (agg(c, k) for k in ("precision", "recall", "f1-score")))
          + f"{sup:>9.0f}")
for row in ("macro avg", "weighted avg"):
    print(f"{row:<10}" + "".join(f"{m:>10.2f} +/-{s:<4.2f}"
          for m, s in (agg(row, k) for k in ("precision", "recall", "f1-score"))))

P = np.sum(cms, axis=0)
print(f"\nPOOLED CONFUSION MATRIX ({P.sum()} test predictions across {len(SEEDS)} seeds)")
print(" " * 9 + "".join(f"{n:>9}" for n in class_names))
for n, r in zip(class_names, P):
    print(f"{n:>9}" + "".join(f"{v:>9}" for v in r))

ndef = P[1:].sum()
esc = P[1:, 0].sum()
fr = P[0, 1:].sum()
print(f"\nescapes (defect predicted good) : {esc}/{ndef} = {100*esc/ndef:.1f}%")
print(f"binary defect recall            : {ndef-esc}/{ndef} = {100*(ndef-esc)/ndef:.1f}%")
print(f"false rejects (good -> defect)  : {fr}/{P[0].sum()} = {100*fr/P[0].sum():.1f}%")

# ---- LaTeX ----------------------------------------------------------------
print("\n" + "=" * 72)
print("PASTE INTO THE PAPER (replaces Table 2)")
print("=" * 72)
print(r"""\begin{table}[h]
\centering
\caption{Per-class classification results on the held-out metal\_nut test
split, mean $\pm$ standard deviation over %d seeds (60/20/20 split;
checkpoints selected on validation, evaluated once on test).}
\label{tab:classification}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Support} \\
\hline""" % len(SEEDS))
for c in class_names:
    sup = np.mean([r[c]["support"] for r in reports])
    cells = " & ".join(f"{m:.2f} $\\pm$ {s:.2f}"
                       for m, s in (agg(c, k) for k in ("precision", "recall", "f1-score")))
    print(f"{c} & {cells} & {sup:.0f} \\\\")
print(r"\hline")
for row, lab in (("macro avg", "Macro Avg"), ("weighted avg", "Weighted Avg")):
    cells = " & ".join(f"{m:.2f} $\\pm$ {s:.2f}"
                       for m, s in (agg(row, k) for k in ("precision", "recall", "f1-score")))
    print(f"\\textbf{{{lab}}} & {cells} & \\\\")
print(r"""\hline
\end{tabular}
\end{table}""")
print(f"\nHeadline sentence: test accuracy {A.mean():.1f}\\% $\\pm$ {A.std(ddof=1):.1f}\\% "
      f"over {len(SEEDS)} seeds.")

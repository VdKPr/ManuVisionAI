"""
regen_autoencoder_figure.py — replacement for Figure 5 (audit item B3).

What was wrong with the current figure
--------------------------------------
In train_autoencoder.py (lines ~380-434) the visualisation has three defects:

  1. HARDCODED VERDICTS. The good loop always prints "VERDICT: PASS" and the
     defect loop always prints "VERDICT: DEFECT". These are ground-truth
     labels, not model decisions -- no threshold is ever applied. The figure
     therefore CANNOT show a misclassification, yet the paper's caption claims
     "some defective parts incorrectly classified as PASS".

  2. WRONG QUANTITY DISPLAYED. The panel prints error_map.mean(), but the
     actual decision rule uses the 95th PERCENTILE of the error map. Those are
     different numbers, which is why the figure shows a good part with higher
     error (0.002506) than a part labelled DEFECT (0.001242) -- an apparent
     contradiction that is really a units mismatch.

  3. LAYOUT BUGS. plt.subplots(4,4) with the good loop writing rows i*2 (0, 2)
     and the defect loop writing rows 2+i (2, 3) means row 1 is never drawn
     (the blank strip of white axes) and row 2 is drawn TWICE (the overlapping
     red/green text).

This script produces two corrected figures using the SAME decision rule as
Table 5: score = 95th percentile of the per-pixel error map, threshold =
mean + 1*std of good-part scores.

  fig_autoencoder_distributions.png  <- RECOMMENDED as the replacement.
      Histogram of good vs defective scores with the threshold marked. The
      paper claims "the reconstruction error distributions for good and
      defective parts overlap significantly" -- this figure PROVES that claim
      directly, which sample images can only hint at.

  fig_autoencoder_samples.png
      Three rows (good correctly passed / defect caught / defect MISSED) with
      real, derived verdicts. Use alongside or instead of the histogram.

It also reprints the Table 5 threshold sweep so you can confirm the numbers
currently in the paper still reproduce.

Run
---
    .venv_ManuVisionAI\\Scripts\\python.exe regen_autoencoder_figure.py

Read-only w.r.t. your existing files: it only writes the two new PNGs.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase\metal_nut"
CKPT = "best_autoencoder.pth"
MULT = 1.0          # paper reports mean + 1 sigma


# ---------------- architecture (identical to train_autoencoder.py) ---------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        c = []
        for i, o in [(3, 32), (32, 64), (64, 128), (128, 256), (256, 256)]:
            c += [nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU()]
        self.encoder = nn.Sequential(*c)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = []
        for i, o in [(256, 256), (256, 128), (128, 64), (64, 32)]:
            d += [nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU()]
        d += [nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*d)

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder = Encoder(), Decoder()
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(256 * 8 * 8, 32)
        self.fc_decode = nn.Linear(32, 256 * 8 * 8)
        self.relu = nn.ReLU()

    def forward(self, x):
        f = self.encoder(x)
        latent = self.relu(self.fc_encode(self.flatten(f)))
        return self.decoder(self.relu(self.fc_decode(latent)).view(-1, 256, 8, 8))


# ---------------- data (identical construction) ----------------------------
test_path = os.path.join(ROOT, "test")
good_test_images = [os.path.join(test_path, "good", f)
                    for f in os.listdir(os.path.join(test_path, "good"))]
defect_images = []
for dt in os.listdir(test_path):
    d = os.path.join(test_path, dt)
    if dt == "good" or not os.path.isdir(d):
        continue
    defect_images += [os.path.join(d, f) for f in os.listdir(d)]

print(f"Good test images : {len(good_test_images)}")
print(f"Defect images    : {len(defect_images)}")

transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()


def analyse(path):
    """Return (score, original HWC, reconstruction HWC, error map) — score is
    the 95th percentile of the per-pixel error map, matching Table 5."""
    x = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        r = model(x)
    err = torch.mean((x - r) ** 2, dim=1).squeeze().cpu()
    score = torch.quantile(err.flatten(), 0.95).item()
    return (score, x.squeeze().cpu().permute(1, 2, 0).numpy(),
            r.squeeze().cpu().permute(1, 2, 0).numpy(), err.numpy())


print("Scoring...")
good = [(p,) + analyse(p) for p in good_test_images]
defect = [(p,) + analyse(p) for p in defect_images]
gs = np.array([g[1] for g in good])
ds = np.array([d[1] for d in defect])

print(f"\nGOOD      p95 error: mean {gs.mean():.6f}  std {gs.std():.6f}")
print(f"DEFECTIVE p95 error: mean {ds.mean():.6f}  std {ds.std():.6f}")

# ---------------- reproduce the Table 5 sweep ------------------------------
print("\n" + "=" * 74)
print("TABLE 5 REPRODUCTION (paper: 68.2/30.1, 86.4/29.0, 90.9/28.0, 95.5/26.9)")
print("=" * 74)
print(f"{'Threshold':<22}{'Good %':>10}{'Defect %':>11}")
for m in (0.5, 1.0, 1.5, 2.0):
    t = gs.mean() + m * gs.std()
    print(f"mean + {m}*sigma{'':<8}{100*(gs<=t).mean():>10.1f}{100*(ds>t).mean():>11.1f}")

THRESH = gs.mean() + MULT * gs.std()
print(f"\nThreshold used for the figures (mean + {MULT}*sigma): {THRESH:.6f}")

# ---------------- FIGURE 1: distributions (recommended) --------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
bins = np.linspace(min(gs.min(), ds.min()), max(gs.max(), ds.max()), 40)
ax.hist(gs, bins=bins, alpha=0.65, label=f"Good (n={len(gs)})", color="#2E7D32",
        edgecolor="white", linewidth=0.5)
ax.hist(ds, bins=bins, alpha=0.65, label=f"Defective (n={len(ds)})", color="#C62828",
        edgecolor="white", linewidth=0.5)
ax.axvline(THRESH, color="black", ls="--", lw=1.6,
           label=f"Threshold (mean $+$ {MULT}$\\sigma$)")
ax.set_xlabel("Reconstruction error (95th percentile of per-pixel error map)")
ax.set_ylabel("Number of images")
ax.set_title("Autoencoder reconstruction error: good vs defective (metal_nut)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig_autoencoder_distributions.png", dpi=200)
print("\nwrote fig_autoencoder_distributions.png")

# ---------------- FIGURE 2: samples with REAL verdicts ---------------------
caught = [d for d in defect if d[1] > THRESH]
missed = [d for d in defect if d[1] <= THRESH]
passed = [g for g in good if g[1] <= THRESH]
print(f"defects caught {len(caught)} | defects MISSED {len(missed)} | "
      f"good passed {len(passed)}/{len(good)}")

rows = []
if passed:
    rows.append(("GOOD", passed[len(passed) // 2]))
if caught:
    rows.append(("DEFECTIVE", caught[len(caught) // 2]))
if missed:
    rows.append(("DEFECTIVE", missed[len(missed) // 2]))

fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3.8 * len(rows)))
axes = np.atleast_2d(axes)
for r, (label, rec) in enumerate(rows):
    _, score, orig, recon, err = rec
    verdict = "DEFECT" if score > THRESH else "PASS"
    ok = (verdict == "DEFECT") == (label == "DEFECTIVE")
    for c, (im, ttl, cm) in enumerate([(orig, f"{label} — original", None),
                                       (recon, "reconstruction", None),
                                       (err, "error map", "hot")]):
        axes[r, c].imshow(im, cmap=cm)
        axes[r, c].set_title(ttl, fontsize=11)
        axes[r, c].axis("off")
    axes[r, 3].text(0.5, 0.5,
                    f"p95 error: {score:.6f}\nthreshold: {THRESH:.6f}\n"
                    f"VERDICT: {verdict}\n({'correct' if ok else 'MISSED'})",
                    ha="center", va="center", fontsize=13,
                    color=("#2E7D32" if ok else "#C62828"))
    axes[r, 3].axis("off")
fig.suptitle("Autoencoder anomaly detection — verdicts derived from the "
             f"mean $+$ {MULT}$\\sigma$ threshold", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig_autoencoder_samples.png", dpi=200)
print("wrote fig_autoencoder_samples.png")

print(f"""
Suggested caption for the distribution figure:

  Distribution of autoencoder reconstruction error (95th percentile of the
  per-pixel error map) for {len(gs)} defect-free and {len(ds)} defective
  metal_nut images. The dashed line marks the mean $+$ {MULT}$\\sigma$
  operating point of Table 5. The two distributions overlap across almost
  their entire range: no threshold separates them, which is why no operating
  point achieves both >80\\% good identification and >50\\% defect detection.
""")

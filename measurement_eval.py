"""
measurement_eval.py — validation of the dimensional measurement stage.
Closes audit item M5.

The problem
-----------
The paper claims automated dimensional measurement and Figure 2 shows concrete
figures ("Length: 9.4mm, Area: 46.46mm2"). But the pixel-to-millimetre factor s
is a constant the operator types into the interface (0.10 in that screenshot);
there is no calibration step and no comparison against ground truth anywhere.
The mm figures are therefore arbitrary rescalings of pixel counts, and the
measurement stage has never been validated at all.

What this measures
------------------
Predicted measurements against mask-derived ground truth, in PIXEL space.
Working in pixels is the honest choice: relative error is scale-invariant, so
the percentages reported here hold for any calibration s, while absolute mm
errors would merely inherit whatever s the operator happened to enter.

Measurements follow app.py exactly: connected components, then per region
area = pixel count and max_length = max(bbox height, bbox width).

Three quantities are compared per image, because these are what actually drive
the tolerance decision:

    total defect area     sum over regions
    maximum region length the quantity compared against L_max
    region count          over- or under-segmentation

and then the end-to-end question the stage exists to answer:

    tolerance agreement   does the predicted measurement produce the same
                          ACCEPT/REJECT disposition as ground truth, swept
                          across a range of thresholds?

That last one matters more than the raw error: a measurement can be
substantially wrong and still yield the correct disposition, or nearly right
and still flip the verdict near a threshold.

Run
---
    .venv_ManuVisionAI\\Scripts\\python.exe measurement_eval.py

Read-only.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_closing

ROOT     = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase\metal_nut"
CKPT     = "best_segmentation_model.pth"
IMG_SIZE, GT_BIN, MIN_SIZE, THR = 256, 0.19, 50, 0.25


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


def measure(mask):
    """Identical to measure_defect() in app.py, but in pixels."""
    lab, n = ndimage.label(mask)
    regions = []
    for i in range(1, n + 1):
        r = (lab == i); a = int(r.sum())
        if a == 0:
            continue
        c = np.where(r)
        h = int(c[0].max() - c[0].min())
        w = int(c[1].max() - c[1].min())
        regions.append((a, max(h, w)))
    if not regions:
        return 0, 0, 0
    return sum(a for a, _ in regions), max(l for _, l in regions), len(regions)


def post(prob):
    m = (prob > THR).astype(np.uint8)
    m = binary_opening(m, structure=np.ones((5, 5))).astype(np.uint8)
    m = binary_closing(m, structure=np.ones((3, 3))).astype(np.uint8)
    lab, n = ndimage.label(m)
    for i in range(1, n + 1):
        if (lab == i).sum() < MIN_SIZE:
            m[lab == i] = 0
    return m


# ---- same split as train_segmentation.py / eval_segmentation.py -----------
test_path, gt_path = os.path.join(ROOT, "test"), os.path.join(ROOT, "ground_truth")
images, masks = [], []
for dt in os.listdir(gt_path):
    idir, mdir = os.path.join(test_path, dt), os.path.join(gt_path, dt)
    for i_f, m_f in zip(sorted(os.listdir(idir)), sorted(os.listdir(mdir))):
        images.append(os.path.join(idir, i_f)); masks.append(os.path.join(mdir, m_f))
good = os.path.join(test_path, "good")
if os.path.exists(good):
    for f in os.listdir(good):
        images.append(os.path.join(good, f)); masks.append(None)

_, vimg, _, vmask = train_test_split(images, masks, test_size=0.2, random_state=42)
pairs = [(i, m) for i, m in zip(vimg, vmask) if m is not None]
print(f"validation split: {len(vimg)} images, {len(pairs)} with ground-truth masks\n")

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

rows = []
with torch.no_grad():
    for ip, mp in pairs:
        x = tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(device)
        pred = post(model(x).squeeze().cpu().numpy())
        gtm = np.array(Image.open(mp).convert("L").resize((IMG_SIZE, IMG_SIZE)))
        gtm = (gtm / 255.0 > GT_BIN).astype(np.uint8)
        rows.append((os.path.basename(os.path.dirname(ip)), measure(pred), measure(gtm)))

print(f"{'defect':<9}{'GT area':>9}{'pred':>8}{'err%':>8}"
      f"{'GT len':>9}{'pred':>7}{'err%':>8}{'GTn':>5}{'pn':>4}")
print("-" * 68)
misses = []
for dt, (pa, pl, pn), (ga, gl, gn) in rows:
    ae = 100 * (pa - ga) / ga if ga else float("nan")
    le = 100 * (pl - gl) / gl if gl else float("nan")
    print(f"{dt:<9}{ga:>9}{pa:>8}{ae:>+8.1f}{gl:>9}{pl:>7}{le:>+8.1f}{gn:>5}{pn:>4}")
    if pa == 0:
        misses.append(dt)

det = [(dt, p, g) for dt, p, g in rows if p[0] > 0]
ga = np.array([g[0] for _, _, g in det], float); pa = np.array([p[0] for _, p, _ in det], float)
gl = np.array([g[1] for _, _, g in det], float); pl = np.array([p[1] for _, p, _ in det], float)

print("\n" + "=" * 68)
print(f"MEASUREMENT ACCURACY  ({len(det)} of {len(rows)} images with a detected defect)")
print("=" * 68)
if misses:
    print(f"segmentation produced an empty mask on {len(misses)}: {', '.join(misses)}")
    print("(measurement is undefined there -- these are segmentation misses,")
    print(" and they are excluded from the error statistics below)\n")


def stats(g, p, name, unit):
    err = p - g; rel = 100 * err / g
    print(f"{name}")
    print(f"  ground truth   mean {g.mean():8.1f} {unit}   range {g.min():.0f}-{g.max():.0f}")
    print(f"  predicted      mean {p.mean():8.1f} {unit}")
    print(f"  bias           {err.mean():+8.1f} {unit}   ({rel.mean():+.1f}% mean relative)")
    print(f"  MAE            {np.abs(err).mean():8.1f} {unit}")
    print(f"  MAPE           {np.abs(rel).mean():8.1f}%")
    print(f"  median abs err {np.median(np.abs(rel)):8.1f}%")
    if len(g) > 2:
        print(f"  correlation r  {np.corrcoef(g, p)[0,1]:8.3f}")
    print()


stats(ga, pa, "TOTAL DEFECT AREA", "px  ")
stats(gl, pl, "MAXIMUM REGION LENGTH", "px  ")

gn = np.array([g[2] for _, _, g in det]); pn = np.array([p[2] for _, p, _ in det])
print(f"REGION COUNT: ground truth {gn.sum()} regions, predicted {pn.sum()}"
      f"  ({100*(pn.sum()-gn.sum())/gn.sum():+.0f}%)")
print(f"  exact match on {int((gn==pn).sum())}/{len(gn)} images\n")

# ---- the decision that actually matters -----------------------------------
print("=" * 68)
print("TOLERANCE-DECISION AGREEMENT")
print("=" * 68)
print("Fraction of images where the predicted measurement yields the same")
print("ACCEPT/REJECT disposition as ground truth, swept across thresholds.\n")
print(f"{'L_max (px)':>12}{'GT reject':>11}{'pred reject':>13}{'agreement':>12}")
for t in (20, 40, 60, 80, 100, 120):
    g_r, p_r = gl > t, pl > t
    print(f"{t:>12}{int(g_r.sum()):>11}{int(p_r.sum()):>13}"
          f"{100*(g_r==p_r).mean():>11.0f}%")
print()
print(f"{'A_max (px)':>12}{'GT reject':>11}{'pred reject':>13}{'agreement':>12}")
for t in (500, 1000, 2000, 3000, 5000, 8000):
    g_r, p_r = ga > t, pa > t
    print(f"{t:>12}{int(g_r.sum()):>11}{int(p_r.sum()):>13}"
          f"{100*(g_r==p_r).mean():>11.0f}%")

print("\n" + "=" * 68)
print("LATEX")
print("=" * 68)
print(r"""\begin{table}[h]
\centering
\caption{Dimensional measurement accuracy against mask-derived ground truth on
the %d validation images with a detected defect. Errors are reported in pixels;
relative error is invariant to the pixel-to-millimetre calibration $s$, whereas
absolute millimetre error would simply inherit whichever value of $s$ the
operator enters.}
\label{tab:measurement}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Quantity} & \textbf{GT mean} & \textbf{Bias} & \textbf{MAE} & \textbf{MAPE} \\
\hline
Total defect area & %.0f px & %+.0f px & %.0f px & %.0f\%% \\
\hline
Max region length & %.0f px & %+.0f px & %.0f px & %.0f\%% \\
\hline
\end{tabular}
\end{table}""" % (len(det),
                  ga.mean(), (pa-ga).mean(), np.abs(pa-ga).mean(), np.abs(100*(pa-ga)/ga).mean(),
                  gl.mean(), (pl-gl).mean(), np.abs(pl-gl).mean(), np.abs(100*(pl-gl)/gl).mean()))

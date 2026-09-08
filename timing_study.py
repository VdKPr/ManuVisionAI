"""
timing_study.py — per-stage latency measurement for the inspection pipeline.
Closes audit item M4.

The problem
-----------
Experiment 6 claims the agent "reduces average inspection time for conforming
parts by approximately 60%". There is no timing code anywhere in the repo and
no latency figure in the paper -- the number is an estimate presented as a
result. It also appears again in the Conclusion.

What this measures
------------------
Each stage of the pipeline, timed separately over repeated runs:

    classify        ResNet18 forward pass
    segment         U-Net forward pass
    post-process    morphological opening / closing / min-size filtering
    measure         connected components + area/length
    tolerance       threshold comparison
    root cause      GPT-4o-mini call via LangChain   <-- network bound

then the two end-to-end paths the agent chooses between:

    conforming path   classify -> report                    (2 stages)
    defective path    classify -> segment -> measure ->
                      tolerance -> root cause               (5 stages)

The saving is reported as measured, both including and excluding the LLM call,
because the LLM call is network-bound and will dominate everything else.

Cost note
---------
Makes N_LLM (default 5) calls to gpt-4o-mini. At current pricing this is well
under one US cent in total. Set N_LLM = 0 to skip the LLM entirely and time
only the local stages.

Run
---
    .venv_ManuVisionAI\\Scripts\\python.exe timing_study.py
"""

import os
import time
import statistics as st
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_closing

ROOT     = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase\metal_nut"
CLS_CKPT = "best_defect_model_metalnut.pth"
SEG_CKPT = "best_segmentation_model.pth"
N_WARM, N_RUNS, N_LLM = 3, 20, 5
CLASSES  = ["good", "bent", "color", "flip", "scratch"]


# ---------------- U-Net (identical to train_segmentation.py) ---------------
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
if device.type == "cuda":
    print(f"gpu   : {torch.cuda.get_device_name(0)}")

cls_tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
seg_tf = transforms.Compose([
    transforms.Resize((256, 256)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

clf = models.resnet18(weights=None)
clf.fc = nn.Linear(clf.fc.in_features, len(CLASSES))
clf.load_state_dict(torch.load(CLS_CKPT, map_location=device))
clf = clf.to(device).eval()

seg = UNet().to(device)
seg.load_state_dict(torch.load(SEG_CKPT, map_location=device))
seg.eval()

# one conforming and one defective sample
good_p = os.path.join(ROOT, "test", "good")
scr_p = os.path.join(ROOT, "test", "scratch")
img_good = Image.open(os.path.join(good_p, sorted(os.listdir(good_p))[0])).convert("RGB")
img_def = Image.open(os.path.join(scr_p, sorted(os.listdir(scr_p))[0])).convert("RGB")


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, n=N_RUNS):
    for _ in range(N_WARM):
        fn()
    sync()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return st.mean(ts), (st.stdev(ts) if len(ts) > 1 else 0.0)


# ---------------- stage definitions ----------------------------------------
def stage_classify(img):
    x = cls_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = clf(x)
    p = torch.softmax(out, 1)
    return CLASSES[p.argmax(1).item()], p.max().item()


def stage_segment(img):
    x = seg_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        m = seg(x)
    return m.squeeze().cpu().numpy()


def stage_post(prob):
    m = (prob > 0.25).astype(np.uint8)
    m = binary_opening(m, structure=np.ones((5, 5))).astype(np.uint8)
    m = binary_closing(m, structure=np.ones((3, 3))).astype(np.uint8)
    lab, n = ndimage.label(m)
    for i in range(1, n + 1):
        if (lab == i).sum() < 50:
            m[lab == i] = 0
    return m


def stage_measure(mask, px=0.1):
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        r = (lab == i); a = r.sum()
        if a == 0:
            continue
        c = np.where(r)
        h = (c[0].max() - c[0].min()) * px
        w = (c[1].max() - c[1].min()) * px
        out.append({"area_mm2": a * px * px, "max_length_mm": max(h, w)})
    return out


def stage_tolerance(meas, lmax=2.0, amax=5.0):
    return [m for m in meas
            if m["max_length_mm"] > lmax or m["area_mm2"] > amax]


print("\nbenchmarking local stages...")
prob = stage_segment(img_def)
mask = stage_post(prob)
meas = stage_measure(mask)

rows = [
    ("classify (ResNet18)",      *bench(lambda: stage_classify(img_def))),
    ("segment (U-Net)",          *bench(lambda: stage_segment(img_def))),
    ("post-process (morph.)",    *bench(lambda: stage_post(prob))),
    ("measure (conn. comp.)",    *bench(lambda: stage_measure(mask))),
    ("tolerance check",          *bench(lambda: stage_tolerance(meas))),
]

# ---------------- LLM stage ------------------------------------------------
llm_mean = llm_std = 0.0
if N_LLM > 0:
    try:
        from root_cause import get_root_cause_analysis
        print(f"timing {N_LLM} GPT-4o-mini calls (network bound)...")
        ts = []
        for i in range(N_LLM):
            t0 = time.perf_counter()
            get_root_cause_analysis("scratch", meas, 0.95)
            ts.append((time.perf_counter() - t0) * 1000)
            print(f"  call {i+1}/{N_LLM}: {ts[-1]:.0f} ms")
        llm_mean = st.mean(ts)
        llm_std = st.stdev(ts) if len(ts) > 1 else 0.0
        rows.append(("root cause (GPT-4o-mini)", llm_mean, llm_std))
    except Exception as e:
        print(f"  LLM timing skipped: {type(e).__name__}: {e}")

# ---------------- report ---------------------------------------------------
print("\n" + "=" * 66)
print("PER-STAGE LATENCY")
print("=" * 66)
print(f"{'Stage':<28}{'Mean (ms)':>12}{'Std (ms)':>12}")
print("-" * 66)
for n, m, s in rows:
    print(f"{n:<28}{m:>12.1f}{s:>12.1f}")

local = {n: m for n, m, _ in rows if "root cause" not in n}
conforming = local["classify (ResNet18)"]
defective_local = sum(local.values())
defective_full = defective_local + llm_mean

print("\n" + "=" * 66)
print("END-TO-END PATHS")
print("=" * 66)
print(f"conforming (classify -> report)      : {conforming:>9.1f} ms")
print(f"defective, local stages only         : {defective_local:>9.1f} ms")
if llm_mean:
    print(f"defective, including LLM             : {defective_full:>9.1f} ms")

sav_local = 100 * (1 - conforming / defective_local)
print(f"\nsaving on conforming parts, local only : {sav_local:>6.1f}%")
if llm_mean:
    sav_full = 100 * (1 - conforming / defective_full)
    print(f"saving on conforming parts, incl. LLM  : {sav_full:>6.1f}%")
    print(f"LLM share of the defective path        : {100*llm_mean/defective_full:>6.1f}%")
print("\n(paper currently claims 'approximately 60%')")

# ---------------- LaTeX ----------------------------------------------------
print("\n" + "=" * 66)
print("LATEX TABLE")
print("=" * 66)
print(r"""\begin{table}[h]
\centering
\caption{Measured per-stage latency on an NVIDIA RTX 4060, mean $\pm$
standard deviation over %d runs (%d runs for the network-bound LLM call).
The agent executes only the first stage for conforming parts.}
\label{tab:latency}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Stage} & \textbf{Mean (ms)} & \textbf{Std (ms)} \\
\hline""" % (N_RUNS, N_LLM))
for n, m, s in rows:
    print(f"{n.replace('_',chr(92)+'_')} & {m:.1f} & {s:.1f} \\\\")
print(r"\hline")
print(f"\\textbf{{Conforming path (1 stage)}} & \\textbf{{{conforming:.1f}}} & \\\\")
print(f"\\textbf{{Defective path (5 stages)}} & \\textbf{{{defective_full:.1f}}} & \\\\")
print(r"""\hline
\end{tabular}
\end{table}""")

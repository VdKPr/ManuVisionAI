"""
ae_by_type.py — autoencoder detection rate broken down by defect type.

The histogram shows the defective distribution is bimodal: one mode sitting on
top of the good distribution, plus a long right tail. The hypothesis is that
the tail is GLOBAL appearance changes (flip) and the overlapping mode is
LOCALISED defects (scratch, small bent). If that holds, "the autoencoder fails"
becomes the much sharper "the autoencoder detects only global appearance
change, and localised defects -- the ones that matter for surface quality --
are invisible to it".

That also lines up with the classifier: flip was its ONLY perfect class
(F1 1.00), while bent and scratch were its weakest. Two independent methods
failing on the same defect types is a real finding.

Run
---
    .venv_ManuVisionAI\\Scripts\\python.exe ae_by_type.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

ROOT = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase\metal_nut"
CKPT, MULT = "best_autoencoder.pth", 1.0


class Encoder(nn.Module):
    def __init__(self):
        super().__init__(); c = []
        for i, o in [(3, 32), (32, 64), (64, 128), (128, 256), (256, 256)]:
            c += [nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU()]
        self.encoder = nn.Sequential(*c)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__(); d = []
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
        l = self.relu(self.fc_encode(self.flatten(f)))
        return self.decoder(self.relu(self.fc_decode(l)).view(-1, 256, 8, 8))


tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()


def score(p):
    x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        r = model(x)
    e = torch.mean((x - r) ** 2, dim=1).squeeze().cpu()
    return torch.quantile(e.flatten(), 0.95).item()


test = os.path.join(ROOT, "test")
good = np.array([score(os.path.join(test, "good", f))
                 for f in os.listdir(os.path.join(test, "good"))])
THRESH = good.mean() + MULT * good.std()

print(f"Good: n={len(good)}  mean {good.mean():.6f}  std {good.std():.6f}")
print(f"Threshold (mean + {MULT}sigma): {THRESH:.6f}")
print(f"Good correctly passed: {(good<=THRESH).sum()}/{len(good)} "
      f"({100*(good<=THRESH).mean():.1f}%)\n")

print("=" * 66)
print("DETECTION RATE BY DEFECT TYPE")
print("=" * 66)
print(f"{'Defect type':<12}{'n':>5}{'detected':>10}{'rate':>9}{'mean p95':>12}")
print("-" * 66)
rows, alls = [], []
for dt in sorted(os.listdir(test)):
    d = os.path.join(test, dt)
    if dt == "good" or not os.path.isdir(d):
        continue
    s = np.array([score(os.path.join(d, f)) for f in os.listdir(d)])
    alls.append(s)
    det = int((s > THRESH).sum())
    rows.append((dt, len(s), det, 100 * det / len(s), s.mean()))
    print(f"{dt:<12}{len(s):>5}{det:>10}{100*det/len(s):>8.1f}%{s.mean():>12.6f}")
tot = np.concatenate(alls)
print("-" * 66)
print(f"{'ALL':<12}{len(tot):>5}{int((tot>THRESH).sum()):>10}"
      f"{100*(tot>THRESH).mean():>8.1f}%{tot.mean():>12.6f}")
print("\n(paper reports 29.0% overall at mean + 1sigma -- ALL row should match)")

print("\n" + "=" * 66)
print("LATEX TABLE")
print("=" * 66)
print(r"""\begin{table}[h]
\centering
\caption{Autoencoder detection rate by defect type (metal\_nut,
threshold mean $+$ 1$\sigma$).}
\label{tab:ae_by_type}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Defect type} & \textbf{Images} & \textbf{Detected} & \textbf{Rate} \\
\hline""")
for dt, n, det, rate, mu in rows:
    print(f"{dt} & {n} & {det} & {rate:.0f}\\% \\\\")
print(r"\hline")
print(f"\\textbf{{All}} & \\textbf{{{len(tot)}}} & "
      f"\\textbf{{{int((tot>THRESH).sum())}}} & "
      f"\\textbf{{{100*(tot>THRESH).mean():.0f}\\%}} \\\\")
print(r"""\hline
\end{tabular}
\end{table}""")

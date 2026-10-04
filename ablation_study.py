"""
ablation_study.py — does augmentation help? does weighted cross-entropy help?

Same protocol as lr_study.py at its chosen setting: stratified 60/20/20 split,
seeds 42/43/44, ResNet18 (ImageNet), Adam lr=1e-4, 20 epochs, batch 16,
checkpoint picked on VALIDATION accuracy, scored ONCE on TEST.

What's new: every run is fully seeded, and every configuration uses the SAME
three splits — so the only thing that changes between rows is the config.
That makes seed-paired differences against the baseline meaningful.

Configurations
--------------
  baseline     no augmentation, plain cross-entropy (what you ship today)
  aug          safe augmentation (see SAFE_AUG), plain cross-entropy
  wce          no augmentation, class-weighted cross-entropy
  aug+wce      both
  aug+mirror   safe augmentation PLUS horizontal/vertical mirroring.
               NEGATIVE CONTROL. Hypothesis: MVTec's "flip" class is the nut
               upside down, which looks like a mirror image of a good nut, so
               mirroring good parts creates flip-like images labelled "good".
               Expect flip recall to fall. Check a good and a flip image by eye.

Why augmentation is "safe" here
-------------------------------
  * Rotation: a nut can sit at any angle, and rotation never mirrors it.
  * Small shift/scale: placement and zoom vary slightly on a real line.
  * Brightness/contrast only: NO hue/saturation jitter, because "color" is a
    defect class — shifting colour could create or erase that defect.
  * Light blur: focus variation.

Outputs (in results/ablation_<timestamp>/)
------------------------------------------
  runs.csv       one row per config x seed
  summary.md     table for the README / DEVLOG / interviews
  summary.json   everything, machine-readable
  plus a full log of the console output

Nothing touches your shipped checkpoint. Checkpoints are NOT saved unless
SAVE_CKPTS = True, and then only inside the results folder.

Operating point: besides argmax, each run picks a cutoff tau on VALIDATION —
flag a part when P(good) < tau — minimising 10 x escapes + 1 x false rejects,
then applies it once to TEST. This is how you'd trade false rejects for escapes.

Run (about 30-40 min on an RTX 4060 for 5 configs x 3 seeds; ~20 min for run 2):
    .venv_ManuVisionAI\\Scripts\\python.exe ablation_study.py
"""
import os, sys, json, csv, random, atexit
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

# ----------------------------------------------------------------------------- settings
DATASET = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase\metal_nut"
# Run 1 (4 Oct): seeds [42, 43, 44], all five configs -> aug+wce and aug+mirror looked best.
# Run 2 (this default): FRESH seeds, only the configs chosen in run 1, so the
# winners are confirmed on splits that played no part in picking them.
SEEDS = [42]
EPOCHS, LR, BATCH = 20, 1e-4, 16
CONFIGS = ["aug+mirror"] #["baseline", "aug+wce", "aug+mirror"]
COST_ESCAPE, COST_FALSE_REJECT = 10.0, 1.0      # an escape costs 10x a false reject
SAVE_CKPTS = True #False
NUM_WORKERS = 0            # keep 0 on Windows unless you know you need more
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", f"ablation_{datetime.now():%Y%m%d_%H%M%S}")
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------------------- logging
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            if not s.closed:
                s.write(data); s.flush()
    def flush(self):
        for s in self.streams:
            if not s.closed:
                s.flush()
    def __getattr__(self, name):
        return getattr(self.streams[0], name)

_log = open(os.path.join(OUT, "log.txt"), "w", encoding="utf-8")
atexit.register(_log.close)
sys.stdout = Tee(sys.__stdout__, _log)
sys.stderr = Tee(sys.__stderr__, _log)


# ----------------------------------------------------------------------------- data
class DefectDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def load_metal_nut():
    images, labels, names = [], [], ["good"]
    for split in ["train", "test"]:                      # same order as lr_study.py
        p = os.path.join(DATASET, split, "good")
        for f in sorted(os.listdir(p)):
            images.append(os.path.join(p, f)); labels.append(0)
    lid = 1
    for dt in sorted(os.listdir(os.path.join(DATASET, "test"))):
        if dt == "good":
            continue
        names.append(dt)
        d = os.path.join(DATASET, "test", dt)
        for f in sorted(os.listdir(d)):
            images.append(os.path.join(d, f)); labels.append(lid)
        lid += 1
    return images, labels, names


EVAL_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                              transforms.Normalize(MEAN, STD)])

SAFE_AUG = [
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=180, fill=0),      # fill=0 assumes a dark background — check
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=0),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # no hue/saturation: "color" is a class
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
]
MIRROR = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
TAIL = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]


def train_transform(cfg):
    if cfg in ("aug", "aug+wce"):
        return transforms.Compose(SAFE_AUG + TAIL)
    if cfg == "aug+mirror":
        return transforms.Compose(SAFE_AUG + MIRROR + TAIL)
    return EVAL_TF                                        # baseline, wce


# ----------------------------------------------------------------------------- helpers
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_weights(train_labels, k):
    counts = np.bincount(train_labels, minlength=k).astype(float)
    return counts.sum() / (k * counts)                    # inverse frequency


def metrics_from(gt, pr, k):
    """Inspection metrics from test labels and predictions (0 = good)."""
    gt, pr = np.asarray(gt), np.asarray(pr)
    cm = confusion_matrix(gt, pr, labels=list(range(k)))
    n_def, n_good = cm[1:].sum(), cm[0].sum()
    return {
        "acc": float((gt == pr).mean()),
        "bal_acc": float(balanced_accuracy_score(gt, pr)),
        "macro_f1": float(f1_score(gt, pr, labels=list(range(k)), average="macro", zero_division=0)),
        "escape_rate": float(cm[1:, 0].sum() / n_def) if n_def else 0.0,      # defect -> good
        "false_reject": float(cm[0, 1:].sum() / n_good) if n_good else 0.0,   # good -> defect
        "per_class_recall": [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0 for i in range(k)],
        "cm": cm.tolist(),
    }


def pick_threshold(p_good, y):
    """Flag a part as defective when P(good) < tau. Choose tau on VALIDATION to
    minimise COST_ESCAPE * escapes + COST_FALSE_REJECT * false rejects."""
    p, y = np.asarray(p_good, float), np.asarray(y)
    cands = np.unique(np.concatenate([[0.0, 1.0 + 1e-9], p, p + 1e-9]))
    best_tau, best_cost = 0.5, np.inf
    for tau in cands:
        flag = p < tau
        cost = COST_ESCAPE * np.sum((y != 0) & ~flag) + COST_FALSE_REJECT * np.sum((y == 0) & flag)
        if cost < best_cost:
            best_tau, best_cost = float(tau), cost
    return best_tau


def binary_at(p_good, y, tau):
    p, y = np.asarray(p_good, float), np.asarray(y)
    flag = p < tau
    return {"escapes": int(np.sum((y != 0) & ~flag)), "false_rejects": int(np.sum((y == 0) & flag)),
            "n_def": int(np.sum(y != 0)), "n_good": int(np.sum(y == 0))}


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval(); probs, gt = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy()); gt.extend(y.numpy())
    return np.array(gt), np.concatenate(probs)


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); pr, gt = [], []
    for x, y in loader:
        pr.extend(model(x.to(device)).argmax(1).cpu().numpy()); gt.extend(y.numpy())
    return gt, pr


def run_one(cfg, seed, images, labels, k, device):
    tr_x, tmp_x, tr_y, tmp_y = train_test_split(images, labels, test_size=0.4,
                                                random_state=seed, stratify=labels)
    va_x, te_x, va_y, te_y = train_test_split(tmp_x, tmp_y, test_size=0.5,
                                              random_state=seed, stratify=tmp_y)
    seed_everything(seed)
    g = torch.Generator(); g.manual_seed(seed)
    tr = DataLoader(DefectDataset(tr_x, tr_y, train_transform(cfg)), batch_size=BATCH,
                    shuffle=True, generator=g, num_workers=NUM_WORKERS)
    va = DataLoader(DefectDataset(va_x, va_y, EVAL_TF), batch_size=BATCH, num_workers=NUM_WORKERS)
    te = DataLoader(DefectDataset(te_x, te_y, EVAL_TF), batch_size=BATCH, num_workers=NUM_WORKERS)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, k)
    model = model.to(device)
    if "wce" in cfg:
        w = torch.tensor(class_weights(tr_y, k), dtype=torch.float32, device=device)
        crit = nn.CrossEntropyLoss(weight=w)
    else:
        crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best, state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        g_va, p_va = predict(model, va, device)
        v = float(np.mean(np.array(g_va) == np.array(p_va)))
        if v > best:
            best, state = v, {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}

    model.load_state_dict(state)
    if SAVE_CKPTS:
        torch.save(state, os.path.join(OUT, f"{cfg.replace('+', '_')}_seed{seed}.pth"))
    gt, prob = predict_proba(model, te, device)
    m = metrics_from(gt, prob.argmax(1), k)
    m["val_acc"] = best
    g_va, prob_va = predict_proba(model, va, device)
    tau = pick_threshold(prob_va[:, 0], g_va)
    m["tuned"] = {"tau": tau, **binary_at(prob[:, 0], gt, tau)}
    return m


def fmt(vals, scale=100, nd=1):
    a = np.array(vals) * scale
    return f"{a.mean():.{nd}f} ± {a.std(ddof=1):.{nd}f}" if len(a) > 1 else f"{a.mean():.{nd}f}"


# ----------------------------------------------------------------------------- main
def main():
    images, labels, names = load_metal_nut()
    k = len(names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(images)} images, classes {names}, device {device}")
    print(f"configs {CONFIGS}, seeds {SEEDS}, results -> {OUT}\n")

    res, rows = {}, []
    for cfg in CONFIGS:
        res[cfg] = []
        for seed in SEEDS:
            m = run_one(cfg, seed, images, labels, k, device)
            res[cfg].append(m)
            rows.append({"config": cfg, "seed": seed, **{x: m[x] for x in
                         ("val_acc", "acc", "bal_acc", "macro_f1", "escape_rate", "false_reject")},
                         **{f"recall_{n}": r for n, r in zip(names, m["per_class_recall"])}})
            print(f"{cfg:<11} seed {seed}: test acc {m['acc']*100:5.1f}  macroF1 {m['macro_f1']:.2f}  "
                  f"escapes {m['escape_rate']*100:4.1f}%  false rejects {m['false_reject']*100:4.1f}%  | "
                  f"tuned tau {m['tuned']['tau']:.3f}: escapes {m['tuned']['escapes']}, "
                  f"false rejects {m['tuned']['false_rejects']}")
        print()

    with open(os.path.join(OUT, "runs.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    # ---- summary table (mean ± std over seeds, pooled escapes) --------------
    lines = ["| Config | Test accuracy % | Balanced acc % | Macro F1 | Escape rate % | False-reject rate % | Pooled escapes |",
             "|---|---|---|---|---|---|---|"]
    for cfg in CONFIGS:
        R = res[cfg]
        P = np.sum([np.array(m["cm"]) for m in R], axis=0)
        esc, nd = int(P[1:, 0].sum()), int(P[1:].sum())
        lines.append(f"| {cfg} | {fmt([m['acc'] for m in R])} | {fmt([m['bal_acc'] for m in R])} | "
                     f"{fmt([m['macro_f1'] for m in R], scale=1, nd=2)} | {fmt([m['escape_rate'] for m in R])} | "
                     f"{fmt([m['false_reject'] for m in R])} | {esc}/{nd} |")

    # ---- seed-paired differences vs baseline (same splits, so this is fair) -
    lines += ["", "Seed-paired change vs baseline (mean ± std over seeds, percentage points):", "",
              "| Config | Δ accuracy | Δ macro F1 (×100) | Δ escape rate | Δ false-reject rate |", "|---|---|---|---|---|"]
    if "baseline" in res:
        B = res["baseline"]
        for cfg in CONFIGS:
            if cfg == "baseline":
                continue
            d = lambda key: [res[cfg][i][key] - B[i][key] for i in range(len(SEEDS))]
            lines.append(f"| {cfg} | {fmt(d('acc'))} | {fmt(d('macro_f1'))} | "
                         f"{fmt(d('escape_rate'))} | {fmt(d('false_reject'))} |")

    # ---- per-class recall ----------------------------------------------------
    lines += ["", "Per-class recall % (mean over seeds):", "",
              "| Config | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for cfg in CONFIGS:
        rec = np.mean([m["per_class_recall"] for m in res[cfg]], axis=0) * 100
        lines.append(f"| {cfg} | " + " | ".join(f"{r:.0f}" for r in rec) + " |")

    lines += ["", f"Operating point tuned on validation (escape cost = {COST_ESCAPE:g}x a false reject; "
              "part flagged defective when P(good) < tau):", "",
              "| Config | tau (per seed) | Escapes, argmax | Escapes, tuned tau | False rejects, argmax | False rejects, tuned tau |",
              "|---|---|---|---|---|---|"]
    for cfg in CONFIGS:
        R = res[cfg]
        P = np.sum([np.array(m["cm"]) for m in R], axis=0)
        e_t = sum(m["tuned"]["escapes"] for m in R); f_t = sum(m["tuned"]["false_rejects"] for m in R)
        nd = sum(m["tuned"]["n_def"] for m in R); ng = sum(m["tuned"]["n_good"] for m in R)
        taus = ", ".join(f"{m['tuned']['tau']:.3f}" for m in R)
        lines.append(f"| {cfg} | {taus} | {int(P[1:, 0].sum())}/{nd} | {e_t}/{nd} | "
                     f"{int(P[0, 1:].sum())}/{ng} | {f_t}/{ng} |")

    n_test = sum(np.array(res[CONFIGS[0]][0]["cm"]).sum(axis=1))
    lines += ["", f"Each test split has {n_test} images; one image is {100/n_test:.1f} accuracy points.",
              "Differences smaller than the seed-to-seed spread are noise, not findings."]
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"settings": {"seeds": SEEDS, "epochs": EPOCHS, "lr": LR, "batch": BATCH,
                                "classes": names}, "results": res}, f, indent=2)

    for cfg in CONFIGS:
        P = np.sum([np.array(m["cm"]) for m in res[cfg]], axis=0)
        print(f"\nPooled confusion matrix — {cfg} (rows = true, cols = predicted)")
        print(" " * 9 + "".join(f"{n:>9}" for n in names))
        for n, r in zip(names, P):
            print(f"{n:>9}" + "".join(f"{v:>9}" for v in r))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()

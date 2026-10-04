"""
calibrate_scale.py — derive millimetres-per-pixel from the part itself.

MVTec AD ships no physical scale, so the app's 0.1 mm/px was an arbitrary number.
This replaces it with a scale derived from a stated assumption about the part:

    ASSUMPTION: the metal nut is an M10 nut, 16 mm across flats (ISO 4032).
    (DIN 934 M10 is 17 mm. Change REF_MM if you assume a different part.)

Method
------
1. For every train/good image: Otsu threshold -> largest outer contour = the nut.
2. Measure a reference feature in pixels (default: short side of the minimum-area
   rectangle, which equals "across flats" for a hexagonal outline).
3. mm_per_px = REF_MM / median(reference_px) over all good images. Using the median
   of many good parts mimics a calibration step: the camera is fixed, so one scale
   applies to every image, and a defect can't distort it.
4. Report the spread. If the coefficient of variation is above ~2%, either the
   segmentation is unstable or the parts are not imaged at a constant scale.

Check the overlay images it saves. If the red box doesn't hug the feature you
meant to calibrate on, change REF_FEATURE.

Honest framing for interviews: "scale from an assumed nominal part size", not
"calibrated measurement". ISO 4032 allows 15.73-16.00 mm across flats for M10,
so the part tolerance alone adds about 1.7% scale uncertainty. A real line would
use a calibration target in the part plane, and ideally a telecentric lens.

Run:
    .venv_ManuVisionAI\\Scripts\\python.exe calibrate_scale.py
"""
import os, sys, json, atexit, glob
from datetime import datetime

import cv2
import numpy as np

DATASET = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase\metal_nut"
REF_MM = 16.0                        # M10 across flats, ISO 4032 (DIN 934: 17.0)
REF_FEATURE = "minrect_short"        # or "minrect_long", "enclosing_circle"
MASK_SIZE = 256                      # the U-Net works at 256x256; the app measures there
N_OVERLAYS = 4

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", f"calibration_{datetime.now():%Y%m%d_%H%M%S}")
os.makedirs(OUT, exist_ok=True)


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


def part_contour(bgr):
    """Largest outer contour of the part, whichever way the contrast goes."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([th[0], th[-1], th[:, 0], th[:, -1]])
    if border.mean() > 127:                       # background came out white -> invert
        th = cv2.bitwise_not(th)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def measure(contour):
    (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
    _, radius = cv2.minEnclosingCircle(contour)
    return {"minrect_short": min(w, h), "minrect_long": max(w, h),
            "enclosing_circle": 2 * radius, "area_px": cv2.contourArea(contour)}


def overlay(bgr, contour, path):
    vis = bgr.copy()
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)                       # green: outline
    box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32)
    cv2.drawContours(vis, [box], -1, (0, 0, 255), 2)                           # red: min-area rect
    (x, y), r = cv2.minEnclosingCircle(contour)
    cv2.circle(vis, (int(x), int(y)), int(r), (255, 0, 0), 2)                  # blue: enclosing circle
    cv2.imwrite(path, vis)


def main():
    files = sorted(glob.glob(os.path.join(DATASET, "train", "good", "*.png")))
    if not files:
        sys.exit(f"No images found under {DATASET}")
    refs, sizes, failed = [], set(), 0
    for i, f in enumerate(files):
        bgr = cv2.imread(f)
        sizes.add(bgr.shape[:2])
        c = part_contour(bgr)
        if c is None or cv2.contourArea(c) < 0.05 * bgr.shape[0] * bgr.shape[1]:
            failed += 1
            continue
        refs.append(measure(c)[REF_FEATURE])
        if i < N_OVERLAYS:
            overlay(bgr, c, os.path.join(OUT, f"calib_check_{i}.png"))

    refs = np.array(refs)
    med = float(np.median(refs))
    q1, q3 = np.percentile(refs, [25, 75])
    cv_pct = float(refs.std(ddof=1) / med * 100)
    h, w = next(iter(sizes)) if len(sizes) == 1 else max(sizes)
    mm_px = REF_MM / med
    mm_px_mask = mm_px * (w / MASK_SIZE)

    print(f"images measured : {len(refs)} of {len(files)} (failed segmentation: {failed})")
    print(f"image size      : {w} x {h}" + ("" if len(sizes) == 1 else f"  (WARNING: sizes vary: {sizes})"))
    print(f"feature         : {REF_FEATURE} = {REF_MM} mm (assumed)")
    print(f"reference px    : median {med:.1f}, IQR {q1:.1f}-{q3:.1f}, CV {cv_pct:.2f}%")
    if cv_pct > 2:
        print("WARNING: spread above 2% — check the overlays before trusting this scale.")
    print(f"\nmm per pixel, original {w}x{h}   : {mm_px:.4f}")
    print(f"mm per pixel, {MASK_SIZE}x{MASK_SIZE} mask      : {mm_px_mask:.4f}   <- use this in app.py / api.py")
    print(f"old arbitrary value               : 0.1000  (off by {mm_px_mask / 0.1:.2f}x)")
    print(f"\nscale uncertainty from part tolerance alone (ISO 4032 M10, 15.73-16.00 mm): ~1.7%")

    cal = {"assumption": f"part = M10 nut, {REF_FEATURE} = {REF_MM} mm (ISO 4032 across flats)",
           "ref_feature": REF_FEATURE, "ref_mm": REF_MM, "n_images": int(len(refs)),
           "median_ref_px": med, "iqr_px": [float(q1), float(q3)], "cv_percent": cv_pct,
           "image_size": [w, h], "mask_size": MASK_SIZE,
           "mm_per_px_original": mm_px, "mm_per_px_mask": mm_px_mask,
           "created": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(OUT, "calibration.json"), "w", encoding="utf-8") as fh:
        json.dump(cal, fh, indent=2)
    with open(os.path.join(HERE, "calibration.json"), "w", encoding="utf-8") as fh:
        json.dump(cal, fh, indent=2)                      # copy next to app.py for the app to load
    print(f"\nSaved calibration.json and {min(N_OVERLAYS, len(files))} overlay images to {OUT}")


if __name__ == "__main__":
    main()

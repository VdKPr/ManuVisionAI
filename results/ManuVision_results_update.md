# ManuVision AI — results update (4 October 2026)

Three blocks to paste: a DEVLOG entry, a README results section, and resume bullets. Every number comes from the logs you ran on 4 October 2026.

---

## 1. DEVLOG.md — append as Level 8

## Level 8: Seeded evaluation, augmentation ablation, segmentation, measurement, latency and calibration (4 Oct 2026)

**Protocol.** Stratified 60/20/20 split, fully seeded (Python, NumPy, torch, DataLoader), identical splits for every configuration. ResNet18 (ImageNet weights), Adam lr 1e-4, 20 epochs, batch 16. Checkpoint chosen on validation accuracy; test scored once. 67 test images per split (18–19 defects).

### Run 1 — five configurations, seeds 42–44

| Configuration | Test accuracy % | Macro F1 | Escapes (pooled) | False rejects (pooled) |
|---|---|---|---|---|
| Baseline (no augmentation, plain CE) | 92.5 ± 6.5 | 0.84 ± 0.11 | 4/54 | 5/147 |
| Augmentation | 95.5 ± 2.6 | 0.88 ± 0.08 | 5/54 | 1/147 |
| Weighted CE | 92.5 ± 1.5 | 0.84 ± 0.07 | 5/54 | 5/147 |
| Augmentation + weighted CE | 98.0 ± 2.3 | 0.97 ± 0.03 | 2/54 | 2/147 |
| Augmentation + mirroring | 99.0 ± 1.7 | 0.98 ± 0.04 | 2/54 | 0/147 |

### Run 2 — confirmation on fresh seeds 45–47 (configurations fixed before running)

| Configuration | Test accuracy % | Macro F1 | Escapes (pooled) | False rejects (pooled) |
|---|---|---|---|---|
| Baseline | 88.6 ± 3.8 | 0.75 ± 0.10 | 15/57 | 2/144 |
| Augmentation + weighted CE | 94.0 ± 1.5 | 0.86 ± 0.04 | 8/57 | 1/144 |
| Augmentation + mirroring | 95.0 ± 3.4 | 0.88 ± 0.08 | 6/57 | 0/144 |

Augmentations: rotation ±180°, shift 5%, scale 0.95–1.05, brightness/contrast 0.2, Gaussian blur (p 0.3), horizontal and vertical mirroring. No hue or saturation jitter, because "color" is a defect class.

### Findings

- Augmentation with mirroring improved accuracy and macro F1 on all six seeds. Over six seeds: accuracy 90.6% ± 5.2% → 97.0% ± 3.3%, macro F1 0.79 → 0.93, escapes 19/111 (17%) → 8/111 (7%), false rejects 7/291 → 0/291.
- The gains landed on the weak classes. Fresh-seed recall: bent 47% → 87%, color 47% → 62%, scratch 60% → 80%. Flip stayed at 100% throughout.
- Weighted cross-entropy alone made no measurable difference.
- Hypothesis refuted: I expected mirroring to confuse the flip class (flip = nut upside down), so I ran it as a negative control. Flip recall stayed at 100%, and the mirrored setup came out best.
- The seeded baseline (90.6% ± 5.2%) sits below the earlier unseeded 93.5% ± 3.1% for the same configuration. Training randomness alone moves results by several points.
- Operating point tuned on validation (escape cost 10× a false reject). For the baseline, escapes fell from 15 to 8 of 57 while false rejects rose from 2 to 14 of 144. For augmentation + mirroring there was no gain (6 → 7). With about 18 validation defects the cutoff is underdetermined; a margin-based or cross-validated cutoff is next.

### Segmentation (eval_segmentation.py — 18 defect and 5 good validation images)

| Configuration | Dice | IoU | Precision | Recall | Good images with a false alarm |
|---|---|---|---|---|---|
| Threshold 0.50, raw | 0.822 | 0.722 | 0.959 | 0.956 | 0/5 |
| Threshold 0.25, raw | 0.830 | 0.729 | 0.925 | 0.973 | 5/5 |
| Threshold 0.25 + open + close + min-size (deployed) | 0.833 | 0.733 | 0.929 | 0.971 | 0/5 |

Post-processing adds only 0.003 Dice. Its real effect is removing the speckle false alarms that the lower threshold introduces. On this checkpoint, threshold 0.5 with no cleanup is nearly as good. Caveat: this validation split also selected the checkpoint, so these numbers are optimistic.

### Measurement (measurement_eval.py — pixel space, 18 validation images)

| Quantity | Median absolute error | Mean absolute % error | Bias |
|---|---|---|---|
| Maximum region length | 3.0% | 10.8% | −7.8% |
| Total defect area | 13.7% | 22.2% | +4.4% |

- Region count: 35 predicted vs 25 in ground truth (+40%); exact match on 13 of 18 images.
- Accept/reject agreement with ground truth, across a sweep of limits: 83–100%.
- The largest errors were on thin and small defects: one scratch measured 53% under in area, one color defect 56% under in length.
- Flip masks cover the whole nut (about 49% of the 256² image), so a flip's "defect area" is really the part's area.

### Latency (timing_study.py — RTX 4060 Laptop GPU)

| Stage | Mean (ms) — logged run | Std (ms) |
|---|---|---|
| Classify (ResNet18) | 4.7 | 0.9 |
| Segment (U-Net) | 15.6 | 1.7 |
| Post-process | 2.1 | 0.2 |
| Measure | 0.3 | 0.0 |
| Root cause (GPT-4o-mini, 5 calls) | 8,571 | 948 |

The good-part path takes 4.7 ms. The defective path takes 22.7 ms locally and 8.6 s with the LLM, which is 99.7% of that time. An earlier run gave 21.6 ms and 11.2 s, because the LLM call is network-bound and varies. Skipping segmentation for good parts saves about 79% of local time (79.2% in the logged run); the earlier "about 60%" was an unmeasured estimate. Log: results/timing_study.txt.

### Calibration (calibrate_scale.py)

- The part's outline width (short side of its minimum-area rectangle) was measured on 220 good images: median 565 px, IQR 563.9–565.8, spread 0.34%.
- Assuming a 16 mm part width (the size of an M10 nut) gives 0.0283 mm/px at 700×700 and 0.0774 mm/px on the 256×256 mask.
- The old 0.1 mm/px default overstated lengths by about 29% and areas by about 67%.
- This is a nominal-size assumption, not a metrology calibration. The outline has eight corners, so the part isn't a standard hex nut.

### Next

- Ship the augmentation + mirroring checkpoint and load the pixel size from calibration.json.
- Fix the PASS-on-empty-mask bug so that case returns REVIEW.
- Evaluate segmentation on a held-out test split.
- Reduce escapes further: higher input resolution for color and scratch, plus an anomaly-detector check on parts predicted good.

---

## 2. README.md — replace the Results section

## Results (held-out and seeded — details in DEVLOG, Level 8)

- **Classifier** (ResNet18, MVTec AD metal_nut, 5 classes): 97.0% ± 3.3% test accuracy over six seeds with augmentation (95.0% on fresh seeds), macro F1 0.93, escaped defects 7% (8/111), false rejects 0/291. Without augmentation: 90.6% ± 5.2%, escapes 17%.
- **Segmentation** (U-Net): per-image Dice 0.833 and IoU 0.733 on 18 validation defect images; no false alarms on good parts after post-processing.
- **Measurement**: length median error 3%, area median error 14%; accept/reject agreement with ground truth 83–100%.
- **Latency** (RTX 4060): about 5 ms for good parts and about 23 ms for defective parts locally; the optional LLM root-cause step adds 8–11 s.
- **Scale**: 0.0283 mm per pixel, derived from an assumed 16 mm part width.

README housekeeping: rename ManuVision_README.md to README.md (today's README.md is only the dataset citation), fix the Quick Start clone URL (it says ManuVision-AI), tick the finished roadmap items, and remove the unsourced market statistics.

---

## 3. Resume — replace the ManuVision AI bullets

- Built a defect-inspection pipeline on MVTec AD (metal_nut): ResNet18 classifier, U-Net segmentation (per-image Dice 0.83), dimensional measurement with a part-derived mm scale, and LLM root-cause notes; served with FastAPI and Streamlit.
- Ran a seeded augmentation ablation, confirmed on fresh splits: accuracy 90.6% → 97.0% and escaped defects 17% → 7% over six seeds; my hypothesis that mirroring would hurt the flip class was refuted.
- Measured every stage: about 22 ms local inspection on an RTX 4060 (the LLM step is over 99% of the defective path); LangGraph routing skips segmentation for good parts (about 79% faster locally).

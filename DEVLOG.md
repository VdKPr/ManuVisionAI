# ManuVision AI — Development Log

## Level 1: Classification
- Trained ResNet18 on MVTec AD metal_nut, 20 epochs, 89% accuracy
- Built Streamlit app with SQL logging and dashboard

## Level 2: Segmentation
- Trained U-Net on ground truth masks
- Problem: mask completely black at threshold 0.5
- Debug: raw output max=0.99, model detecting but threshold too high
- Tried 0.15 → entire part surface detected (too sensitive)
- Tried 0.2 → scattered noise
- Fix: post-processing (binary_opening + binary_closing + min-size filter)
- Result at 0.25: detects defect regions with acceptable noise

## Level 3: LLM Root Cause Analysis
- GPT-4o-mini via LangChain with manufacturing-specific prompt
- Generates: probable cause, corrective actions, severity, process parameters

## Level 4: FastAPI REST Backend
- /inspect, /dashboard, /stats, /health endpoints
- Swagger auto-documentation

## Level 5: Docker Containerization
- Dockerfile + docker-compose for reproducible deployment

## Level 6: LangGraph Agentic AI
- Agent autonomously decides which tools to invoke
- Good parts: skip segmentation (2 steps instead of 5)

## Level 7: Multi-Product Experiments

### Attempt 1: Multi-class classifier (73 classes)
- Trained ResNet18 across all 15 MVTec categories
- Result: 80% accuracy but macro F1 = 0.20
- Problem: model predicted "good" for everything (4116 good vs 2-30 per defect)
- Learning: class imbalance makes multi-class impractical with small defect samples

### Attempt 2: Binary classifier (good vs defective)
- Simplified to 2 classes across all categories
- Result: 76% accuracy but 0% defect recall
- Problem: "good" bottle looks nothing like "good" metal_nut — no single boundary works
- Learning: cross-product generalization needs per-category models

### Attempt 3: Autoencoder anomaly detection (all categories)
- Trained on 4096 good images across all categories
- Result: 93% good detection but only 3% defect detection
- Problem: model became generic image reconstructor, defects reconstructed equally well

### Attempt 4: Autoencoder (per-category, metal_nut only)
- Trained on 220 good metal_nut images only
- Added tight bottleneck (16384 → 32 latent dims)
- Used 95th percentile error instead of mean
- Tested multiple thresholds (0.5×std to 2×std)
- Best result: 86% good, 29% defect detection
- Problem: error distributions still overlap — basic autoencoder insufficient
- Learning: state-of-art methods (PatchCore, EfficientAD) use pretrained features, not raw reconstruction

### Conclusion
- Per-category ResNet18 classifier (89%) remains the working solution
- Each new product line needs its own training run
- For production: PatchCore or EfficientAD for unsupervised anomaly detection


---

## Correction notice (post-audit)

The figures above are the ORIGINAL run and several are superseded. Corrected
values, obtained under a held-out 60/20/20 protocol at lr=1e-4:

| Claim above | Corrected |
|---|---|
| Per-category ResNet18 89% | 93.5% +/- 3.1% held-out (3 splits) |
| Multi-class 73-class: 80% acc, macro F1 0.20 | 90.4% acc, macro F1 0.62 |
| Binary: 76% acc, 0% defect recall | 94.5% acc, 85.3% defect recall |
| Autoencoder per-category 29% defect detection | unchanged -- this result is real |

Two of the three apparent failures were artifacts, not findings:

* The 73-class result came from a label bug -- `label_id` started at 0, the
  same index as `good`, merging 20 bottle_broken_large images into the
  conforming class and misaligning every later label.
* The binary result came from `lr=0.001`, an order of magnitude too high for
  fine-tuning a pretrained ResNet18. It collapses the model onto the majority
  class, which looks exactly like "the task is impossible".

The autoencoder failure survives scrutiny because it has a mechanism:
reconstruction error is ANTI-correlated with defectiveness for bent, scratch
and color (they reconstruct with LOWER error than good parts), so no threshold
can work. Detection is 100% on flip and 4/70 on everything else.

Also fixed: app.py / api.py / agent.py loaded `best_defect_model.pth`, which the
multi-product scripts overwrite. Since Sep 4 that file held a 2-class model, so
loading it into the 5-class head raised a size-mismatch error at startup. The
training scripts now write distinct filenames.


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
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

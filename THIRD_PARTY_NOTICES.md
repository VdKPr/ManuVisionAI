# Third-Party Notices

The MIT licence in `LICENSE` covers the source code in this repository. It does **not** cover the dataset this project is trained and evaluated on, which carries its own terms.

---

## MVTec Anomaly Detection dataset (MVTec AD)

**Licence:** [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**Copyright:** © 2019–2021 MVTec Software GmbH

The dataset is **not redistributed** in this repository. It is excluded by `.gitignore` and must be obtained directly from MVTec:
<https://www.mvtec.com/company/research/datasets/mvtec-ad>

### Required citation

MVTec asks that both papers be cited:

> Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger.
> **"MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection."**
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 9592–9600.

> Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger.
> **"The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection."**
> *International Journal of Computer Vision*, 129(4):1038–1059, 2021.

### What the licence means for this project

| Term | Consequence here |
|---|---|
| **BY** — Attribution | The citations above must appear wherever this work is published, including the README and any paper or presentation. |
| **NC** — NonCommercial | This project may be used for research, study, and portfolio purposes only. It may not be used commercially, sold, or deployed in a revenue-generating setting without separate permission from MVTec. |
| **SA** — ShareAlike | Adapted material must be shared under the same licence. Figures in this repository that display or are derived from dataset images (for example sample grids, segmentation overlays, defect visualisations) are adapted material and inherit CC BY-NC-SA 4.0 — they are **not** covered by the MIT licence. |

### Trained model weights

Whether model weights trained on CC BY-NC-SA data inherit that licence is **unsettled**. There is no controlling case law and reasonable readings differ.

This repository takes the conservative position: checkpoints (`*.pth`) are excluded by `.gitignore` and are not distributed. If they are ever published, they should be treated as carrying the NonCommercial restriction, not the MIT licence.

---

## Other dependencies

Python package dependencies are listed in `requirements.txt` and are each governed by their own licences (predominantly MIT, BSD-3-Clause, and Apache-2.0). They are not vendored into this repository.

The root-cause analysis stage calls the OpenAI API. Use of that service is governed by OpenAI's terms, and an API key is required; no key is included in this repository.

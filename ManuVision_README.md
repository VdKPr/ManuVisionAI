# 🔍 ManuVision AI — AI Manufacturing Quality Inspector

An AI-powered manufacturing defect detection system that identifies product defects using computer vision and logs inspection results with an analytics dashboard.

Trained on the MVTec Anomaly Detection dataset. Built with PyTorch, Streamlit, and SQLite.

## What It Does

1. **Upload** a product image from the production line
2. **AI detects** whether the part is good or defective — classifies defect type (bent, scratch, color, flip) with confidence scores
3. **Logs** every inspection to a SQL database automatically
4. **Dashboard** shows total inspections, defect rate, defect type breakdown, and recent inspection history

## Demo

![Defect Detected](D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\Screenshot1.png)
![Dashboard](D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\Screenshot2.png)

## Results

- **Model:** ResNet18 (pretrained on ImageNet, fine-tuned on MVTec AD)
- **Dataset:** MVTec Anomaly Detection — metal_nut category
- **Validation Accuracy:** 89%
- **Defect Types Detected:** bent, color, flip, scratch
- **Inference:** Real-time, <100ms per image on CPU

## Tech Stack

- **Detection Model:** PyTorch, ResNet18, Transfer Learning
- **Dataset:** MVTec AD (industry-standard anomaly detection benchmark)
- **Database:** SQLite (inspection logging + analytics)
- **Frontend:** Streamlit (interactive dashboard)
- **Language:** Python 3.10+

## Architecture

```
Product Image → ResNet18 Classifier → Defect/Good + Confidence Score
                                    ↓
                              SQLite Database (log every inspection)
                                    ↓
                              Streamlit Dashboard (trends, defect rates)
```

## Quick Start

```bash
git clone https://github.com/VdKPr/ManuVision-AI.git
cd ManuVision-AI
pip install torch torchvision streamlit Pillow matplotlib scikit-learn
streamlit run app.py
```

## Project Structure

```
ManuVision-AI/
├── explore_mvtec.py        # Dataset exploration and visualization
├── train_detector.py       # Model training script
├── app.py                  # Streamlit app with dashboard
├── best_defect_model.pth   # Trained model weights
├── defect_logs.db          # SQLite inspection database
├── mvtec_samples.png       # Sample defect images
└── README.md               # This file
```

## Roadmap

- [ ] LLM-powered root cause analysis — AI explains WHY the defect happened and recommends process fixes
- [ ] FastAPI backend for production deployment
- [ ] Docker containerization
- [ ] Multi-product support (currently metal_nut only)
- [ ] Real-time camera feed integration
- [ ] Deploy to cloud with public API

## Why This Project Matters

Manufacturing quality inspection is a $4.5B market. Human inspectors miss 20-30% of defects and fatigue within 2 hours. This system achieves 89% accuracy with zero fatigue, logging every decision for traceability — addressing real industry pain points in automated quality control.

## Author

**Varad Pawar** — M.Tech, IIT Bombay | GATE AIR 208 | AI/ML Engineer
- LinkedIn: https://linkedin.com/in/varadkpawar
- GitHub: https://github.com/VdKPr

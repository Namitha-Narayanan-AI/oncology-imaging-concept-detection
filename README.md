# Oncology Imaging Concept Detection

A PyTorch medical imaging pipeline built toward AI-assisted oncology — starting with a chest X-ray classification baseline and progressing to lung nodule detection.

---

## Overview

Medical imaging pipelines for oncology present unique evaluation challenges — lesion detection, class imbalance, and asymmetric misclassification costs all require clinical context to interpret correctly. This project builds toward that space systematically, beginning with a reproducible chest X-ray baseline before progressing to oncology-specific imaging and evaluation.

---

## Phase 1 — Chest X-Ray Baseline

Chest X-ray classification (pneumonia vs. normal) serves as a controlled starting point. The data is well-understood, publicly available, and widely benchmarked — suitable for validating the pipeline architecture and surfacing the metric trade-offs that matter in clinical AI.

### Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) · Kermany et al., *Cell* 2018  
5,216 training images · 624 test images · Classes: `NORMAL`, `PNEUMONIA`

### Model

- `SimpleCNN` — 3-layer convolutional backbone  
- Input: `[B, 3, 224, 224]` 
- Output: binary logits `[B, 2]`  
- Loss: Cross-Entropy · Optimiser: Adam (`lr=1e-3`)

### Training

| Epoch | Train Loss | Train Acc | Val Acc |
|---|---|---|---|
| 1 | 0.1924 | 91.93% | 56.25% |
| 2 | 0.0757 | 97.26% | 68.75% |
| 3 | 0.0564 | 97.89% | 93.75% |

### Test Set Evaluation (624 images)

| Metric | Score |
|---|---|
| Accuracy | 73.24% |
| Precision | 80.30% |
| Recall | 73.24% |
| F1-Score | 68.40% |

**Confusion matrix:**

```
                    Pred: Normal   Pred: Pneumonia
Actual: Normal            69              165
Actual: Pneumonia          2              388
```

The model misses 2 pneumonia cases out of 390 (high sensitivity) but over-flags 165 normal cases as diseased (low specificity). In a clinical screening context, this is an expected trade-off — missing a true positive carries greater risk than a false alarm that triggers further investigation.

This result makes concrete a foundational challenge in clinical AI: **accuracy alone is not a meaningful metric** under class imbalance or asymmetric misclassification costs. Sensitivity, specificity, and their trade-offs must be interpreted relative to the clinical use case — whether screening, diagnosis, or triage — each carrying different tolerance thresholds.

---

## Outputs

### Confusion Matrix
![Confusion Matrix](results/figures/confusion_matrix.png)

### Prediction Examples
![Prediction Examples](results/figures/prediction_examples.png)

---

## Project Structure

```
oncology-imaging-concept-detection/
├── src/
│   ├── dataset.py
│   ├── download_dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize_results.py
├── results/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   └── prediction_examples.png
│   └── metrics/
│       └── test_metrics.json
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── requirements.txt
└── README.md
```

---

## Direction

The baseline establishes the core workflow. The project will next focus on oncology-relevant imaging, with emphasis on lung imaging, radiomics features, visual explainability, and clinically meaningful evaluation.

---

## Setup

```bash
git clone https://github.com/Namitha-Narayanan-AI/oncology-imaging-concept-detection
cd oncology-imaging-concept-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python src/download_dataset.py   # downloads chest X-ray data via KaggleHub
python src/train.py
python src/evaluate.py
python src/visualize_results.py
```

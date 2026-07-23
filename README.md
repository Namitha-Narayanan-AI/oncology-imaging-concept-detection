# Oncology Imaging Concept Detection

A reproducible PyTorch medical-imaging research pipeline built toward **AI-assisted oncology**.

The project begins with a controlled chest X-ray classification baseline to establish the full evaluation workflow — training, clinical metrics, error analysis, explainability, and decision-threshold analysis — before moving into **oncology-specific imaging, radiomics, and lesion-level analysis**.

---

## Research Goal

Medical-imaging models should not be judged by accuracy alone.

This project investigates how model behaviour changes when we examine:

- class-specific errors,
- sensitivity and specificity,
- false-positive and false-negative trade-offs,
- visual explanations,
- prediction confidence,
- and decision thresholds.

**Phase 1** uses pneumonia-vs-normal chest X-rays as a controlled baseline.  
**Phase 2** will transition to cancer-specific imaging.

---

## Phase 1 — Chest X-Ray Baseline

**Status:** Completed

### Dataset

- **Task:** `NORMAL` vs `PNEUMONIA`
- **Training images:** 5,216
- **Test images:** 624
- **Source:** Chest X-Ray Images (Pneumonia), associated with Kermany et al., *Cell* (2018)
- **License:** CC BY 4.0

Dataset links:

- [Kaggle dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [Mendeley source](https://data.mendeley.com/datasets/rscbjbr9sj/2)

### Baseline Model

| Component | Configuration |
|---|---|
| Architecture | `SimpleCNN` |
| Input | `3 × 224 × 224` |
| Output | 2 logits |
| Loss | Cross-Entropy |
| Optimiser | Adam |
| Learning rate | `1e-3` |
| Epochs | 3 |

The simple architecture is intentional: Phase 1 focuses on building and interrogating the evaluation pipeline rather than claiming state-of-the-art pneumonia performance.

---

## Baseline Results

### Standard Test Metrics

| Metric | Score |
|---|---:|
| Accuracy | **73.24%** |
| Weighted Precision | **80.30%** |
| Weighted Recall | **73.24%** |
| Weighted F1 | **68.40%** |

### Confusion Matrix

| Actual | Predicted Normal | Predicted Pneumonia |
|---|---:|---:|
| Normal | 69 | 165 |
| Pneumonia | 2 | 388 |

![Confusion Matrix](results/figures/confusion_matrix.png)

The model detects almost every pneumonia case, but it also labels many normal images as pneumonia. This is a useful example of why aggregate accuracy does not fully describe clinical behaviour.

---

## Clinical Error Profile

| Clinical Metric | Score |
|---|---:|
| Sensitivity | **99.49%** |
| Specificity | **29.49%** |
| False-negative rate | **0.51%** |
| False-positive rate | **70.51%** |
| Positive predictive value | **70.16%** |
| Negative predictive value | **97.18%** |
| Balanced accuracy | **64.49%** |

At the default decision rule, the model behaves like a **high-sensitivity, low-specificity classifier**.

- Only **2 / 390** pneumonia cases are missed.
- **165 / 234** normal cases are incorrectly labelled as pneumonia.

The acceptable operating point depends on the intended clinical use case and on the relative cost of false positives and false negatives.

---

## Explainability — Grad-CAM

Grad-CAM was used to inspect whether misclassified images were being judged from clinically meaningful lung regions or from possible shortcut features.

### False-positive examples

<p align="center">
  <img src="results/figures/gradcam/false_positive/0_NORMAL_pred_PNEUMONIA.png" width="47%" />
  <img src="results/figures/gradcam/false_positive/1_NORMAL_pred_PNEUMONIA.png" width="47%" />
</p>

### False-negative examples

<p align="center">
  <img src="results/figures/gradcam/false_negative/390_PNEUMONIA_pred_NORMAL.png" width="47%" />
  <img src="results/figures/gradcam/false_negative/391_PNEUMONIA_pred_NORMAL.png" width="47%" />
</p>

Several false-positive examples show broad activation across lung fields, ribs, and image boundaries rather than a clearly localised abnormality. One false-negative example shows attention near an image marker rather than the lung field.

These observations suggest possible **shortcut learning or sensitivity to acquisition-related cues**, although Grad-CAM alone cannot establish causal model reasoning.

Detailed notes: [`docs/error_analysis_notes.md`](docs/error_analysis_notes.md)

---

## Decision-Threshold Analysis

The baseline originally predicts the class with the larger output score. For a two-class softmax model, this corresponds to an implicit probability threshold near `0.50`.

To examine the model as a clinical decision system, pneumonia probabilities were collected and the same trained network was evaluated across multiple thresholds.

**No retraining was performed.**

### Default threshold: `0.50`

| Metric | Score |
|---|---:|
| Sensitivity | **99.49%** |
| Specificity | **29.49%** |
| Balanced accuracy | **64.49%** |
| Pneumonia precision | **70.16%** |
| Pneumonia F1 | **82.29%** |
| False positives | **165** |
| False negatives | **2** |

### Highest exploratory balanced accuracy: `0.999`

| Metric | Score |
|---|---:|
| Accuracy | **82.37%** |
| Sensitivity | **84.10%** |
| Specificity | **79.49%** |
| Balanced accuracy | **81.79%** |
| Pneumonia precision | **87.23%** |
| Pneumonia F1 | **85.64%** |
| False positives | **48** |
| False negatives | **62** |

![Threshold Analysis](results/figures/threshold_analysis.png)

Increasing the threshold greatly reduces false positives, but at very high thresholds the number of missed pneumonia cases rises sharply.

> **There is no universally best threshold. The appropriate operating point depends on the clinical objective and the relative cost of different errors.**

### Methodological note

The threshold sweep was performed **post hoc on the existing test predictions**. Therefore, `0.999` is **not** presented as an optimised clinical threshold.

A rigorous workflow would:

1. train on the training set,
2. select the operating threshold on an independent validation cohort,
3. freeze that threshold,
4. evaluate once on an untouched test cohort.

---

## What Phase 1 Established

```text
Dataset
  ↓
Training
  ↓
Standard evaluation
  ↓
Clinical metrics
  ↓
Error analysis
  ↓
Grad-CAM
  ↓
Probability extraction
  ↓
Threshold analysis
  ↓
Clinical interpretation
```

The main lesson is that medical-imaging performance must be interpreted through the **error profile and intended clinical use**, not through accuracy alone.

---

## Repository Structure

```text
oncology-imaging-concept-detection/
├── docs/
│   ├── notes.md
│   └── error_analysis_notes.md
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── results/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   ├── prediction_examples.png
│   │   ├── threshold_analysis.png
│   │   └── gradcam/
│   └── metrics/
│       ├── test_metrics.json
│       ├── clinical_metrics.json
│       ├── error_analysis.csv
│       └── threshold_analysis.csv
├── src/
│   ├── dataset.py
│   ├── download_dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── error_analysis.py
│   ├── gradcam_analysis.py
│   ├── threshold_analysis.py
│   └── visualize_results.py
├── requirements.txt
└── README.md
```

---

## Reproduce the Baseline

### 1. Clone and install

```bash
git clone https://github.com/Namitha-Narayanan-AI/oncology-imaging-concept-detection
cd oncology-imaging-concept-detection

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python src/download_dataset.py
python src/train.py
python src/evaluate.py
python src/error_analysis.py
python src/gradcam_analysis.py
python src/threshold_analysis.py
python src/visualize_results.py
```

---

## Research Direction

### Phase 2 — Oncology Imaging

The next stage will move from the controlled pneumonia baseline to **cancer-specific imaging**.

Planned direction:

- thoracic CT,
- lung nodule analysis,
- lesion-level classification,
- radiomic feature extraction,
- malignancy prediction,
- tumour-focused explainability,
- clinically meaningful validation.

The aim is to evolve this repository into an oncology-focused computational imaging project while preserving the same emphasis on **reproducibility, interpretability, and clinically relevant evaluation**.

---

## Limitations

- Phase 1 is a pneumonia benchmark, not an oncology task.
- The baseline uses a small custom CNN rather than a clinically validated architecture.
- The dataset is imbalanced.
- The predefined validation split is limited.
- Threshold analysis is exploratory and uses test predictions.
- Grad-CAM is a coarse post-hoc explanation method.
- External generalisation has not yet been evaluated.

---

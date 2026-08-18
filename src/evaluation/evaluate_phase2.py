"""Evaluation utilities for binary radiologist-assessed malignancy risk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, roc_auc_score,
)


def predict(model, loader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
            for i, probability in enumerate(probabilities):
                rows.append({
                    "annotation_id": batch["annotation_id"][i],
                    "patient_id": batch["patient_id"][i],
                    "malignancy_rating": int(batch["malignancy_rating"][i]),
                    "reference_label": int(batch["target"][i]),
                    "probability": float(probability),
                })
    return pd.DataFrame(rows)


def binary_metrics(labels, probabilities, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    two_classes = np.unique(labels).size == 2
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(labels, probabilities)) if labels.size else float("nan"),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision_ppv": float(precision_score(labels, predictions, zero_division=0)),
        "npv": npv,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)) if two_classes else float("nan"),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def save_evaluation(predictions: pd.DataFrame, threshold: float, output_dir: str | Path) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    predictions = predictions.copy()
    predictions["prediction"] = (predictions["probability"] >= threshold).astype(int)
    predictions[[
        "annotation_id", "malignancy_rating", "reference_label",
        "probability", "prediction",
    ]].to_csv(output / "test_predictions.csv", index=False)
    metrics = binary_metrics(predictions.reference_label, predictions.probability, threshold)
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")
    return metrics

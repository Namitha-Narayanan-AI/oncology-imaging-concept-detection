"""Generate the final paper-facing Model 1 versus Model 2 figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


BASELINE = Path("results/phase2/baseline_final")
MULTITASK = Path("results/phase2/multitask_final")
FIGURES = MULTITASK / "figures"
COLORS = {"m1": "#2774AE", "m2": "#D97706", "low": "#4C78A8", "high": "#E45756"}


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _style(axis, title, xlabel, ylabel):
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel(xlabel); axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2); axis.spines[["top", "right"]].set_visible(False)


def _save(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    baseline_pred = pd.read_csv(BASELINE / "test_predictions.csv")
    multitask_pred = pd.read_csv(MULTITASK / "test_predictions.csv")
    baseline_metrics = _json(BASELINE / "test_metrics.json")
    multitask_metrics = _json(MULTITASK / "test_metrics.json")

    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    for label, frame, y, p, metrics, color in (
        ("Model 1 — Single-task", baseline_pred, "reference_label", "probability", baseline_metrics, COLORS["m1"]),
        ("Model 2 — Multi-label concept", multitask_pred, "malignancy_label", "malignancy_probability", multitask_metrics, COLORS["m2"]),
    ):
        fpr, tpr, _ = roc_curve(frame[y], frame[p])
        axis.plot(fpr, tpr, linewidth=2.2, color=color, label=f"{label} (AUC {metrics['roc_auc']:.3f})")
    axis.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
    _style(axis, "Held-out malignancy-risk ROC", "False-positive rate", "True-positive rate")
    axis.legend(frameon=False); _save(figure, "model_comparison_roc.png")

    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    for label, frame, y, p, metrics, color in (
        ("Model 1 — Single-task", baseline_pred, "reference_label", "probability", baseline_metrics, COLORS["m1"]),
        ("Model 2 — Multi-label concept", multitask_pred, "malignancy_label", "malignancy_probability", multitask_metrics, COLORS["m2"]),
    ):
        precision, recall, _ = precision_recall_curve(frame[y], frame[p])
        axis.plot(recall, precision, linewidth=2.2, color=color, label=f"{label} (AP {metrics['pr_auc']:.3f})")
    axis.axhline(multitask_pred.malignancy_label.mean(), linestyle="--", color="#777777", linewidth=1, label="Prevalence")
    _style(axis, "Held-out malignancy-risk precision–recall", "Recall", "Precision")
    axis.legend(frameon=False); _save(figure, "model_comparison_pr.png")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, title, frame, y, p in (
        (axes[0], "Model 1 — Single-task", baseline_pred, "reference_label", "probability"),
        (axes[1], "Model 2 — Multi-label concept", multitask_pred, "malignancy_label", "malignancy_probability"),
    ):
        values = [frame.loc[frame[y] == value, p] for value in (0, 1)]
        parts = axis.violinplot(values, positions=[0, 1], showmedians=True)
        for body, color in zip(parts["bodies"], [COLORS["low"], COLORS["high"]]): body.set_facecolor(color); body.set_alpha(0.7)
        axis.set_xticks([0, 1], ["Low risk", "High risk"])
        _style(axis, title, "Reference class", "Predicted high-risk probability")
    figure.tight_layout(); _save(figure, "probability_separation.png")

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for axis, title, metrics in ((axes[0], "Model 1", baseline_metrics), (axes[1], "Model 2", multitask_metrics)):
        matrix = np.asarray(metrics["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for col in range(2):
                axis.text(
                    col, row, f"{matrix[row,col]}\n{(('TN','FP'),('FN','TP'))[row][col]}",
                    ha="center", va="center", fontweight="bold",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black",
                )
        axis.set_xticks([0,1],["Low","High"]); axis.set_yticks([0,1],["Low","High"])
        axis.set_title(title, fontweight="bold"); axis.set_xlabel("Predicted"); axis.set_ylabel("Reference")
    figure.tight_layout(); _save(figure, "confusion_matrices.png")

    history = pd.read_csv(MULTITASK / "training_history.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(history.epoch, history.train_total_loss, "o-", label="Train")
    axes[0].plot(history.epoch, history.val_total_loss, "o-", label="Validation")
    _style(axes[0], "Total masked loss", "Epoch", "Loss"); axes[0].legend(frameon=False)
    for task, color in zip(("malignancy", "spiculation", "lobulation"), ("#2774AE", "#D97706", "#2E8B57")):
        axes[1].plot(history.epoch, history[f"train_{task}_loss"], "--", color=color, alpha=0.7)
        axes[1].plot(history.epoch, history[f"val_{task}_loss"], "o-", color=color, label=task.capitalize())
    _style(axes[1], "Validation task losses", "Epoch", "BCE loss"); axes[1].legend(frameon=False, loc="lower right")
    figure.tight_layout(); _save(figure, "model2_training.png")

    auxiliary = _json(MULTITASK / "auxiliary_metrics.json")
    metrics = ["sensitivity", "specificity", "f1", "balanced_accuracy"]
    x = np.arange(len(metrics)); width = 0.34
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.bar(x-width/2, [auxiliary["spiculation"][m] for m in metrics], width, label="Spiculation", color="#2774AE")
    axis.bar(x+width/2, [auxiliary["lobulation"][m] for m in metrics], width, label="Lobulation", color="#D97706")
    axis.set_xticks(x, ["Sensitivity", "Specificity", "F1", "Balanced acc."])
    axis.set_ylim(0, 1); _style(axis, "Held-out auxiliary concept performance", "Metric", "Score")
    axis.legend(frameon=False); _save(figure, "concept_performance.png")
    print(f"Saved final figures to {FIGURES}")


if __name__ == "__main__":
    main()

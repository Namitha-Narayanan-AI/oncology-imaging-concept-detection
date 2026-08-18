"""Create reproducible, lightweight figures for the completed Phase 2 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


COLORS = {
    "blue": "#2774AE",
    "orange": "#D97706",
    "green": "#2E8B57",
    "red": "#B91C1C",
    "gray": "#6B7280",
}


def load_results(results_dir: Path):
    history = pd.read_csv(results_dir / "training_history.csv")
    predictions = pd.read_csv(results_dir / "test_predictions.csv")
    validation = json.loads((results_dir / "validation_metrics.json").read_text())
    test = json.loads((results_dir / "test_metrics.json").read_text())
    return history, predictions, validation, test


def style_axis(axis, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot_loss(axis, history: pd.DataFrame) -> None:
    axis.plot(history.epoch, history.train_loss, marker="o", linewidth=2,
              color=COLORS["blue"], label="Training loss")
    axis.plot(history.epoch, history.val_loss, marker="o", linewidth=2,
              color=COLORS["orange"], label="Validation loss")
    best = history.loc[history.val_loss.idxmin()]
    axis.scatter(best.epoch, best.val_loss, s=80, color=COLORS["green"], zorder=4,
                 label=f"Best epoch: {int(best.epoch)}")
    axis.set_xticks(history.epoch)
    style_axis(axis, "Training trajectory", "Epoch", "BCE loss")
    axis.legend(frameon=False, fontsize=8)


def plot_roc(axis, predictions: pd.DataFrame, test: dict) -> None:
    fpr, tpr, _ = roc_curve(predictions.reference_label, predictions.probability)
    axis.plot(fpr, tpr, linewidth=2.2, color=COLORS["blue"],
              label=f"ROC-AUC = {test['roc_auc']:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"],
              linewidth=1.2, label="Chance")
    axis.set_xlim(0, 1); axis.set_ylim(0, 1)
    style_axis(axis, "Held-out ROC curve", "False-positive rate", "True-positive rate")
    axis.legend(frameon=False, fontsize=8, loc="lower right")


def plot_pr(axis, predictions: pd.DataFrame, test: dict) -> None:
    precision, recall, _ = precision_recall_curve(
        predictions.reference_label, predictions.probability
    )
    prevalence = float(predictions.reference_label.mean())
    axis.plot(recall, precision, linewidth=2.2, color=COLORS["orange"],
              label=f"PR-AUC = {test['pr_auc']:.3f}")
    axis.axhline(prevalence, linestyle="--", color=COLORS["gray"], linewidth=1.2,
                 label=f"High-risk prevalence = {prevalence:.3f}")
    axis.set_xlim(0, 1); axis.set_ylim(0, 1)
    style_axis(axis, "Held-out precision–recall curve", "Recall (sensitivity)", "Precision")
    axis.legend(frameon=False, fontsize=8, loc="upper right")


def plot_confusion(axis, test: dict) -> None:
    matrix = np.asarray(test["confusion_matrix"], dtype=int)
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            label = (("TN", "FP"), ("FN", "TP"))[row][column]
            axis.text(column, row, f"{matrix[row, column]}\n{label}",
                      ha="center", va="center", fontweight="bold",
                      color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    axis.set_xticks([0, 1], ["Low risk", "High risk"])
    axis.set_yticks([0, 1], ["Low risk", "High risk"])
    axis.set_xlabel("Predicted class"); axis.set_ylabel("Reference class")
    axis.set_title("Held-out confusion matrix", loc="left", fontweight="bold")
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def plot_probabilities(axis, predictions: pd.DataFrame, threshold: float) -> None:
    low = predictions.loc[predictions.reference_label == 0, "probability"]
    high = predictions.loc[predictions.reference_label == 1, "probability"]
    bins = np.linspace(predictions.probability.min(), predictions.probability.max(), 16)
    axis.hist(low, bins=bins, alpha=0.68, color=COLORS["blue"], label="Low-risk reference")
    axis.hist(high, bins=bins, alpha=0.68, color=COLORS["orange"], label="High-risk reference")
    axis.axvline(threshold, color=COLORS["red"], linestyle="--", linewidth=1.8,
                 label=f"Threshold = {threshold:.6f}")
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    style_axis(axis, "Prediction-score distribution", "Predicted high-risk probability",
               "Reader annotations")
    axis.legend(frameon=False, fontsize=8)


def save_figure(figure, path: Path) -> None:
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path,
                        default=Path("results/phase2/baseline_final"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    history, predictions, validation, test = load_results(args.results_dir)
    threshold = float(validation["threshold"])

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    plot_loss(axis, history); save_figure(figure, output_dir / "training_history.png")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    plot_roc(axes[0], predictions, test); plot_pr(axes[1], predictions, test)
    figure.tight_layout(); save_figure(figure, output_dir / "roc_pr_curves.png")

    figure, axis = plt.subplots(figsize=(5.8, 4.8))
    plot_confusion(axis, test); save_figure(figure, output_dir / "confusion_matrix.png")

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    plot_probabilities(axis, predictions, threshold)
    save_figure(figure, output_dir / "probability_distribution.png")

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    plot_loss(axes[0, 0], history)
    plot_roc(axes[0, 1], predictions, test)
    plot_pr(axes[1, 0], predictions, test)
    plot_confusion(axes[1, 1], test)
    figure.suptitle(
        "Phase 2: 3D CT radiologist-assessed malignancy-risk baseline",
        fontsize=14, fontweight="bold", y=1.01,
    )
    figure.tight_layout()
    save_figure(figure, output_dir / "phase2_results_overview.png")
    print(f"Saved Phase 2 figures to {output_dir}")


if __name__ == "__main__":
    main()

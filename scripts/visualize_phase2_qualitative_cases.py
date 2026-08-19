"""Visualize a deterministic, balanced set of held-out Phase 2 cases."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from src.lidc.dataset import LIDCNoduleDataset


BASELINE_DIR = Path("results/phase2/baseline_final")
MULTITASK_DIR = Path("results/phase2/multitask_final")
OUTPUT = MULTITASK_DIR / "figures" / "qualitative_case_comparison.png"
SELECTION_OUTPUT = MULTITASK_DIR / "qualitative_case_selection.csv"


def load_aligned_predictions() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_DIR / "test_predictions.csv").rename(
        columns={
            "reference_label": "malignancy_label",
            "probability": "model1_malignancy_probability",
            "prediction": "model1_malignancy_prediction",
        }
    )
    model2 = pd.read_csv(MULTITASK_DIR / "test_predictions.csv").drop(
        columns=["malignancy_rating"], errors="ignore"
    )
    aligned = baseline.merge(
        model2,
        on=["annotation_id", "malignancy_label"],
        validate="one_to_one",
    )
    aligned["model1_correct"] = aligned.model1_malignancy_prediction.eq(
        aligned.malignancy_label
    )
    aligned["model2_correct"] = aligned.malignancy_prediction.eq(
        aligned.malignancy_label
    )
    return aligned


def select_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    """Choose the first unused annotation ID satisfying each fixed category."""
    categories = (
        ("Both malignancy predictions correct", predictions.model1_correct & predictions.model2_correct),
        ("Model 1 incorrect; Model 2 correct", ~predictions.model1_correct & predictions.model2_correct),
        ("Model 1 correct; Model 2 incorrect", predictions.model1_correct & ~predictions.model2_correct),
        ("Both malignancy predictions incorrect", ~predictions.model1_correct & ~predictions.model2_correct),
        (
            "Valid high spiculation; Model 2 correct",
            predictions.spiculation_valid
            & predictions.spiculation_label.eq(1)
            & predictions.spiculation_prediction.eq(1),
        ),
        (
            "Valid high lobulation; Model 2 correct",
            predictions.lobulation_valid
            & predictions.lobulation_label.eq(1)
            & predictions.lobulation_prediction.eq(1),
        ),
    )
    selected = []
    used: set[str] = set()
    for category, mask in categories:
        candidates = predictions.loc[mask & ~predictions.annotation_id.isin(used)].sort_values(
            "annotation_id"
        )
        if candidates.empty:
            print(f"No unused held-out case exists for: {category}")
            continue
        row = candidates.iloc[0].copy()
        row["selection_category"] = category
        row["case_label"] = f"Case {chr(ord('A') + len(selected))}"
        selected.append(row)
        used.add(str(row.annotation_id))
    return pd.DataFrame(selected).reset_index(drop=True)


def concept_text(row: pd.Series, concept: str) -> str:
    rating = int(row[f"{concept}_rating"])
    probability = float(row[f"{concept}_probability"])
    prediction = "high" if int(row[f"{concept}_prediction"]) else "low"
    if not bool(row[f"{concept}_valid"]):
        return (
            f"{concept.capitalize()}: indeterminate / not evaluated (rating {rating})\n"
            f"Model 2: p={probability:.6f} → {prediction} (not scored)"
        )
    reference = "high" if int(row[f"{concept}_label"]) else "low"
    outcome = "correct" if prediction == reference else "incorrect"
    return (
        f"{concept.capitalize()}: {reference} (rating {rating})\n"
        f"Model 2: p={probability:.6f} → {prediction} — {outcome}"
    )


def case_text(row: pd.Series) -> str:
    reference = "high" if int(row.malignancy_label) else "low"
    model1_prediction = "high" if int(row.model1_malignancy_prediction) else "low"
    model2_prediction = "high" if int(row.malignancy_prediction) else "low"
    return "\n".join(
        (
            "RADIOLOGIST-ASSESSED REFERENCE",
            f"Malignancy risk: {reference} (rating {int(row.malignancy_rating)})",
            "",
            "MODEL 1 — MALIGNANCY RISK ONLY",
            f"p={float(row.model1_malignancy_probability):.6f} → {model1_prediction}",
            "correct" if bool(row.model1_correct) else "incorrect",
            "",
            "MODEL 2 — MULTI-LABEL CONCEPT",
            f"Malignancy: p={float(row.malignancy_probability):.6f} → {model2_prediction}",
            "correct" if bool(row.model2_correct) else "incorrect",
            "",
            concept_text(row, "spiculation"),
            "",
            concept_text(row, "lobulation"),
        )
    )


def main() -> None:
    aligned = load_aligned_predictions()
    selected = select_cases(aligned)
    if selected.empty:
        raise RuntimeError("No qualifying held-out cases were found")

    manifest = pd.read_csv("data/lidc/manifest.csv")
    manifest = manifest.loc[manifest.annotation_id.isin(selected.annotation_id)]
    records = selected[["annotation_id"]].merge(
        manifest, on="annotation_id", how="left", validate="one_to_one"
    )
    if records.series_dir.isna().any():
        raise ValueError("A selected prediction is missing from the manifest")
    dataset = LIDCNoduleDataset(
        records,
        crop_size=(64, 64, 64),
        target_spacing=(1.0, 1.0, 1.0),
        hu_clip_range=(-1000.0, 400.0),
    )

    rows = (len(selected) + 1) // 2
    figure = plt.figure(figsize=(17, 5.1 * rows), constrained_layout=True)
    outer = figure.add_gridspec(rows, 2, hspace=0.18, wspace=0.12)
    for index, (_, prediction) in enumerate(selected.iterrows()):
        item = dataset[index]
        axial = item["image"][0, item["image"].shape[1] // 2].numpy()
        inner = outer[index // 2, index % 2].subgridspec(1, 2, width_ratios=(1.0, 1.35))
        image_axis = figure.add_subplot(inner[0, 0])
        text_axis = figure.add_subplot(inner[0, 1])
        image_axis.imshow(axial, cmap="gray", vmin=0.0, vmax=1.0)
        image_axis.set_title(
            f"{prediction.case_label}\n{prediction.selection_category}",
            loc="left", fontsize=11, fontweight="bold",
        )
        image_axis.set_axis_off()
        text_axis.text(
            0.01, 0.98, case_text(prediction), va="top", ha="left",
            fontsize=9.2, linespacing=1.18, family="sans-serif",
        )
        text_axis.set_axis_off()

    figure.suptitle(
        "Held-out pulmonary-nodule case comparison",
        fontsize=17, fontweight="bold",
    )
    figure.text(
        0.5,
        0.002,
        "Frozen validation thresholds: Model 1 malignancy 0.491633; "
        "Model 2 malignancy 0.502755, spiculation 0.502532, lobulation 0.522503.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    selected.to_csv(SELECTION_OUTPUT, index=False)
    summary = selected[[
        "case_label", "selection_category", "annotation_id", "malignancy_rating",
        "malignancy_label", "model1_malignancy_probability",
        "model1_malignancy_prediction", "malignancy_probability",
        "malignancy_prediction", "spiculation_rating", "spiculation_valid",
        "spiculation_probability", "spiculation_prediction", "lobulation_rating",
        "lobulation_valid", "lobulation_probability", "lobulation_prediction",
    ]]
    print(summary.to_json(orient="records", indent=2))
    print(f"Saved {len(selected)} cases to {OUTPUT}")


if __name__ == "__main__":
    main()

"""Create final comparison, patient-bootstrap and heterogeneity summaries."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pydicom
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


BASELINE = Path("results/phase2/baseline_final")
MULTITASK = Path("results/phase2/multitask_final")
METRICS = (
    "roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity",
    "precision_ppv", "npv", "f1", "balanced_accuracy",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _aligned_predictions() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE / "test_predictions.csv").rename(columns={
        "reference_label": "label", "probability": "baseline_probability",
        "prediction": "baseline_prediction",
    })
    multitask = pd.read_csv(MULTITASK / "test_predictions.csv")
    manifest = pd.read_csv("data/lidc/manifest.csv")[
        ["annotation_id", "patient_id", "series_instance_uid", "series_dir"]
    ].drop_duplicates("annotation_id")
    aligned = baseline.merge(
        multitask[["annotation_id", "malignancy_label", "malignancy_probability", "malignancy_prediction"]],
        on="annotation_id", validate="one_to_one",
    ).merge(manifest, on="annotation_id", validate="one_to_one")
    if len(aligned) != 628 or not np.array_equal(aligned.label, aligned.malignancy_label):
        raise ValueError("Model 1 and Model 2 held-out populations are not identical")
    return aligned


def save_comparison() -> pd.DataFrame:
    baseline = _json(BASELINE / "test_metrics.json")
    multitask = _json(MULTITASK / "test_metrics.json")
    table = pd.DataFrame([
        {
            "metric": metric,
            "model_1_single_task": baseline[metric],
            "model_2_multi_label_concept": multitask[metric],
            "model_2_minus_model_1": multitask[metric] - baseline[metric],
        }
        for metric in METRICS
    ])
    table.to_csv(MULTITASK / "model_comparison.csv", index=False)
    return table


def _bootstrap_metrics(frame: pd.DataFrame, probability: str, prediction: str) -> tuple[float, float, float]:
    labels = frame.label.to_numpy(dtype=int)
    probabilities = frame[probability].to_numpy(dtype=float)
    predictions = frame[prediction].to_numpy(dtype=int)
    return (
        float(roc_auc_score(labels, probabilities)),
        float(average_precision_score(labels, probabilities)),
        float(balanced_accuracy_score(labels, predictions)),
    )


def save_patient_bootstrap(aligned: pd.DataFrame, iterations: int = 2000, seed: int = 42) -> dict:
    patients = aligned.patient_id.unique()
    groups = {patient: aligned.loc[aligned.patient_id == patient] for patient in patients}
    rng = np.random.default_rng(seed); samples = []
    for _ in range(iterations):
        draw = rng.choice(patients, size=len(patients), replace=True)
        frame = pd.concat([groups[patient] for patient in draw], ignore_index=True)
        if frame.label.nunique() < 2:
            continue
        model_1 = _bootstrap_metrics(frame, "baseline_probability", "baseline_prediction")
        model_2 = _bootstrap_metrics(frame, "malignancy_probability", "malignancy_prediction")
        samples.append((*model_1, *model_2, *(b - a for a, b in zip(model_1, model_2))))
    values = np.asarray(samples)
    names = ("roc_auc", "pr_auc", "balanced_accuracy")
    result = {
        "method": "patient-level percentile bootstrap",
        "seed": seed,
        "requested_iterations": iterations,
        "valid_iterations": len(samples),
        "patients": len(patients),
        "models": {},
        "model_2_minus_model_1": {},
    }
    for model, offset in (("model_1_single_task", 0), ("model_2_multi_label_concept", 3)):
        result["models"][model] = {
            name: {"lower_95": float(np.percentile(values[:, offset + i], 2.5)),
                   "upper_95": float(np.percentile(values[:, offset + i], 97.5))}
            for i, name in enumerate(names)
        }
    result["model_2_minus_model_1"] = {
        name: {"lower_95": float(np.percentile(values[:, 6 + i], 2.5)),
               "upper_95": float(np.percentile(values[:, 6 + i], 97.5))}
        for i, name in enumerate(names)
    }
    (MULTITASK / "confidence_intervals.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _slice_thickness(series_dir: str) -> float:
    for path in sorted(Path(series_dir).iterdir()):
        if not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["SliceThickness"])
        except Exception:
            continue
        if hasattr(dataset, "SliceThickness"):
            return float(dataset.SliceThickness)
    return float("nan")


def save_heterogeneity(aligned: pd.DataFrame) -> pd.DataFrame:
    thickness = {
        uid: _slice_thickness(group.series_dir.iloc[0])
        for uid, group in aligned.groupby("series_instance_uid")
    }
    aligned = aligned.copy()
    aligned["slice_thickness_mm"] = aligned.series_instance_uid.map(thickness)
    aligned["slice_thickness_group"] = np.where(
        aligned.slice_thickness_mm <= 2.0, "<=2 mm", ">2 mm"
    )
    rows = []
    for group_name, group in aligned.groupby("slice_thickness_group", sort=True):
        for model, probability, prediction in (
            ("Model 1 - Single-task", "baseline_probability", "baseline_prediction"),
            ("Model 2 - Multi-label concept", "malignancy_probability", "malignancy_prediction"),
        ):
            roc = float(roc_auc_score(group.label, group[probability])) if group.label.nunique() == 2 else float("nan")
            rows.append({
                "slice_thickness_group": group_name,
                "model": model,
                "patients": group.patient_id.nunique(),
                "reader_records": len(group),
                "high_risk_prevalence": float(group.label.mean()),
                "roc_auc": roc,
                "balanced_accuracy_global_threshold": float(
                    balanced_accuracy_score(group.label, group[prediction])
                ),
            })
    result = pd.DataFrame(rows)
    result.to_csv(MULTITASK / "heterogeneity_slice_thickness.csv", index=False)
    return result


def save_label_audit() -> None:
    manifest = pd.read_csv("data/lidc/manifest.csv")
    primary = manifest.loc[manifest.malignancy.isin([1, 2, 4, 5])]
    rows = []
    for concept in ("spiculation", "lobulation"):
        for split in ("train", "val", "test", "total"):
            frame = primary if split == "total" else primary.loc[primary.split == split]
            counts = frame[concept].value_counts(dropna=False)
            row = {"concept": concept, "split": split}
            row.update({f"rating_{rating}": int(counts.get(rating, 0)) for rating in range(1, 6)})
            row["missing_or_invalid"] = int(len(frame) - frame[concept].isin(range(1, 6)).sum())
            rows.append(row)
    pd.DataFrame(rows).to_csv(MULTITASK / "concept_label_audit.csv", index=False)


def main() -> None:
    aligned = _aligned_predictions()
    comparison = save_comparison()
    intervals = save_patient_bootstrap(aligned)
    heterogeneity = save_heterogeneity(aligned)
    save_label_audit()
    print(comparison.to_string(index=False))
    print(json.dumps(intervals, indent=2))
    print(heterogeneity.to_string(index=False))


if __name__ == "__main__":
    main()

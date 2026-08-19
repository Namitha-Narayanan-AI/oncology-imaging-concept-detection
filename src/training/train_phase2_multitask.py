"""Train the final multi-label pulmonary-nodule concept model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from src.evaluation.evaluate_phase2 import binary_metrics
from src.evaluation.threshold_analysis_phase2 import select_balanced_accuracy_threshold
from src.lidc.dataset import LIDCNoduleDataset
from src.models.multitask_3d_cnn import MultiTask3DCNN


TASKS = ("malignancy", "spiculation", "lobulation")


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def masked_task_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    criteria: dict[str, nn.Module],
) -> dict[str, torch.Tensor]:
    """Return available BCE task losses; auxiliary rating 3/missing is masked."""
    losses = {
        "malignancy": criteria["malignancy"](
            outputs["malignancy"].squeeze(1), batch["malignancy_target"]
        )
    }
    for task in ("spiculation", "lobulation"):
        valid = batch[f"{task}_valid"].bool()
        if bool(valid.any()):
            losses[task] = criteria[task](
                outputs[task].squeeze(1)[valid], batch[f"{task}_target"][valid]
            )
    return losses


def _run_epoch(model, loader, criteria, device, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    task_sums = {task: 0.0 for task in TASKS}
    task_counts = {task: 0 for task in TASKS}
    total_sum = 0.0; sample_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for source_batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in source_batch.items()
            }
            if training:
                optimizer.zero_grad(set_to_none=True)
            losses = masked_task_losses(model(batch["image"]), batch, criteria)
            total_loss = torch.stack(tuple(losses.values())).mean()
            if training:
                total_loss.backward(); optimizer.step()
            batch_size = len(batch["malignancy_target"])
            total_sum += float(total_loss.detach().cpu()) * batch_size
            sample_count += batch_size
            for task, loss in losses.items():
                count = batch_size if task == "malignancy" else int(batch[f"{task}_valid"].sum())
                task_sums[task] += float(loss.detach().cpu()) * count
                task_counts[task] += count
    result = {"total_loss": total_sum / sample_count}
    result.update({
        f"{task}_loss": task_sums[task] / task_counts[task]
        if task_counts[task] else float("nan")
        for task in TASKS
    })
    return result


def predict_concepts(model, loader, device) -> pd.DataFrame:
    model.eval(); rows = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image"].to(device))
            probabilities = {
                task: torch.sigmoid(outputs[task].squeeze(1)).cpu().numpy()
                for task in TASKS
            }
            for index in range(len(batch["malignancy_target"])):
                row = {
                    "annotation_id": batch["annotation_id"][index],
                    "patient_id": batch["patient_id"][index],
                    "malignancy_rating": int(batch["malignancy_rating"][index]),
                    "malignancy_label": int(batch["malignancy_target"][index]),
                    "malignancy_probability": float(probabilities["malignancy"][index]),
                }
                for task in ("spiculation", "lobulation"):
                    row[f"{task}_rating"] = int(batch[f"{task}_rating"][index])
                    row[f"{task}_valid"] = bool(batch[f"{task}_valid"][index])
                    row[f"{task}_label"] = int(batch[f"{task}_target"][index])
                    row[f"{task}_probability"] = float(probabilities[task][index])
                rows.append(row)
    return pd.DataFrame(rows)


def _task_metrics(frame: pd.DataFrame, task: str, threshold: float) -> dict:
    valid = frame if task == "malignancy" else frame.loc[frame[f"{task}_valid"]]
    metrics = binary_metrics(valid[f"{task}_label"], valid[f"{task}_probability"], threshold)
    metrics.update({
        "n_evaluated": len(valid),
        "negative_count": int((valid[f"{task}_label"] == 0).sum()),
        "positive_count": int((valid[f"{task}_label"] == 1).sum()),
    })
    return metrics


def train_multitask(config: dict, epochs_override=None, sample_limits=None) -> dict:
    seed = int(config["seed"]); seed_everything(seed)
    training = config["training"]; manifest = pd.read_csv(config["data"]["manifest_path"])
    # Preserve exactly Model 1's primary population.
    manifest = manifest.loc[manifest["malignancy"].isin([1, 2, 4, 5])].copy()
    manifest["malignancy_risk_label"] = manifest["malignancy"].isin([4, 5]).astype(int)
    limits = sample_limits or {}; datasets = {}
    for split in ("train", "val", "test"):
        records = manifest.loc[manifest.split == split]
        if split == "train":
            groups = [group for _, group in records.groupby("series_instance_uid", sort=True)]
            random.Random(seed).shuffle(groups); records = pd.concat(groups, ignore_index=True)
        if limits.get(split):
            records = records.head(limits[split])
        datasets[split] = LIDCNoduleDataset(
            records,
            crop_size=tuple(config["preprocessing"]["crop_size_voxels"]),
            target_spacing=tuple(config["preprocessing"]["target_spacing_mm"]),
            hu_clip_range=tuple(config["preprocessing"]["hu_clip_range"]),
        )
        if not len(datasets[split]):
            raise ValueError(f"The {split} split has no modelling records")
    loaders = {
        split: DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=False, num_workers=0)
        for split, dataset in datasets.items()
    }
    train_records = datasets["train"].records
    counts = {}
    for task in TASKS:
        ratings = train_records[task]
        negative = int(ratings.isin([1, 2]).sum()); positive = int(ratings.isin([4, 5]).sum())
        if not negative or not positive:
            raise ValueError(f"Training split lacks a binary class for {task}")
        counts[task] = {"valid_negatives": negative, "valid_positives": positive, "pos_weight": negative / positive}
    device = torch.device("cpu")
    model = MultiTask3DCNN().to(device)
    criteria = {
        task: nn.BCEWithLogitsLoss(pos_weight=torch.tensor([counts[task]["pos_weight"]], device=device))
        for task in TASKS
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    output = Path(config["outputs"]["multitask_dir"]); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best_validation_checkpoint.pt"
    history = []; best_val = float("inf"); best_epoch = 0
    for epoch in range(1, int(epochs_override or training["epochs"]) + 1):
        train_values = _run_epoch(model, loaders["train"], criteria, device, optimizer)
        val_values = _run_epoch(model, loaders["val"], criteria, device)
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_values.items()})
        row.update({f"val_{key}": value for key, value in val_values.items()})
        history.append(row); print(json.dumps(row), flush=True)
        if row["val_malignancy_loss"] < best_val:
            best_val = row["val_malignancy_loss"]; best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    (output / "frozen_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    validation = predict_concepts(model, loaders["val"], device)
    thresholds = {}
    validation_metrics = {}
    for task in TASKS:
        valid = validation if task == "malignancy" else validation.loc[validation[f"{task}_valid"]]
        threshold, score = select_balanced_accuracy_threshold(valid[f"{task}_label"], valid[f"{task}_probability"])
        thresholds[task] = threshold
        validation_metrics[task] = _task_metrics(validation, task, threshold)
        validation_metrics[task]["selection_objective"] = "validation_balanced_accuracy"
    (output / "validation_metrics.json").write_text(json.dumps({"selected_epoch": best_epoch, "thresholds": thresholds, "metrics": validation_metrics}, indent=2) + "\n")
    test = predict_concepts(model, loaders["test"], device)
    test["malignancy_prediction"] = (test.malignancy_probability >= thresholds["malignancy"]).astype(int)
    for task in ("spiculation", "lobulation"):
        test[f"{task}_prediction"] = (test[f"{task}_probability"] >= thresholds[task]).astype(int)
    test.to_csv(output / "test_predictions.csv", index=False)
    primary_metrics = _task_metrics(test, "malignancy", thresholds["malignancy"])
    primary_metrics["sample_count"] = len(test)
    primary_metrics["high_risk_prevalence"] = float(test.malignancy_label.mean())
    (output / "test_metrics.json").write_text(json.dumps(primary_metrics, indent=2) + "\n")
    auxiliary = {task: _task_metrics(test, task, thresholds[task]) for task in ("spiculation", "lobulation")}
    (output / "auxiliary_metrics.json").write_text(json.dumps(auxiliary, indent=2) + "\n")
    return {
        "device": str(device), "best_epoch": best_epoch, "best_validation_malignancy_loss": best_val,
        "samples": {split: len(dataset) for split, dataset in datasets.items()},
        "patients": {split: int(dataset.records.patient_id.nunique()) for split, dataset in datasets.items()},
        "class_weights": counts, "parameter_count": sum(p.numel() for p in model.parameters()),
        "thresholds": thresholds, "test_metrics": primary_metrics, "auxiliary_metrics": auxiliary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_multitask_final.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    limits = {"train": args.max_train_samples, "val": args.max_val_samples, "test": args.max_test_samples}
    print(json.dumps(train_multitask(config, args.epochs, limits), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

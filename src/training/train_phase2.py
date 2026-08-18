"""Train the compact Phase 2 single-task 3D CNN."""

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

from src.evaluation.evaluate_phase2 import binary_metrics, predict, save_evaluation
from src.evaluation.threshold_analysis_phase2 import select_balanced_accuracy_threshold
from src.lidc.dataset import LIDCNoduleDataset
from src.models.simple_3d_cnn import Simple3DCNN


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def choose_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_baseline(
    config: dict,
    epochs_override: int | None = None,
    sample_limits: dict[str, int] | None = None,
) -> dict:
    seed = int(config["seed"]); seed_everything(seed)
    training = config["training"]
    manifest = pd.read_csv(config["data"]["manifest_path"])
    limits = sample_limits or {}
    datasets = {}
    for split in ("train", "val", "test"):
        records = manifest.loc[manifest.split == split]
        if split == "train":
            groups = [group for _, group in records.groupby("series_instance_uid", sort=True)]
            random.Random(seed).shuffle(groups)
            records = pd.concat(groups, ignore_index=True)
        if limits.get(split):
            records = records.head(limits[split])
        datasets[split] = LIDCNoduleDataset(
            records,
            crop_size=tuple(config["preprocessing"]["crop_size_voxels"]),
            target_spacing=tuple(config["preprocessing"]["target_spacing_mm"]),
            hu_clip_range=tuple(config["preprocessing"]["hu_clip_range"]),
        )
        if len(datasets[split]) == 0:
            raise ValueError(f"The {split} split has no modelling records")
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=False,
            num_workers=0,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }
    train_labels = datasets["train"].records.malignancy_risk_label.astype(int)
    positives = int(train_labels.sum()); negatives = int((train_labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Training smoke subset must contain both low- and high-risk records")
    device = choose_device(); model = Simple3DCNN().to(device)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    epochs = int(epochs_override or training["epochs"])
    output = Path(config["outputs"]["baseline_dir"]); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best_validation_checkpoint.pt"
    history = []; best_val = float("inf"); best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train(); train_total = 0.0
        for batch in loaders["train"]:
            images = batch["image"].to(device); targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets); loss.backward(); optimizer.step()
            train_total += float(loss.detach().cpu()) * len(targets)
        model.eval(); val_total = 0.0
        with torch.no_grad():
            for batch in loaders["val"]:
                targets = batch["target"].to(device)
                loss = criterion(model(batch["image"].to(device)), targets)
                val_total += float(loss.detach().cpu()) * len(targets)
        row = {"epoch": epoch, "train_loss": train_total / len(datasets["train"]), "val_loss": val_total / len(datasets["val"])}
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]; best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    (output / "frozen_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    validation_predictions = predict(model, loaders["val"], device)
    threshold, validation_balanced_accuracy = select_balanced_accuracy_threshold(
        validation_predictions.reference_label, validation_predictions.probability
    )
    threshold_record = {"threshold": threshold, "selection_split": "validation", "objective": "balanced_accuracy", "validation_balanced_accuracy": validation_balanced_accuracy}
    threshold_record["validation_metrics"] = binary_metrics(
        validation_predictions.reference_label,
        validation_predictions.probability,
        threshold,
    )
    (output / "validation_metrics.json").write_text(json.dumps(threshold_record, indent=2, allow_nan=True) + "\n")
    test_predictions = predict(model, loaders["test"], device)
    metrics = save_evaluation(test_predictions, threshold, output)
    summary = {
        "device": str(device), "epochs": epochs,
        "samples": {split: len(dataset) for split, dataset in datasets.items()},
        "training_class_counts": {"low_risk": negatives, "high_risk": positives},
        "pos_weight": negatives / positives,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_validation_loss": best_val, "best_epoch": best_epoch,
        "best_epoch_train_loss": history[best_epoch - 1]["train_loss"],
        "final_train_loss": history[-1]["train_loss"],
        "selected_threshold": threshold, "test_metrics": metrics,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_lidc_vertical_slice.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    limits = {"train": args.max_train_samples, "val": args.max_val_samples, "test": args.max_test_samples}
    print(json.dumps(train_baseline(config, args.epochs, limits), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

"""Create deterministic patient-level LIDC-IDRI split files."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src.lidc.splits import assign_patient_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_lidc_vertical_slice.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    manifest = pd.read_csv(config["data"]["manifest_path"])
    split_config = config["splits"]
    assignments = assign_patient_splits(
        manifest["patient_id"].astype(str).tolist(),
        split_config["train_fraction"], split_config["val_fraction"],
        split_config["test_fraction"], config["seed"],
    )
    manifest["split"] = manifest["patient_id"].astype(str).map(assignments)
    manifest.to_csv(config["data"]["manifest_path"], index=False)
    output_dir = Path(config["data"]["splits_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        patients = sorted(patient for patient, assigned in assignments.items() if assigned == split)
        (output_dir / f"{split}_patients.txt").write_text("\n".join(patients) + ("\n" if patients else ""))
    summary = {
        split: {
            "patients": len(set(manifest.loc[manifest.split == split, "patient_id"])),
            "reader_annotations": int((manifest.split == split).sum()),
        }
        for split in ("train", "val", "test")
    }
    summary["patient_overlap"] = False
    summary_path = Path(config["outputs"]["split_summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

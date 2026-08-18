"""Build the Phase 2 reader-level LIDC-IDRI manifest."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.lidc.manifest import build_reader_manifest, index_dicom_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_lidc_vertical_slice.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    data = config["data"]
    index = index_dicom_series(data["raw_dicom_dir"])
    manifest = build_reader_manifest(
        data["annotation_dir"], index, config["vertical_slice"].get("max_patients")
    )
    output = Path(data["manifest_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)

    summary = {
        "manifest_level": "reader_annotation",
        "patients": int(manifest["patient_id"].nunique()),
        "series": int(manifest["series_instance_uid"].nunique()),
        "reader_annotations": len(manifest),
        "binary_target": "ratings_1_2_low_risk_0__ratings_4_5_high_risk_1__rating_3_excluded",
        "sampling_unit": "characterized_reader_level_nodule_annotation",
        "malignancy_rating_counts": {
            str(rating): int((manifest["malignancy"] == rating).sum())
            for rating in (1, 2, 4, 5)
        },
        "malignancy_rating_3_excluded": int(manifest.attrs.get("excluded_rating_3", 0)),
        "invalid_malignancy_ratings_excluded": int(manifest.attrs.get("excluded_invalid_ratings", 0)),
        "class_counts": {
            "low_risk_0": int((manifest["malignancy_risk_label"] == 0).sum()),
            "high_risk_1": int((manifest["malignancy_risk_label"] == 1).sum()),
        },
        "label_semantics": "subjective_radiologist_assessment_not_pathology_confirmed_diagnosis",
    }
    summary_path = Path(config["outputs"]["manifest_summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

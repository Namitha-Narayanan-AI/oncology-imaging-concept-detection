"""Inspect one manifest-linked LIDC-IDRI DICOM series."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.lidc.dicom_io import load_ct_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/lidc/manifest.csv")
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()
    row = pd.read_csv(args.manifest).iloc[args.row]
    series = load_ct_series(row.series_dir)
    print(json.dumps({
        "patient_id": str(row.patient_id),
        "series_instance_uid": series.series_instance_uid,
        "volume_shape_zyx": list(series.volume.shape),
        "voxel_spacing_zyx_mm": list(series.voxel_spacing),
        "hu_min": float(series.volume.min()),
        "hu_max": float(series.volume.max()),
        "linked_reader_annotations": int((pd.read_csv(args.manifest).series_instance_uid == row.series_instance_uid).sum()),
    }, indent=2))


if __name__ == "__main__":
    main()

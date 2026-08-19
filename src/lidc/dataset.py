"""Lazy PyTorch dataset for reader-level LIDC nodule annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .annotations import parse_annotation_file
from .crop_extraction import extract_nodule_crop, preprocess_ct_crop
from .dicom_io import load_ct_series


def binary_concept_target(rating: Any) -> tuple[float, bool]:
    """Map LIDC ratings 1-2/4-5 to 0/1 and mask 3 or missing values."""
    if pd.isna(rating):
        return 0.0, False
    value = int(rating)
    if value in (1, 2):
        return 0.0, True
    if value in (4, 5):
        return 1.0, True
    return 0.0, False


class LIDCNoduleDataset(Dataset):
    """Load one raw CT series and create one nodule crop on each item request."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        split: str | None = None,
        crop_size: tuple[int, int, int] = (64, 64, 64),
        target_spacing: tuple[float, float, float] | None = (1.0, 1.0, 1.0),
        hu_clip_range: tuple[float, float] = (-1000.0, 400.0),
    ) -> None:
        records = pd.read_csv(manifest) if isinstance(manifest, (str, Path)) else manifest.copy()
        if split is not None:
            records = records.loc[records["split"] == split]
        if "malignancy_risk_label" not in records:
            raise ValueError("Manifest lacks malignancy_risk_label")
        self.records = records.reset_index(drop=True)
        self.crop_size = tuple(int(v) for v in crop_size)
        self.target_spacing = None if target_spacing is None else tuple(float(v) for v in target_spacing)
        self.hu_clip_range = tuple(float(v) for v in hu_clip_range)
        # Keep only the most recently used raw series in RAM. Manifest rows are
        # series-grouped, so reader annotations can share one load without any
        # persistent cache or processed-data copy.
        self._last_series_dir: str | None = None
        self._last_series = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records.iloc[index]
        annotation_file = parse_annotation_file(row["xml_path"])
        annotation = next(
            (
                n for n in annotation_file.nodules
                if n.reading_session_index == int(row["reading_session_index"])
                and str(n.nodule_id) == str(row["reader_nodule_id"])
            ),
            None,
        )
        if annotation is None:
            raise KeyError(f"Annotation {row['annotation_id']} was not found in XML")
        series_dir = str(row["series_dir"])
        if self._last_series_dir != series_dir:
            self._last_series = load_ct_series(series_dir)
            self._last_series_dir = series_dir
        series = self._last_series
        if series.series_instance_uid != str(row["series_instance_uid"]):
            raise ValueError("Manifest series UID does not match loaded DICOM series")
        crop = extract_nodule_crop(
            series, annotation, self.crop_size, self.target_spacing
        )
        values = preprocess_ct_crop(crop.values, self.hu_clip_range)
        spiculation_target, spiculation_valid = binary_concept_target(
            row.get("spiculation")
        )
        lobulation_target, lobulation_valid = binary_concept_target(
            row.get("lobulation")
        )
        return {
            "image": torch.from_numpy(values).unsqueeze(0),
            "target": torch.tensor(float(row["malignancy_risk_label"]), dtype=torch.float32),
            "malignancy_target": torch.tensor(
                float(row["malignancy_risk_label"]), dtype=torch.float32
            ),
            "spiculation_target": torch.tensor(spiculation_target, dtype=torch.float32),
            "spiculation_valid": torch.tensor(spiculation_valid, dtype=torch.bool),
            "lobulation_target": torch.tensor(lobulation_target, dtype=torch.float32),
            "lobulation_valid": torch.tensor(lobulation_valid, dtype=torch.bool),
            "annotation_id": str(row["annotation_id"]),
            "patient_id": str(row["patient_id"]),
            "malignancy_rating": int(row["malignancy"]),
            "spiculation_rating": (
                int(row["spiculation"]) if not pd.isna(row.get("spiculation")) else -1
            ),
            "lobulation_rating": (
                int(row["lobulation"]) if not pd.isna(row.get("lobulation")) else -1
            ),
        }

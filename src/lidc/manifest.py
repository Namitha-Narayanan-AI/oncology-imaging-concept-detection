"""Build a reader-level LIDC-IDRI annotation manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pydicom

from .annotations import parse_annotation_file


TARGETS = ("malignancy", "spiculation", "lobulation", "texture", "margin", "sphericity")


def index_dicom_series(raw_dir: str | Path) -> dict[str, dict[str, str]]:
    """Map SeriesInstanceUID to its patient and on-disk directory."""

    index: dict[str, dict[str, str]] = {}
    patient_dirs = sorted(
        path
        for path in Path(raw_dir).rglob("LIDC-IDRI-[0-9][0-9][0-9][0-9]")
        if path.is_dir()
    )
    for patient_dir in patient_dirs:
        # TCIA's layout is patient/study/series. Avoid recursively considering
        # every patient and study directory as a possible series.
        for study_dir in sorted(path for path in patient_dir.iterdir() if path.is_dir()):
            for series_dir in sorted(path for path in study_dir.iterdir() if path.is_dir()):
                files = (path for path in series_dir.iterdir() if path.is_file())
                first_file = next(files, None)
                if first_file is None:
                    continue
                try:
                    ds = pydicom.dcmread(
                        first_file,
                        stop_before_pixels=True,
                        specific_tags=["PatientID", "SeriesInstanceUID", "Modality"],
                    )
                except Exception:
                    continue
                if getattr(ds, "Modality", None) != "CT" or not hasattr(ds, "SeriesInstanceUID"):
                    continue
                patient_id = str(getattr(ds, "PatientID", patient_dir.name))
                index[str(ds.SeriesInstanceUID)] = {
                    "patient_id": patient_id,
                    "series_dir": str(series_dir.resolve()),
                }
    return index


def build_reader_manifest(
    annotation_dir: str | Path,
    series_index: dict[str, dict[str, str]],
    max_patients: int | None = None,
) -> pd.DataFrame:
    """Return one row per characterized reader annotation."""

    rows: list[dict] = []
    excluded_rating_3 = 0
    excluded_invalid_ratings = 0
    selected_patients: set[str] = set()
    for xml_path in sorted(Path(annotation_dir).rglob("*.xml")):
        annotation_file = parse_annotation_file(xml_path)
        series_uid = annotation_file.series_instance_uid
        series = series_index.get(series_uid or "")
        if series is None:
            continue
        patient_id = series["patient_id"]
        if patient_id not in selected_patients:
            if max_patients is not None and len(selected_patients) >= max_patients:
                continue
            selected_patients.add(patient_id)

        for nodule in annotation_file.nodules:
            if nodule.malignancy == 3:
                excluded_rating_3 += 1
                continue
            if nodule.malignancy is None:
                continue
            if nodule.malignancy not in {1, 2, 4, 5}:
                excluded_invalid_ratings += 1
                continue
            points = [point for roi in nodule.rois for point in roi.boundary_points]
            z_values = [roi.z_position for roi in nodule.rois if roi.z_position is not None]
            row = {
                "patient_id": patient_id,
                "study_instance_uid": annotation_file.study_instance_uid,
                "series_instance_uid": series_uid,
                "series_dir": series["series_dir"],
                "xml_path": str(xml_path.resolve()),
                "reader_id": nodule.reader_id,
                "reading_session_index": nodule.reading_session_index,
                "reader_nodule_id": nodule.nodule_id,
                "annotation_id": f"{series_uid}:R{nodule.reading_session_index}:{nodule.nodule_id}",
                "roi_count": len(nodule.rois),
                "center_x_pixel": sum(x for x, _ in points) / len(points) if points else None,
                "center_y_pixel": sum(y for _, y in points) / len(points) if points else None,
                "center_z_mm": sum(z_values) / len(z_values) if z_values else None,
                "malignancy_risk_label": 0 if nodule.malignancy <= 2 else 1,
            }
            row.update({target: getattr(nodule, target) for target in TARGETS})
            rows.append(row)
    manifest = pd.DataFrame(rows)
    manifest.attrs["excluded_rating_3"] = excluded_rating_3
    manifest.attrs["excluded_invalid_ratings"] = excluded_invalid_ratings
    return manifest

"""DICOM loading utilities for LIDC-IDRI CT series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError


class DicomSeriesError(ValueError):
    """Base exception for invalid CT DICOM series inputs."""


class EmptyDicomSeriesError(DicomSeriesError):
    """Raised when a directory contains no readable DICOM slices."""


class InconsistentDimensionsError(DicomSeriesError):
    """Raised when slices in one series have different image dimensions."""


class InconsistentOrientationError(DicomSeriesError):
    """Raised when slices in one series have different orientations."""


class MixedSeriesError(DicomSeriesError):
    """Raised when a directory contains multiple SeriesInstanceUID values."""


@dataclass(frozen=True)
class DicomSeries:
    """Loaded CT series volume and geometry metadata."""

    volume: np.ndarray
    voxel_spacing: tuple[float, float, float]
    origin: tuple[float, float, float] | None
    orientation: tuple[float, ...]
    series_instance_uid: str
    image_positions: tuple[tuple[float, float, float] | None, ...]
    sop_instance_uids: tuple[str, ...]
    slice_coordinates: tuple[float, ...]


def load_ct_series(series_dir: str | Path) -> DicomSeries:
    """Load one CT DICOM series as a HU volume sorted by slice geometry."""

    slices = _read_dicom_slices(series_dir)
    _validate_single_series(slices)
    _validate_dimensions(slices)
    _validate_orientation(slices)

    sorted_slices = _sort_slices(slices)
    volume = np.stack([_to_hounsfield_units(ds) for ds in sorted_slices])

    first = sorted_slices[0]
    row_spacing, col_spacing = [float(value) for value in first.PixelSpacing]
    slice_spacing = _slice_spacing(sorted_slices)
    origin = _image_position(first)
    slice_normal = _slice_normal(first)
    image_positions = tuple(_image_position(ds) for ds in sorted_slices)
    slice_coordinates = tuple(
        float(np.dot(np.asarray(position), slice_normal))
        if position is not None
        else float(ds.InstanceNumber)
        for ds, position in zip(sorted_slices, image_positions)
    )

    return DicomSeries(
        volume=volume,
        voxel_spacing=(slice_spacing, row_spacing, col_spacing),
        origin=origin,
        orientation=tuple(float(value) for value in first.ImageOrientationPatient),
        series_instance_uid=str(first.SeriesInstanceUID),
        image_positions=image_positions,
        sop_instance_uids=tuple(str(ds.SOPInstanceUID) for ds in sorted_slices),
        slice_coordinates=slice_coordinates,
    )


def _read_dicom_slices(series_dir: str | Path) -> list[pydicom.Dataset]:
    path = Path(series_dir)
    if not path.exists() or not path.is_dir():
        raise EmptyDicomSeriesError(f"No DICOM directory found at {path}")

    slices = []
    for file_path in sorted(path.iterdir()):
        if not file_path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(file_path)
        except (InvalidDicomError, OSError):
            continue
        if hasattr(dataset, "PixelData"):
            slices.append(dataset)

    if not slices:
        raise EmptyDicomSeriesError(f"No readable DICOM slices found in {path}")

    return slices


def _validate_single_series(slices: list[pydicom.Dataset]) -> None:
    series_uids = {str(ds.SeriesInstanceUID) for ds in slices}
    if len(series_uids) != 1:
        raise MixedSeriesError(
            f"Expected one SeriesInstanceUID, found {len(series_uids)}"
        )


def _validate_dimensions(slices: list[pydicom.Dataset]) -> None:
    dimensions = {(int(ds.Rows), int(ds.Columns)) for ds in slices}
    if len(dimensions) != 1:
        raise InconsistentDimensionsError(
            f"Expected consistent image dimensions, found {sorted(dimensions)}"
        )


def _validate_orientation(slices: list[pydicom.Dataset]) -> None:
    reference = np.asarray(slices[0].ImageOrientationPatient, dtype=float)
    for ds in slices[1:]:
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        if not np.allclose(reference, orientation, atol=1e-5):
            raise InconsistentOrientationError(
                "Expected all slices to have the same ImageOrientationPatient"
            )


def _sort_slices(slices: list[pydicom.Dataset]) -> list[pydicom.Dataset]:
    positions = [_image_position(ds) for ds in slices]
    if all(position is not None for position in positions):
        slice_normal = _slice_normal(slices[0])
        return [
            ds
            for _, ds in sorted(
                zip(
                    [
                        float(np.dot(np.asarray(position), slice_normal))
                        for position in positions
                    ],
                    slices,
                ),
                key=lambda item: item[0],
            )
        ]

    if any(position is not None for position in positions):
        raise DicomSeriesError(
            "ImagePositionPatient is missing from only some slices"
        )

    return sorted(slices, key=lambda ds: int(ds.InstanceNumber))


def _to_hounsfield_units(dataset: pydicom.Dataset) -> np.ndarray:
    pixels = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    return pixels * slope + intercept


def _slice_spacing(slices: list[pydicom.Dataset]) -> float:
    if len(slices) < 2:
        return float(getattr(slices[0], "SliceThickness", 1.0))

    positions = [_image_position(ds) for ds in slices]
    if all(position is not None for position in positions):
        slice_normal = _slice_normal(slices[0])
        projected = [
            float(np.dot(np.asarray(position), slice_normal))
            for position in positions
        ]
        distances = np.diff(projected)
        nonzero = [abs(value) for value in distances if not np.isclose(value, 0.0)]
        if nonzero:
            return float(np.median(nonzero))

    return float(
        getattr(
            slices[0],
            "SpacingBetweenSlices",
            getattr(slices[0], "SliceThickness", 1.0),
        )
    )


def _image_position(dataset: pydicom.Dataset) -> tuple[float, float, float] | None:
    if not hasattr(dataset, "ImagePositionPatient"):
        return None
    return tuple(float(value) for value in dataset.ImagePositionPatient)


def _slice_normal(dataset: pydicom.Dataset) -> np.ndarray:
    orientation = np.asarray(dataset.ImageOrientationPatient, dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    norm = np.linalg.norm(normal)
    if np.isclose(norm, 0.0):
        raise InconsistentOrientationError("ImageOrientationPatient has no slice normal")
    return normal / norm

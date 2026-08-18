"""In-memory alignment, extraction and preprocessing of LIDC nodule crops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .annotations import NoduleAnnotation, ROI
from .dicom_io import DicomSeries


class ROIAlignmentError(ValueError):
    """Raised when an XML ROI cannot be matched unambiguously to a CT slice."""


@dataclass(frozen=True)
class NoduleCrop:
    values: np.ndarray
    center_zyx: tuple[float, float, float]
    roi_slice_indices: tuple[int, ...]
    native_crop_shape: tuple[int, int, int]


def resolve_roi_slice(roi: ROI, series: DicomSeries, tolerance_mm: float | None = None) -> int:
    """Resolve an ROI to volume z-index by SOP UID, then patient-space geometry."""
    if roi.sop_instance_uid:
        matches = [i for i, uid in enumerate(series.sop_instance_uids) if uid == roi.sop_instance_uid]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ROIAlignmentError(f"Duplicate SOPInstanceUID {roi.sop_instance_uid}")

    if roi.z_position is None or not series.image_positions:
        raise ROIAlignmentError("ROI has no matching SOPInstanceUID or geometric position")

    orientation = np.asarray(series.orientation, dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    normal /= np.linalg.norm(normal)
    reference = next((p for p in series.image_positions if p is not None), None)
    if reference is None:
        raise ROIAlignmentError("DICOM series has no ImagePositionPatient values")
    roi_position = np.asarray((reference[0], reference[1], roi.z_position), dtype=float)
    roi_coordinate = float(np.dot(roi_position, normal))
    distances = np.abs(np.asarray(series.slice_coordinates) - roi_coordinate)
    best = int(np.argmin(distances))
    allowed = tolerance_mm if tolerance_mm is not None else max(series.voxel_spacing[0] * 0.51, 1e-3)
    if float(distances[best]) > allowed:
        raise ROIAlignmentError(
            f"Nearest CT slice is {float(distances[best]):.3f} mm away; tolerance is {allowed:.3f} mm"
        )
    tied = np.flatnonzero(np.isclose(distances, distances[best], atol=1e-6))
    if len(tied) != 1:
        raise ROIAlignmentError("ROI position is equally close to multiple CT slices")
    return best


def nodule_center_zyx(annotation: NoduleAnnotation, series: DicomSeries) -> tuple[float, float, float]:
    """Return the contour-point-weighted centre of included, non-empty ROIs."""
    coordinates: list[tuple[int, int, int]] = []
    for roi in annotation.rois:
        if roi.inclusion is False or not roi.boundary_points:
            continue
        z_index = resolve_roi_slice(roi, series)
        coordinates.extend((z_index, y, x) for x, y in roi.boundary_points)
    if not coordinates:
        raise ValueError("Nodule annotation has no included contour points")
    return tuple(float(value) for value in np.mean(coordinates, axis=0))


def extract_nodule_crop(
    series: DicomSeries,
    annotation: NoduleAnnotation,
    crop_size: tuple[int, int, int] = (64, 64, 64),
    target_spacing: tuple[float, float, float] | None = (1.0, 1.0, 1.0),
    padding_value_hu: float = -1000.0,
) -> NoduleCrop:
    """Extract a fixed-size z/y/x crop in memory, padding with air HU as needed."""
    center = nodule_center_zyx(annotation, series)
    if target_spacing is None:
        native_shape = tuple(int(v) for v in crop_size)
    else:
        native_shape = tuple(
            max(1, int(round(out * target / native)))
            for out, target, native in zip(crop_size, target_spacing, series.voxel_spacing)
        )
    native = _crop_with_padding(series.volume, center, native_shape, padding_value_hu)
    if native.shape != crop_size:
        tensor = torch.from_numpy(native).float()[None, None]
        native = F.interpolate(tensor, size=crop_size, mode="trilinear", align_corners=False)[0, 0].numpy()
    indices = tuple(resolve_roi_slice(roi, series) for roi in annotation.rois if roi.inclusion is not False)
    return NoduleCrop(native.astype(np.float32, copy=False), center, indices, native_shape)


def preprocess_ct_crop(
    crop_hu: np.ndarray, hu_clip_range: tuple[float, float] = (-1000.0, 400.0)
) -> np.ndarray:
    """Clip CT values and min-max normalize them to [0, 1]."""
    low, high = map(float, hu_clip_range)
    if high <= low:
        raise ValueError("HU clip maximum must exceed minimum")
    clipped = np.clip(crop_hu.astype(np.float32, copy=False), low, high)
    return ((clipped - low) / (high - low)).astype(np.float32, copy=False)


def _crop_with_padding(
    volume: np.ndarray,
    center: tuple[float, float, float],
    shape: tuple[int, int, int],
    padding_value: float,
) -> np.ndarray:
    result = np.full(shape, padding_value, dtype=volume.dtype)
    starts = [int(round(c - (size - 1) / 2)) for c, size in zip(center, shape)]
    source_slices = []
    target_slices = []
    for start, size, limit in zip(starts, shape, volume.shape):
        source_start = max(start, 0)
        source_end = min(start + size, limit)
        target_start = source_start - start
        target_end = target_start + max(0, source_end - source_start)
        source_slices.append(slice(source_start, source_end))
        target_slices.append(slice(target_start, target_end))
    result[tuple(target_slices)] = volume[tuple(source_slices)]
    return result

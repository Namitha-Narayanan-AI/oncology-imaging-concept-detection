from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.lidc.annotations import AnnotationFile, NoduleAnnotation, ROI
from src.lidc.crop_extraction import (
    NoduleCrop, ROIAlignmentError, extract_nodule_crop, resolve_roi_slice,
)
from src.lidc.dataset import LIDCNoduleDataset
from src.lidc.dicom_io import DicomSeries
from src.models.simple_3d_cnn import Simple3DCNN


def make_series(shape=(5, 8, 8)):
    return DicomSeries(
        volume=np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        voxel_spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        series_instance_uid="series-1",
        image_positions=tuple((0.0, 0.0, float(z)) for z in range(shape[0])),
        sop_instance_uids=tuple(f"slice-{z}" for z in range(shape[0])),
        slice_coordinates=tuple(float(z) for z in range(shape[0])),
    )


def make_annotation(x=4, y=4, z=2):
    return NoduleAnnotation(
        nodule_id="n1", reader_id="reader", reading_session_index=1,
        malignancy=4, subtlety=3, spiculation=2, lobulation=2, margin=3,
        texture=4, sphericity=3, calcification=1, internal_structure=1,
        rois=(ROI(float(z), f"slice-{z}", True, ((x, y), (x + 1, y), (x, y + 1))),),
    )


def test_roi_slice_matching_prefers_sop_and_falls_back_to_geometry():
    series = make_series()
    assert resolve_roi_slice(ROI(0.0, "slice-3", True, ((1, 1),)), series) == 3
    assert resolve_roi_slice(ROI(2.0, "missing", True, ((1, 1),)), series) == 2
    with pytest.raises(ROIAlignmentError):
        resolve_roi_slice(ROI(20.0, None, True, ((1, 1),)), series)


def test_crop_extraction_has_fixed_shape_and_contains_center():
    series = make_series()
    crop = extract_nodule_crop(series, make_annotation(), (3, 5, 5), None)
    assert crop.values.shape == (3, 5, 5)
    assert crop.values[1, 2, 2] == series.volume[2, 4, 4]


def test_crop_boundary_is_padded_with_air_hu():
    crop = extract_nodule_crop(make_series(), make_annotation(0, 0, 0), (5, 5, 5), None)
    assert crop.values.shape == (5, 5, 5)
    assert np.any(crop.values == -1000.0)


def test_lazy_dataset_returns_conv3d_tensor(monkeypatch, tmp_path: Path):
    annotation = make_annotation()
    annotation_file = AnnotationFile("patient", "study", "series-1", (annotation,))
    monkeypatch.setattr("src.lidc.dataset.parse_annotation_file", lambda _: annotation_file)
    monkeypatch.setattr("src.lidc.dataset.load_ct_series", lambda _: make_series())
    monkeypatch.setattr(
        "src.lidc.dataset.extract_nodule_crop",
        lambda *args, **kwargs: NoduleCrop(np.zeros((4, 6, 8), np.float32), (2, 4, 4), (2,), (4, 6, 8)),
    )
    frame = pd.DataFrame([{
        "xml_path": str(tmp_path / "sample.xml"), "series_dir": str(tmp_path),
        "series_instance_uid": "series-1", "reading_session_index": 1,
        "reader_nodule_id": "n1", "annotation_id": "a1", "patient_id": "p1",
        "malignancy": 4, "malignancy_risk_label": 1, "split": "train",
    }])
    item = LIDCNoduleDataset(frame, split="train", crop_size=(4, 6, 8))[0]
    assert item["image"].shape == (1, 4, 6, 8)
    assert item["image"].dtype == torch.float32
    assert item["target"].item() == 1.0


def test_simple_3d_cnn_forward_shape():
    model = Simple3DCNN()
    assert model(torch.zeros(2, 1, 32, 32, 32)).shape == (2,)

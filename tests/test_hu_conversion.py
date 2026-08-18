from pathlib import Path

import numpy as np
import pytest
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from src.lidc.dicom_io import (
    InconsistentDimensionsError,
    MixedSeriesError,
    load_ct_series,
)


def write_slice(
    path: Path,
    pixels: np.ndarray,
    *,
    series_uid: str,
    instance_number: int,
    z_position: float,
    slope: float = 1.0,
    intercept: float = 0.0,
):
    """Create a minimal synthetic CT DICOM slice for loader tests."""

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CT"
    dataset.SeriesInstanceUID = series_uid
    dataset.StudyInstanceUID = generate_uid()
    dataset.FrameOfReferenceUID = generate_uid()
    dataset.InstanceNumber = instance_number
    dataset.ImagePositionPatient = [0.0, 0.0, z_position]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.PixelSpacing = [0.7, 0.8]
    dataset.SliceThickness = 2.5
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.RescaleSlope = slope
    dataset.RescaleIntercept = intercept
    dataset.PixelData = pixels.astype(np.int16).tobytes()

    pydicom.dcmwrite(path, dataset)


def test_hu_conversion(tmp_path):
    series_uid = generate_uid()
    pixels = np.array([[0, 100], [-100, 50]], dtype=np.int16)

    write_slice(
        tmp_path / "slice.dcm",
        pixels,
        series_uid=series_uid,
        instance_number=1,
        z_position=0.0,
        slope=2.0,
        intercept=-1024.0,
    )

    series = load_ct_series(tmp_path)

    expected = pixels.astype(np.float32) * 2.0 - 1024.0
    np.testing.assert_array_equal(series.volume[0], expected)
    assert series.volume.dtype == np.float32
    assert series.voxel_spacing == (2.5, 0.7, 0.8)
    assert series.origin == (0.0, 0.0, 0.0)
    assert series.orientation == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert series.series_instance_uid == series_uid


def test_geometric_ordering_uses_image_position_patient(tmp_path):
    series_uid = generate_uid()
    tmp_path.joinpath("notes.txt").write_text("not a dicom file")

    write_slice(
        tmp_path / "late.dcm",
        np.full((2, 2), 30, dtype=np.int16),
        series_uid=series_uid,
        instance_number=1,
        z_position=10.0,
    )
    write_slice(
        tmp_path / "early.dcm",
        np.full((2, 2), 10, dtype=np.int16),
        series_uid=series_uid,
        instance_number=2,
        z_position=-5.0,
    )
    write_slice(
        tmp_path / "middle.dcm",
        np.full((2, 2), 20, dtype=np.int16),
        series_uid=series_uid,
        instance_number=3,
        z_position=2.5,
    )

    series = load_ct_series(tmp_path)

    assert series.volume[:, 0, 0].tolist() == [10.0, 20.0, 30.0]
    assert series.voxel_spacing == (7.5, 0.7, 0.8)
    assert series.origin == (0.0, 0.0, -5.0)


def test_mixed_series_rejection(tmp_path):
    write_slice(
        tmp_path / "a.dcm",
        np.ones((2, 2), dtype=np.int16),
        series_uid=generate_uid(),
        instance_number=1,
        z_position=0.0,
    )
    write_slice(
        tmp_path / "b.dcm",
        np.ones((2, 2), dtype=np.int16),
        series_uid=generate_uid(),
        instance_number=2,
        z_position=1.0,
    )

    with pytest.raises(MixedSeriesError):
        load_ct_series(tmp_path)


def test_inconsistent_dimensions_rejection(tmp_path):
    series_uid = generate_uid()
    write_slice(
        tmp_path / "a.dcm",
        np.ones((2, 2), dtype=np.int16),
        series_uid=series_uid,
        instance_number=1,
        z_position=0.0,
    )
    write_slice(
        tmp_path / "b.dcm",
        np.ones((3, 2), dtype=np.int16),
        series_uid=series_uid,
        instance_number=2,
        z_position=1.0,
    )

    with pytest.raises(InconsistentDimensionsError):
        load_ct_series(tmp_path)

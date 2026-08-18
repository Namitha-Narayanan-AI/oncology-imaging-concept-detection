"""Parser for LIDC-IDRI XML nodule annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class ROI:
    """One 2D contour ROI on a CT slice."""

    z_position: float | None
    sop_instance_uid: str | None
    inclusion: bool | None
    boundary_points: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class NoduleAnnotation:
    """Radiologist annotation for one LIDC-IDRI nodule."""

    nodule_id: str | None
    reader_id: str | None
    reading_session_index: int
    malignancy: int | None
    subtlety: int | None
    spiculation: int | None
    lobulation: int | None
    margin: int | None
    texture: int | None
    sphericity: int | None
    calcification: int | None
    internal_structure: int | None
    rois: tuple[ROI, ...]


@dataclass(frozen=True)
class AnnotationFile:
    """Parsed contents of one LIDC-IDRI annotation XML file."""

    patient_id: str | None
    study_instance_uid: str | None
    series_instance_uid: str | None
    nodules: tuple[NoduleAnnotation, ...]


def parse_annotation_file(xml_path: str | Path) -> AnnotationFile:
    """Parse one LIDC-IDRI XML file into nodule annotation dataclasses."""

    root = ET.parse(xml_path).getroot()
    return AnnotationFile(
        patient_id=_first_text(root, "PatientID", "patientID", "PatientId"),
        study_instance_uid=_first_text(root, "StudyInstanceUID"),
        series_instance_uid=_first_text(root, "SeriesInstanceUid", "SeriesInstanceUID"),
        nodules=tuple(
            _parse_nodule(nodule, reader_id, session_index)
            for session_index, session in enumerate(
                _direct_children(root, "readingSession"), start=1
            )
            for reader_id in [_first_text(session, "servicingRadiologistID")]
            for nodule in _direct_children(session, "unblindedReadNodule")
        ),
    )


def _parse_nodule(
    element: ET.Element, reader_id: str | None, reading_session_index: int
) -> NoduleAnnotation:
    characteristics = _first_child(element, "characteristics")
    return NoduleAnnotation(
        nodule_id=_first_text(element, "noduleID"),
        reader_id=reader_id,
        reading_session_index=reading_session_index,
        malignancy=_int_child(characteristics, "malignancy"),
        subtlety=_int_child(characteristics, "subtlety"),
        spiculation=_int_child(characteristics, "spiculation"),
        lobulation=_int_child(characteristics, "lobulation"),
        margin=_int_child(characteristics, "margin"),
        texture=_int_child(characteristics, "texture"),
        sphericity=_int_child(characteristics, "sphericity"),
        calcification=_int_child(characteristics, "calcification"),
        internal_structure=_int_child(characteristics, "internalStructure"),
        rois=tuple(_parse_roi(roi) for roi in _iter_children(element, "roi")),
    )


def _parse_roi(element: ET.Element) -> ROI:
    return ROI(
        z_position=_float_text(_first_text(element, "imageZposition")),
        sop_instance_uid=_first_text(element, "imageSOP_UID"),
        inclusion=_bool_text(_first_text(element, "inclusion")),
        boundary_points=tuple(
            point
            for edge_map in _iter_children(element, "edgeMap")
            if (point := _parse_edge_map(edge_map)) is not None
        ),
    )


def _parse_edge_map(element: ET.Element) -> tuple[int, int] | None:
    x_coord = _int_text(_first_text(element, "xCoord"))
    y_coord = _int_text(_first_text(element, "yCoord"))
    if x_coord is None or y_coord is None:
        return None
    return x_coord, y_coord


def _int_child(element: ET.Element | None, tag_name: str) -> int | None:
    if element is None:
        return None
    return _int_text(_first_text(element, tag_name))


def _first_child(element: ET.Element, tag_name: str) -> ET.Element | None:
    return next(_iter_children(element, tag_name), None)


def _iter_children(element: ET.Element, tag_name: str):
    for child in element.iter():
        if _local_name(child.tag) == tag_name:
            yield child


def _direct_children(element: ET.Element, tag_name: str):
    for child in element:
        if _local_name(child.tag) == tag_name:
            yield child


def _first_text(element: ET.Element, *tag_names: str) -> str | None:
    wanted = set(tag_names)
    for child in element.iter():
        if _local_name(child.tag) in wanted and child.text:
            text = child.text.strip()
            if text:
                return text
    return None


def _int_text(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _float_text(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _bool_text(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]

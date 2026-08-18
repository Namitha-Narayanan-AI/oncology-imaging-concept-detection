import pytest

from src.lidc.splits import assign_patient_splits


def test_patient_assignments_are_disjoint_and_reproducible():
    patients = [f"P{i:03d}" for i in range(20)]
    first = assign_patient_splits(patients + patients[:3], seed=42)
    second = assign_patient_splits(list(reversed(patients)), seed=42)
    assert first == second
    assert set(first) == set(patients)
    assert {split for split in first.values()} == {"train", "val", "test"}


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        assign_patient_splits(["P1"], 0.8, 0.2, 0.2)

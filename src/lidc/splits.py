"""Deterministic patient-level split utilities."""

from __future__ import annotations

import random


def assign_patient_splits(
    patient_ids: list[str],
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """Assign each unique patient to exactly one reproducible split."""

    if any(value < 0 for value in (train_fraction, val_fraction, test_fraction)):
        raise ValueError("Split fractions must be non-negative")
    if abs(train_fraction + val_fraction + test_fraction - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to 1")

    patients = sorted(set(patient_ids))
    random.Random(seed).shuffle(patients)
    count = len(patients)
    train_end = round(count * train_fraction)
    val_end = train_end + round(count * val_fraction)
    val_end = min(val_end, count)

    return {
        patient_id: split
        for split, subset in (
            ("train", patients[:train_end]),
            ("val", patients[train_end:val_end]),
            ("test", patients[val_end:]),
        )
        for patient_id in subset
    }

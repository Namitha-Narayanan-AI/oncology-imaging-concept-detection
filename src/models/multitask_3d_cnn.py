"""Compact multi-label pulmonary-nodule concept model."""

from __future__ import annotations

import torch
from torch import nn


class MultiTask3DCNN(nn.Module):
    """Predict malignancy risk, spiculation and lobulation from one 3D crop."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.malignancy_head = nn.Linear(64, 1)
        self.spiculation_head = nn.Linear(64, 1)
        self.lobulation_head = nn.Linear(64, 1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        representation = self.features(inputs).flatten(1)
        return {
            "malignancy": self.malignancy_head(representation),
            "spiculation": self.spiculation_head(representation),
            "lobulation": self.lobulation_head(representation),
        }

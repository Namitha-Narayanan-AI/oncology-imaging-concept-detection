"""Compact single-task 3D CNN for radiologist-assessed malignancy risk."""

from __future__ import annotations

import torch
from torch import nn


class Simple3DCNN(nn.Module):
    """Map one-channel 3D CT crops to one binary malignancy-risk logit."""

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
        self.classifier = nn.Linear(64, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs).flatten(1)
        return self.classifier(features).squeeze(1)

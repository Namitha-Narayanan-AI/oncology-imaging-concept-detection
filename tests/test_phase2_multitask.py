import numpy as np
import pandas as pd
import torch
from torch import nn

from src.lidc.dataset import binary_concept_target
from src.models.multitask_3d_cnn import MultiTask3DCNN
from src.training.train_phase2_multitask import masked_task_losses


def test_multitask_output_shapes():
    outputs = MultiTask3DCNN()(torch.zeros(2, 1, 32, 32, 32))
    assert {key: value.shape for key, value in outputs.items()} == {
        "malignancy": (2, 1), "spiculation": (2, 1), "lobulation": (2, 1)
    }


def test_binary_concept_rating_conversion():
    assert binary_concept_target(1) == (0.0, True)
    assert binary_concept_target(2) == (0.0, True)
    assert binary_concept_target(3) == (0.0, False)
    assert binary_concept_target(4) == (1.0, True)
    assert binary_concept_target(5) == (1.0, True)
    assert binary_concept_target(np.nan) == (0.0, False)
    assert binary_concept_target(None) == (0.0, False)


def test_masked_losses_are_finite_and_backward_when_auxiliary_missing():
    model = MultiTask3DCNN()
    batch = {
        "image": torch.randn(2, 1, 16, 16, 16),
        "malignancy_target": torch.tensor([0.0, 1.0]),
        "spiculation_target": torch.tensor([0.0, 0.0]),
        "spiculation_valid": torch.tensor([False, False]),
        "lobulation_target": torch.tensor([0.0, 1.0]),
        "lobulation_valid": torch.tensor([True, True]),
    }
    criteria = {task: nn.BCEWithLogitsLoss() for task in ("malignancy", "spiculation", "lobulation")}
    losses = masked_task_losses(model(batch["image"]), batch, criteria)
    assert set(losses) == {"malignancy", "lobulation"}
    total = torch.stack(tuple(losses.values())).mean()
    assert torch.isfinite(total)
    total.backward()
    assert model.features[0].weight.grad is not None

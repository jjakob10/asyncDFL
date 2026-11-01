import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.training import ClassRecall, classification_confusion


@pytest.mark.parametrize("kind,inputs,targets,expected", [
    ("binary", [-9., 9., 0.], [0., 0., 1.], [[1, 1], [0, 1]]),
    ("multilabel", [[-9., 9.], [9., -9.], [0., -9.]],
     [[0., 1.], [0., 1.], [1., 0.]], [[[1, 1], [0, 1]], [[1, 0], [1, 1]]]),
])
def test_generic_binary_and_multilabel_confusion(kind, inputs, targets, expected):
    loader = DataLoader(TensorDataset(torch.tensor(inputs), torch.tensor(targets)), batch_size=2)
    result = classification_confusion(nn.Identity(), {}, loader, kind, ["0", "1"])
    assert result["counts"] == expected
    assert result["sample_count"] == 3


@pytest.mark.parametrize("kind,outputs,targets,expected", [
    ("binary", [-9., 9., 0.], [0., 0., 1.], [.5, 1.]),
    ("multilabel", [[-9., 9.], [9., -9.], [0., -9.]],
     [[0., 1.], [0., 1.], [1., 0.]], [1., .5]),
])
def test_generic_recall_preserves_threshold_and_zero_support(kind, outputs, targets, expected):
    recall = ClassRecall(2, kind)
    assert recall.metrics()["recall_0"] == 0
    recall.update(torch.tensor(outputs), torch.tensor(targets))
    assert [recall.metrics()[f"recall_{i}"] for i in range(2)] == expected

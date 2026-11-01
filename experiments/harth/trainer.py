from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.collection import CollectionSource, digest
from app.config import Settings
from app.federation import StateDict
from app.training import ClassRecall, DataBundle, classification_confusion, class_distribution
from experiments.harth.data import (
    CLASS_CODES, CLASS_NAMES, HarthWindows, read_metadata, split_participants, download, prepare,
)


COLLECTION_CLASS_CODES = CLASS_CODES


def prepare_data(data_dir: Path) -> None:
    root = data_dir / "harth"
    prepared = root / "prepared"
    if (prepared / "metadata.json").is_file():
        read_metadata(prepared, verify=True)
    else:
        prepare(download(root), prepared)


def create_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(6, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, len(CLASS_CODES)),
    )


def load_data(settings: Settings) -> DataBundle:
    if settings.client_count != 6 or settings.empty_client_ids:
        raise ValueError("HARTH subject protocol requires six clients with local stocks")
    root = Path(settings.data_dir) / "harth" / "prepared"
    metadata = read_metadata(root, verify=True)
    test_subjects, groups = split_participants(metadata)
    local = HarthWindows(root, groups[settings.client_id], metadata, settings.max_train_samples)
    test = HarthWindows(root, test_subjects, metadata, settings.max_test_samples)
    if not len(local) or not len(test):
        raise ValueError("HARTH train and test must both have complete windows")
    train_loader = DataLoader(local, batch_size=settings.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(settings.seed + settings.client_id))
    test_loader = DataLoader(test, batch_size=max(settings.batch_size, 256), shuffle=False)
    source = CollectionSource(local, local.labels, local.sample_ids, CLASS_CODES, True, {
        "dataset": "harth", "preparation_hash": digest(metadata),
        "split_seed": 42, "test_participants": list(test_subjects),
        "client_participants": [list(group) for group in groups],
        "protocol": "project_example_subject_holdout",
    })
    encoded = [CLASS_CODES.index(int(code)) for code in local.labels]
    return DataBundle(train_loader, test_loader, len(local),
                      class_distribution(encoded, list(CLASS_NAMES)), source)


def train_local(base_state: StateDict, train_loader: DataLoader,
                settings: Settings) -> tuple[StateDict, dict[str, float]]:
    model = create_model()
    model.load_state_dict(base_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.learning_rate,
                                momentum=settings.momentum)
    criterion = nn.CrossEntropyLoss()
    loss_sum = correct = seen = 0
    for _ in range(settings.local_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            seen += len(targets)
            loss_sum += loss.item() * len(targets)
            correct += int((outputs.argmax(1) == targets).sum())
    return ({key: value.detach().cpu() for key, value in model.state_dict().items()},
            {"loss": loss_sum / max(1, seen), "accuracy": correct / max(1, seen)})


@torch.inference_mode()
def evaluate(state: StateDict, test_loader: DataLoader) -> dict[str, float]:
    model = create_model()
    model.load_state_dict(state)
    model.eval()
    recall = ClassRecall(len(CLASS_CODES))
    loss_sum = correct = seen = 0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        recall.update(outputs, targets)
        loss_sum += float(nn.functional.cross_entropy(outputs, targets, reduction="sum"))
        correct += int((outputs.argmax(1) == targets).sum())
        seen += len(targets)
    return {**recall.metrics(), "loss": loss_sum / max(1, seen), "accuracy": correct / max(1, seen)}


def evaluate_confusion(state: StateDict, test_loader: DataLoader) -> dict:
    return classification_confusion(create_model(), state, test_loader,
                                    "multiclass", list(CLASS_NAMES))

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from app.config import Settings
from app.federation import StateDict
from app.training import ClassRecall, classification_confusion, class_distribution, DataBundle, make_partitions


class MnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def create_model() -> nn.Module:
    return MnistCNN()


def load_data(settings: Settings) -> DataBundle:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        settings.data_dir,
        train=True,
        download=settings.download_data,
        transform=transform,
    )
    test_dataset: Dataset = datasets.MNIST(
        settings.data_dir,
        train=False,
        download=settings.download_data,
        transform=transform,
    )

    labels = train_dataset.targets.numpy()
    candidate_indices = np.arange(len(labels))
    if settings.max_train_samples:
        rng = np.random.default_rng(settings.seed)
        candidate_indices = rng.permutation(candidate_indices)[
            : settings.max_train_samples
        ]
        labels = labels[candidate_indices]

    partitions = make_partitions(
        labels,
        settings.client_count,
        settings.partition_mode,
        settings.seed,
        settings.dirichlet_alpha,
        min(
            settings.min_partition_size,
            max(1, len(labels) // settings.client_count // 2),
        ),
        empty_client_ids=settings.empty_client_ids,
        client_classes=settings.client_classes,
    )
    local_indices = candidate_indices[partitions[settings.client_id]]
    train_subset = Subset(train_dataset, local_indices.tolist())

    if settings.max_test_samples:
        test_indices = np.random.default_rng(settings.seed).permutation(
            len(test_dataset)
        )[: settings.max_test_samples]
        test_dataset = Subset(test_dataset, test_indices.tolist())

    generator = torch.Generator().manual_seed(settings.seed + settings.client_id)
    train_loader = None
    if len(train_subset):
        train_loader = DataLoader(
            train_subset,
            batch_size=settings.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(settings.batch_size, 256),
        shuffle=False,
        num_workers=0,
    )
    return DataBundle(
        train_loader, test_loader, len(train_subset),
        class_distribution(labels[partitions[settings.client_id]], [str(item) for item in range(10)]),
    )


def train_local(
    base_state: StateDict,
    train_loader: DataLoader,
    settings: Settings,
) -> tuple[StateDict, dict[str, float]]:
    model = create_model()
    model.load_state_dict(base_state)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=settings.learning_rate,
        momentum=settings.momentum,
    )
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for _ in range(settings.local_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_seen += batch_size
    return (
        {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        {
            "loss": total_loss / max(total_seen, 1),
            "accuracy": total_correct / max(total_seen, 1),
        },
    )


@torch.inference_mode()
def evaluate(state: StateDict, test_loader: DataLoader) -> dict[str, float]:
    model = create_model()
    model.load_state_dict(state)
    model.eval()
    recall = ClassRecall(10, "multiclass")
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        recall.update(outputs, targets)
        total_loss += criterion(outputs, targets).item()
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_seen += targets.size(0)
    return {
        **recall.metrics(),
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
    }


def evaluate_confusion(
    state: StateDict, test_loader: DataLoader,
) -> dict[str, Any]:
    """Evaluate final confusion counts on this client's test set."""
    return classification_confusion(
        create_model(), state, test_loader, "multiclass", [str(i) for i in range(10)],
    )

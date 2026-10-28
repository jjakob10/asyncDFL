"""Shared Johann adapter; reuse the existing architectures and preprocessing."""
from __future__ import annotations

import time
from importlib import import_module
from types import ModuleType

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from app.config import Settings
from app.federation import StateDict
from app.run_control import RunControl
from app.training import DataBundle, class_distribution
from experiments.johann.data import index_hash, quantitative_partitions, split_training, stratified_subset


def base_trainer(dataset: str) -> ModuleType:
    return import_module(f"experiments.{dataset}.trainer")


def load_data(settings: Settings, dataset: str) -> DataBundle:
    if settings.client_count != 8 or settings.empty_client_ids:
        raise ValueError("Johann requires eight clients with local data")
    if dataset == "mnist":
        factory = datasets.MNIST
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
    else:
        factory = datasets.CIFAR10
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            (.4914, .4822, .4465), (.2470, .2435, .2616))])
    training = factory(settings.data_dir, train=True, download=settings.download_data, transform=transform)
    labels = np.asarray(training.targets, dtype=np.int64)
    train_indices, validation_indices = split_training(labels, settings.seed, settings.max_train_samples)
    parts = quantitative_partitions(labels[train_indices], settings.johann_distribution, settings.seed)
    local = train_indices[parts[settings.client_id]]
    names = [str(i) for i in range(10)] if dataset == "mnist" else list(training.classes)
    distribution = class_distribution(labels[local], names)
    distribution.update(
        training_indices_sha256=index_hash(train_indices),
        validation_indices_sha256=index_hash(validation_indices),
        local_indices_sha256=index_hash(local), local_indices=local.tolist(),
        validation_sample_count=len(validation_indices),
        global_training_counts=np.bincount(labels[train_indices], minlength=10).tolist(),
        tracked_class_ids=[3, 8], tracked_class_names=[names[3], names[8]],
        johann_distribution=settings.johann_distribution,
    )
    if settings.evaluation_split == "validation":
        evaluation = Subset(training, validation_indices.tolist())
        eval_labels = labels[validation_indices]
    else:
        evaluation = factory(settings.data_dir, train=False, download=settings.download_data, transform=transform)
        eval_labels = np.asarray(evaluation.targets)
    if settings.max_test_samples:
        subset = stratified_subset(eval_labels, min(settings.max_test_samples, len(eval_labels)), settings.seed + 20000)
        evaluation = Subset(evaluation, subset.tolist())
    distribution["evaluation_sample_count"] = len(evaluation)
    loader = DataLoader(Subset(training, local.tolist()), batch_size=settings.batch_size,
                        shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(settings.seed + settings.client_id))
    test_loader = DataLoader(evaluation, batch_size=max(256, settings.batch_size), shuffle=False,
                             generator=torch.Generator().manual_seed(settings.seed + 30000))
    return DataBundle(loader, test_loader, len(local), distribution)


def train_controlled(base: StateDict, loader: DataLoader, settings: Settings,
                     control: RunControl, dataset: str) -> tuple[StateDict, dict[str, float]]:
    model = base_trainer(dataset).create_model()
    model.load_state_dict(base)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.learning_rate, momentum=settings.momentum)
    total_loss = correct = seen = 0
    for _ in range(settings.local_epochs):
        for inputs, targets in loader:
            control.checkpoint()
            began = time.monotonic()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            count = len(targets)
            total_loss += loss.item() * count
            correct += (outputs.argmax(1) == targets).sum().item()
            seen += count
            control.batch_finished(count, time.monotonic() - began)
    control.checkpoint()
    return ({name: value.detach().cpu() for name, value in model.state_dict().items()},
            {"loss": total_loss / max(1, seen), "accuracy": correct / max(1, seen)})


def metrics_from_counts(counts: np.ndarray, loss: float = 0.0) -> dict[str, float | int]:
    counts = np.asarray(counts, dtype=np.int64)
    actual, predicted = counts.sum(1), counts.sum(0)
    diagonal = np.diag(counts)
    recall = np.divide(diagonal, actual, out=np.zeros(10, dtype=float), where=actual > 0)
    f1 = np.divide(2 * diagonal, actual + predicted, out=np.zeros(10, dtype=float), where=(actual + predicted) > 0)
    result = {"loss": float(loss), "accuracy": float(diagonal.sum() / max(1, counts.sum())),
              "macro_f1": float(f1.mean()), "recall_tracked": float(recall[[3, 8]].mean()),
              "recall_other": float(np.delete(recall, [3, 8]).mean())}
    result.update({f"recall_{i}": float(recall[i]) for i in range(10)})
    result.update({f"support_{i}": int(actual[i]) for i in range(10)})
    return result


@torch.inference_mode()
def evaluate_detailed(state: StateDict, loader: DataLoader, dataset: str) -> dict[str, float | int]:
    model = base_trainer(dataset).create_model()
    model.load_state_dict(state)
    model.eval()
    counts = torch.zeros(10, 10, dtype=torch.int64)
    loss = 0.0
    for inputs, targets in loader:
        outputs = model(inputs)
        loss += nn.functional.cross_entropy(outputs, targets, reduction="sum").item()
        counts += torch.bincount(targets.long() * 10 + outputs.argmax(1), minlength=100).reshape(10, 10)
    return metrics_from_counts(counts.numpy(), loss / max(1, int(counts.sum())))

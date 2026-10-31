"""cifar10 adapter for controlled availability experiments."""
from experiments.cifar10.trainer import create_model, train_local, evaluate, evaluate_confusion
from experiments.johann import trainer as shared


def load_data(settings):
    return shared.load_data(settings, "cifar10")


def train_local_controlled(base, loader, settings, control):
    return shared.train_controlled(base, loader, settings, control, "cifar10")


def evaluate_detailed(state, loader):
    return shared.evaluate_detailed(state, loader, "cifar10")


def train_reintegration(base, previous, loader, settings, control):
    return shared.train_reintegration(base, previous, loader, settings, control, "cifar10")

"""mnist adapter for controlled availability experiments."""
from experiments.mnist.trainer import create_model, train_local, evaluate, evaluate_confusion
from experiments.johann import trainer as shared


def load_data(settings):
    return shared.load_data(settings, "mnist")


def train_local_controlled(base, loader, settings, control):
    return shared.train_controlled(base, loader, settings, control, "mnist")


def evaluate_detailed(state, loader):
    return shared.evaluate_detailed(state, loader, "mnist")


def train_reintegration(base, previous, loader, settings, control):
    return shared.train_reintegration(base, previous, loader, settings, control, "mnist")

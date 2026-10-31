from app.config import Settings
from app.training import DataBundle
from experiments.cifar10 import trainer
from experiments.cifar10.trainer import create_model, evaluate, evaluate_confusion, train_local
from experiments.simulated_collection.images import load_collection_data


def load_data(settings: Settings) -> DataBundle:
    return load_collection_data(trainer, "cifar10", settings)

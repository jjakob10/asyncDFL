from app.config import Settings
from app.training import DataBundle
from experiments.mnist import trainer
from experiments.mnist.trainer import create_model, evaluate, evaluate_confusion, train_local
from experiments.simulated_collection.images import load_collection_data


def load_data(settings: Settings) -> DataBundle:
    return load_collection_data(trainer, "mnist", settings)

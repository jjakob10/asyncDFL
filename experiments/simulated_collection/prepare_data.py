"""Dataset preparation entry point used by the dedicated images."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.harth.data import download, prepare, read_metadata


def prepare_dataset(dataset: str, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if dataset == "harth":
        prepared = root / "harth" / "prepared"
        if (prepared / "metadata.json").is_file():
            read_metadata(prepared, verify=True)
            return
        prepare(download(root / "harth"), prepared)
    else:
        from torchvision import datasets

        factory = datasets.MNIST if dataset == "mnist" else datasets.CIFAR10
        if dataset == "cifar10":
            # Same archive and checksum as the existing CIFAR-10 image.
            factory.url = ("https://huggingface.co/datasets/xingslong/cifar-10-batches-py/"
                           "resolve/main/cifar-10-python.tar.gz")
        factory(str(root), train=True, download=True)
        factory(str(root), train=False, download=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("mnist", "cifar10", "harth"))
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    args = parser.parse_args()
    prepare_dataset(args.dataset, args.data_dir)

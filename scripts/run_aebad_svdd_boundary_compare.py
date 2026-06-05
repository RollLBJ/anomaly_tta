from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

THIS_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_robustad_svdd_boundary_compare as core  # noqa: E402


def train_split_dir(data_root: Path, category: str) -> Path:
    path = data_root / category / "train"
    if not path.exists():
        raise FileNotFoundError(f"Missing AeBAD train split: {path}")
    return path


def test_split_dirs(data_root: Path, category: str, splits: Sequence[str]) -> list[Path]:
    test_root = data_root / category / "test"
    paths = [test_root / split for split in splits] if splits else sorted(path for path in test_root.iterdir() if path.is_dir())
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing AeBAD test splits: {missing}")
    if not paths:
        raise FileNotFoundError(f"No AeBAD test splits found for category={category}")
    return paths


def split_token(path: Path, category: str) -> str:
    del category
    return path.name


def domain_name(split: str) -> str:
    return split


class AeBADImageDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        category: str,
        resize_size: int,
        crop_size: int,
        label_filter: int | None = None,
        stream_order: str = "random",
        seed: int = 0,
    ) -> None:
        del category
        self.split_dir = split_dir
        self.transform = core.build_transform(resize_size=resize_size, crop_size=crop_size)
        samples: list[tuple[Path, int]] = []
        for folder_name, label in (("good", 0), ("anomaly", 1)):
            if label_filter is not None and int(label_filter) != int(label):
                continue
            folder = split_dir / folder_name
            if folder.exists():
                samples.extend((path, int(label)) for path in core.image_paths(folder))
        if not samples:
            raise RuntimeError(f"No AeBAD images found in {split_dir}")
        samples = sorted(samples, key=lambda item: str(item[0]))
        if str(stream_order) == "random":
            rng = np.random.default_rng(int(seed))
            order = np.arange(len(samples))
            rng.shuffle(order)
            samples = [samples[int(index)] for index in order]
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return {
            "image": self.transform(image),
            "label": torch.tensor(int(label), dtype=torch.long),
            "path": str(path),
        }


core.RobustADImageDataset = AeBADImageDataset
core.train_split_dir = train_split_dir
core.test_split_dirs = test_split_dirs
core.split_token = split_token
core.domain_name = domain_name


if __name__ == "__main__":
    core.main()

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_robustad_svdd_boundary_compare as core  # noqa: E402


def _category_from_marker(path: Path, split: str) -> str:
    prefix = f"__m2ad_{split}__"
    if not path.name.startswith(prefix):
        raise ValueError(f"Unexpected M2AD virtual split path: {path}")
    return path.name[len(prefix) :]


def _meta_path(data_root: Path) -> Path:
    path = data_root / "jsons" / "meta_unsupervised.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing M2AD split metadata: {path}")
    return path


def train_split_dir(data_root: Path, category: str) -> Path:
    _meta_path(data_root)
    return data_root / f"__m2ad_train__{category}"


def test_split_dirs(data_root: Path, category: str, splits: Sequence[str]) -> list[Path]:
    if splits and any(str(split) != "test" for split in splits):
        raise ValueError(f"M2AD single-view split only exposes split='test', got {list(splits)}")
    _meta_path(data_root)
    return [data_root / f"__m2ad_test__{category}"]


def split_token(path: Path, category: str) -> str:
    del category
    if path.name.startswith("__m2ad_test__"):
        return "test"
    if path.name.startswith("__m2ad_train__"):
        return "train"
    return path.name


def domain_name(split: str) -> str:
    return "illumination_shift" if split == "test" else split


class M2ADImageDataset(Dataset):
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
        split = split_token(split_dir, category="")
        category_name = _category_from_marker(split_dir, split=split)
        root = split_dir.parent
        self.transform = core.build_transform(resize_size=resize_size, crop_size=crop_size)
        with _meta_path(root).open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        rows = list(meta[split][category_name])
        if label_filter is not None:
            rows = [row for row in rows if int(row["image_anomaly"]) == int(label_filter)]

        samples: list[dict[str, Any]] = []
        for row in rows:
            image_path = root / row["img_path"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing M2AD image referenced by metadata: {image_path}")
            samples.append({"path": image_path, "label": int(row["image_anomaly"])})
        if not samples:
            raise RuntimeError(f"No M2AD images found for category={category_name} split={split} root={root}")

        samples = sorted(samples, key=lambda item: str(item["path"]))
        if str(stream_order) == "random":
            rng = np.random.default_rng(int(seed))
            order = np.arange(len(samples))
            rng.shuffle(order)
            samples = [samples[int(index)] for index in order]
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        return {
            "image": self.transform(image),
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
            "path": str(sample["path"]),
        }


core.RobustADImageDataset = M2ADImageDataset
core.train_split_dir = train_split_dir
core.test_split_dirs = test_split_dirs
core.split_token = split_token
core.domain_name = domain_name


if __name__ == "__main__":
    core.main()

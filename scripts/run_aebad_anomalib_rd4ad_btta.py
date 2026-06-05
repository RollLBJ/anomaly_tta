from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


THIS_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SCRIPTS = Path("/home/qilab/byeongju_lee/bt_recsvm_research/scripts")
if str(UPSTREAM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SCRIPTS))

import run_robustad_anomalib_rd4ad_bt_recsvm as runner  # noqa: E402


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def image_paths(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(folder.rglob(pattern))
    return sorted(paths)


def train_split_dir(data_root: Path, category: str) -> Path:
    path = data_root / category / "train"
    if not path.exists():
        raise FileNotFoundError(f"Missing AeBAD train split: {path}")
    return path


def test_split_dirs(data_root: Path, category: str, splits: Sequence[str]) -> list[Path]:
    test_root = data_root / category / "test"
    if splits:
        paths = [test_root / split for split in splits]
    else:
        paths = sorted(path for path in test_root.iterdir() if path.is_dir())
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing AeBAD test splits: {missing}")
    if not paths:
        raise FileNotFoundError(f"No AeBAD test splits found for category={category}")
    return paths


def split_token(path: Path, category: str) -> str:
    return path.name


def domain_name(category: str, split: str) -> str:
    return split


def build_transform(resize_size: int, crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((int(resize_size), int(resize_size)), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((int(crop_size), int(crop_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class AeBADImageDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        resize_size: int,
        crop_size: int,
        label_filter: int | None = None,
        stream_order: str = "sequential",
        seed: int = 0,
    ) -> None:
        self.split_dir = split_dir
        self.transform = build_transform(resize_size=resize_size, crop_size=crop_size)
        samples: list[tuple[Path, int]] = []
        for folder_name, label in (("good", 0), ("anomaly", 1)):
            if label_filter is not None and int(label_filter) != label:
                continue
            folder = split_dir / folder_name
            if folder.exists():
                samples.extend((path, label) for path in image_paths(folder))
        if not samples:
            raise RuntimeError(f"No AeBAD images found in {split_dir}")
        samples = sorted(samples, key=lambda item: str(item[0]))
        if stream_order == "random":
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


runner.RobustADImageDataset = AeBADImageDataset
runner.train_split_dir = train_split_dir
runner.test_split_dirs = test_split_dirs
runner.split_token = split_token
runner.domain_name = domain_name
runner.TABLE2_RD4AD_AUROC = {}


_write_report = runner.write_report


def write_report(*args: Any, **kwargs: Any) -> None:
    _write_report(*args, **kwargs)
    output_root = kwargs.get("output_root", args[0] if args else None)
    if output_root is None:
        return
    report_path = Path(output_root) / "report.md"
    if not report_path.is_file():
        return
    text = report_path.read_text(encoding="utf-8")
    text = text.replace("# RobustAD Anomalib RD4AD TTA", "# AeBAD Anomalib RD4AD BTTA")
    text = text.replace(
        "- Dataset: RobustAD single-view imagefolder splits.",
        "- Dataset: AeBAD imagefolder splits.",
    )
    report_path.write_text(text, encoding="utf-8")


runner.write_report = write_report


if __name__ == "__main__":
    runner.main()

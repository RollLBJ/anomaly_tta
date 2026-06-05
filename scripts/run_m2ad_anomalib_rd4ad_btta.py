from __future__ import annotations

import json
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


def _category_from_marker(path: Path, split: str) -> str:
    prefix = f"__m2ad_{split}__"
    if not path.name.startswith(prefix):
        raise ValueError(f"Unexpected M2AD virtual split path: {path}")
    return path.name[len(prefix) :]


def _m2ad_meta_root(marker_path: Path) -> Path:
    return marker_path.parent


def train_split_dir(data_root: Path, category: str) -> Path:
    meta_path = data_root / "jsons" / "meta_unsupervised.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing M2AD split metadata: {meta_path}")
    return data_root / f"__m2ad_train__{category}"


def test_split_dirs(data_root: Path, category: str, splits: Sequence[str]) -> list[Path]:
    if splits and any(split != "test" for split in splits):
        raise ValueError(f"M2AD single-view split only exposes split='test', got {list(splits)}")
    meta_path = data_root / "jsons" / "meta_unsupervised.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing M2AD split metadata: {meta_path}")
    return [data_root / f"__m2ad_test__{category}"]


def split_token(path: Path, category: str) -> str:
    if path.name.startswith("__m2ad_test__"):
        return "test"
    if path.name.startswith("__m2ad_train__"):
        return "train"
    return path.name


def domain_name(category: str, split: str) -> str:
    return "illumination_shift" if split == "test" else split


class M2ADImageDataset(Dataset):
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
        split = split_token(split_dir, category="")
        category = _category_from_marker(split_dir, split=split)
        root = _m2ad_meta_root(split_dir)
        with (root / "jsons" / "meta_unsupervised.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        rows = list(meta[split][category])
        if label_filter is not None:
            rows = [row for row in rows if int(row["image_anomaly"]) == int(label_filter)]
        samples: list[dict[str, Any]] = []
        for row in rows:
            image_path = root / row["img_path"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing M2AD image referenced by metadata: {image_path}")
            samples.append({"path": image_path, "label": int(row["image_anomaly"])})
        if not samples:
            raise RuntimeError(f"No M2AD images found for category={category} split={split} root={root}")
        samples = sorted(samples, key=lambda item: str(item["path"]))
        if stream_order == "random":
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


runner.RobustADImageDataset = M2ADImageDataset
runner.train_split_dir = train_split_dir
runner.test_split_dirs = test_split_dirs
runner.split_token = split_token
runner.domain_name = domain_name
runner.TABLE2_RD4AD_AUROC = {}


def build_transform(resize_size: int, crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((int(resize_size), int(resize_size)), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((int(crop_size), int(crop_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


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
    text = text.replace("# RobustAD Anomalib RD4AD TTA", "# M2AD Anomalib RD4AD BTTA")
    text = text.replace(
        "- Dataset: RobustAD single-view imagefolder splits.",
        "- Dataset: M2AD single-view illumination-shift metadata splits.",
    )
    report_path.write_text(text, encoding="utf-8")


runner.write_report = write_report


if __name__ == "__main__":
    runner.main()

from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from imagecorruptions import corrupt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


THIS_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SCRIPTS = Path("/home/qilab/byeongju_lee/bt_recsvm_research/scripts")
if str(UPSTREAM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SCRIPTS))

import run_robustad_anomalib_rd4ad_bt_recsvm as runner  # noqa: E402


MVTEC_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

CORRUPTION_NAMES = {
    "brightness": "brightness",
    "contrast": "contrast",
    "defocus_blur": "defocus_blur",
    "gaussian_noise": "gaussian_noise",
}
CURRENT_MVTEC_C_SEVERITY = 5


def build_corruption_by_split(severity: int) -> dict[str, str]:
    return {f"{name}_s{int(severity)}": corruption for name, corruption in CORRUPTION_NAMES.items()}


CORRUPTION_BY_SPLIT = build_corruption_by_split(CURRENT_MVTEC_C_SEVERITY)
DEFAULT_TEST_SPLITS = ("test0", *CORRUPTION_BY_SPLIT.keys())
SEVERITY_BY_SPLIT = {name: CURRENT_MVTEC_C_SEVERITY for name in CORRUPTION_BY_SPLIT}
MARKER_RE = re.compile(r"^__mvtec_c_(?P<split>.+)__$")


def configure_mvtec_c_severity(severity: int) -> None:
    global CORRUPTION_BY_SPLIT, DEFAULT_TEST_SPLITS, SEVERITY_BY_SPLIT, CURRENT_MVTEC_C_SEVERITY
    if not 1 <= int(severity) <= 5:
        raise ValueError(f"MVTec-C severity must be in [1, 5], got {severity}")
    CURRENT_MVTEC_C_SEVERITY = int(severity)
    CORRUPTION_BY_SPLIT = build_corruption_by_split(CURRENT_MVTEC_C_SEVERITY)
    DEFAULT_TEST_SPLITS = ("test0", *CORRUPTION_BY_SPLIT.keys())
    SEVERITY_BY_SPLIT = {name: CURRENT_MVTEC_C_SEVERITY for name in CORRUPTION_BY_SPLIT}


def pop_mvtec_c_severity(argv: list[str]) -> int:
    severity = CURRENT_MVTEC_C_SEVERITY
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--mvtec-c-severity":
            if index + 1 >= len(argv):
                raise ValueError("--mvtec-c-severity requires an integer value")
            severity = int(argv[index + 1])
            del argv[index : index + 2]
            continue
        if arg.startswith("--mvtec-c-severity="):
            severity = int(arg.split("=", 1)[1])
            del argv[index]
            continue
        index += 1
    return severity


def image_paths(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(folder.rglob(pattern))
    return sorted(paths)


def train_split_dir(data_root: Path, category: str) -> Path:
    path = data_root / category / "train"
    if not path.exists():
        raise FileNotFoundError(f"Missing MVTec train split: {path}")
    return path


def test_split_dirs(data_root: Path, category: str, splits: Sequence[str]) -> list[Path]:
    requested = tuple(splits) if splits else DEFAULT_TEST_SPLITS
    invalid = [split for split in requested if split != "test0" and split not in CORRUPTION_BY_SPLIT]
    if invalid:
        raise ValueError(f"Unsupported MVTec-C split(s): {invalid}; valid={list(DEFAULT_TEST_SPLITS)}")
    category_root = data_root / category
    if not (category_root / "test").exists():
        raise FileNotFoundError(f"Missing MVTec test split: {category_root / 'test'}")
    return [category_root / f"__mvtec_c_{split}__" for split in requested]


def split_token(path: Path, category: str) -> str:
    match = MARKER_RE.match(path.name)
    if match:
        return match.group("split")
    if path.name == "train":
        return "train"
    return path.name


def domain_name(category: str, split: str) -> str:
    return "clean_test" if split == "test0" else split


def build_transform(resize_size: int, crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((int(resize_size), int(resize_size)), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((int(crop_size), int(crop_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
    )


def _corrupt_image(image: Image.Image, split: str, path: Path) -> Image.Image:
    corruption_name = CORRUPTION_BY_SPLIT.get(split)
    if corruption_name is None:
        return image
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    seed = zlib.crc32(f"{split}:{path}".encode("utf-8")) & 0xFFFF_FFFF
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        corrupted = corrupt(array, corruption_name=corruption_name, severity=int(SEVERITY_BY_SPLIT[split]))
    finally:
        np.random.set_state(rng_state)
    return Image.fromarray(np.asarray(corrupted, dtype=np.uint8), mode="RGB")


class MVTecCImageDataset(Dataset):
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
        self.split = split_token(split_dir, category="")
        self.transform = build_transform(resize_size=resize_size, crop_size=crop_size)

        if self.split == "train":
            root = split_dir
            folder_specs = (("good", 0),)
        else:
            root = split_dir.parent / "test"
            folder_specs = tuple((folder.name, 0 if folder.name == "good" else 1) for folder in sorted(root.iterdir()) if folder.is_dir())

        samples: list[tuple[Path, int]] = []
        for folder_name, label in folder_specs:
            if label_filter is not None and int(label_filter) != int(label):
                continue
            folder = root / folder_name
            if folder.exists():
                samples.extend((path, int(label)) for path in image_paths(folder))
        if not samples:
            raise RuntimeError(f"No MVTec images found in {split_dir}")
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
        image = _corrupt_image(image, split=self.split, path=path)
        return {
            "image": self.transform(image),
            "label": torch.tensor(int(label), dtype=torch.long),
            "path": str(path),
        }


_parse_args = runner.parse_args


def parse_args() -> Any:
    explicit_categories = "--categories" in sys.argv
    explicit_data_root = "--data-root" in sys.argv
    explicit_output_root = "--output-root" in sys.argv
    severity = pop_mvtec_c_severity(sys.argv)
    configure_mvtec_c_severity(severity)
    args = _parse_args()
    args.mvtec_c_severity = int(severity)
    if not explicit_categories:
        args.categories = list(MVTEC_CATEGORIES)
    if not explicit_data_root:
        args.data_root = str(THIS_ROOT / "data" / "mvtec_ad")
    if not explicit_output_root:
        args.output_root = str(THIS_ROOT / "outputs" / f"mvtec_c_s{int(severity)}_rd4ad_btta")
    return args


runner.parse_args = parse_args
runner.RobustADImageDataset = MVTecCImageDataset
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
    text = text.replace("# RobustAD Anomalib RD4AD TTA", "# MVTec-C Anomalib RD4AD BTTA")
    text = text.replace(
        "- Dataset: RobustAD single-view imagefolder splits.",
        f"- Dataset: MVTec AD clean train/test with ImageNet-C corruptions at severity {CURRENT_MVTEC_C_SEVERITY}.",
    )
    report_path.write_text(text, encoding="utf-8")


runner.write_report = write_report


if __name__ == "__main__":
    runner.main()

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from imagecorruptions import corrupt
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
ADSHIFT_ROOT = ROOT / "external" / "ADShift"
if str(ADSHIFT_ROOT) not in sys.path:
    sys.path.insert(0, str(ADSHIFT_ROOT))

from dataset import AugMixDatasetMVTec  # noqa: E402
from de_resnet import de_wide_resnet50_2  # noqa: E402
from resnet import wide_resnet50_2 as train_wide_resnet50_2  # noqa: E402
from resnet_TTA import wide_resnet50_2 as tta_wide_resnet50_2  # noqa: E402


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
CORRUPTIONS = ("brightness", "contrast", "defocus_blur", "gaussian_noise")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
METHOD = "gnl_adshift"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADShift GNL/DINL+ATTA on MVTec-C s5.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "mvtec_ad"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs/mvtec_c_s5_gnl_adshift_seed012_randomstream_256"))
    parser.add_argument("--categories", nargs="+", default=list(MVTEC_CATEGORIES))
    parser.add_argument("--splits", nargs="+", default=["test0", *(f"{name}_s5" for name in CORRUPTIONS)])
    parser.add_argument("--stream-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--stream-order", choices=("random", "sequential"), default="random")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lamda", type=float, default=0.5)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.stream_order != "random":
        print("[warn] project default is random stream; non-random stream requested explicitly", file=sys.stderr)
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_paths(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            paths.append(path)
    return sorted(paths)


def split_corruption(split: str) -> str | None:
    if split in {"train", "test0"}:
        return None
    for corruption in CORRUPTIONS:
        if split == f"{corruption}_s5":
            return corruption
    raise ValueError(f"Unsupported split: {split}")


def split_domain(split: str) -> str:
    return "clean_test" if split == "test0" else split


def corrupt_image(image: Image.Image, split: str, path: Path, severity: int) -> Image.Image:
    corruption = split_corruption(split)
    if corruption is None:
        return image
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    seed = zlib.crc32(f"{split}:{path}".encode("utf-8")) & 0xFFFF_FFFF
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        corrupted = corrupt(array, corruption_name=corruption, severity=int(severity))
    finally:
        np.random.set_state(rng_state)
    return Image.fromarray(np.asarray(corrupted, dtype=np.uint8), mode="RGB")


class NormalPILDataset(Dataset):
    def __init__(self, data_root: Path, category: str, image_size: int) -> None:
        self.paths = image_paths(data_root / category / "train" / "good")
        if not self.paths:
            raise RuntimeError(f"No normal training images for {category}")
        self.resize = transforms.Resize((int(image_size), int(image_size)))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.resize(image), 0


class MVTecCEvalDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        category: str,
        split: str,
        image_size: int,
        severity: int,
        stream_order: str,
        seed: int,
    ) -> None:
        self.split = split
        self.severity = int(severity)
        self.transform = transforms.Compose(
            [
                transforms.Resize((int(image_size), int(image_size))),
                transforms.CenterCrop((int(image_size), int(image_size))),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
        )
        test_root = data_root / category / "test"
        samples: list[tuple[Path, int]] = []
        for folder in sorted(test_root.iterdir()):
            if not folder.is_dir():
                continue
            label = 0 if folder.name == "good" else 1
            samples.extend((path, label) for path in image_paths(folder))
        if not samples:
            raise RuntimeError(f"No test images for {category}/{split}")
        samples = sorted(samples, key=lambda item: str(item[0]))
        if stream_order == "random":
            rng = np.random.default_rng(int(seed))
            order = np.arange(len(samples))
            rng.shuffle(order)
            samples = [samples[int(index)] for index in order]
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = corrupt_image(image=image, split=self.split, path=path, severity=self.severity)
        return self.transform(image), torch.tensor(int(label), dtype=torch.long), str(path)


def loss_function(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
    cos_loss = torch.nn.CosineSimilarity()
    loss = torch.zeros((), device=a[0].device)
    for index in range(len(a)):
        loss = loss + torch.mean(
            1 - cos_loss(a[index].view(a[index].shape[0], -1), b[index].view(b[index].shape[0], -1)),
        )
    return loss


def loss_function_last(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
    cos_loss = torch.nn.CosineSimilarity()
    return torch.mean(1 - cos_loss(a[0].view(a[0].shape[0], -1), b[0].view(b[0].shape[0], -1)))


def train_category(category: str, args: argparse.Namespace, device: torch.device, checkpoint_path: Path) -> None:
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
    )
    train_data = AugMixDatasetMVTec(
        NormalPILDataset(Path(args.data_root), category=category, image_size=int(args.image_size)),
        preprocess,
    )
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    loader = DataLoader(
        train_data,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        generator=generator,
    )
    encoder, bn = train_wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device).eval()
    bn = bn.to(device)
    decoder = de_wide_resnet50_2(pretrained=False).to(device)
    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=float(args.lr), betas=(0.5, 0.999))

    for epoch in range(int(args.epochs)):
        bn.train()
        decoder.train()
        losses: list[float] = []
        for normal, augmix_img, gray_img in loader:
            normal = normal.to(device, non_blocking=True)
            augmix_img = augmix_img.to(device, non_blocking=True)
            gray_img = gray_img.to(device, non_blocking=True)
            with torch.no_grad():
                inputs_normal = encoder(normal)
                inputs_augmix = encoder(augmix_img)
                inputs_gray = encoder(gray_img)
            bn_normal = bn(inputs_normal)
            outputs_normal = decoder(bn_normal)
            bn_augmix = bn(inputs_augmix)
            outputs_augmix = decoder(bn_augmix)
            bn_gray = bn(inputs_gray)
            outputs_gray = decoder(bn_gray)
            loss_bn = loss_function([bn_normal], [bn_augmix]) + loss_function([bn_normal], [bn_gray])
            loss_last = loss_function_last(outputs_normal, outputs_augmix) + loss_function_last(outputs_normal, outputs_gray)
            loss_normal = loss_function(inputs_normal, outputs_normal)
            loss = loss_normal * 0.9 + loss_bn * 0.05 + loss_last * 0.05
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[train] {category} epoch={epoch + 1}/{args.epochs} loss={np.mean(losses):.6f}", flush=True)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"bn": bn.state_dict(), "decoder": decoder.state_dict()}, checkpoint_path)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    encoder, bn = tta_wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device).eval()
    bn = bn.to(device)
    decoder = de_wide_resnet50_2(pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    bn_state = dict(checkpoint["bn"])
    for key in list(bn_state):
        if "memory" in key:
            bn_state.pop(key)
    decoder.load_state_dict(checkpoint["decoder"])
    bn.load_state_dict(bn_state)
    bn.eval()
    decoder.eval()
    return encoder, bn, decoder


def normal_reference_image(data_root: Path, category: str, image_size: int, device: torch.device) -> torch.Tensor:
    paths = image_paths(data_root / category / "train" / "good")
    if not paths:
        raise RuntimeError(f"No normal reference image for {category}")
    transform = transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
    )
    return transform(Image.open(paths[0]).convert("RGB")).unsqueeze(0).to(device)


def scores_from_feature_lists(fs_list: Sequence[torch.Tensor], ft_list: Sequence[torch.Tensor], out_size: int) -> np.ndarray:
    maps: torch.Tensor | None = None
    for fs, ft in zip(fs_list, ft_list, strict=True):
        amap = 1 - F.cosine_similarity(fs, ft)
        amap = F.interpolate(amap.unsqueeze(1), size=int(out_size), mode="bilinear", align_corners=True)
        maps = amap if maps is None else maps + amap
    if maps is None:
        return np.asarray([], dtype=np.float64)
    maps_np = maps[:, 0].detach().cpu().numpy()
    return np.asarray([float(np.max(gaussian_filter(item, sigma=4))) for item in maps_np], dtype=np.float64)


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "ap": float("nan"), "f1_max": float("nan"), "fpr95": float("nan")}
    auroc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision * recall / np.clip(precision + recall, 1e-12, None)
    f1_max = float(np.nanmax(f1))
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[valid[0]]) if valid.size else 1.0
    return {"auroc": auroc, "ap": ap, "f1_max": f1_max, "fpr95": fpr95}


@torch.no_grad()
def evaluate_split(
    category: str,
    split: str,
    stream_seed: int,
    encoder: torch.nn.Module,
    bn: torch.nn.Module,
    decoder: torch.nn.Module,
    normal_image: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    dataset = MVTecCEvalDataset(
        Path(args.data_root),
        category=category,
        split=split,
        image_size=int(args.image_size),
        severity=int(args.severity),
        stream_order=str(args.stream_order),
        seed=int(stream_seed),
    )
    loader = DataLoader(dataset, batch_size=int(args.eval_batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=True)
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for images, batch_labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        normal_batch = normal_image.expand(images.shape[0], -1, -1, -1).contiguous()
        features = encoder(images, normal_batch, "EFDM_test", lamda=float(args.lamda))
        outputs = decoder(bn(features))
        labels.append(batch_labels.cpu().numpy().astype(np.int64))
        scores.append(scores_from_feature_lists(features, outputs, out_size=int(args.image_size)))
    label_array = np.concatenate(labels)
    score_array = np.concatenate(scores)
    metrics = binary_metrics(label_array, score_array)
    return {
        "method": METHOD,
        "category": category,
        "split": split,
        "domain": split_domain(split),
        "seed": int(args.seed),
        "stream_seed": int(stream_seed),
        "stream_order": str(args.stream_order),
        "image_size": int(args.image_size),
        "n_train_normal": len(image_paths(Path(args.data_root) / category / "train" / "good")),
        "n_images": int(label_array.shape[0]),
        "n_normal": int(np.sum(label_array == 0)),
        "n_anomaly": int(np.sum(label_array == 1)),
        "image_auroc": metrics["auroc"],
        "image_ap": metrics["ap"],
        "image_f1_max": metrics["f1_max"],
        "image_fpr95": metrics["fpr95"],
        "selected_pseudo_normal_count": 0,
        "selected_pseudo_normal_purity": float("nan"),
        "optimizer_steps": 0,
        "tta_lr": float(args.lr),
        "tta_steps": 0,
        "tta_param_scope": "adshift_gnl_dinl",
        "tta_score_source": "efdm_reference",
        "tta_score_ema_decay": float("nan"),
        "lamda": float(args.lamda),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "method",
        "category",
        "split",
        "domain",
        "seed",
        "stream_seed",
        "stream_order",
        "image_size",
        "n_train_normal",
        "n_images",
        "n_normal",
        "n_anomaly",
        "image_auroc",
        "image_ap",
        "image_f1_max",
        "image_fpr95",
        "selected_pseudo_normal_count",
        "selected_pseudo_normal_purity",
        "optimizer_steps",
        "tta_lr",
        "tta_steps",
        "tta_param_scope",
        "tta_score_source",
        "tta_score_ema_decay",
        "lamda",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(output_root: Path) -> None:
    detail_path = output_root / "detailed.csv"
    if not detail_path.is_file():
        return
    rows = list(csv.DictReader(detail_path.open(newline="", encoding="utf-8")))
    summary: list[dict[str, Any]] = []
    for stream_seed in sorted({int(row["stream_seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["stream_seed"]) == stream_seed]
        clean = [float(row["image_auroc"]) * 100.0 for row in seed_rows if row["split"] == "test0"]
        shift = [float(row["image_auroc"]) * 100.0 for row in seed_rows if row["split"] != "test0"]
        summary.append(
            {
                "method": METHOD,
                "stream_seed": stream_seed,
                "clean_auroc": f"{np.mean(clean):.6f}" if clean else "nan",
                "shift_s5_auroc": f"{np.mean(shift):.6f}" if shift else "nan",
                "n_clean_rows": len(clean),
                "n_shift_rows": len(shift),
            },
        )
    summary_path = output_root / "summary_by_seed.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()) if summary else [])
        if summary:
            writer.writeheader()
            writer.writerows(summary)
    lines = ["# MVTec-C s5 GNL ADShift", "", "| method | stream_seed | clean AUROC | shift s5 AUROC | rows |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['stream_seed']} | {float(row['clean_auroc']):.2f} | "
            f"{float(row['shift_s5_auroc']):.2f} | {row['n_shift_rows']} |",
        )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    output_root = Path(args.output_root).resolve()
    detail_path = output_root / "detailed.csv"
    rows: list[dict[str, Any]] = []
    if detail_path.is_file():
        rows = list(csv.DictReader(detail_path.open(newline="", encoding="utf-8")))
    completed = {(row["category"], row["split"], int(row["stream_seed"])) for row in rows}
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    start = time.time()
    checkpoint_root = output_root / "checkpoints"

    for category in args.categories:
        checkpoint_path = checkpoint_root / f"mvtec_GNL_{category}_seed{int(args.seed)}_epoch{int(args.epochs) - 1}.pth"
        if not args.eval_only and not checkpoint_path.exists():
            train_category(category, args, device, checkpoint_path)
        encoder, bn, decoder = load_model(checkpoint_path, device)
        normal_image = normal_reference_image(Path(args.data_root), category, int(args.image_size), device)
        for stream_seed in args.stream_seeds:
            for split in args.splits:
                key = (category, split, int(stream_seed))
                if key in completed:
                    continue
                row = evaluate_split(category, split, int(stream_seed), encoder, bn, decoder, normal_image, args, device)
                rows.append(row)
                completed.add(key)
                write_csv(detail_path, rows)
                summarize(output_root)
                print(
                    f"[eval] seed={stream_seed} {category}/{split} auroc={row['image_auroc'] * 100:.2f}",
                    flush=True,
                )
        encoder.to("cpu")
        bn.to("cpu")
        decoder.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summarize(output_root)
    print(f"[done] elapsed={time.time() - start:.1f}s output={detail_path}", flush=True)


if __name__ == "__main__":
    main()

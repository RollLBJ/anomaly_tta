from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset import AugMixDatasetMVTec
from de_resnet import de_wide_resnet50_2
from resnet import wide_resnet50_2 as train_wide_resnet50_2
from resnet_TTA import wide_resnet50_2 as tta_wide_resnet50_2


CATEGORIES = ("MetalParts", "PCB", "PiledBags")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADShift GNL/DINL+ATTA on RobustAD.")
    parser.add_argument("--data-root", default="/home/qilab/byeongju_lee/anomaly_tta/data/robustad")
    parser.add_argument("--output-root", default="/home/qilab/byeongju_lee/anomaly_tta/outputs/robustad_gnl_adshift_seed0")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stream-seed", type=int, default=0)
    parser.add_argument("--stream-order", choices=("random", "sequential"), default="random")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lamda", type=float, default=0.5)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


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


def train_split_dir(data_root: Path, category: str) -> Path:
    matches = sorted((data_root / category).glob("*_data_dir_train"))
    if not matches:
        raise FileNotFoundError(f"Missing RobustAD train split for {category}")
    return matches[0]


def test_split_dirs(data_root: Path, category: str) -> list[Path]:
    return sorted((data_root / category).glob("*_data_dir_test*"), key=split_token)


def split_token(path: Path) -> str:
    name = path.name
    if "_test" in name:
        return "test" + name.rsplit("_test", 1)[1]
    if name.endswith("_train"):
        return "train"
    return name


class NormalPILDataset(Dataset):
    def __init__(self, split_dir: Path, image_size: int) -> None:
        self.paths = image_paths(split_dir / "normal")
        if not self.paths:
            raise RuntimeError(f"No normal training images found in {split_dir / 'normal'}")
        self.resize = transforms.Resize((int(image_size), int(image_size)))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.resize(image), 0


class RobustADEvalDataset(Dataset):
    def __init__(self, split_dir: Path, image_size: int, stream_order: str, seed: int) -> None:
        self.split_dir = split_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize((int(image_size), int(image_size))),
                transforms.ToTensor(),
                transforms.CenterCrop(int(image_size)),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
        )
        samples: list[tuple[Path, int]] = []
        samples.extend((path, 0) for path in image_paths(split_dir / "normal"))
        samples.extend((path, 1) for path in image_paths(split_dir / "anomaly"))
        if not samples:
            raise RuntimeError(f"No test images found in {split_dir}")
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
        return self.transform(image), torch.tensor(label, dtype=torch.long), str(path)


def loss_function(a: list[torch.Tensor], b: list[torch.Tensor]) -> torch.Tensor:
    cos_loss = torch.nn.CosineSimilarity()
    loss = torch.zeros((), device=a[0].device)
    for item in range(len(a)):
        loss = loss + torch.mean(
            1 - cos_loss(a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1)),
        )
    return loss


def loss_function_last(a: list[torch.Tensor], b: list[torch.Tensor]) -> torch.Tensor:
    cos_loss = torch.nn.CosineSimilarity()
    return torch.mean(1 - cos_loss(a[0].view(a[0].shape[0], -1), b[0].view(b[0].shape[0], -1)))


def train_category(category: str, args: argparse.Namespace, device: torch.device, checkpoint_path: Path) -> None:
    train_dir = train_split_dir(Path(args.data_root), category)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
    )
    train_data = AugMixDatasetMVTec(NormalPILDataset(train_dir, image_size=int(args.image_size)), preprocess)
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
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
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


def cal_anomaly_map(fs_list: list[torch.Tensor], ft_list: list[torch.Tensor], out_size: int) -> np.ndarray:
    anomaly_map = np.zeros([out_size, out_size])
    for index in range(len(ft_list)):
        a_map = 1 - F.cosine_similarity(fs_list[index], ft_list[index])
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
        anomaly_map += a_map[0, 0].detach().cpu().numpy()
    return anomaly_map


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    encoder, bn = tta_wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    encoder.eval()
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
    train_dir = train_split_dir(data_root, category)
    paths = image_paths(train_dir / "normal")
    if not paths:
        raise RuntimeError(f"No normal reference image for {category}")
    transform = transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
    )
    image = Image.open(paths[0]).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


@torch.no_grad()
def evaluate_split(
    category: str,
    split_dir: Path,
    encoder: torch.nn.Module,
    bn: torch.nn.Module,
    decoder: torch.nn.Module,
    normal_image: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    dataset = RobustADEvalDataset(
        split_dir,
        image_size=int(args.image_size),
        stream_order=str(args.stream_order),
        seed=int(args.stream_seed),
    )
    loader = DataLoader(dataset, batch_size=int(args.eval_batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=True)
    labels: list[int] = []
    scores: list[float] = []
    for images, batch_labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        for index in range(images.shape[0]):
            image = images[index : index + 1]
            inputs = encoder(image, normal_image, "EFDM_test", lamda=float(args.lamda))
            outputs = decoder(bn(inputs))
            anomaly_map = cal_anomaly_map(inputs, outputs, int(image.shape[-1]))
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            labels.append(int(batch_labels[index].item()))
            scores.append(float(np.max(anomaly_map)))
    auroc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
    return {
        "category": category,
        "split": split_token(split_dir),
        "domain": "source" if split_token(split_dir) == "test0" else "target",
        "method": "ADShift_GNL_DINL_ATTA",
        "seed": int(args.seed),
        "stream_seed": int(args.stream_seed),
        "stream_order": str(args.stream_order),
        "n_samples": len(labels),
        "n_normal": int(np.sum(np.asarray(labels) == 0)),
        "n_anomaly": int(np.sum(np.asarray(labels) == 1)),
        "auroc": auroc,
    }


def write_outputs(rows: list[dict[str, Any]], output_root: Path, elapsed_sec: float, args: argparse.Namespace) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    detail_path = output_root / "robustad_gnl_detailed.csv"
    summary_path = output_root / "robustad_gnl_summary.csv"
    report_path = output_root / "report.md"
    columns = [
        "category",
        "split",
        "domain",
        "method",
        "seed",
        "stream_seed",
        "stream_order",
        "n_samples",
        "n_normal",
        "n_anomaly",
        "auroc",
    ]
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary: list[dict[str, Any]] = []
    categories = sorted({str(row["category"]) for row in rows})
    for category in categories:
        cat_rows = [row for row in rows if row["category"] == category and row["domain"] == "target"]
        summary.append(
            {
                "category": category,
                "method": "ADShift_GNL_DINL_ATTA",
                "target_splits": len(cat_rows),
                "target_auroc_mean": float(np.mean([row["auroc"] for row in cat_rows])) if cat_rows else float("nan"),
            },
        )
    all_target = [row["auroc"] for row in rows if row["domain"] == "target"]
    summary.append(
        {
            "category": "OVERALL_TARGET",
            "method": "ADShift_GNL_DINL_ATTA",
            "target_splits": len(all_target),
            "target_auroc_mean": float(np.mean(all_target)) if all_target else float("nan"),
        },
    )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "method", "target_splits", "target_auroc_mean"])
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# RobustAD GNL ADShift",
        "",
        f"- Data root: `{args.data_root}`",
        f"- Seed: `{args.seed}`",
        f"- Stream seed/order: `{args.stream_seed}` / `{args.stream_order}`",
        f"- Image size: `{args.image_size}`",
        f"- Epochs: `{args.epochs}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Elapsed seconds: `{elapsed_sec:.1f}`",
        "",
        "| category | target splits | target AUROC mean |",
        "| --- | ---: | ---: |",
    ]
    for row in summary:
        lines.append(f"| {row['category']} | {row['target_splits']} | {row['target_auroc_mean']:.4f} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    output_root = Path(args.output_root)
    checkpoint_root = output_root / "checkpoints"
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    data_root = Path(args.data_root)
    start = time.time()
    rows: list[dict[str, Any]] = []
    for category in args.categories:
        checkpoint_path = checkpoint_root / f"robustad_GNL_{category}_seed{args.seed}_epoch{int(args.epochs) - 1}.pth"
        if not args.eval_only and not checkpoint_path.exists():
            train_category(category, args, device, checkpoint_path)
        encoder, bn, decoder = load_model(checkpoint_path, device)
        normal_image = normal_reference_image(data_root, category, int(args.image_size), device)
        for split_dir in test_split_dirs(data_root, category):
            row = evaluate_split(category, split_dir, encoder, bn, decoder, normal_image, args, device)
            rows.append(row)
            print(
                f"[eval] {category}/{row['split']} auroc={row['auroc'] * 100:.2f} "
                f"n={row['n_samples']}",
                flush=True,
            )
    write_outputs(rows, output_root, time.time() - start, args)


if __name__ == "__main__":
    main()

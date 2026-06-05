from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from imagecorruptions import corrupt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_robustad_svdd_boundary_compare as core  # noqa: E402


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
DETAIL_COLUMNS = (
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
    "feature_family",
    "boundary_model",
    "selected_pseudo_normal_count",
    "selected_pseudo_normal_purity",
    "pseudo_only_count",
    "pseudo_only_purity",
    "optimizer_steps",
    "active_label_count",
    "active_label_normal_count",
    "active_label_anomaly_count",
    "tta_lr",
    "tta_steps",
    "tta_param_scope",
    "tta_score_source",
    "tta_score_ema_decay",
    "negative_label_weight",
    "active_svm_confidence_threshold",
    "active_svm_query_side_mode",
    "active_svm_tail_pseudo_label_fraction",
    "active_svm_lower_tail_pseudo_normal_weight",
    "active_svm_upper_tail_pseudo_anomaly_weight",
    "active_memory_weight_mode",
    "active_memory_size_final",
    "active_memory_weight_mean",
    "active_memory_weight_min",
    "active_memory_weight_max",
    "active_memory_selected_weight_sum",
    "active_memory_weighted_purity",
    "active_label_ce_mode",
    "active_label_ce_targets",
    "active_label_ce_weight",
    "active_label_ce_update",
    "active_label_ce_pseudo_weight_mode",
    "active_label_ce_pseudo_weight_mean",
    "active_label_ce_pseudo_weight_min",
    "active_label_ce_pseudo_weight_max",
    "active_label_ce_steps",
    "checkpoint_path",
)


def build_transform(resize_size: int, crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((int(resize_size), int(resize_size)), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((int(crop_size), int(crop_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
    )


def image_paths(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(folder.rglob(pattern))
    return sorted(paths)


def split_domain(split: str) -> str:
    return "clean_test" if split == "test0" else split


def split_corruption(split: str) -> str | None:
    if split in {"train", "test0"}:
        return None
    for corruption in CORRUPTIONS:
        if split == f"{corruption}_s5":
            return corruption
    raise ValueError(f"Unsupported split: {split}")


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


class MVTecCImageDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        category: str,
        split: str,
        resize_size: int,
        crop_size: int,
        severity: int,
        label_filter: int | None = None,
        stream_order: str = "random",
        seed: int = 0,
    ) -> None:
        self.data_root = data_root
        self.category = category
        self.split = split
        self.severity = int(severity)
        self.transform = build_transform(resize_size=resize_size, crop_size=crop_size)
        category_root = data_root / category
        if split == "train":
            folder_specs = (("train/good", 0),)
        else:
            test_root = category_root / "test"
            folder_specs = tuple(
                (f"test/{folder.name}", 0 if folder.name == "good" else 1)
                for folder in sorted(test_root.iterdir())
                if folder.is_dir()
            )
        samples: list[tuple[Path, int]] = []
        for relative, label in folder_specs:
            if label_filter is not None and int(label_filter) != int(label):
                continue
            folder = category_root / relative
            samples.extend((path, int(label)) for path in image_paths(folder))
        if not samples:
            raise RuntimeError(f"No images for {category}/{split}")
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
        image = corrupt_image(image=image, split=self.split, path=path, severity=self.severity)
        return {
            "image": self.transform(image),
            "label": torch.tensor(int(label), dtype=torch.long),
            "path": str(path),
        }


def robust_zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return array
    center = float(np.median(array))
    scale = float(np.median(np.abs(array - center))) * 1.4826
    if not math.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(array))
    if not math.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    return (array - center) / scale


def entropy_from_maps(maps_np: np.ndarray) -> np.ndarray:
    flat = maps_np.reshape(maps_np.shape[0], -1).astype(np.float64, copy=False)
    mean = flat.mean(axis=1, keepdims=True)
    std = np.maximum(flat.std(axis=1, keepdims=True), 1e-6)
    prob = 1.0 / (1.0 + np.exp(-np.clip((flat - mean) / std, -12.0, 12.0)))
    ent = -(prob * np.log(np.clip(prob, 1e-6, 1.0)) + (1.0 - prob) * np.log(np.clip(1.0 - prob, 1e-6, 1.0)))
    return ent.mean(axis=1).astype(np.float64)


def median_score_query(scores: np.ndarray) -> int:
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = float(np.quantile(score_array, 0.50))
    return int(np.argsort(np.abs(score_array - target))[0])


def ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    core.update_ema_model(ema_model, model, decay=float(decay))


def load_source_model(category: str, checkpoint_roots: Sequence[Path], device: torch.device) -> tuple[core.AnomalibReverseDistillationModel, Path]:
    for root in checkpoint_roots:
        path = root / category / "anomalib_reverse_distillation.pt"
        if path.is_file():
            payload = torch.load(path, map_location=device, weights_only=False)
            model = core.AnomalibReverseDistillationModel().to(device)
            missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
            missing = [key for key in missing if not key.startswith("decoder_adapters.")]
            unexpected = [key for key in unexpected if not key.startswith("decoder_adapters.")]
            if missing or unexpected:
                raise RuntimeError(f"Checkpoint mismatch for {path}: missing={missing}, unexpected={unexpected}")
            model.eval()
            return model, path
    raise FileNotFoundError(f"No checkpoint found for {category} under {checkpoint_roots}")


def make_core_args(args: argparse.Namespace, stream_seed: int) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    ns.seed = 0
    ns.stream_seed = int(stream_seed)
    ns.tta_param_scope = str(args.tta_param_scope)
    ns.tta_distill_l2_weight = 0.0
    ns.tta_bn_anchor_weight = 0.0
    ns.feature_family = str(args.feature_family)
    ns.boundary_model = str(args.boundary_model)
    ns.tta_grad_clip = float(args.tta_grad_clip)
    return ns


def adapt_one_sample_signed(
    model: core.AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
    image: torch.Tensor,
    args: argparse.Namespace,
    loss_sign: float,
) -> tuple[int, float]:
    if int(args.tta_steps) <= 0:
        return 0, float("nan")
    last_loss = float("nan")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for _step in range(int(args.tta_steps)):
        core.set_tta_train_mode(model, args)
        optimizer.zero_grad(set_to_none=True)
        reconstruction_loss = model.reconstruction_loss(
            image,
            distill_l2_weight=float(args.tta_distill_l2_weight),
        )
        loss = float(loss_sign) * reconstruction_loss
        loss.backward()
        if float(args.tta_grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(args.tta_grad_clip))
        optimizer.step()
        last_loss = float(reconstruction_loss.detach().item())
    model.eval()
    return int(args.tta_steps), last_loss


def evaluate_label_tta(
    source_model: core.AnomalibReverseDistillationModel,
    loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    method: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model = copy.deepcopy(source_model).to(device).eval()
    score_model = copy.deepcopy(model).to(device).eval()
    trainable = core.configure_tta_parameters(model, args)
    optimizer = torch.optim.Adam(trainable, lr=float(args.tta_lr))
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    selected_total = 0
    selected_normal_total = 0
    pseudo_only_total = 0
    pseudo_only_normal_total = 0
    optimizer_steps = 0
    active_total = 0
    active_normal = 0
    active_anomaly = 0

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy().astype(np.int64)
        with torch.no_grad():
            maps_np = core.anomaly_maps_np(score_model, images, image_size=int(args.crop_size))
        scores = core.scores_from_maps(maps_np)
        if method in {"eatta", "eatta_neg"} and str(args.eatta_query_mode) == "feature_perturb_sensitivity":
            query = core.feature_perturbation_query(score_model, images, scores, args, batch_index=batch_index)
        else:
            query = median_score_query(scores)
        query_label = int(labels[query])
        active_total += 1
        active_normal += int(query_label == 0)
        active_anomaly += int(query_label == 1)

        positive_indices: list[int] = []
        negative_indices: list[int] = []
        pseudo_indices_for_stats: list[int] = []
        if query_label == 0:
            positive_indices.append(int(query))
        elif method in {"atta_neg", "eatta_neg"}:
            negative_indices.append(int(query))
        if method in {"eatta", "eatta_neg"}:
            entropy = entropy_from_maps(maps_np)
            generator = torch.Generator(device=images.device)
            generator.manual_seed(int(args.stream_seed) + 10_003 * int(batch_index))
            noisy = images + torch.randn(images.shape, generator=generator, device=images.device, dtype=images.dtype) * float(args.eatta_noise_std)
            with torch.no_grad():
                noisy_maps = core.anomaly_maps_np(score_model, noisy, image_size=int(args.crop_size))
            stability = np.abs(core.scores_from_maps(noisy_maps) - scores)
            score_thr = float(np.quantile(scores, float(args.eatta_pseudo_normal_score_q)))
            ent_thr = float(np.quantile(entropy, float(args.eatta_pseudo_normal_entropy_q)))
            stab_thr = float(np.quantile(stability, float(args.eatta_pseudo_normal_stability_q)))
            candidate_mask = (scores <= score_thr) & (entropy <= ent_thr) & (stability <= stab_thr)
            candidate_mask[int(query)] = False
            candidates = np.flatnonzero(candidate_mask)
            max_count = min(
                int(candidates.size),
                max(1, int(math.ceil(float(scores.size) * float(args.eatta_pseudo_normal_max_fraction)))),
            )
            if max_count > 0:
                rank = robust_zscore(scores) + robust_zscore(entropy) + robust_zscore(stability)
                selected_candidates = [int(index) for index in candidates[np.argsort(rank[candidates])[:max_count]]]
                positive_indices.extend(selected_candidates)
                pseudo_indices_for_stats.extend(selected_candidates)
        positive_indices = sorted(set(positive_indices))
        negative_indices = sorted(set(negative_indices) - set(positive_indices))
        pseudo_indices_for_stats = sorted(set(pseudo_indices_for_stats))
        if pseudo_indices_for_stats:
            pseudo_only_total += len(pseudo_indices_for_stats)
            pseudo_only_normal_total += int(np.sum(labels[np.asarray(pseudo_indices_for_stats, dtype=np.int64)] == 0))
        if positive_indices or negative_indices:
            selected_indices = positive_indices + negative_indices
            selected_total += len(selected_indices)
            selected_normal_total += int(np.sum(labels[np.asarray(selected_indices, dtype=np.int64)] == 0))
            for index in positive_indices:
                steps, _ = adapt_one_sample_signed(model, optimizer, images[index : index + 1], args, loss_sign=1.0)
                optimizer_steps += int(steps)
            for index in negative_indices:
                steps, _ = adapt_one_sample_signed(
                    model,
                    optimizer,
                    images[index : index + 1],
                    args,
                    loss_sign=-float(args.negative_label_weight),
                )
                optimizer_steps += int(steps)
            ema_update(score_model, model, decay=float(args.tta_score_ema_decay))
        with torch.no_grad():
            tta_maps = core.anomaly_maps_np(model, images, image_size=int(args.crop_size))
        labels_all.append(labels)
        scores_all.append(core.scores_from_maps(tta_maps))

    return np.concatenate(labels_all), np.concatenate(scores_all), {
        "selected_total": float(selected_total),
        "selected_normal_total": float(selected_normal_total),
        "pseudo_only_total": float(pseudo_only_total),
        "pseudo_only_normal_total": float(pseudo_only_normal_total),
        "optimizer_steps": float(optimizer_steps),
        "active_label_total": float(active_total),
        "active_label_normal_total": float(active_normal),
        "active_label_anomaly_total": float(active_anomaly),
    }


def evaluate_btta_negative_query(
    source_model: core.AnomalibReverseDistillationModel,
    loader: torch.utils.data.DataLoader,
    source_stats: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model = copy.deepcopy(source_model).to(device)
    score_model = copy.deepcopy(source_model).to(device) if str(args.tta_score_source) == "adapted_ema" else None
    trainable = core.configure_tta_parameters(model, args)
    optimizer = torch.optim.Adam(trainable, lr=float(args.tta_lr))
    active_features: list[np.ndarray] = []
    active_labels: list[int] = []
    active_weights: list[float] = []
    stream_tail_features: list[np.ndarray] = []
    stream_tail_scores: list[float] = []
    stream_tail_labels: list[int] = []
    stream_tail_selected_ids: set[int] = set()
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    selected_total = 0
    selected_normal_total = 0
    pseudo_only_total = 0
    pseudo_only_normal_total = 0
    optimizer_steps = 0
    active_label_total = 0
    active_label_normal_total = 0
    active_label_anomaly_total = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy().astype(np.int64)
        scorer = source_model
        if str(args.tta_score_source) == "adapted":
            scorer = model
        elif str(args.tta_score_source) == "adapted_ema" and score_model is not None:
            scorer = score_model
        with torch.no_grad():
            selection_maps = core.anomaly_maps_np(scorer, images, image_size=int(args.crop_size))
        selection_scores = core.scores_from_maps(selection_maps)
        selection_features = core.map_stat_features_from_maps(
            selection_maps,
            source_stats,
            feature_family=str(args.feature_family),
        )
        batch_global_ids = np.arange(
            len(stream_tail_scores),
            len(stream_tail_scores) + int(labels.shape[0]),
            dtype=np.int64,
        )
        for feature, score, label in zip(selection_features, selection_scores, labels, strict=True):
            stream_tail_features.append(feature.copy())
            stream_tail_scores.append(float(score))
            stream_tail_labels.append(int(label))

        fit_before = core.fit_active_boundary(active_features, active_labels, active_weights, str(args.boundary_model))
        if fit_before is None:
            query_index = int(np.argsort(selection_scores)[len(selection_scores) // 2])
        else:
            decisions_before = core.boundary_decision(fit_before, selection_features)
            query_index = core.active_query_index_from_decisions(
                decisions_before,
                mode=str(args.active_svm_query_side_mode),
                query_count=int(active_label_total),
            )
        query_label = int(labels[query_index])
        active_features.append(selection_features[query_index].copy())
        active_labels.append(query_label)
        active_weights.append(1.0)
        active_label_total += 1
        active_label_normal_total += int(query_label == 0)
        active_label_anomaly_total += int(query_label == 1)

        tail_ids, tail_pseudo = core.select_stream_tail_pseudo_labels(
            stream_tail_scores,
            fraction=float(args.active_svm_tail_pseudo_label_fraction),
            excluded_ids={int(batch_global_ids[query_index])},
            selected_ids=stream_tail_selected_ids,
        )
        tail_current_positions: list[int] = []
        for global_id, pseudo_label in zip(tail_ids, tail_pseudo, strict=True):
            stream_tail_selected_ids.add(int(global_id))
            active_features.append(stream_tail_features[int(global_id)].copy())
            active_labels.append(int(pseudo_label))
            active_weights.append(
                float(args.active_svm_lower_tail_pseudo_normal_weight)
                if int(pseudo_label) == 0
                else float(args.active_svm_upper_tail_pseudo_anomaly_weight)
            )
            current = np.flatnonzero(batch_global_ids == int(global_id))
            if current.size:
                tail_current_positions.append(int(current[0]))

        fit_after = core.fit_active_boundary(active_features, active_labels, active_weights, str(args.boundary_model))
        positive_indices: list[int] = []
        negative_indices: list[int] = []
        if query_label == 0:
            positive_indices.append(int(query_index))
        else:
            negative_indices.append(int(query_index))
        if fit_after is not None:
            decisions = core.boundary_decision(fit_after, selection_features)
            excluded = {int(query_index), *tail_current_positions}
            pseudo_mask = decisions <= -float(args.active_svm_confidence_threshold)
            for index, selected in enumerate(pseudo_mask.tolist()):
                if selected and int(index) not in excluded:
                    positive_indices.append(int(index))
        pseudo_only_indices = sorted(set(positive_indices) - {int(query_index)})
        if pseudo_only_indices:
            pseudo_only_total += len(pseudo_only_indices)
            pseudo_only_normal_total += int(np.sum(labels[np.asarray(pseudo_only_indices, dtype=np.int64)] == 0))
        positive_indices = sorted(set(positive_indices))
        negative_indices = sorted(set(negative_indices) - set(positive_indices))
        if positive_indices or negative_indices:
            selected_indices = positive_indices + negative_indices
            selected_total += len(selected_indices)
            selected_normal_total += int(np.sum(labels[np.asarray(selected_indices, dtype=np.int64)] == 0))
            for index in positive_indices:
                steps, _ = adapt_one_sample_signed(model, optimizer, images[index : index + 1], args, loss_sign=1.0)
                optimizer_steps += int(steps)
            for index in negative_indices:
                steps, _ = adapt_one_sample_signed(
                    model,
                    optimizer,
                    images[index : index + 1],
                    args,
                    loss_sign=-float(args.negative_label_weight),
                )
                optimizer_steps += int(steps)
        if score_model is not None and (positive_indices or negative_indices):
            ema_update(score_model, model, decay=float(args.tta_score_ema_decay))
        with torch.no_grad():
            tta_maps = core.anomaly_maps_np(model, images, image_size=int(args.crop_size))
        labels_all.append(labels)
        scores_all.append(core.scores_from_maps(tta_maps))

    return np.concatenate(labels_all), np.concatenate(scores_all), {
        "selected_total": float(selected_total),
        "selected_normal_total": float(selected_normal_total),
        "pseudo_only_total": float(pseudo_only_total),
        "pseudo_only_normal_total": float(pseudo_only_normal_total),
        "optimizer_steps": float(optimizer_steps),
        "active_label_total": float(active_label_total),
        "active_label_normal_total": float(active_label_normal_total),
        "active_label_anomaly_total": float(active_label_anomaly_total),
    }


def row_from_metrics(
    method: str,
    category: str,
    split: str,
    stream_seed: int,
    args: argparse.Namespace,
    labels: np.ndarray,
    metrics: dict[str, float],
    checkpoint: Path,
    n_train_normal: int,
    stats: dict[str, float] | None = None,
) -> dict[str, Any]:
    stats = stats or {}
    selected = int(stats.get("selected_total", 0))
    selected_normal = int(stats.get("selected_normal_total", 0))
    pseudo_only = int(stats.get("pseudo_only_total", 0))
    pseudo_only_normal = int(stats.get("pseudo_only_normal_total", 0))
    return {
        "method": method,
        "category": category,
        "split": split,
        "domain": split_domain(split),
        "seed": 0,
        "stream_seed": int(stream_seed),
        "stream_order": str(args.stream_order),
        "image_size": int(args.crop_size),
        "n_train_normal": int(n_train_normal),
        "n_images": int(labels.shape[0]),
        "n_normal": int(np.sum(labels == 0)),
        "n_anomaly": int(np.sum(labels == 1)),
        "image_auroc": metrics["auroc"],
        "image_ap": metrics["ap"],
        "image_f1_max": metrics["f1_max"],
        "image_fpr95": metrics["fpr95"],
        "feature_family": str(args.feature_family),
        "boundary_model": str(args.boundary_model),
        "selected_pseudo_normal_count": selected,
        "selected_pseudo_normal_purity": float(selected_normal / selected) if selected > 0 else float("nan"),
        "pseudo_only_count": pseudo_only,
        "pseudo_only_purity": float(pseudo_only_normal / pseudo_only) if pseudo_only > 0 else float("nan"),
        "optimizer_steps": int(stats.get("optimizer_steps", 0)),
        "active_label_count": int(stats.get("active_label_total", 0)),
        "active_label_normal_count": int(stats.get("active_label_normal_total", 0)),
        "active_label_anomaly_count": int(stats.get("active_label_anomaly_total", 0)),
        "tta_lr": float(args.tta_lr),
        "tta_steps": int(args.tta_steps),
        "tta_param_scope": str(args.tta_param_scope),
        "tta_score_source": str(args.tta_score_source),
        "tta_score_ema_decay": float(args.tta_score_ema_decay),
        "negative_label_weight": float(args.negative_label_weight),
        "active_svm_confidence_threshold": float(args.active_svm_confidence_threshold),
        "active_svm_query_side_mode": str(args.active_svm_query_side_mode),
        "active_svm_tail_pseudo_label_fraction": float(args.active_svm_tail_pseudo_label_fraction),
        "active_svm_lower_tail_pseudo_normal_weight": float(args.active_svm_lower_tail_pseudo_normal_weight),
        "active_svm_upper_tail_pseudo_anomaly_weight": float(args.active_svm_upper_tail_pseudo_anomaly_weight),
        "active_memory_weight_mode": str(args.active_memory_weight_mode),
        "active_memory_size_final": int(stats.get("active_memory_size_final", 0)),
        "active_memory_weight_mean": float(stats.get("active_memory_weight_mean", float("nan"))),
        "active_memory_weight_min": float(stats.get("active_memory_weight_min", float("nan"))),
        "active_memory_weight_max": float(stats.get("active_memory_weight_max", float("nan"))),
        "active_memory_selected_weight_sum": float(stats.get("active_memory_selected_weight_sum", 0.0)),
        "active_memory_weighted_purity": (
            float(stats.get("active_memory_selected_normal_weight_sum", 0.0))
            / float(stats.get("active_memory_selected_weight_sum", 0.0))
            if float(stats.get("active_memory_selected_weight_sum", 0.0)) > 0.0
            else float("nan")
        ),
        "active_label_ce_mode": str(args.active_label_ce_mode),
        "active_label_ce_targets": str(args.active_label_ce_targets),
        "active_label_ce_weight": float(args.active_label_ce_weight),
        "active_label_ce_update": str(args.active_label_ce_update),
        "active_label_ce_pseudo_weight_mode": str(args.active_label_ce_pseudo_weight_mode),
        "active_label_ce_pseudo_weight_mean": float(stats.get("active_label_ce_pseudo_weight_mean", float("nan"))),
        "active_label_ce_pseudo_weight_min": float(stats.get("active_label_ce_pseudo_weight_min", float("nan"))),
        "active_label_ce_pseudo_weight_max": float(stats.get("active_label_ce_pseudo_weight_max", float("nan"))),
        "active_label_ce_steps": int(stats.get("active_label_ce_steps", 0)),
        "checkpoint_path": str(checkpoint),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(output_root: Path) -> None:
    detail_path = output_root / "detailed.csv"
    if not detail_path.is_file():
        return
    with detail_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary: list[dict[str, Any]] = []
    for method in sorted({row["method"] for row in rows}):
        for stream_seed in sorted({int(row["stream_seed"]) for row in rows if row["method"] == method}):
            method_rows = [row for row in rows if row["method"] == method and int(row["stream_seed"]) == stream_seed]
            clean = [float(row["image_auroc"]) * 100.0 for row in method_rows if row["split"] == "test0"]
            shift = [float(row["image_auroc"]) * 100.0 for row in method_rows if row["split"] != "test0"]
            selected = [float(row["selected_pseudo_normal_count"]) for row in method_rows if row["split"] != "test0"]
            purity = [
                float(row["selected_pseudo_normal_purity"]) * 100.0
                for row in method_rows
                if row["split"] != "test0" and row["selected_pseudo_normal_purity"].lower() != "nan"
            ]
            summary.append(
                {
                    "method": method,
                    "stream_seed": stream_seed,
                    "clean_auroc": f"{np.mean(clean):.6f}" if clean else "nan",
                    "shift_s5_auroc": f"{np.mean(shift):.6f}" if shift else "nan",
                    "n_clean_rows": len(clean),
                    "n_shift_rows": len(shift),
                    "selected_mean_shift": f"{np.mean(selected):.6f}" if selected else "nan",
                    "purity_mean_shift": f"{np.mean(purity):.6f}" if purity else "nan",
                },
            )
    summary_path = output_root / "summary_by_seed.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()) if summary else [])
        if summary:
            writer.writeheader()
            writer.writerows(summary)
    report = ["# MVTec-C s5 Seed Sweep", ""]
    report.append("| method | stream_seed | clean AUROC | shift s5 AUROC | rows | selected | purity |")
    report.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary:
        report.append(
            f"| {row['method']} | {row['stream_seed']} | {float(row['clean_auroc']):.2f} | "
            f"{float(row['shift_s5_auroc']):.2f} | {row['n_shift_rows']} | "
            f"{float(row['selected_mean_shift']):.1f} | "
            f"{float(row['purity_mean_shift']):.1f} |"
        )
    (output_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MVTec-C s5 source/ATTA/EATTA/BTTA seed sweep.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "mvtec_ad"))
    parser.add_argument(
        "--checkpoint-roots",
        nargs="+",
        default=[
            str(ROOT / "outputs/mvtec_c_s5_btta5d_seed0_randomstream_256e20_bs8_gpu0/rd4ad_checkpoints"),
            str(ROOT / "outputs/mvtec_c_s5_btta5d_seed0_randomstream_256e20_bs8_gpu1/rd4ad_checkpoints"),
        ],
    )
    parser.add_argument("--output-root", default=str(ROOT / "outputs/mvtec_c_s5_seed012_base_atta_eatta_btta5d"))
    parser.add_argument("--categories", nargs="+", default=list(MVTEC_CATEGORIES))
    parser.add_argument("--splits", nargs="+", default=["test0", *(f"{name}_s5" for name in CORRUPTIONS)])
    parser.add_argument("--methods", nargs="+", default=["source", "atta", "eatta", "btta_5d_sptail1"])
    parser.add_argument("--stream-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--stream-order", choices=("random", "sequential"), default="random")
    parser.add_argument("--tta-lr", type=float, default=0.02)
    parser.add_argument("--tta-steps", type=int, default=1)
    parser.add_argument("--tta-grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--tta-param-scope",
        choices=(
            "bn_only",
            "bottleneck_decoder_bn_only",
            "bottleneck_bn_only",
            "decoder_bn_only",
            "bottleneck_full",
            "decoder_full",
            "view_adapter_decoder",
        ),
        default="bn_only",
    )
    parser.add_argument("--tta-score-source", choices=("adapted_ema",), default="adapted_ema")
    parser.add_argument("--tta-score-ema-decay", type=float, default=0.95)
    parser.add_argument(
        "--feature-family",
        choices=(
            "base5",
            "score1d",
            "multiscale",
            "spatial",
            "frequency",
            "multiscale_frequency",
            "multiscale_frequency_nosource",
            "msfreq_no_low",
            "compact_msfreq",
            "edge_msfreq",
            "all",
        ),
        default="base5",
    )
    parser.add_argument("--boundary-model", choices=("linear_svm", "svdd"), default="linear_svm")
    parser.add_argument("--active-svm-source-pixel-q", type=float, default=0.99)
    parser.add_argument("--active-svm-confidence-threshold", type=float, default=0.10)
    parser.add_argument(
        "--active-svm-query-side-mode",
        choices=("boundary_nearest", "below_nearest", "above_nearest", "alternate_nearest"),
        default="boundary_nearest",
    )
    parser.add_argument("--active-svm-tail-pseudo-label-fraction", type=float, default=0.01)
    parser.add_argument("--active-svm-lower-tail-pseudo-normal-weight", type=float, default=0.5)
    parser.add_argument("--active-svm-upper-tail-pseudo-anomaly-weight", type=float, default=0.2)
    parser.add_argument("--active-memory-weight-mode", choices=("none", "normal_nearest"), default="none")
    parser.add_argument("--active-memory-max-size", type=int, default=64)
    parser.add_argument("--active-memory-min-size", type=int, default=3)
    parser.add_argument("--active-memory-distance-scale", type=float, default=1.0)
    parser.add_argument("--active-memory-weight-min", type=float, default=0.25)
    parser.add_argument("--active-memory-weight-max", type=float, default=1.5)
    parser.add_argument("--active-label-ce-mode", choices=("none", "anomaly_only", "all"), default="none")
    parser.add_argument("--active-label-ce-targets", choices=("active", "active_pseudo"), default="active")
    parser.add_argument("--active-label-ce-weight", type=float, default=1.0)
    parser.add_argument("--active-label-ce-update", choices=("separate", "joint"), default="separate")
    parser.add_argument("--active-label-ce-pseudo-weight-mode", choices=("none", "svm_margin"), default="none")
    parser.add_argument("--eatta-noise-std", type=float, default=0.01)
    parser.add_argument(
        "--eatta-query-mode",
        choices=("median_score", "feature_perturb_sensitivity"),
        default="median_score",
    )
    parser.add_argument("--eatta-feature-perturb-std", type=float, default=0.01)
    parser.add_argument("--eatta-pseudo-normal-score-q", type=float, default=0.25)
    parser.add_argument("--eatta-pseudo-normal-entropy-q", type=float, default=0.5)
    parser.add_argument("--eatta-pseudo-normal-stability-q", type=float, default=0.5)
    parser.add_argument("--eatta-pseudo-normal-max-fraction", type=float, default=0.5)
    parser.add_argument("--paper-oracle-num", type=int, default=1)
    parser.add_argument("--paper-score-center-q", type=float, default=0.99)
    parser.add_argument("--paper-logit-scale-mult", type=float, default=1.0)
    parser.add_argument("--paper-memory-size", type=int, default=512)
    parser.add_argument("--paper-replay-batch-size", type=int, default=4)
    parser.add_argument("--atta-entropy-high-threshold", type=float, default=0.01)
    parser.add_argument("--atta-pseudo-normal-entropy-q", type=float, default=0.5)
    parser.add_argument("--atta-pseudo-normal-max-fraction", type=float, default=0.5)
    parser.add_argument("--atta-cluster-increase", type=int, default=1)
    parser.add_argument("--atta-cluster-budget", type=int, default=300)
    parser.add_argument("--eatta-entropy-margin", type=float, default=0.2772588722239781)
    parser.add_argument("--eatta-class-history", type=int, default=1)
    parser.add_argument("--eatta-gnd-momentum", type=float, default=0.8)
    parser.add_argument("--negative-label-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.stream_order != "random":
        print("[warn] project default is random stream; non-random stream requested explicitly", file=sys.stderr)
    if args.eatta_feature_perturb_std < 0.0:
        parser.error("--eatta-feature-perturb-std must be >= 0")
    if args.active_memory_max_size < 1:
        parser.error("--active-memory-max-size must be >= 1")
    if args.active_memory_min_size < 1:
        parser.error("--active-memory-min-size must be >= 1")
    if args.active_memory_distance_scale <= 0.0:
        parser.error("--active-memory-distance-scale must be > 0")
    if args.active_memory_weight_min <= 0.0:
        parser.error("--active-memory-weight-min must be > 0")
    if args.active_memory_weight_max < args.active_memory_weight_min:
        parser.error("--active-memory-weight-max must be >= --active-memory-weight-min")
    if args.active_label_ce_weight < 0.0:
        parser.error("--active-label-ce-weight must be >= 0")
    if args.paper_oracle_num < 1:
        parser.error("--paper-oracle-num must be >= 1")
    if args.paper_logit_scale_mult <= 0.0:
        parser.error("--paper-logit-scale-mult must be > 0")
    if args.paper_memory_size < 1:
        parser.error("--paper-memory-size must be >= 1")
    if args.paper_replay_batch_size < 1:
        parser.error("--paper-replay-batch-size must be >= 1")
    if args.atta_cluster_increase < 1:
        parser.error("--atta-cluster-increase must be >= 1")
    if args.atta_cluster_budget < 1:
        parser.error("--atta-cluster-budget must be >= 1")
    if args.eatta_entropy_margin < 0.0:
        parser.error("--eatta-entropy-margin must be >= 0")
    if args.eatta_class_history < 1:
        parser.error("--eatta-class-history must be >= 1")
    if not 0.0 <= float(args.eatta_gnd_momentum) < 1.0:
        parser.error("--eatta-gnd-momentum must be in [0, 1)")
    for name in (
        "eatta_pseudo_normal_score_q",
        "eatta_pseudo_normal_entropy_q",
        "eatta_pseudo_normal_stability_q",
        "eatta_pseudo_normal_max_fraction",
        "paper_score_center_q",
        "atta_pseudo_normal_entropy_q",
        "atta_pseudo_normal_max_fraction",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    detail_path = output_root / "detailed.csv"
    rows: list[dict[str, Any]] = []
    if detail_path.is_file():
        with detail_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    completed = {(row["method"], row["category"], row["split"], int(row["stream_seed"])) for row in rows}
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    checkpoint_roots = [Path(path).resolve() for path in args.checkpoint_roots]
    data_root = Path(args.data_root).resolve()
    start = time.time()
    for stream_seed in args.stream_seeds:
        core_args = make_core_args(args, stream_seed=stream_seed)
        paper_faithful_methods = {"atta_paper_faithful", "eatta_paper_faithful"}
        for category in args.categories:
            train_dataset = MVTecCImageDataset(
                data_root=data_root,
                category=category,
                split="train",
                resize_size=int(args.resize_size),
                crop_size=int(args.crop_size),
                severity=int(args.severity),
                label_filter=0,
                stream_order="sequential",
                seed=0,
            )
            source_model, checkpoint = load_source_model(category, checkpoint_roots=checkpoint_roots, device=device)
            source_stats = (
                core.compute_source_stats(source_model, train_dataset, core_args, device)
                if core.feature_family_needs_source_stats(str(args.feature_family))
                or bool(paper_faithful_methods.intersection(set(str(method) for method in args.methods)))
                else {}
            )
            for split in args.splits:
                dataset = MVTecCImageDataset(
                    data_root=data_root,
                    category=category,
                    split=split,
                    resize_size=int(args.resize_size),
                    crop_size=int(args.crop_size),
                    severity=int(args.severity),
                    stream_order=str(args.stream_order),
                    seed=int(stream_seed),
                )
                loader = core.make_loader(dataset, batch_size=int(args.batch_size), num_workers=int(args.num_workers), shuffle=False)
                source_labels: np.ndarray | None = None
                source_scores: np.ndarray | None = None
                for method in args.methods:
                    normalized_method = "source" if method == "base" else method
                    key = (method, category, split, int(stream_seed))
                    if key in completed:
                        continue
                    if normalized_method == "source":
                        if source_labels is None or source_scores is None:
                            source_labels, source_scores = core.evaluate_source(source_model, loader, core_args, device)
                        labels, scores, stats = source_labels, source_scores, {}
                    elif normalized_method in {"atta", "atta_neg", "eatta", "eatta_neg"}:
                        labels, scores, stats = evaluate_label_tta(source_model, loader, core_args, device, method=normalized_method)
                    elif normalized_method in {
                        "atta_paper",
                        "eatta_paper",
                        "atta_paper_faithful",
                        "eatta_paper_faithful",
                        "atta_paper_hybrid",
                        "eatta_paper_hybrid",
                    }:
                        labels, scores, stats = core.evaluate_label_tta(
                            source_model,
                            loader,
                            core_args,
                            device,
                            method=normalized_method,
                            source_stats=source_stats,
                        )
                    elif normalized_method in {"btta_5d_sptail1", "btta_active_boundary"}:
                        labels, scores, stats = core.evaluate_tta(source_model, loader, source_stats, core_args, device)
                    elif normalized_method == "btta_5d_sptail1_neg":
                        labels, scores, stats = evaluate_btta_negative_query(
                            source_model,
                            loader,
                            source_stats,
                            core_args,
                            device,
                        )
                    else:
                        raise ValueError(f"Unsupported method: {method}")
                    metrics = core.binary_metrics(labels=labels, scores=scores)
                    rows.append(
                        row_from_metrics(
                            method=method,
                            category=category,
                            split=split,
                            stream_seed=int(stream_seed),
                            args=core_args,
                            labels=labels,
                            metrics=metrics,
                            checkpoint=checkpoint,
                            n_train_normal=len(train_dataset),
                            stats=stats,
                        ),
                    )
                    completed.add(key)
                    write_csv(detail_path, rows)
                    summarize(output_root)
                    print(
                        f"[{method}] seed={stream_seed} {category}/{split} auroc={metrics['auroc'] * 100:.2f} "
                        f"selected={int(stats.get('selected_total', 0))}",
                        flush=True,
                    )
            source_model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summarize(output_root)
    print(f"[done] elapsed={time.time() - start:.1f}s output={detail_path}", flush=True)


if __name__ == "__main__":
    main()

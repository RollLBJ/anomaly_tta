from __future__ import annotations

import argparse
import copy
import csv
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as connected_components
from scipy.fft import dctn
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.resnet import BasicBlock, Bottleneck


DETAIL_COLUMNS = (
    "category",
    "split",
    "domain",
    "method",
    "boundary_model",
    "feature_family",
    "feature_dim",
    "stream_order",
    "seed",
    "stream_seed",
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
    "pseudo_only_count",
    "pseudo_only_purity",
    "optimizer_steps",
    "active_label_count",
    "active_label_normal_count",
    "active_label_anomaly_count",
    "active_tail_pseudo_label_count",
    "active_tail_pseudo_label_normal_count",
    "active_tail_pseudo_label_anomaly_count",
    "active_tail_pseudo_label_accuracy",
    "active_svm_confidence_threshold",
    "active_svm_query_side_mode",
    "active_svm_boundary_ema_decay",
    "active_svm_tail_start_mode",
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
    "active_label_ce_joint_scale_mode",
    "active_label_ce_pseudo_weight_mode",
    "active_label_ce_pseudo_weight_mean",
    "active_label_ce_pseudo_weight_min",
    "active_label_ce_pseudo_weight_max",
    "active_label_ce_steps",
    "active_extra_loss_mode",
    "active_extra_loss_weight",
    "tta_lr",
    "tta_steps",
    "tta_param_scope",
    "tta_score_source",
    "tta_score_ema_decay",
    "tta_distill_l2_weight",
    "tta_bn_anchor_weight",
    "paper_oracle_num",
    "paper_score_center_q",
    "paper_logit_scale_mult",
    "paper_logit_center",
    "paper_logit_scale",
    "paper_memory_size",
    "paper_replay_batch_size",
    "paper_memory_replay_mode",
    "atta_entropy_high_threshold",
    "atta_pseudo_normal_entropy_q",
    "atta_pseudo_normal_max_fraction",
    "atta_cluster_increase",
    "atta_cluster_budget",
    "eatta_pseudo_normal_entropy_q",
    "eatta_pseudo_normal_max_fraction",
    "eatta_entropy_margin",
    "eatta_class_history",
    "eatta_gnd_momentum",
    "eatta_feature_perturb_std",
    "checkpoint_path",
)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def image_paths(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(folder.rglob(pattern))
    return sorted(paths)


def split_prefix(category: str) -> str:
    if category == "MetalParts":
        return "metal_parts"
    if category == "PiledBags":
        return "piled_bags"
    return category.lower()


def train_split_dir(data_root: Path, category: str) -> Path:
    path = data_root / category / f"{split_prefix(category)}_data_dir_train"
    if not path.exists():
        raise FileNotFoundError(f"Missing RobustAD train split: {path}")
    return path


def test_split_dirs(data_root: Path, category: str, splits: Sequence[str]) -> list[Path]:
    root = data_root / category
    if splits:
        paths = [root / f"{split_prefix(category)}_data_dir_{split}" for split in splits]
    else:
        paths = sorted(root.glob(f"{split_prefix(category)}_data_dir_test*"))
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing RobustAD test splits: {missing}")
    return paths


def split_token(path: Path, category: str) -> str:
    prefix = f"{split_prefix(category)}_data_dir_"
    return path.name[len(prefix) :] if path.name.startswith(prefix) else path.name


def domain_name(split: str) -> str:
    return "source_domain" if split == "test0" else split


def build_transform(resize_size: int, crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((int(resize_size), int(resize_size)), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((int(crop_size), int(crop_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
    )


class RobustADImageDataset(Dataset):
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
        self.split_dir = split_dir
        self.category = category
        self.transform = build_transform(resize_size=resize_size, crop_size=crop_size)
        samples: list[tuple[Path, int]] = []
        for folder_name, label in (("normal", 0), ("anomaly", 1)):
            if label_filter is not None and int(label_filter) != int(label):
                continue
            folder = split_dir / folder_name
            if folder.exists():
                samples.extend((path, int(label)) for path in image_paths(folder))
        if not samples:
            raise RuntimeError(f"No RobustAD images found in {split_dir}")
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


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class TimmFeatureExtractor(nn.Module):
    def __init__(self, backbone: str, layers: Sequence[str], pre_trained: bool = False) -> None:
        super().__init__()
        self.layers = list(layers)
        probe = timm.create_model(backbone, pretrained=False, features_only=True, exportable=True)
        layer_names = [info["module"] for info in probe.feature_info.info]
        indices = [layer_names.index(layer) for layer in self.layers]
        self.feature_extractor = timm.create_model(
            backbone,
            pretrained=pre_trained,
            pretrained_cfg=None,
            features_only=True,
            exportable=True,
            out_indices=indices,
        )
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(inputs)
        if isinstance(features, dict):
            return [features[layer] for layer in self.layers]
        return list(features)


class OCBE(nn.Module):
    def __init__(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        layers: int,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)
        self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)
        self.bn1 = norm_layer(128 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn2 = norm_layer(256 * block.expansion)
        self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn3 = norm_layer(256 * block.expansion)
        self.conv4 = conv1x1(256 * block.expansion * 4, 512 * block.expansion, 1)
        self.bn4 = norm_layer(512 * block.expansion)

    def _make_layer(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [
            block(
                self.inplanes * 3,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            ),
        ]
        self.inplanes = planes * block.expansion
        layers.extend(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
            )
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        feature0 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(features[0]))))))
        feature1 = self.relu(self.bn3(self.conv3(features[1])))
        feature_cat = torch.cat([feature0, feature1, features[2]], dim=1)
        return self.bn_layer(feature_cat).contiguous()


class DecoderBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = (
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=stride, groups=groups, bias=False, dilation=dilation)
            if stride == 2
            else conv3x3(width, width, stride, groups, dilation)
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        identity = batch
        out = self.relu(self.bn1(self.conv1(batch)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.upsample is not None:
            identity = self.upsample(batch)
        return self.relu(out + identity)


class ResNetDecoder(nn.Module):
    def __init__(
        self,
        block: type[DecoderBottleneck],
        layers: list[int],
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 512 * block.expansion
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

    def _make_layer(self, block: type[DecoderBottleneck], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=2,
                    stride=stride,
                    groups=self.groups,
                    bias=False,
                    dilation=self.dilation,
                ),
                norm_layer(planes * block.expansion),
            )
        layers = [
            block(self.inplanes, planes, stride, upsample, self.groups, self.base_width, previous_dilation, norm_layer),
        ]
        self.inplanes = planes * block.expansion
        layers.extend(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
            )
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor) -> list[torch.Tensor]:
        feature_a = self.layer1(batch)
        feature_b = self.layer2(feature_a)
        feature_c = self.layer3(feature_b)
        return [feature_c, feature_b, feature_a]


class ResidualFeatureAdapter(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(int(channels), int(channels), kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.proj(feature)


class AnomalibReverseDistillationModel(nn.Module):
    def __init__(self, backbone: str = "wide_resnet50_2", layers: Sequence[str] = ("layer1", "layer2", "layer3")) -> None:
        super().__init__()
        self.encoder = TimmFeatureExtractor(backbone=backbone, layers=layers, pre_trained=False)
        self.bottleneck = OCBE(Bottleneck, 3)
        self.decoder = ResNetDecoder(DecoderBottleneck, [3, 4, 6, 3], width_per_group=128)
        self.decoder_adapters = nn.ModuleList(
            [
                ResidualFeatureAdapter(256),
                ResidualFeatureAdapter(512),
                ResidualFeatureAdapter(1024),
            ],
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.bottleneck.parameters()) + list(self.decoder.parameters())

    def forward_features(self, images: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        encoder_features = self.encoder(images)
        decoder_features = self.decoder(self.bottleneck(encoder_features))
        decoder_features = [adapter(feature) for adapter, feature in zip(self.decoder_adapters, decoder_features, strict=True)]
        return encoder_features, decoder_features

    def reconstruction_loss(self, images: torch.Tensor, distill_l2_weight: float = 0.0) -> torch.Tensor:
        encoder_features, decoder_features = self.forward_features(images)
        cos_loss = nn.CosineSimilarity(dim=1)
        loss_sum = images.new_tensor(0.0)
        for encoder_feature, decoder_feature in zip(encoder_features, decoder_features, strict=True):
            layer_loss = torch.mean(
                1.0
                - cos_loss(
                    encoder_feature.view(encoder_feature.shape[0], -1),
                    decoder_feature.view(decoder_feature.shape[0], -1),
                ),
            )
            if float(distill_l2_weight) > 0.0:
                encoder_flat = F.normalize(encoder_feature.view(encoder_feature.shape[0], -1), dim=1)
                decoder_flat = F.normalize(decoder_feature.view(decoder_feature.shape[0], -1), dim=1)
                layer_loss = layer_loss + float(distill_l2_weight) * F.mse_loss(decoder_flat, encoder_flat)
            loss_sum = loss_sum + layer_loss
        return loss_sum

    def anomaly_maps(self, images: torch.Tensor, image_size: int) -> torch.Tensor:
        encoder_features, decoder_features = self.forward_features(images)
        maps = []
        for encoder_feature, decoder_feature in zip(encoder_features, decoder_features, strict=True):
            layer_map = 1.0 - F.cosine_similarity(encoder_feature, decoder_feature, dim=1).unsqueeze(1)
            maps.append(F.interpolate(layer_map, size=(int(image_size), int(image_size)), mode="bilinear", align_corners=False))
        return torch.stack(maps, dim=0).sum(dim=0)


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle, num_workers=int(num_workers), pin_memory=True)


def load_model(checkpoint_root: Path, category: str, device: torch.device) -> tuple[AnomalibReverseDistillationModel, Path]:
    path = checkpoint_root / safe_name(category) / "anomalib_reverse_distillation.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    model = AnomalibReverseDistillationModel().to(device)
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    unexpected = [key for key in unexpected if not key.startswith("decoder_adapters.")]
    missing = [key for key in missing if not key.startswith("decoder_adapters.")]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch for {path}: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, path


def smooth_maps(maps_np: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    smoothed = np.empty_like(maps_np, dtype=np.float32)
    for index in range(maps_np.shape[0]):
        smoothed[index, 0] = gaussian_filter(maps_np[index, 0], sigma=sigma)
    return smoothed


@torch.no_grad()
def anomaly_maps_np(model: AnomalibReverseDistillationModel, images: torch.Tensor, image_size: int) -> np.ndarray:
    maps = model.anomaly_maps(images, image_size=image_size)
    return smooth_maps(maps.detach().cpu().numpy(), sigma=4.0)


@torch.no_grad()
def anomaly_maps_with_layer_maps_np(
    model: AnomalibReverseDistillationModel,
    images: torch.Tensor,
    image_size: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    encoder_features, decoder_features = model.forward_features(images)
    layer_maps = []
    for encoder_feature, decoder_feature in zip(encoder_features, decoder_features, strict=True):
        layer_map = 1.0 - F.cosine_similarity(encoder_feature, decoder_feature, dim=1).unsqueeze(1)
        layer_map = F.interpolate(
            layer_map,
            size=(int(image_size), int(image_size)),
            mode="bilinear",
            align_corners=False,
        )
        layer_maps.append(layer_map)
    maps = torch.stack(layer_maps, dim=0).sum(dim=0)
    maps_np = smooth_maps(maps.detach().cpu().numpy(), sigma=4.0)
    layer_maps_np = [smooth_maps(layer_map.detach().cpu().numpy(), sigma=4.0) for layer_map in layer_maps]
    return maps_np, layer_maps_np


def scores_from_maps(maps_np: np.ndarray) -> np.ndarray:
    flat = maps_np.reshape(maps_np.shape[0], -1)
    return flat.max(axis=1).astype(np.float64)


def top_fraction_mean(flat_maps: np.ndarray, fraction: float) -> np.ndarray:
    k = max(1, int(math.ceil(float(flat_maps.shape[1]) * float(fraction))))
    k = min(k, int(flat_maps.shape[1]))
    top = np.partition(flat_maps, kth=flat_maps.shape[1] - k, axis=1)[:, -k:]
    return top.mean(axis=1).astype(np.float64)


def downsample_maps_np(maps_np: np.ndarray, factor: int) -> np.ndarray:
    if int(factor) == 1:
        return maps_np
    factor = int(factor)
    n, c, h, w = maps_np.shape
    h2 = h // factor
    w2 = w // factor
    cropped = maps_np[:, :, : h2 * factor, : w2 * factor]
    return cropped.reshape(n, c, h2, factor, w2, factor).mean(axis=(3, 5))


def base5_features_from_maps(
    maps_np: np.ndarray,
    pixel_threshold: float,
    max_mean: float,
    max_std: float,
) -> np.ndarray:
    flat = maps_np.reshape(maps_np.shape[0], -1).astype(np.float64, copy=False)
    score_max = flat.max(axis=1)
    score_mean = flat.mean(axis=1)
    score_std = flat.std(axis=1)
    area_ratio = (flat > float(pixel_threshold)).mean(axis=1).astype(np.float64)
    source_z_max = (score_max - float(max_mean)) / max(float(max_std), 1e-8)
    return np.stack((score_max, score_mean, score_std, area_ratio, source_z_max), axis=1).astype(np.float64)


def raw3_features_from_maps(maps_np: np.ndarray) -> np.ndarray:
    flat = maps_np.reshape(maps_np.shape[0], -1).astype(np.float64, copy=False)
    return np.stack((flat.max(axis=1), flat.mean(axis=1), flat.std(axis=1)), axis=1).astype(np.float64)


def spatial_features_from_maps(maps_np: np.ndarray, pixel_threshold: float) -> np.ndarray:
    features: list[tuple[float, float, float]] = []
    for anomaly_map in maps_np[:, 0]:
        mask = np.asarray(anomaly_map > float(pixel_threshold), dtype=bool)
        labeled, count = connected_components(mask)
        total_area = float(mask.size)
        if count <= 0:
            features.append((0.0, 0.0, 0.0))
            continue
        areas = np.bincount(labeled.reshape(-1), minlength=count + 1)[1:]
        max_area_ratio = float(areas.max() / max(total_area, 1.0))
        ys, xs = np.nonzero(mask)
        if ys.size <= 1:
            dispersion = 0.0
        else:
            y = ys.astype(np.float64) / max(float(mask.shape[0] - 1), 1.0)
            x = xs.astype(np.float64) / max(float(mask.shape[1] - 1), 1.0)
            dispersion = float(np.mean((y - y.mean()) ** 2 + (x - x.mean()) ** 2))
        features.append((float(count), max_area_ratio, dispersion))
    return np.asarray(features, dtype=np.float64)


def frequency_features_from_maps(maps_np: np.ndarray) -> np.ndarray:
    features: list[tuple[float, float, float]] = []
    for anomaly_map in maps_np[:, 0].astype(np.float64, copy=False):
        coeff = dctn(anomaly_map, norm="ortho")
        energy = coeff * coeff
        total = float(energy.sum()) + 1e-12
        h, w = energy.shape
        low = float(energy[: max(1, h // 8), : max(1, w // 8)].sum() / total)
        high = float(energy[h // 4 :, w // 4 :].sum() / total)
        yy, xx = np.indices((h, w), dtype=np.float64)
        radius = np.sqrt((yy / max(float(h - 1), 1.0)) ** 2 + (xx / max(float(w - 1), 1.0)) ** 2)
        centroid = float((energy * radius).sum() / total)
        features.append((low, high, centroid))
    return np.asarray(features, dtype=np.float64)


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


@torch.no_grad()
def pooled_encoder_features_np(model: AnomalibReverseDistillationModel, images: torch.Tensor) -> np.ndarray:
    features = model.encoder(images)
    pooled = [F.adaptive_avg_pool2d(feature, output_size=1).flatten(1) for feature in features]
    return torch.cat(pooled, dim=1).detach().cpu().numpy().astype(np.float64)


@torch.no_grad()
def feature_perturbation_diffs(
    model: AnomalibReverseDistillationModel,
    images: torch.Tensor,
    original_scores: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    encoder_features = model.encoder(images)
    perturbed_features: list[torch.Tensor] = []
    for feature in encoder_features:
        noise = torch.randn(
            feature.size(),
            device=feature.device,
            dtype=feature.dtype,
        ) * float(args.eatta_feature_perturb_std)
        perturbed_features.append(feature.clone().detach() + noise)

    decoder_features = model.decoder(model.bottleneck(perturbed_features))
    decoder_features = [
        adapter(feature) for adapter, feature in zip(model.decoder_adapters, decoder_features, strict=True)
    ]
    maps = []
    for encoder_feature, decoder_feature in zip(perturbed_features, decoder_features, strict=True):
        layer_map = 1.0 - F.cosine_similarity(encoder_feature, decoder_feature, dim=1).unsqueeze(1)
        maps.append(
            F.interpolate(
                layer_map,
                size=(int(args.crop_size), int(args.crop_size)),
                mode="bilinear",
                align_corners=False,
            ),
        )
    maps_np = smooth_maps(torch.stack(maps, dim=0).sum(dim=0).detach().cpu().numpy(), sigma=4.0)
    perturbed_scores = scores_from_maps(maps_np)
    py = torch.as_tensor(original_scores, device=images.device, dtype=torch.float32).reshape(-1)
    py2 = torch.as_tensor(perturbed_scores, device=images.device, dtype=torch.float32).reshape(-1)
    return torch.abs(py - py2).detach().cpu().numpy().astype(np.float64)


@torch.no_grad()
def feature_perturbation_query(
    model: AnomalibReverseDistillationModel,
    images: torch.Tensor,
    original_scores: np.ndarray,
    args: argparse.Namespace,
    batch_index: int,
) -> int:
    del batch_index
    diff = feature_perturbation_diffs(model=model, images=images, original_scores=original_scores, args=args)
    return int(np.argsort(-diff)[0])


def binary_pseudo_labels_from_scores(scores: np.ndarray) -> np.ndarray:
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    threshold = float(np.median(score_array))
    return np.asarray(score_array > threshold, dtype=np.int64)


def binary_entropy_from_scores(scores: np.ndarray) -> np.ndarray:
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    center = float(np.median(score_array))
    scale = float(np.std(score_array))
    if not math.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    anomaly_logit = np.clip((score_array - center) / scale, -10.0, 10.0)
    logits = np.stack((-anomaly_logit, anomaly_logit), axis=1)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return -np.sum(probs * np.log(np.clip(probs, 1e-6, 1.0)), axis=1).astype(np.float64)


def score_logits_from_scores_np(scores: np.ndarray, center: float, scale: float) -> np.ndarray:
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    safe_scale = float(scale)
    if not math.isfinite(safe_scale) or safe_scale <= 1e-8:
        safe_scale = 1.0
    anomaly_logit = np.clip((score_array - float(center)) / safe_scale, -10.0, 10.0)
    return np.stack((-anomaly_logit, anomaly_logit), axis=1).astype(np.float64)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits_array = np.asarray(logits, dtype=np.float64)
    logits_array = logits_array - logits_array.max(axis=1, keepdims=True)
    probs = np.exp(logits_array)
    return probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)


def binary_pseudo_labels_from_fixed_scores(scores: np.ndarray, center: float, scale: float) -> np.ndarray:
    return np.argmax(score_logits_from_scores_np(scores, center=center, scale=scale), axis=1).astype(np.int64)


def binary_entropy_from_fixed_scores(scores: np.ndarray, center: float, scale: float) -> np.ndarray:
    probs = softmax_np(score_logits_from_scores_np(scores, center=center, scale=scale))
    return -np.sum(probs * np.log(np.clip(probs, 1e-6, 1.0)), axis=1).astype(np.float64)


def batch_score_center_scale(scores: np.ndarray) -> tuple[float, float]:
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    center = float(np.median(score_array))
    scale = float(np.std(score_array))
    if not math.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    return center, scale


@torch.no_grad()
def feature_perturbation_confidence_diffs(
    model: AnomalibReverseDistillationModel,
    images: torch.Tensor,
    scores: np.ndarray,
    args: argparse.Namespace,
    logit_center: float,
    logit_scale: float,
) -> np.ndarray:
    logits = score_logits_from_scores_np(scores, center=logit_center, scale=logit_scale)
    probs = softmax_np(logits)
    pseudo_labels = np.argmax(probs, axis=1).astype(np.int64)
    original_confidence = probs[np.arange(probs.shape[0]), pseudo_labels]

    encoder_features = model.encoder(images)
    perturbed_features: list[torch.Tensor] = []
    for feature in encoder_features:
        noise = torch.randn(
            feature.size(),
            device=feature.device,
            dtype=feature.dtype,
        ) * float(args.eatta_feature_perturb_std)
        perturbed_features.append(feature.clone().detach() + noise)

    decoder_features = model.decoder(model.bottleneck(perturbed_features))
    decoder_features = [
        adapter(feature) for adapter, feature in zip(model.decoder_adapters, decoder_features, strict=True)
    ]
    maps = []
    for encoder_feature, decoder_feature in zip(perturbed_features, decoder_features, strict=True):
        layer_map = 1.0 - F.cosine_similarity(encoder_feature, decoder_feature, dim=1).unsqueeze(1)
        maps.append(
            F.interpolate(
                layer_map,
                size=(int(args.crop_size), int(args.crop_size)),
                mode="bilinear",
                align_corners=False,
            ),
        )
    maps_np = smooth_maps(torch.stack(maps, dim=0).sum(dim=0).detach().cpu().numpy(), sigma=4.0)
    perturbed_scores = scores_from_maps(maps_np)
    perturbed_probs = softmax_np(score_logits_from_scores_np(perturbed_scores, center=logit_center, scale=logit_scale))
    perturbed_confidence = perturbed_probs[np.arange(perturbed_probs.shape[0]), pseudo_labels]
    return (original_confidence - perturbed_confidence).astype(np.float64)


@torch.no_grad()
def feature_perturbation_batch_confidence_diffs(
    model: AnomalibReverseDistillationModel,
    images: torch.Tensor,
    scores: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    logit_center, logit_scale = batch_score_center_scale(scores)
    return feature_perturbation_confidence_diffs(
        model=model,
        images=images,
        scores=scores,
        args=args,
        logit_center=logit_center,
        logit_scale=logit_scale,
    )


def eatta_history_select_indices(
    diff: np.ndarray,
    pseudo_labels: np.ndarray,
    state: dict[str, Any],
    args: argparse.Namespace,
) -> np.ndarray:
    sorted_indices = np.argsort(-np.asarray(diff, dtype=np.float64).reshape(-1))
    oracle_num = min(max(1, int(args.paper_oracle_num)), int(sorted_indices.size))
    if sorted_indices.size == 0:
        return np.asarray([], dtype=np.int64)
    recent_labels = state.setdefault("eatta_recent_pseudo_labels", [])
    class_diff = state.setdefault("eatta_class_diff", {})
    if not recent_labels:
        selected = sorted_indices[:oracle_num].tolist()
    else:
        selected: list[int] = []
        recent = set(int(label) for label in recent_labels[-max(1, int(args.eatta_class_history)) :])
        top = sorted_indices[:oracle_num]
        tail = sorted_indices[oracle_num:]
        for index in top.tolist():
            label = int(pseudo_labels[index])
            previous = float(class_diff.get(label, -float("inf")))
            if label not in recent or float(diff[index]) >= previous:
                selected.append(int(index))
                continue
            replacement = next((int(other) for other in tail.tolist() if int(pseudo_labels[other]) not in recent), None)
            selected.append(int(index if replacement is None else replacement))
        if not selected:
            selected = sorted_indices[:oracle_num].tolist()
    selected = selected[:oracle_num]
    for index in selected:
        label = int(pseudo_labels[int(index)])
        class_diff[label] = float(diff[int(index)])
        recent_labels.append(label)
    history = max(1, int(args.eatta_class_history))
    del recent_labels[:-history]
    return np.asarray(selected, dtype=np.int64)


def simatta_query_indices(
    features: np.ndarray,
    uncertainty: np.ndarray,
    anchor_features: Sequence[np.ndarray],
    args: argparse.Namespace,
    batch_index: int,
) -> np.ndarray:
    feature_array = np.asarray(features, dtype=np.float64)
    uncertainty_array = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    query_count = min(max(1, int(args.paper_oracle_num)), int(feature_array.shape[0]))
    if feature_array.shape[0] <= query_count:
        return np.arange(feature_array.shape[0], dtype=np.int64)

    candidates = np.flatnonzero(uncertainty_array >= float(args.atta_entropy_high_threshold))
    if candidates.size == 0:
        candidates = np.arange(feature_array.shape[0], dtype=np.int64)
    candidate_features = feature_array[candidates]
    anchor_array = (
        np.asarray([np.asarray(feature, dtype=np.float64).reshape(-1) for feature in anchor_features], dtype=np.float64)
        if anchor_features
        else np.empty((0, feature_array.shape[1]), dtype=np.float64)
    )
    combined = np.concatenate((anchor_array, candidate_features), axis=0)
    center = np.median(combined, axis=0)
    scale = np.std(combined, axis=0)
    fallback = float(np.std(combined))
    if not math.isfinite(fallback) or fallback <= 1e-8:
        fallback = 1.0
    scale = np.where(scale > 1e-8, scale, fallback)
    combined_scaled = (combined - center) / scale
    n_anchor = int(anchor_array.shape[0])
    n_clusters = min(max(query_count, n_anchor + query_count), int(combined_scaled.shape[0]))

    if n_clusters <= 1:
        order = np.argsort(-uncertainty_array[candidates])
        return candidates[order[:query_count]].astype(np.int64)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=int(args.stream_seed) + 10_003 * int(batch_index))
    labels = kmeans.fit_predict(combined_scaled)
    selected: list[int] = []
    for cluster_id in range(n_clusters):
        members = np.flatnonzero(labels == cluster_id)
        if members.size == 0 or np.any(members < n_anchor):
            continue
        center_vec = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
        distances = np.linalg.norm(combined_scaled[members] - center_vec, axis=1)
        representative = int(members[int(np.argmin(distances))] - n_anchor)
        if representative >= 0:
            selected.append(int(candidates[representative]))
    if selected:
        selected = sorted(set(selected), key=lambda index: float(uncertainty_array[index]), reverse=True)
    if len(selected) < query_count:
        for index in candidates[np.argsort(-uncertainty_array[candidates])].tolist():
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) >= query_count:
                break
    return np.asarray(selected[:query_count], dtype=np.int64)


def simatta_incremental_query_indices(
    features: np.ndarray,
    uncertainty: np.ndarray,
    state: dict[str, Any],
    args: argparse.Namespace,
    batch_index: int,
) -> np.ndarray:
    feature_array = np.asarray(features, dtype=np.float64)
    uncertainty_array = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    query_count = min(max(1, int(args.paper_oracle_num)), int(feature_array.shape[0]))
    if feature_array.shape[0] == 0:
        return np.asarray([], dtype=np.int64)
    if feature_array.shape[0] <= query_count:
        return np.arange(feature_array.shape[0], dtype=np.int64)

    candidates = np.flatnonzero(uncertainty_array >= float(args.atta_entropy_high_threshold))
    if candidates.size == 0:
        candidates = np.arange(feature_array.shape[0], dtype=np.int64)
    candidate_features = feature_array[candidates]
    anchor_features = state.setdefault("simatta_anchor_features", [])
    anchor_weights = state.setdefault("simatta_anchor_weights", [])
    anchor_array = (
        np.asarray([np.asarray(feature, dtype=np.float64).reshape(-1) for feature in anchor_features], dtype=np.float64)
        if anchor_features
        else np.empty((0, feature_array.shape[1]), dtype=np.float64)
    )
    combined = np.concatenate((anchor_array, candidate_features), axis=0)
    center = np.median(combined, axis=0)
    scale = np.std(combined, axis=0)
    fallback = float(np.std(combined))
    if not math.isfinite(fallback) or fallback <= 1e-8:
        fallback = 1.0
    scale = np.where(scale > 1e-8, scale, fallback)
    combined_scaled = (combined - center) / scale
    n_anchor = int(anchor_array.shape[0])
    cluster_increase = max(1, int(args.atta_cluster_increase))
    cluster_budget = max(query_count, int(args.atta_cluster_budget))
    n_clusters = min(max(query_count, n_anchor + cluster_increase), cluster_budget, int(combined_scaled.shape[0]))
    if n_clusters <= 1:
        selected = candidates[np.argsort(-uncertainty_array[candidates])[:query_count]]
        state["simatta_pending_anchor_weights"] = {int(index): 1.0 for index in selected.tolist()}
        return selected.astype(np.int64)

    weights = np.concatenate(
        (
            np.asarray(anchor_weights, dtype=np.float64).reshape(-1)[:n_anchor] if n_anchor > 0 else np.empty(0),
            np.ones(int(candidate_features.shape[0]), dtype=np.float64),
        ),
        axis=0,
    )
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=int(args.stream_seed) + 10_003 * int(batch_index))
    labels = kmeans.fit_predict(combined_scaled, sample_weight=weights)
    selected: list[int] = []
    pending_weights: dict[int, float] = {}
    updated_anchor_weights = list(float(weight) for weight in weights[:n_anchor])
    for cluster_id in range(n_clusters):
        members = np.flatnonzero(labels == cluster_id)
        if members.size == 0:
            continue
        anchor_members = members[members < n_anchor]
        cluster_weight = float(np.sum(weights[members]))
        if anchor_members.size > 0:
            share = cluster_weight / float(anchor_members.size)
            for member in anchor_members.tolist():
                updated_anchor_weights[int(member)] = max(float(updated_anchor_weights[int(member)]), share)
            continue
        center_vec = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
        distances = np.linalg.norm(combined_scaled[members] - center_vec, axis=1)
        representative = int(members[int(np.argmin(distances))] - n_anchor)
        if representative >= 0:
            index = int(candidates[representative])
            selected.append(index)
            pending_weights[index] = cluster_weight
    if selected:
        selected = sorted(set(selected), key=lambda index: float(uncertainty_array[index]), reverse=True)
    if len(selected) < query_count:
        for index in candidates[np.argsort(-uncertainty_array[candidates])].tolist():
            if int(index) not in selected:
                selected.append(int(index))
                pending_weights[int(index)] = 1.0
            if len(selected) >= query_count:
                break
    state["simatta_anchor_weights"] = updated_anchor_weights
    state["simatta_pending_anchor_weights"] = pending_weights
    return np.asarray(selected[:query_count], dtype=np.int64)


def simatta_add_anchors(
    state: dict[str, Any],
    features: np.ndarray,
    query_indices: np.ndarray,
    args: argparse.Namespace,
) -> None:
    anchor_features = state.setdefault("simatta_anchor_features", [])
    anchor_weights = state.setdefault("simatta_anchor_weights", [])
    pending_weights = state.get("simatta_pending_anchor_weights", {})
    for index in np.asarray(query_indices, dtype=np.int64).tolist():
        anchor_features.append(np.asarray(features[int(index)], dtype=np.float64).copy())
        anchor_weights.append(float(pending_weights.get(int(index), 1.0)))
    max_size = max(1, int(args.atta_cluster_budget))
    overflow = max(0, len(anchor_features) - max_size)
    if overflow > 0:
        del anchor_features[:overflow]
        del anchor_weights[:overflow]


def paper_low_entropy_indices(
    entropy: np.ndarray,
    excluded_indices: np.ndarray,
    args: argparse.Namespace,
    entropy_q: float | None = None,
    max_fraction: float | None = None,
) -> np.ndarray:
    entropy_array = np.asarray(entropy, dtype=np.float64).reshape(-1)
    q = float(args.atta_pseudo_normal_entropy_q) if entropy_q is None else float(entropy_q)
    fraction = float(args.atta_pseudo_normal_max_fraction) if max_fraction is None else float(max_fraction)
    if entropy_array.size == 0 or fraction <= 0.0:
        return np.asarray([], dtype=np.int64)
    entropy_threshold = float(np.quantile(entropy_array, q))
    mask = entropy_array <= entropy_threshold
    if excluded_indices.size > 0:
        mask[np.asarray(excluded_indices, dtype=np.int64)] = False
    candidates = np.flatnonzero(mask)
    max_count = min(
        int(candidates.size),
        max(1, int(math.ceil(float(entropy_array.size) * fraction))),
    )
    if max_count <= 0:
        return np.asarray([], dtype=np.int64)
    return candidates[np.argsort(entropy_array[candidates])[:max_count]].astype(np.int64)


def paper_memory_append(
    memory: list[tuple[torch.Tensor, int]],
    images: torch.Tensor,
    indices: np.ndarray,
    labels: np.ndarray,
    max_size: int,
) -> None:
    for index in np.asarray(indices, dtype=np.int64).tolist():
        memory.append((images[int(index)].detach().cpu(), int(labels[int(index)])))
    overflow = max(0, len(memory) - max(1, int(max_size)))
    if overflow > 0:
        del memory[:overflow]


def paper_memory_tensor(
    memory: list[tuple[torch.Tensor, int]],
    limit: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not memory:
        return None, None
    entries = memory[-max(1, int(limit)) :]
    images = torch.stack([image for image, _label in entries], dim=0).to(device, non_blocking=True)
    labels = torch.as_tensor([label for _image, label in entries], device=device, dtype=torch.long)
    return images, labels


def paper_memory_entry_tensor(
    entries: Sequence[tuple[torch.Tensor, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([image for image, _label in entries], dim=0).to(device, non_blocking=True)
    labels = torch.as_tensor([label for _image, label in entries], device=device, dtype=torch.long)
    return images, labels


def paper_memory_chunks(
    memory: list[tuple[torch.Tensor, int]],
    chunk_size: int,
) -> list[list[tuple[torch.Tensor, int]]]:
    size = max(1, int(chunk_size))
    return [memory[start : start + size] for start in range(0, len(memory), size)]


def anomaly_map_to_prob(anomaly_maps: torch.Tensor) -> torch.Tensor:
    mean = anomaly_maps.mean(dim=(-1, -2), keepdim=True)
    std = anomaly_maps.std(dim=(-1, -2), keepdim=True, unbiased=False).clamp_min(1e-6)
    return torch.sigmoid(((anomaly_maps - mean) / std).clamp(-12.0, 12.0))


def anomaly_map_sample_entropy(anomaly_maps: torch.Tensor) -> torch.Tensor:
    prob = anomaly_map_to_prob(anomaly_maps).clamp(1e-6, 1.0 - 1e-6)
    entropy = -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())
    return entropy.mean(dim=(1, 2, 3))


def softmax_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    return -(probabilities * logits.log_softmax(dim=1)).sum(dim=1)


def anomaly_map_tv_loss(anomaly_maps: torch.Tensor) -> torch.Tensor:
    dy = (anomaly_maps[:, :, 1:, :] - anomaly_maps[:, :, :-1, :]).abs().mean()
    dx = (anomaly_maps[:, :, :, 1:] - anomaly_maps[:, :, :, :-1]).abs().mean()
    return dx + dy


def binary_logits_from_anomaly_maps(anomaly_maps: torch.Tensor) -> torch.Tensor:
    scores = anomaly_maps.flatten(1).amax(dim=1)
    center = scores.detach().median()
    scale = scores.detach().std(unbiased=False).clamp_min(1e-3)
    anomaly_logit = ((scores - center) / scale).clamp(-10.0, 10.0)
    return torch.stack((-anomaly_logit, anomaly_logit), dim=1)


def binary_logits_from_anomaly_maps_fixed(
    anomaly_maps: torch.Tensor,
    center: float,
    scale: float,
) -> torch.Tensor:
    scores = anomaly_maps.flatten(1).amax(dim=1)
    safe_scale = float(scale)
    if not math.isfinite(safe_scale) or safe_scale <= 1e-8:
        safe_scale = 1.0
    anomaly_logit = ((scores - float(center)) / safe_scale).clamp(-10.0, 10.0)
    return torch.stack((-anomaly_logit, anomaly_logit), dim=1)


def gradient_norm(loss: torch.Tensor, params: Sequence[nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    norms = [grad.norm() for grad in grads if grad is not None]
    if not norms:
        return loss.new_zeros(())
    return torch.norm(torch.stack(norms))


def eatta_weighted_loss(
    loss_ent: torch.Tensor,
    loss_ce: torch.Tensor,
    has_entropy: bool,
    state: dict[str, Any],
    params: Sequence[nn.Parameter],
    args: argparse.Namespace,
) -> torch.Tensor:
    if not has_entropy:
        return loss_ce
    grad_ent = gradient_norm(loss_ent, params)
    grad_ce = gradient_norm(loss_ce, params)
    denom = grad_ent + grad_ce + loss_ce.new_tensor(1e-12)
    w_ent = float((2.0 * grad_ce / denom).detach().item())
    w_ce = float((2.0 * grad_ent / denom).detach().item())
    momentum = float(args.eatta_gnd_momentum)
    if "eatta_entropy_weight_ema" not in state or "eatta_supervised_weight_ema" not in state:
        state["eatta_entropy_weight_ema"] = w_ent
        state["eatta_supervised_weight_ema"] = w_ce
    else:
        state["eatta_entropy_weight_ema"] = momentum * float(state["eatta_entropy_weight_ema"]) + (1.0 - momentum) * w_ent
        state["eatta_supervised_weight_ema"] = momentum * float(state["eatta_supervised_weight_ema"]) + (1.0 - momentum) * w_ce
    return float(state["eatta_entropy_weight_ema"]) * loss_ent + float(state["eatta_supervised_weight_ema"]) * loss_ce


def adapt_paper_label_batch(
    model: AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: np.ndarray,
    active_indices: np.ndarray,
    pseudo_normal_indices: np.ndarray,
    pseudo_target_labels: np.ndarray | None,
    entropy_indices: np.ndarray,
    args: argparse.Namespace,
    state: dict[str, Any],
    method: str,
    logit_center: float | None = None,
    logit_scale: float | None = None,
) -> tuple[int, float]:
    if int(args.tta_steps) <= 0:
        return 0, float("nan")
    active_tensor = torch.as_tensor(active_indices, device=images.device, dtype=torch.long)
    pseudo_tensor = torch.as_tensor(pseudo_normal_indices, device=images.device, dtype=torch.long)
    if pseudo_target_labels is None:
        pseudo_label_tensor = torch.zeros_like(pseudo_tensor, dtype=torch.long)
    else:
        pseudo_label_tensor = torch.as_tensor(pseudo_target_labels, device=images.device, dtype=torch.long)
    entropy_tensor = torch.as_tensor(entropy_indices, device=images.device, dtype=torch.long)
    labels_tensor = torch.as_tensor(labels, device=images.device, dtype=torch.long)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    last_loss = float("nan")
    for _step in range(int(args.tta_steps)):
        set_tta_train_mode(model, args)
        optimizer.zero_grad(set_to_none=True)
        anomaly_maps = model.anomaly_maps(images, image_size=int(args.crop_size))
        logits = (
            binary_logits_from_anomaly_maps(anomaly_maps)
            if logit_center is None or logit_scale is None
            else binary_logits_from_anomaly_maps_fixed(anomaly_maps, center=float(logit_center), scale=float(logit_scale))
        )
        zero = logits.new_tensor(0.0)
        has_active = int(active_tensor.numel()) > 0
        has_pseudo = int(pseudo_tensor.numel()) > 0
        has_entropy = int(entropy_tensor.numel()) > 0
        loss_target = F.cross_entropy(logits[active_tensor], labels_tensor[active_tensor]) if has_active else zero
        if method == "atta_paper":
            loss_source = (
                F.cross_entropy(logits[pseudo_tensor], pseudo_label_tensor)
                if has_pseudo
                else zero
            )
            if has_active and has_pseudo:
                alpha = float(active_tensor.numel()) / float(active_tensor.numel() + pseudo_tensor.numel())
                loss = (1.0 - alpha) * loss_source + alpha * loss_target
            elif has_active:
                loss = loss_target
            else:
                loss = loss_source
        else:
            sample_entropy = softmax_entropy_from_logits(logits)
            loss_ent = sample_entropy[entropy_tensor].mean() if has_entropy else zero
            loss = eatta_weighted_loss(
                loss_ent=loss_ent,
                loss_ce=loss_target,
                has_entropy=has_entropy and has_active,
                state=state,
                params=params,
                args=args,
            )
        loss.backward()
        if float(args.tta_grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.tta_grad_clip))
        optimizer.step()
        last_loss = float(loss.detach().item())
    model.eval()
    return int(args.tta_steps), last_loss


def adapt_simatta_memory_batch(
    model: AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
    low_memory: list[tuple[torch.Tensor, int]],
    high_memory: list[tuple[torch.Tensor, int]],
    args: argparse.Namespace,
    logit_center: float | None,
    logit_scale: float | None,
    device: torch.device,
) -> tuple[int, float]:
    if int(args.tta_steps) <= 0 or (not low_memory and not high_memory):
        return 0, float("nan")
    replay_limit = max(1, int(args.paper_replay_batch_size))
    replay_mode = str(getattr(args, "paper_memory_replay_mode", "recent"))
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    last_loss = float("nan")
    if replay_mode == "full":
        memory_chunks = paper_memory_chunks(low_memory + high_memory, replay_limit)
        total_items = max(1, len(low_memory) + len(high_memory))
        update_count = 0
        for _step in range(int(args.tta_steps)):
            set_tta_train_mode(model, args)
            optimizer.zero_grad(set_to_none=True)
            stepped = False
            weighted_loss_value = 0.0
            for entries in memory_chunks:
                if not entries:
                    continue
                memory_images, memory_labels = paper_memory_entry_tensor(entries, device)
                memory_maps = model.anomaly_maps(memory_images, image_size=int(args.crop_size))
                memory_logits = (
                    binary_logits_from_anomaly_maps(memory_maps)
                    if logit_center is None or logit_scale is None
                    else binary_logits_from_anomaly_maps_fixed(memory_maps, center=float(logit_center), scale=float(logit_scale))
                )
                chunk_loss = F.cross_entropy(memory_logits, memory_labels)
                weight = float(memory_labels.numel()) / float(total_items)
                loss = chunk_loss * weight
                loss.backward()
                weighted_loss_value += float(loss.detach().item())
                stepped = True
            if not stepped:
                return 0, float("nan")
            if float(args.tta_grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.tta_grad_clip))
            optimizer.step()
            update_count += 1
            last_loss = weighted_loss_value
        model.eval()
        return update_count, last_loss

    if low_memory and high_memory:
        low_limit = max(1, replay_limit // 2)
        high_limit = max(1, replay_limit - low_limit)
    elif low_memory:
        low_limit = replay_limit
        high_limit = 0
    else:
        low_limit = 0
        high_limit = replay_limit
    low_images, low_labels = paper_memory_tensor(low_memory, low_limit, device) if low_limit > 0 else (None, None)
    high_images, high_labels = paper_memory_tensor(high_memory, high_limit, device) if high_limit > 0 else (None, None)
    for _step in range(int(args.tta_steps)):
        set_tta_train_mode(model, args)
        optimizer.zero_grad(set_to_none=True)
        zero: torch.Tensor | None = None
        losses: list[torch.Tensor] = []
        weights: list[float] = []
        if logit_center is None or logit_scale is None:
            image_parts = [image_part for image_part in (low_images, high_images) if image_part is not None]
            label_parts = [label_part for label_part in (low_labels, high_labels) if label_part is not None]
            if not image_parts or not label_parts:
                return 0, float("nan")
            memory_images = torch.cat(image_parts, dim=0)
            memory_labels = torch.cat(label_parts, dim=0)
            memory_maps = model.anomaly_maps(memory_images, image_size=int(args.crop_size))
            memory_logits = binary_logits_from_anomaly_maps(memory_maps)
            loss = F.cross_entropy(memory_logits, memory_labels)
            loss.backward()
            if float(args.tta_grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.tta_grad_clip))
            optimizer.step()
            last_loss = float(loss.detach().item())
            continue
        if low_images is not None and low_labels is not None:
            low_maps = model.anomaly_maps(low_images, image_size=int(args.crop_size))
            low_logits = binary_logits_from_anomaly_maps_fixed(low_maps, center=logit_center, scale=logit_scale)
            low_loss = F.cross_entropy(low_logits, low_labels)
            losses.append(low_loss)
            weights.append(float(low_labels.numel()))
            zero = low_loss.new_tensor(0.0)
        if high_images is not None and high_labels is not None:
            high_maps = model.anomaly_maps(high_images, image_size=int(args.crop_size))
            high_logits = binary_logits_from_anomaly_maps_fixed(high_maps, center=logit_center, scale=logit_scale)
            high_loss = F.cross_entropy(high_logits, high_labels)
            losses.append(high_loss)
            weights.append(float(high_labels.numel()))
            zero = high_loss.new_tensor(0.0)
        if not losses:
            return 0, float("nan")
        total = max(sum(weights), 1.0)
        loss = losses[0] * (weights[0] / total)
        for loss_item, weight in zip(losses[1:], weights[1:], strict=True):
            loss = loss + loss_item * (weight / total)
        if zero is not None:
            loss = loss + zero
        loss.backward()
        if float(args.tta_grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.tta_grad_clip))
        optimizer.step()
        last_loss = float(loss.detach().item())
    model.eval()
    return int(args.tta_steps), last_loss


def multiscale_feature_chunks(
    maps_np: np.ndarray,
    source_stats: dict[str, Any],
) -> list[np.ndarray]:
    scale_stats = source_stats["scale_stats"]
    chunks = []
    for factor in (1, 2, 4):
        scaled = downsample_maps_np(maps_np, factor)
        stats = scale_stats[str(factor)]
        chunks.append(
            base5_features_from_maps(
                scaled,
                pixel_threshold=float(stats["pixel_threshold"]),
                max_mean=float(stats["max_mean"]),
                max_std=float(stats["max_std"]),
            ),
        )
    return chunks


def multiscale_nosource_feature_chunks(maps_np: np.ndarray, factors: Sequence[int] = (1, 2, 4)) -> list[np.ndarray]:
    return [raw3_features_from_maps(downsample_maps_np(maps_np, factor)) for factor in factors]


def multiscale_frequency_nosource_features(maps_np: np.ndarray, factors: Sequence[int] = (1, 2, 4)) -> np.ndarray:
    chunks = [frequency_features_from_maps(downsample_maps_np(maps_np, factor)) for factor in factors]
    return np.concatenate(chunks, axis=1).astype(np.float64)


def feature_family_needs_layer_maps(feature_family: str) -> bool:
    return str(feature_family) in {
        "encoder_raw9",
        "encoder_raw9_finalfreq3",
        "hybrid_21d",
    }


def encoder_raw3_features_from_layer_maps(layer_maps_np: Sequence[np.ndarray]) -> np.ndarray:
    if len(layer_maps_np) < 3:
        raise ValueError(f"Expected 3 encoder layer maps, got {len(layer_maps_np)}")
    chunks = [raw3_features_from_maps(np.asarray(layer_maps_np[index])) for index in range(3)]
    return np.concatenate(chunks, axis=1).astype(np.float64)


def map_stat_features_from_maps(
    maps_np: np.ndarray,
    source_stats: dict[str, Any],
    feature_family: str,
    layer_maps_np: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    family = str(feature_family)
    if family == "score1d":
        return scores_from_maps(maps_np).reshape(-1, 1).astype(np.float64)
    if family == "final_raw3":
        return raw3_features_from_maps(maps_np).astype(np.float64)
    if family in {"multiscale_frequency_nosource", "current_12d"}:
        multiscale = np.concatenate(multiscale_nosource_feature_chunks(maps_np), axis=1)
        frequency = frequency_features_from_maps(maps_np)
        return np.concatenate((multiscale, frequency), axis=1).astype(np.float64)
    if family == "current_scale1freq_6d":
        raw = raw3_features_from_maps(maps_np)
        frequency = frequency_features_from_maps(maps_np)
        return np.concatenate((raw, frequency), axis=1).astype(np.float64)
    if family == "score_freq_4d":
        score = scores_from_maps(maps_np).reshape(-1, 1).astype(np.float64)
        frequency = frequency_features_from_maps(maps_np)
        return np.concatenate((score, frequency), axis=1).astype(np.float64)
    if family == "score_mean_freq_5d":
        raw = raw3_features_from_maps(maps_np)
        frequency = frequency_features_from_maps(maps_np)
        return np.concatenate((raw[:, [0, 1]], frequency), axis=1).astype(np.float64)
    if family == "score_std_freq_5d":
        raw = raw3_features_from_maps(maps_np)
        frequency = frequency_features_from_maps(maps_np)
        return np.concatenate((raw[:, [0, 2]], frequency), axis=1).astype(np.float64)
    if family == "scale1_no_low_5d":
        raw = raw3_features_from_maps(maps_np)
        frequency = frequency_features_from_maps(maps_np)[:, 1:3]
        return np.concatenate((raw, frequency), axis=1).astype(np.float64)
    if family == "current_no_low_11d":
        multiscale = np.concatenate(multiscale_nosource_feature_chunks(maps_np), axis=1)
        frequency = frequency_features_from_maps(maps_np)[:, 1:3]
        return np.concatenate((multiscale, frequency), axis=1).astype(np.float64)
    if family == "current_edge_13d":
        multiscale = np.concatenate(multiscale_nosource_feature_chunks(maps_np), axis=1)
        frequency = frequency_features_from_maps(maps_np)
        high_minus_low = (frequency[:, 1] - frequency[:, 0]).reshape(-1, 1)
        return np.concatenate((multiscale, frequency, high_minus_low), axis=1).astype(np.float64)
    if family == "current_scale8_15d":
        multiscale = np.concatenate(multiscale_nosource_feature_chunks(maps_np, factors=(1, 2, 4, 8)), axis=1)
        frequency = frequency_features_from_maps(maps_np)
        return np.concatenate((multiscale, frequency), axis=1).astype(np.float64)
    if family == "current_msfreq_18d":
        multiscale = np.concatenate(multiscale_nosource_feature_chunks(maps_np), axis=1)
        frequency = multiscale_frequency_nosource_features(maps_np, factors=(1, 2, 4))
        return np.concatenate((multiscale, frequency), axis=1).astype(np.float64)
    if family in {"multiscale_nosource", "final_multiscale_raw9"}:
        return np.concatenate(multiscale_nosource_feature_chunks(maps_np), axis=1).astype(np.float64)
    if family == "frequency_nosource":
        return frequency_features_from_maps(maps_np).astype(np.float64)
    if family == "encoder_raw9":
        if layer_maps_np is None:
            raise ValueError("encoder_raw9 requires encoder layer maps")
        return encoder_raw3_features_from_layer_maps(layer_maps_np)
    if family == "encoder_raw9_finalfreq3":
        if layer_maps_np is None:
            raise ValueError("encoder_raw9_finalfreq3 requires encoder layer maps")
        encoder = encoder_raw3_features_from_layer_maps(layer_maps_np)
        frequency = frequency_features_from_maps(maps_np)
        return np.concatenate((encoder, frequency), axis=1).astype(np.float64)
    if family == "hybrid_21d":
        if layer_maps_np is None:
            raise ValueError("hybrid_21d requires encoder layer maps")
        current = map_stat_features_from_maps(maps_np, source_stats, "current_12d")
        encoder = encoder_raw3_features_from_layer_maps(layer_maps_np)
        return np.concatenate((current, encoder), axis=1).astype(np.float64)
    scale_stats = source_stats["scale_stats"]
    base = base5_features_from_maps(
        maps_np,
        pixel_threshold=float(scale_stats["1"]["pixel_threshold"]),
        max_mean=float(scale_stats["1"]["max_mean"]),
        max_std=float(scale_stats["1"]["max_std"]),
    )
    if family == "base5":
        return base
    if family == "multiscale":
        return np.concatenate(multiscale_feature_chunks(maps_np, source_stats), axis=1).astype(np.float64)
    if family == "spatial":
        spatial = spatial_features_from_maps(maps_np, pixel_threshold=float(scale_stats["1"]["pixel_threshold"]))
        return np.concatenate((base, spatial), axis=1).astype(np.float64)
    if family == "frequency":
        frequency = frequency_features_from_maps(maps_np)
        return np.concatenate((base, frequency), axis=1).astype(np.float64)
    if family == "multiscale_frequency":
        multiscale = map_stat_features_from_maps(maps_np, source_stats, "multiscale")
        frequency = frequency_features_from_maps(maps_np)
        return np.concatenate((multiscale, frequency), axis=1).astype(np.float64)
    if family == "msfreq_no_low":
        multiscale = map_stat_features_from_maps(maps_np, source_stats, "multiscale")
        frequency = frequency_features_from_maps(maps_np)[:, 1:3]
        return np.concatenate((multiscale, frequency), axis=1).astype(np.float64)
    if family == "compact_msfreq":
        chunks = multiscale_feature_chunks(maps_np, source_stats)
        compact_scales = [chunks[0]]
        compact_scales.extend(chunk[:, [0, 3, 4]] for chunk in chunks[1:])
        frequency = frequency_features_from_maps(maps_np)[:, 1:3]
        return np.concatenate((*compact_scales, frequency), axis=1).astype(np.float64)
    if family == "edge_msfreq":
        chunks = multiscale_feature_chunks(maps_np, source_stats)
        edge_scales = [chunks[0]]
        edge_scales.extend(chunk[:, [0, 2, 3, 4]] for chunk in chunks[1:])
        frequency = frequency_features_from_maps(maps_np)
        high_minus_low = (frequency[:, 1] - frequency[:, 0]).reshape(-1, 1)
        return np.concatenate((*edge_scales, frequency[:, 1:3], high_minus_low), axis=1).astype(np.float64)
    if family == "all":
        multiscale = map_stat_features_from_maps(maps_np, source_stats, "multiscale")
        spatial = spatial_features_from_maps(maps_np, pixel_threshold=float(scale_stats["1"]["pixel_threshold"]))
        frequency = frequency_features_from_maps(maps_np)
        return np.concatenate((multiscale, spatial, frequency), axis=1).astype(np.float64)
    raise ValueError(f"Unsupported feature family: {feature_family}")


def feature_family_needs_source_stats(feature_family: str) -> bool:
    return str(feature_family) not in {
        "score1d",
        "final_raw3",
        "multiscale_nosource",
        "final_multiscale_raw9",
        "frequency_nosource",
        "multiscale_frequency_nosource",
        "current_12d",
        "current_scale1freq_6d",
        "score_freq_4d",
        "score_mean_freq_5d",
        "score_std_freq_5d",
        "scale1_no_low_5d",
        "current_no_low_11d",
        "current_edge_13d",
        "current_scale8_15d",
        "current_msfreq_18d",
        "encoder_raw9",
        "encoder_raw9_finalfreq3",
        "hybrid_21d",
    }


@torch.no_grad()
def compute_source_stats(
    model: AnomalibReverseDistillationModel,
    train_dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    loader = make_loader(train_dataset, batch_size=int(args.batch_size), num_workers=int(args.num_workers), shuffle=False)
    scale_pixels: dict[int, list[np.ndarray]] = {1: [], 2: [], 4: []}
    scale_max_scores: dict[int, list[np.ndarray]] = {1: [], 2: [], 4: []}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        maps_np = anomaly_maps_np(model, images, image_size=int(args.crop_size))
        for factor in (1, 2, 4):
            scaled = downsample_maps_np(maps_np, factor)
            flat = scaled.reshape(scaled.shape[0], -1).astype(np.float64, copy=False)
            scale_max_scores[factor].append(flat.max(axis=1))
            scale_pixels[factor].append(flat.reshape(-1))
    scale_stats: dict[str, dict[str, float]] = {}
    for factor in (1, 2, 4):
        max_all = np.concatenate(scale_max_scores[factor])
        pixel_all = np.concatenate(scale_pixels[factor])
        scale_stats[str(factor)] = {
            "pixel_threshold": float(np.quantile(pixel_all, float(args.active_svm_source_pixel_q))),
            "max_threshold": float(np.quantile(max_all, float(getattr(args, "paper_score_center_q", 0.99)))),
            "max_mean": float(np.mean(max_all)),
            "max_std": float(np.std(max_all)),
        }
    return {
        "scale_stats": scale_stats,
        "pixel_threshold": scale_stats["1"]["pixel_threshold"],
        "max_threshold": scale_stats["1"]["max_threshold"],
        "max_mean": scale_stats["1"]["max_mean"],
        "max_std": scale_stats["1"]["max_std"],
    }


def paper_score_calibration(source_stats: dict[str, Any], args: argparse.Namespace) -> tuple[float, float]:
    center = float(source_stats.get("max_threshold", source_stats.get("max_mean", 0.0)))
    scale = float(source_stats.get("max_std", 1.0)) * float(args.paper_logit_scale_mult)
    if not math.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    return center, scale


def add_unique_parameter(parameter: nn.Parameter, trainable: list[nn.Parameter], seen: set[int]) -> None:
    if id(parameter) in seen:
        return
    parameter.requires_grad_(True)
    trainable.append(parameter)
    seen.add(id(parameter))


def add_module_parameters(module: nn.Module, trainable: list[nn.Parameter], seen: set[int]) -> None:
    for parameter in module.parameters():
        add_unique_parameter(parameter, trainable, seen)


def add_batch_norm_parameters(module: nn.Module, trainable: list[nn.Parameter], seen: set[int]) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.train()
            for parameter in child.parameters(recurse=False):
                add_unique_parameter(parameter, trainable, seen)


def train_batch_norm_modules(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.train()


def configure_tta_parameters(model: AnomalibReverseDistillationModel, args: argparse.Namespace) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable: list[nn.Parameter] = []
    seen: set[int] = set()
    scope = str(args.tta_param_scope)
    if scope == "bn_only":
        add_batch_norm_parameters(model, trainable, seen)
    elif scope == "bottleneck_decoder_bn_only":
        add_batch_norm_parameters(model.bottleneck, trainable, seen)
        add_batch_norm_parameters(model.decoder, trainable, seen)
    elif scope == "bottleneck_bn_only":
        add_batch_norm_parameters(model.bottleneck, trainable, seen)
    elif scope == "decoder_bn_only":
        add_batch_norm_parameters(model.decoder, trainable, seen)
    elif scope == "bottleneck_full":
        add_module_parameters(model.bottleneck, trainable, seen)
    elif scope == "decoder_full":
        add_module_parameters(model.decoder, trainable, seen)
    elif scope == "view_adapter_decoder":
        add_module_parameters(model.decoder_adapters, trainable, seen)
    else:
        raise ValueError(f"Unsupported TTA parameter scope: {args.tta_param_scope}")
    if not trainable:
        raise RuntimeError(f"No TTA parameters selected for scope={args.tta_param_scope}")
    model.encoder.eval()
    return trainable


def set_tta_train_mode(model: AnomalibReverseDistillationModel, args: argparse.Namespace) -> None:
    model.eval()
    model.encoder.eval()
    scope = str(args.tta_param_scope)
    if scope == "bn_only":
        train_batch_norm_modules(model)
    elif scope == "bottleneck_decoder_bn_only":
        train_batch_norm_modules(model.bottleneck)
        train_batch_norm_modules(model.decoder)
    elif scope == "bottleneck_bn_only":
        train_batch_norm_modules(model.bottleneck)
    elif scope == "decoder_bn_only":
        train_batch_norm_modules(model.decoder)
    elif scope == "bottleneck_full":
        model.bottleneck.train()
    elif scope == "decoder_full":
        model.decoder.train()
    elif scope == "view_adapter_decoder":
        model.decoder_adapters.train()


def source_anchor_parameters(
    model: AnomalibReverseDistillationModel,
    args: argparse.Namespace,
) -> list[tuple[nn.Parameter, torch.Tensor]]:
    bn_scopes = {"bn_only", "bottleneck_decoder_bn_only", "bottleneck_bn_only", "decoder_bn_only"}
    if float(args.tta_bn_anchor_weight) <= 0.0 or str(args.tta_param_scope) not in bn_scopes:
        return []
    anchors: list[tuple[nn.Parameter, torch.Tensor]] = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for parameter in module.parameters(recurse=False):
                if parameter.requires_grad:
                    anchors.append((parameter, parameter.detach().clone()))
    return anchors


def bn_anchor_loss(
    image: torch.Tensor,
    anchors: Sequence[tuple[nn.Parameter, torch.Tensor]],
) -> torch.Tensor:
    if not anchors:
        return image.new_tensor(0.0)
    loss = image.new_tensor(0.0)
    for parameter, source_value in anchors:
        loss = loss + F.mse_loss(parameter, source_value)
    return loss / float(len(anchors))


def adapt_one_sample(
    model: AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
    image: torch.Tensor,
    args: argparse.Namespace,
    anchors: Sequence[tuple[nn.Parameter, torch.Tensor]] | None = None,
    loss_weight: float = 1.0,
    ce_images: torch.Tensor | None = None,
    ce_indices: Sequence[int] | None = None,
    ce_target_labels: Sequence[int] | None = None,
    ce_weights: Sequence[float] | None = None,
    ce_weight_scale: float = 1.0,
) -> tuple[int, float]:
    if int(args.tta_steps) <= 0:
        return 0, float("nan")
    last_loss = float("nan")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for step in range(int(args.tta_steps)):
        set_tta_train_mode(model, args)
        optimizer.zero_grad(set_to_none=True)
        loss = float(loss_weight) * model.reconstruction_loss(image, distill_l2_weight=float(args.tta_distill_l2_weight))
        if str(args.active_extra_loss_mode) != "none" and float(args.active_extra_loss_weight) > 0.0:
            maps_extra = model.anomaly_maps(image, image_size=int(args.crop_size))
            extra_mode = str(args.active_extra_loss_mode)
            if extra_mode == "score_mean":
                extra_loss = maps_extra.mean()
            elif extra_mode == "score_max":
                extra_loss = maps_extra.flatten(1).amax(dim=1).mean()
            elif extra_mode == "map_entropy":
                extra_loss = anomaly_map_sample_entropy(maps_extra).mean()
            elif extra_mode == "map_tv":
                extra_loss = anomaly_map_tv_loss(maps_extra)
            else:
                raise ValueError(f"Unsupported active_extra_loss_mode={args.active_extra_loss_mode}")
            loss = loss + float(args.active_extra_loss_weight) * extra_loss
        if (
            ce_images is not None
            and ce_indices is not None
            and ce_target_labels is not None
            and float(ce_weight_scale) > 0.0
        ):
            index_tensor = torch.as_tensor(ce_indices, device=ce_images.device, dtype=torch.long)
            target = torch.as_tensor(ce_target_labels, device=ce_images.device, dtype=torch.long)
            weights = (
                torch.as_tensor(ce_weights, device=ce_images.device, dtype=torch.float32)
                if ce_weights is not None
                else torch.ones_like(target, dtype=torch.float32)
            )
            maps = model.anomaly_maps(ce_images, image_size=int(args.crop_size))
            logits = binary_logits_from_anomaly_maps(maps)
            ce_loss = F.cross_entropy(logits[index_tensor], target, reduction="none")
            loss = loss + float(args.active_label_ce_weight) * float(ce_weight_scale) * (ce_loss * weights).mean()
        if anchors and float(args.tta_bn_anchor_weight) > 0.0:
            loss = loss + float(args.tta_bn_anchor_weight) * bn_anchor_loss(image, anchors)
        loss.backward()
        if float(args.tta_grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(args.tta_grad_clip))
        optimizer.step()
        last_loss = float(loss.detach().item())
    model.eval()
    return int(args.tta_steps), last_loss


def active_label_ce_items(
    indices: Sequence[int],
    target_labels: Sequence[int],
    args: argparse.Namespace,
    sample_weights: Sequence[float] | None = None,
) -> tuple[list[int], list[int], list[float]]:
    if str(args.active_label_ce_mode) == "none":
        return [], [], []
    index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
    target_array = np.asarray(target_labels, dtype=np.int64).reshape(-1)
    if target_array.size != index_array.size:
        raise ValueError("CE target_labels must match indices")
    weight_array = (
        np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if sample_weights is not None
        else np.ones_like(target_array, dtype=np.float64)
    )
    if weight_array.size != index_array.size:
        raise ValueError("CE sample_weights must match indices")
    if str(args.active_label_ce_mode) == "anomaly_only":
        keep = target_array == 1
        index_array = index_array[keep]
        target_array = target_array[keep]
        weight_array = weight_array[keep]
    return index_array.astype(np.int64).tolist(), target_array.astype(np.int64).tolist(), weight_array.astype(np.float64).tolist()


def robust_center_scale(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return 0.0, 1.0
    center = float(np.median(array))
    mad = float(np.median(np.abs(array - center)) * 1.4826)
    if not math.isfinite(mad) or mad <= 1e-8:
        q25, q75 = np.quantile(array, [0.25, 0.75])
        mad = float((q75 - q25) / 1.349)
    if not math.isfinite(mad) or mad <= 1e-8:
        mad = float(np.std(array))
    if not math.isfinite(mad) or mad <= 1e-8:
        mad = 1.0
    return center, mad


def sigmoid_float(value: float) -> float:
    clipped = float(np.clip(value, -50.0, 50.0))
    return float(1.0 / (1.0 + math.exp(-clipped)))


def svm_margin_pseudo_ce_weights(
    fit: dict[str, Any] | None,
    current_decisions: np.ndarray | None,
    stream_features: Sequence[np.ndarray],
    pseudo_indices: Sequence[int],
    args: argparse.Namespace,
) -> dict[int, float]:
    indices = [int(index) for index in pseudo_indices]
    if not indices:
        return {}
    if str(args.active_label_ce_pseudo_weight_mode) == "none" or fit is None or current_decisions is None:
        return {index: 1.0 for index in indices}
    if str(args.active_label_ce_pseudo_weight_mode) != "svm_margin":
        raise ValueError(f"Unsupported active_label_ce_pseudo_weight_mode={args.active_label_ce_pseudo_weight_mode}")
    reference = np.asarray(stream_features, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[0] < 2:
        reference_evidence = -np.asarray(current_decisions, dtype=np.float64).reshape(-1)
    else:
        reference_evidence = -boundary_decision(fit, reference)
    center, scale = robust_center_scale(reference_evidence)
    decisions = np.asarray(current_decisions, dtype=np.float64).reshape(-1)
    weights: dict[int, float] = {}
    for index in indices:
        normal_evidence = -float(decisions[index])
        weights[index] = sigmoid_float((normal_evidence - center) / scale)
    return weights


def adapt_active_label_ce(
    model: AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    indices: Sequence[int],
    target_labels: Sequence[int],
    args: argparse.Namespace,
    sample_weights: Sequence[float] | None = None,
    anchors: Sequence[tuple[nn.Parameter, torch.Tensor]] | None = None,
) -> tuple[int, float]:
    if int(args.tta_steps) <= 0 or str(args.active_label_ce_mode) == "none":
        return 0, float("nan")
    index_list, target_list, weight_list = active_label_ce_items(indices, target_labels, args, sample_weights=sample_weights)
    if not index_list:
        return 0, float("nan")
    last_loss = float("nan")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    index_tensor = torch.as_tensor(index_list, device=images.device, dtype=torch.long)
    target = torch.as_tensor(target_list, device=images.device, dtype=torch.long)
    weight = torch.as_tensor(weight_list, device=images.device, dtype=torch.float32)
    for _step in range(int(args.tta_steps)):
        set_tta_train_mode(model, args)
        optimizer.zero_grad(set_to_none=True)
        maps = model.anomaly_maps(images, image_size=int(args.crop_size))
        logits = binary_logits_from_anomaly_maps(maps)
        ce_loss = F.cross_entropy(logits[index_tensor], target, reduction="none")
        loss = float(args.active_label_ce_weight) * (ce_loss * weight).mean()
        if anchors and float(args.tta_bn_anchor_weight) > 0.0:
            loss = loss + float(args.tta_bn_anchor_weight) * bn_anchor_loss(images[index_tensor], anchors)
        loss.backward()
        if float(args.tta_grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(args.tta_grad_clip))
        optimizer.step()
        last_loss = float(loss.detach().item())
    model.eval()
    return int(args.tta_steps), last_loss


def adapt_one_sample_signed(
    model: AnomalibReverseDistillationModel,
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
        set_tta_train_mode(model, args)
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


@torch.no_grad()
def update_ema_model(ema_model: AnomalibReverseDistillationModel, model: AnomalibReverseDistillationModel, decay: float) -> None:
    decay = float(decay)
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for key, value in ema_state.items():
        source = model_state[key]
        if torch.is_floating_point(value):
            value.mul_(decay).add_(source.detach(), alpha=1.0 - decay)
        else:
            value.copy_(source)


def standardized_active_features(features: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray([np.asarray(feature, dtype=np.float64).reshape(-1) for feature in features], dtype=np.float64)
    center = np.median(array, axis=0)
    scale = np.std(array, axis=0)
    fallback = np.std(array)
    if not math.isfinite(float(fallback)) or fallback <= 1e-8:
        fallback = 1.0
    scale = np.where(scale > 1e-8, scale, fallback)
    return array, center.astype(np.float64), scale.astype(np.float64)


def active_boundary_raw_decision(fit: dict[str, Any], scaled: np.ndarray) -> np.ndarray:
    model_type = str(fit["model_type"])
    if model_type == "linear_svm":
        if "linear_scaled_coef" in fit:
            coef = np.asarray(fit["linear_scaled_coef"], dtype=np.float64).reshape(-1)
            intercept = float(fit["linear_scaled_intercept"])
            return scaled @ coef + intercept
        model = fit["model"]
        decision = model.decision_function(scaled).reshape(-1).astype(np.float64)
        if int(model.classes_[-1]) != 1:
            decision = -decision
        return decision
    if model_type == "svdd":
        center = np.asarray(fit["svdd_center"], dtype=np.float64).reshape(1, -1)
        radius2 = float(fit["svdd_radius2"])
        return np.sum((scaled - center) ** 2, axis=1).astype(np.float64) - radius2
    raise ValueError(f"Unsupported boundary model: {model_type}")


def finish_boundary_fit(
    fit: dict[str, Any],
    feature_array: np.ndarray,
    train_x: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> dict[str, Any]:
    raw_train = active_boundary_raw_decision(fit, train_x)
    if str(fit["model_type"]) == "linear_svm":
        decision_scale = 1.0
    else:
        q25, q75 = np.quantile(raw_train, [0.25, 0.75])
        decision_scale = float(q75 - q25)
        if decision_scale <= 1e-8:
            decision_scale = float(np.std(raw_train))
        if decision_scale <= 1e-8:
            decision_scale = 1.0
    fit["center"] = center.tolist()
    fit["scale"] = scale.tolist()
    fit["decision_scale"] = float(decision_scale)
    fit["feature_dim"] = int(feature_array.shape[1])
    if str(fit["model_type"]) == "linear_svm":
        coef = np.asarray(fit["linear_scaled_coef"], dtype=np.float64).reshape(-1)
        scale_array = np.asarray(scale, dtype=np.float64).reshape(-1)
        center_array = np.asarray(center, dtype=np.float64).reshape(-1)
        raw_coef = coef / scale_array
        raw_intercept = float(fit["linear_scaled_intercept"]) - float(np.dot(center_array, raw_coef))
        fit["linear_raw_coef"] = raw_coef.tolist()
        fit["linear_raw_intercept"] = float(raw_intercept)
    return fit


def fit_active_boundary(
    features: Sequence[np.ndarray],
    labels: Sequence[int],
    sample_weights: Sequence[float],
    model_name: str,
) -> dict[str, Any] | None:
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    weight_array = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    keep = np.isfinite(weight_array) & (weight_array > 0.0)
    if not np.all(keep):
        feature_list = list(features)
        features = [feature_list[index] for index, use in enumerate(keep) if bool(use)]
        label_array = label_array[keep]
        weight_array = weight_array[keep]
    if len(features) < 2:
        return None
    feature_array, center, scale = standardized_active_features(features)
    train_x = (feature_array - center.reshape(1, -1)) / scale.reshape(1, -1)
    if str(model_name) == "linear_svm":
        if np.unique(label_array).size < 2:
            return None
        try:
            model = SVC(kernel="linear", C=1.0, class_weight="balanced")
            model.fit(train_x, label_array, sample_weight=weight_array)
        except Exception:
            return None
        class_sign = 1.0 if int(model.classes_[-1]) == 1 else -1.0
        coef = np.asarray(model.coef_[0], dtype=np.float64) * class_sign
        intercept = float(model.intercept_[0]) * class_sign
        fit = {
            "model_type": "linear_svm",
            "model": model,
            "coef_norm": float(np.linalg.norm(coef)),
            "coef_score_max": float(coef[0]) if coef.size else float("nan"),
            "intercept": float(intercept),
            "linear_scaled_coef": coef.tolist(),
            "linear_scaled_intercept": float(intercept),
        }
        return finish_boundary_fit(fit, feature_array, train_x, center, scale)
    if str(model_name) == "svdd":
        normal_mask = label_array == 0
        if int(np.sum(normal_mask)) < 2:
            return None
        normal_x = train_x[normal_mask]
        normal_w = weight_array[normal_mask]
        normal_w = normal_w / max(float(np.sum(normal_w)), 1e-8)
        svdd_center = np.sum(normal_x * normal_w.reshape(-1, 1), axis=0)
        dist2 = np.sum((normal_x - svdd_center.reshape(1, -1)) ** 2, axis=1)
        radius2 = float(np.quantile(dist2, 0.90))
        fit = {
            "model_type": "svdd",
            "svdd_center": svdd_center.tolist(),
            "svdd_radius2": radius2,
            "coef_norm": float("nan"),
            "coef_score_max": float("nan"),
            "intercept": -radius2,
        }
        return finish_boundary_fit(fit, feature_array, train_x, center, scale)
    raise ValueError(f"Unsupported boundary model: {model_name}")


def boundary_decision(fit: dict[str, Any], features: np.ndarray) -> np.ndarray:
    feature_array = np.asarray(features, dtype=np.float64)
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(1, -1)
    if str(fit.get("model_type")) == "linear_svm" and "linear_raw_coef" in fit:
        coef = np.asarray(fit["linear_raw_coef"], dtype=np.float64).reshape(-1)
        raw = feature_array @ coef + float(fit["linear_raw_intercept"])
        return raw / max(float(fit.get("decision_scale", 1.0)), 1e-8)
    center = np.asarray(fit["center"], dtype=np.float64).reshape(1, -1)
    scale = np.asarray(fit["scale"], dtype=np.float64).reshape(1, -1)
    scaled = (feature_array - center) / scale
    raw = active_boundary_raw_decision(fit, scaled)
    return raw / max(float(fit.get("decision_scale", 1.0)), 1e-8)


def ema_linear_boundary_fit(
    previous: dict[str, Any] | None,
    current: dict[str, Any] | None,
    decay: float,
) -> dict[str, Any] | None:
    if current is None:
        return previous
    if float(decay) <= 0.0 or str(current.get("model_type")) != "linear_svm":
        return current
    if previous is None or str(previous.get("model_type")) != "linear_svm":
        return copy.deepcopy(current)
    prev_coef = np.asarray(previous.get("linear_raw_coef", []), dtype=np.float64).reshape(-1)
    curr_coef = np.asarray(current.get("linear_raw_coef", []), dtype=np.float64).reshape(-1)
    if prev_coef.shape != curr_coef.shape:
        return copy.deepcopy(current)
    blended = copy.deepcopy(current)
    alpha = float(decay)
    coef = alpha * prev_coef + (1.0 - alpha) * curr_coef
    intercept = alpha * float(previous.get("linear_raw_intercept", 0.0)) + (
        1.0 - alpha
    ) * float(current.get("linear_raw_intercept", 0.0))
    blended["linear_raw_coef"] = coef.tolist()
    blended["linear_raw_intercept"] = float(intercept)
    return blended


def active_query_index_from_decisions(decisions: np.ndarray, mode: str, query_count: int) -> int:
    decision_array = np.asarray(decisions, dtype=np.float64).reshape(-1)
    if decision_array.size == 0:
        return 0
    mode = str(mode)
    if mode == "below_nearest":
        candidates = np.flatnonzero(decision_array <= 0.0)
    elif mode == "above_nearest":
        candidates = np.flatnonzero(decision_array >= 0.0)
    elif mode == "alternate_nearest":
        alternate_step = max(0, int(query_count) - 1)
        prefer_below = alternate_step % 2 == 0
        candidates = np.flatnonzero(decision_array <= 0.0) if prefer_below else np.flatnonzero(decision_array >= 0.0)
        if candidates.size == 0:
            candidates = np.flatnonzero(decision_array >= 0.0) if prefer_below else np.flatnonzero(decision_array <= 0.0)
    else:
        candidates = np.arange(decision_array.size, dtype=np.int64)
    if candidates.size == 0:
        candidates = np.arange(decision_array.size, dtype=np.int64)
    return int(candidates[np.argmin(np.abs(decision_array[candidates]))])


def append_active_normal_memory(
    memory: list[np.ndarray],
    feature: np.ndarray,
    max_size: int,
) -> None:
    if int(max_size) <= 0:
        return
    memory.append(np.asarray(feature, dtype=np.float64).reshape(-1).copy())
    overflow = len(memory) - int(max_size)
    if overflow > 0:
        del memory[:overflow]


def active_memory_normal_nearest_weights(
    fit: dict[str, Any],
    features: np.ndarray,
    active_normal_memory: Sequence[np.ndarray],
    indices: Sequence[int],
    args: argparse.Namespace,
) -> dict[int, float]:
    if str(args.active_memory_weight_mode) != "normal_nearest":
        return {int(index): 1.0 for index in indices}
    if len(active_normal_memory) < int(args.active_memory_min_size):
        return {int(index): 1.0 for index in indices}
    if not indices:
        return {}
    center = np.asarray(fit["center"], dtype=np.float64).reshape(1, -1)
    scale = np.asarray(fit["scale"], dtype=np.float64).reshape(1, -1)
    candidate = np.asarray(features[np.asarray(indices, dtype=np.int64)], dtype=np.float64)
    memory = np.asarray(active_normal_memory, dtype=np.float64)
    candidate_scaled = (candidate - center) / scale
    memory_scaled = (memory - center) / scale
    diff = candidate_scaled[:, None, :] - memory_scaled[None, :, :]
    distances = np.sqrt(np.mean(diff * diff, axis=2))
    nearest = np.min(distances, axis=1)
    sigma = max(float(args.active_memory_distance_scale), 1e-8)
    reliability = np.exp(-0.5 * (nearest / sigma) ** 2)
    weight_min = float(args.active_memory_weight_min)
    weight_max = float(args.active_memory_weight_max)
    weights = np.clip(weight_min + (weight_max - weight_min) * reliability, weight_min, weight_max)
    return {int(index): float(weight) for index, weight in zip(indices, weights, strict=True)}


def select_stream_tail_pseudo_labels(
    scores: Sequence[float],
    fraction: float,
    excluded_ids: set[int],
    selected_ids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if score_array.size == 0 or float(fraction) <= 0.0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    available = [idx for idx in range(int(score_array.size)) if idx not in excluded_ids and idx not in selected_ids]
    if not available:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    k = max(1, int(math.ceil(float(score_array.size) * float(fraction))))
    available_sorted_low = sorted(available, key=lambda idx: (float(score_array[idx]), idx))
    lower = available_sorted_low[: min(k, len(available_sorted_low))]
    remaining = [idx for idx in available if idx not in set(lower)]
    available_sorted_high = sorted(remaining, key=lambda idx: (-float(score_array[idx]), idx))
    upper = available_sorted_high[: min(k, len(available_sorted_high))]
    indices = np.asarray(lower + upper, dtype=np.int64)
    pseudo = np.asarray([0] * len(lower) + [1] * len(upper), dtype=np.int64)
    return indices, pseudo


def select_stream_confidence_pseudo_labels(
    fit: dict[str, Any] | None,
    stream_features: Sequence[np.ndarray],
    margin: float,
    excluded_ids: set[int],
    selected_ids: set[int],
    max_per_side: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Select pseudo-labels using SVM decision confidence instead of score percentile.

    Samples with decision < -margin are pseudo-normal (label=0).
    Samples with decision > +margin are pseudo-anomaly (label=1).
    If max_per_side > 0, limits each side to the most confident max_per_side samples.
    """
    if fit is None or len(stream_features) == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    feature_array = np.asarray(
        [np.asarray(f, dtype=np.float64).reshape(-1) for f in stream_features],
        dtype=np.float64,
    )
    decisions = boundary_decision(fit, feature_array)
    available = [
        idx for idx in range(len(stream_features))
        if idx not in excluded_ids and idx not in selected_ids
    ]
    if not available:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    avail_set = set(available)
    margin = float(margin)
    # Pseudo-normal: decision < -margin (confident normal side)
    normal_candidates = [
        (idx, float(decisions[idx])) for idx in available if float(decisions[idx]) < -margin
    ]
    normal_candidates.sort(key=lambda x: x[1])  # most negative first = most confident
    # Pseudo-anomaly: decision > +margin (confident anomaly side)
    anomaly_candidates = [
        (idx, float(decisions[idx])) for idx in available if float(decisions[idx]) > margin
    ]
    anomaly_candidates.sort(key=lambda x: -x[1])  # most positive first = most confident
    if max_per_side > 0:
        normal_candidates = normal_candidates[:max_per_side]
        anomaly_candidates = anomaly_candidates[:max_per_side]
    lower = [idx for idx, _ in normal_candidates]
    upper = [idx for idx, _ in anomaly_candidates]
    indices = np.asarray(lower + upper, dtype=np.int64)
    pseudo = np.asarray([0] * len(lower) + [1] * len(upper), dtype=np.int64)
    return indices, pseudo


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    metrics: dict[str, float] = {}
    if np.unique(labels).size < 2:
        metrics["auroc"] = float("nan")
        metrics["ap"] = float("nan")
        metrics["fpr95"] = float("nan")
    else:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
        metrics["ap"] = float(average_precision_score(labels, scores))
        fpr, tpr, _ = roc_curve(labels, scores)
        above = np.flatnonzero(tpr >= 0.95)
        metrics["fpr95"] = float(fpr[int(above[0])]) if above.size else 1.0
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    metrics["f1_max"] = float(np.nanmax(f1)) if f1.size else float("nan")
    return metrics


@torch.no_grad()
def evaluate_source(
    model: AnomalibReverseDistillationModel,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels_all.append(batch["label"].cpu().numpy().astype(np.int64))
        maps_np = anomaly_maps_np(model, images, image_size=int(args.crop_size))
        scores_all.append(scores_from_maps(maps_np))
    return np.concatenate(labels_all), np.concatenate(scores_all)


def evaluate_tta(
    source_model: AnomalibReverseDistillationModel,
    loader: DataLoader,
    source_stats: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model = copy.deepcopy(source_model).to(device)
    score_model = copy.deepcopy(source_model).to(device) if str(args.tta_score_source) == "adapted_ema" else None
    trainable = configure_tta_parameters(model, args)
    anchors = source_anchor_parameters(model, args)
    optimizer = torch.optim.Adam(trainable, lr=float(args.tta_lr))
    active_features: list[np.ndarray] = []
    active_labels: list[int] = []
    active_weights: list[float] = []
    active_normal_memory: list[np.ndarray] = []
    stream_reference_features: list[np.ndarray] = []
    stream_tail_features: list[np.ndarray] = []
    stream_tail_scores: list[float] = []
    stream_tail_labels: list[int] = []
    stream_tail_selected_ids: set[int] = set()
    tail_buffer_enabled = str(args.active_svm_tail_start_mode) == "always"
    boundary_ema_fit: dict[str, Any] | None = None
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
    tail_total = 0
    tail_normal_total = 0
    tail_anomaly_total = 0
    tail_correct_total = 0
    active_memory_weight_values: list[float] = []
    active_memory_selected_weight_sum = 0.0
    active_memory_selected_normal_weight_sum = 0.0
    active_label_ce_pseudo_weight_values: list[float] = []
    active_label_ce_steps = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy().astype(np.int64)
        scorer = source_model
        if str(args.tta_score_source) == "adapted":
            scorer = model
        elif str(args.tta_score_source) == "adapted_ema" and score_model is not None:
            scorer = score_model
        with torch.no_grad():
            if feature_family_needs_layer_maps(str(args.feature_family)):
                selection_maps, selection_layer_maps = anomaly_maps_with_layer_maps_np(
                    scorer,
                    images,
                    image_size=int(args.crop_size),
                )
            else:
                selection_maps = anomaly_maps_np(scorer, images, image_size=int(args.crop_size))
                selection_layer_maps = None
        selection_scores = scores_from_maps(selection_maps)
        selection_features = map_stat_features_from_maps(
            selection_maps,
            source_stats,
            feature_family=str(args.feature_family),
            layer_maps_np=selection_layer_maps,
        )
        for feature in selection_features:
            stream_reference_features.append(feature.copy())
        batch_global_ids = np.full(int(labels.shape[0]), -1, dtype=np.int64)
        if tail_buffer_enabled:
            batch_global_ids = np.arange(
                len(stream_tail_scores),
                len(stream_tail_scores) + int(labels.shape[0]),
                dtype=np.int64,
            )
            for feature, score, label in zip(selection_features, selection_scores, labels, strict=True):
                stream_tail_features.append(feature.copy())
                stream_tail_scores.append(float(score))
                stream_tail_labels.append(int(label))

        fit_before = fit_active_boundary(active_features, active_labels, active_weights, str(args.boundary_model))
        if float(args.active_svm_boundary_ema_decay) > 0.0 and boundary_ema_fit is not None:
            fit_before = boundary_ema_fit
        if fit_before is None:
            query_index = int(np.argsort(selection_scores)[len(selection_scores) // 2])
        else:
            decisions_before = boundary_decision(fit_before, selection_features)
            query_index = active_query_index_from_decisions(
                decisions_before,
                mode=str(args.active_svm_query_side_mode),
                query_count=int(active_label_total),
            )
        query_label = int(labels[query_index])
        active_features.append(selection_features[query_index].copy())
        active_labels.append(query_label)
        active_weights.append(1.0)
        if query_label == 0 and str(args.active_memory_weight_mode) != "none":
            append_active_normal_memory(
                active_normal_memory,
                selection_features[query_index],
                max_size=int(args.active_memory_max_size),
            )
        active_label_total += 1
        active_label_normal_total += int(query_label == 0)
        active_label_anomaly_total += int(query_label == 1)

        if (
            not tail_buffer_enabled
            and str(args.active_svm_tail_start_mode) == "after_confident_pseudo_normal"
            and float(args.active_svm_tail_pseudo_label_fraction) > 0.0
        ):
            fit_probe = fit_active_boundary(active_features, active_labels, active_weights, str(args.boundary_model))
            if fit_probe is not None:
                probe_decisions = boundary_decision(fit_probe, selection_features)
                if bool(np.any(probe_decisions <= -float(args.active_svm_confidence_threshold))):
                    tail_buffer_enabled = True
                    batch_global_ids = np.arange(
                        len(stream_tail_scores),
                        len(stream_tail_scores) + int(labels.shape[0]),
                        dtype=np.int64,
                    )
                    for feature, score, label in zip(selection_features, selection_scores, labels, strict=True):
                        stream_tail_features.append(feature.copy())
                        stream_tail_scores.append(float(score))
                        stream_tail_labels.append(int(label))

        if str(args.active_svm_tail_pseudo_label_mode) == "svm_confidence" and fit_before is not None:
            max_per_side = max(1, int(math.ceil(len(stream_tail_scores) * float(args.active_svm_tail_pseudo_label_fraction)))) if float(args.active_svm_tail_pseudo_label_fraction) > 0.0 else 0
            tail_ids, tail_pseudo = select_stream_confidence_pseudo_labels(
                fit_before,
                stream_tail_features,
                margin=float(args.active_svm_tail_confidence_margin),
                excluded_ids={int(batch_global_ids[query_index])},
                selected_ids=stream_tail_selected_ids,
                max_per_side=max_per_side,
            )
        else:
            tail_ids, tail_pseudo = select_stream_tail_pseudo_labels(
                stream_tail_scores,
                fraction=float(args.active_svm_tail_pseudo_label_fraction),
                excluded_ids={int(batch_global_ids[query_index])},
                selected_ids=stream_tail_selected_ids,
            )
        tail_current_positions: list[int] = []
        for global_id, pseudo_label in zip(tail_ids, tail_pseudo, strict=True):
            stream_tail_selected_ids.add(int(global_id))
            true_label = int(stream_tail_labels[int(global_id)])
            active_features.append(stream_tail_features[int(global_id)].copy())
            active_labels.append(int(pseudo_label))
            active_weights.append(
                float(args.active_svm_lower_tail_pseudo_normal_weight)
                if int(pseudo_label) == 0
                else float(args.active_svm_upper_tail_pseudo_anomaly_weight)
            )
            tail_total += 1
            tail_normal_total += int(pseudo_label == 0)
            tail_anomaly_total += int(pseudo_label == 1)
            tail_correct_total += int(true_label == int(pseudo_label))
            current = np.flatnonzero(batch_global_ids == int(global_id))
            if current.size:
                tail_current_positions.append(int(current[0]))

        fit_after = fit_active_boundary(active_features, active_labels, active_weights, str(args.boundary_model))
        if float(args.active_svm_boundary_ema_decay) > 0.0:
            boundary_ema_fit = ema_linear_boundary_fit(
                boundary_ema_fit,
                fit_after,
                decay=float(args.active_svm_boundary_ema_decay),
            )
            fit_after = boundary_ema_fit
        adapt_indices: list[int] = []
        pseudo_adapt_indices: list[int] = []
        if query_label == 0:
            adapt_indices.append(int(query_index))
        decisions: np.ndarray | None = None
        if fit_after is not None:
            decisions = boundary_decision(fit_after, selection_features)
            excluded = {int(query_index), *tail_current_positions}
            pseudo_mask = decisions <= -float(args.active_svm_confidence_threshold)
            for index, selected in enumerate(pseudo_mask.tolist()):
                if selected and int(index) not in excluded:
                    adapt_indices.append(int(index))
                    pseudo_adapt_indices.append(int(index))
        adapt_indices = sorted(set(adapt_indices))
        adapt_weight_by_index = {int(index): 1.0 for index in adapt_indices}
        if fit_after is not None and pseudo_adapt_indices:
            adapt_weight_by_index.update(
                active_memory_normal_nearest_weights(
                    fit_after,
                    selection_features,
                    active_normal_memory,
                    sorted(set(pseudo_adapt_indices)),
                    args,
                ),
            )
        ce_indices: list[int] = []
        ce_targets: list[int] = []
        ce_sample_weights: list[float] = []
        if str(args.active_label_ce_targets) in {"active", "active_pseudo"}:
            ce_indices.append(int(query_index))
            ce_targets.append(int(query_label))
            ce_sample_weights.append(1.0)
        pseudo_ce_weight_by_index = svm_margin_pseudo_ce_weights(
            fit_after,
            decisions,
            stream_reference_features,
            sorted(set(pseudo_adapt_indices)),
            args,
        )
        if str(args.active_label_ce_targets) in {"pseudo", "active_pseudo"}:
            for index in sorted(set(pseudo_adapt_indices)):
                ce_indices.append(int(index))
                ce_targets.append(0)
                pseudo_weight = float(pseudo_ce_weight_by_index.get(int(index), 1.0))
                ce_sample_weights.append(pseudo_weight)
                active_label_ce_pseudo_weight_values.append(pseudo_weight)
        ce_indices, ce_targets, ce_sample_weights = active_label_ce_items(
            ce_indices,
            ce_targets,
            args,
            sample_weights=ce_sample_weights,
        )
        use_joint_ce = str(args.active_label_ce_update) == "joint" and bool(ce_indices)
        if use_joint_ce and adapt_indices:
            joint_ce_weight_scale = (
                1.0
                if str(args.active_label_ce_joint_scale_mode) == "full"
                else 1.0 / float(len(adapt_indices))
            )
        else:
            joint_ce_weight_scale = 0.0
        pseudo_only_indices = sorted(set(pseudo_adapt_indices))
        if pseudo_only_indices:
            pseudo_only_total += len(pseudo_only_indices)
            pseudo_only_normal_total += int(np.sum(labels[np.asarray(pseudo_only_indices, dtype=np.int64)] == 0))
        if adapt_indices:
            selected_total += len(adapt_indices)
            selected_normal_total += int(np.sum(labels[np.asarray(adapt_indices, dtype=np.int64)] == 0))
            for index in adapt_indices:
                weight = float(adapt_weight_by_index.get(int(index), 1.0))
                active_memory_weight_values.append(weight)
                active_memory_selected_weight_sum += weight
                active_memory_selected_normal_weight_sum += weight * float(labels[int(index)] == 0)
                steps, _ = adapt_one_sample(
                    model,
                    optimizer,
                    images[index : index + 1],
                    args,
                    anchors=anchors,
                    loss_weight=weight,
                    ce_images=images if use_joint_ce else None,
                    ce_indices=ce_indices if use_joint_ce else None,
                    ce_target_labels=ce_targets if use_joint_ce else None,
                    ce_weights=ce_sample_weights if use_joint_ce else None,
                    ce_weight_scale=joint_ce_weight_scale,
                )
                optimizer_steps += int(steps)
                if use_joint_ce:
                    active_label_ce_steps += int(steps)
        if str(args.active_label_ce_update) == "separate":
            ce_steps, _ = adapt_active_label_ce(
                model,
                optimizer,
                images,
                ce_indices,
                ce_targets,
                args,
                sample_weights=ce_sample_weights,
                anchors=anchors,
            )
            active_label_ce_steps += int(ce_steps)
            optimizer_steps += int(ce_steps)
        else:
            ce_steps = int(args.tta_steps) if use_joint_ce and adapt_indices else 0
        if score_model is not None and (adapt_indices or ce_steps > 0):
            update_ema_model(score_model, model, decay=float(args.tta_score_ema_decay))
        with torch.no_grad():
            tta_maps = anomaly_maps_np(model, images, image_size=int(args.crop_size))
        labels_all.append(labels)
        scores_all.append(scores_from_maps(tta_maps))

    stats = {
        "selected_total": float(selected_total),
        "selected_normal_total": float(selected_normal_total),
        "pseudo_only_total": float(pseudo_only_total),
        "pseudo_only_normal_total": float(pseudo_only_normal_total),
        "optimizer_steps": float(optimizer_steps),
        "active_label_total": float(active_label_total),
        "active_label_normal_total": float(active_label_normal_total),
        "active_label_anomaly_total": float(active_label_anomaly_total),
        "active_label_ce_steps": float(active_label_ce_steps),
        "active_label_ce_pseudo_weight_mean": (
            float(np.mean(active_label_ce_pseudo_weight_values)) if active_label_ce_pseudo_weight_values else float("nan")
        ),
        "active_label_ce_pseudo_weight_min": (
            float(np.min(active_label_ce_pseudo_weight_values)) if active_label_ce_pseudo_weight_values else float("nan")
        ),
        "active_label_ce_pseudo_weight_max": (
            float(np.max(active_label_ce_pseudo_weight_values)) if active_label_ce_pseudo_weight_values else float("nan")
        ),
        "tail_total": float(tail_total),
        "tail_normal_total": float(tail_normal_total),
        "tail_anomaly_total": float(tail_anomaly_total),
        "tail_correct_total": float(tail_correct_total),
        "feature_dim": float(selection_features.shape[1]) if "selection_features" in locals() else 0.0,
        "active_memory_size_final": float(len(active_normal_memory)),
        "active_memory_weight_mean": float(np.mean(active_memory_weight_values)) if active_memory_weight_values else float("nan"),
        "active_memory_weight_min": float(np.min(active_memory_weight_values)) if active_memory_weight_values else float("nan"),
        "active_memory_weight_max": float(np.max(active_memory_weight_values)) if active_memory_weight_values else float("nan"),
        "active_memory_selected_weight_sum": float(active_memory_selected_weight_sum),
        "active_memory_selected_normal_weight_sum": float(active_memory_selected_normal_weight_sum),
    }
    return np.concatenate(labels_all), np.concatenate(scores_all), stats


def evaluate_label_tta(
    source_model: AnomalibReverseDistillationModel,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    method: str,
    source_stats: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model = copy.deepcopy(source_model).to(device).eval()
    score_source = str(args.tta_score_source)
    use_score_ema = score_source == "adapted_ema"
    if use_score_ema:
        score_model = copy.deepcopy(model).to(device).eval()
    elif score_source == "frozen":
        score_model = copy.deepcopy(source_model).to(device).eval()
    else:
        score_model = model
    trainable = configure_tta_parameters(model, args)
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
    paper_state: dict[str, Any] = {}
    atta_anchor_features: list[np.ndarray] = []
    faithful_paper_methods = {"atta_paper_faithful", "eatta_paper_faithful"}
    paper_logit_center = float("nan")
    paper_logit_scale = float("nan")
    if method in faithful_paper_methods:
        if source_stats is None:
            raise ValueError(f"{method} requires source normal score statistics")
        paper_logit_center, paper_logit_scale = paper_score_calibration(source_stats, args)

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy().astype(np.int64)
        with torch.no_grad():
            maps_np = anomaly_maps_np(score_model, images, image_size=int(args.crop_size))
        scores = scores_from_maps(maps_np)
        map_entropy = entropy_from_maps(maps_np)
        paper_entropy = binary_entropy_from_scores(scores)

        if method == "atta_paper_hybrid":
            with torch.no_grad():
                current_maps = anomaly_maps_np(model, images, image_size=int(args.crop_size))
                source_maps = anomaly_maps_np(source_model, images, image_size=int(args.crop_size))
            current_scores = scores_from_maps(current_maps)
            current_entropy = binary_entropy_from_scores(current_scores)
            features = pooled_encoder_features_np(model, images)
            query_indices = simatta_incremental_query_indices(
                features=features,
                uncertainty=current_entropy,
                state=paper_state,
                args=args,
                batch_index=batch_index,
            )
            simatta_add_anchors(paper_state, features, query_indices, args)

            source_scores = scores_from_maps(source_maps)
            source_entropy = binary_entropy_from_scores(source_scores)
            pseudo_labels = binary_pseudo_labels_from_scores(source_scores)
            pseudo_indices = paper_low_entropy_indices(
                entropy=source_entropy,
                excluded_indices=query_indices,
                args=args,
            )
            high_memory = paper_state.setdefault("simatta_high_memory", [])
            low_memory = paper_state.setdefault("simatta_low_memory", [])
            paper_memory_append(high_memory, images, query_indices, labels, max_size=int(args.paper_memory_size))
            paper_memory_append(low_memory, images, pseudo_indices, pseudo_labels, max_size=int(args.paper_memory_size))

            active_total += int(query_indices.size)
            active_normal += int(np.sum(labels[query_indices] == 0)) if query_indices.size > 0 else 0
            active_anomaly += int(np.sum(labels[query_indices] == 1)) if query_indices.size > 0 else 0
            if pseudo_indices.size > 0:
                pseudo_only_total += int(pseudo_indices.size)
                pseudo_only_normal_total += int(np.sum(labels[pseudo_indices] == 0))
            selected_indices = sorted(set(query_indices.tolist()) | set(pseudo_indices.tolist()))
            if selected_indices:
                selected_total += len(selected_indices)
                selected_normal_total += int(np.sum(labels[np.asarray(selected_indices, dtype=np.int64)] == 0))
                steps, _ = adapt_simatta_memory_batch(
                    model,
                    optimizer,
                    low_memory=low_memory,
                    high_memory=high_memory,
                    args=args,
                    logit_center=None,
                    logit_scale=None,
                    device=device,
                )
                optimizer_steps += int(steps)
            with torch.no_grad():
                tta_maps = anomaly_maps_np(model, images, image_size=int(args.crop_size))
            labels_all.append(labels)
            scores_all.append(scores_from_maps(tta_maps))
            continue

        if method == "eatta_paper_hybrid":
            with torch.no_grad():
                current_maps = anomaly_maps_np(model, images, image_size=int(args.crop_size))
            current_scores = scores_from_maps(current_maps)
            current_entropy = binary_entropy_from_scores(current_scores)
            pseudo_labels = binary_pseudo_labels_from_scores(current_scores)
            diff = feature_perturbation_batch_confidence_diffs(
                model=model,
                images=images,
                scores=current_scores,
                args=args,
            )
            query_indices = eatta_history_select_indices(
                diff=diff,
                pseudo_labels=pseudo_labels,
                state=paper_state,
                args=args,
            )
            entropy_indices = paper_low_entropy_indices(
                entropy=current_entropy,
                excluded_indices=query_indices,
                args=args,
                entropy_q=float(args.eatta_pseudo_normal_entropy_q),
                max_fraction=float(args.eatta_pseudo_normal_max_fraction),
            )
            if entropy_indices.size > 0:
                pseudo_only_total += int(entropy_indices.size)
                pseudo_only_normal_total += int(np.sum(labels[entropy_indices] == 0))
            active_total += int(query_indices.size)
            active_normal += int(np.sum(labels[query_indices] == 0)) if query_indices.size > 0 else 0
            active_anomaly += int(np.sum(labels[query_indices] == 1)) if query_indices.size > 0 else 0
            selected_indices = sorted(set(query_indices.tolist()) | set(entropy_indices.tolist()))
            if selected_indices:
                selected_total += len(selected_indices)
                selected_normal_total += int(np.sum(labels[np.asarray(selected_indices, dtype=np.int64)] == 0))
                steps, _ = adapt_paper_label_batch(
                    model,
                    optimizer,
                    images,
                    labels,
                    active_indices=query_indices,
                    pseudo_normal_indices=np.asarray([], dtype=np.int64),
                    pseudo_target_labels=None,
                    entropy_indices=entropy_indices,
                    args=args,
                    state=paper_state,
                    method="eatta_paper",
                )
                optimizer_steps += int(steps)
            with torch.no_grad():
                tta_maps = anomaly_maps_np(model, images, image_size=int(args.crop_size))
            labels_all.append(labels)
            scores_all.append(scores_from_maps(tta_maps))
            continue

        if method == "atta_paper_faithful":
            with torch.no_grad():
                current_maps = anomaly_maps_np(model, images, image_size=int(args.crop_size))
                source_maps = anomaly_maps_np(source_model, images, image_size=int(args.crop_size))
            current_scores = scores_from_maps(current_maps)
            current_entropy = binary_entropy_from_fixed_scores(
                current_scores,
                center=paper_logit_center,
                scale=paper_logit_scale,
            )
            features = pooled_encoder_features_np(model, images)
            query_indices = simatta_incremental_query_indices(
                features=features,
                uncertainty=current_entropy,
                state=paper_state,
                args=args,
                batch_index=batch_index,
            )
            simatta_add_anchors(paper_state, features, query_indices, args)

            source_scores = scores_from_maps(source_maps)
            source_entropy = binary_entropy_from_fixed_scores(
                source_scores,
                center=paper_logit_center,
                scale=paper_logit_scale,
            )
            pseudo_labels = binary_pseudo_labels_from_fixed_scores(
                source_scores,
                center=paper_logit_center,
                scale=paper_logit_scale,
            )
            pseudo_indices = paper_low_entropy_indices(
                entropy=source_entropy,
                excluded_indices=query_indices,
                args=args,
            )
            high_memory = paper_state.setdefault("simatta_high_memory", [])
            low_memory = paper_state.setdefault("simatta_low_memory", [])
            paper_memory_append(high_memory, images, query_indices, labels, max_size=int(args.paper_memory_size))
            paper_memory_append(low_memory, images, pseudo_indices, pseudo_labels, max_size=int(args.paper_memory_size))

            active_total += int(query_indices.size)
            active_normal += int(np.sum(labels[query_indices] == 0)) if query_indices.size > 0 else 0
            active_anomaly += int(np.sum(labels[query_indices] == 1)) if query_indices.size > 0 else 0
            if pseudo_indices.size > 0:
                pseudo_only_total += int(pseudo_indices.size)
                pseudo_only_normal_total += int(np.sum(labels[pseudo_indices] == 0))
            selected_indices = sorted(set(query_indices.tolist()) | set(pseudo_indices.tolist()))
            if selected_indices:
                selected_total += len(selected_indices)
                selected_normal_total += int(np.sum(labels[np.asarray(selected_indices, dtype=np.int64)] == 0))
                steps, _ = adapt_simatta_memory_batch(
                    model,
                    optimizer,
                    low_memory=low_memory,
                    high_memory=high_memory,
                    args=args,
                    logit_center=paper_logit_center,
                    logit_scale=paper_logit_scale,
                    device=device,
                )
                optimizer_steps += int(steps)
            with torch.no_grad():
                tta_maps = anomaly_maps_np(model, images, image_size=int(args.crop_size))
            labels_all.append(labels)
            scores_all.append(scores_from_maps(tta_maps))
            continue

        if method == "eatta_paper_faithful":
            with torch.no_grad():
                current_maps = anomaly_maps_np(model, images, image_size=int(args.crop_size))
            current_scores = scores_from_maps(current_maps)
            current_entropy = binary_entropy_from_fixed_scores(
                current_scores,
                center=paper_logit_center,
                scale=paper_logit_scale,
            )
            pseudo_labels = binary_pseudo_labels_from_fixed_scores(
                current_scores,
                center=paper_logit_center,
                scale=paper_logit_scale,
            )
            diff = feature_perturbation_confidence_diffs(
                model=model,
                images=images,
                scores=current_scores,
                args=args,
                logit_center=paper_logit_center,
                logit_scale=paper_logit_scale,
            )
            query_indices = eatta_history_select_indices(
                diff=diff,
                pseudo_labels=pseudo_labels,
                state=paper_state,
                args=args,
            )
            entropy_indices = paper_low_entropy_indices(
                entropy=current_entropy,
                excluded_indices=query_indices,
                args=args,
                entropy_q=float(args.eatta_pseudo_normal_entropy_q),
                max_fraction=float(args.eatta_pseudo_normal_max_fraction),
            )
            if entropy_indices.size > 0:
                pseudo_only_total += int(entropy_indices.size)
                pseudo_only_normal_total += int(np.sum(labels[entropy_indices] == 0))
            active_total += int(query_indices.size)
            active_normal += int(np.sum(labels[query_indices] == 0)) if query_indices.size > 0 else 0
            active_anomaly += int(np.sum(labels[query_indices] == 1)) if query_indices.size > 0 else 0
            selected_indices = sorted(set(query_indices.tolist()) | set(entropy_indices.tolist()))
            if selected_indices:
                selected_total += len(selected_indices)
                selected_normal_total += int(np.sum(labels[np.asarray(selected_indices, dtype=np.int64)] == 0))
                steps, _ = adapt_paper_label_batch(
                    model,
                    optimizer,
                    images,
                    labels,
                    active_indices=query_indices,
                    pseudo_normal_indices=np.asarray([], dtype=np.int64),
                    pseudo_target_labels=None,
                    entropy_indices=entropy_indices,
                    args=args,
                    state=paper_state,
                    method="eatta_paper",
                    logit_center=paper_logit_center,
                    logit_scale=paper_logit_scale,
                )
                optimizer_steps += int(steps)
            with torch.no_grad():
                tta_maps = anomaly_maps_np(model, images, image_size=int(args.crop_size))
            labels_all.append(labels)
            scores_all.append(scores_from_maps(tta_maps))
            continue

        if method == "atta_paper":
            features = pooled_encoder_features_np(score_model, images)
            query_indices = simatta_query_indices(
                features=features,
                uncertainty=paper_entropy,
                anchor_features=atta_anchor_features,
                args=args,
                batch_index=batch_index,
            )
            for index in query_indices.tolist():
                atta_anchor_features.append(features[int(index)].copy())
            pseudo_labels = binary_pseudo_labels_from_scores(scores)
            pseudo_indices = paper_low_entropy_indices(
                entropy=paper_entropy,
                excluded_indices=query_indices,
                args=args,
            )
            entropy_indices = np.asarray([], dtype=np.int64)
            active_total += int(query_indices.size)
            active_normal += int(np.sum(labels[query_indices] == 0)) if query_indices.size > 0 else 0
            active_anomaly += int(np.sum(labels[query_indices] == 1)) if query_indices.size > 0 else 0
            selected_indices = sorted(set(query_indices.tolist()) | set(pseudo_indices.tolist()))
            if pseudo_indices.size > 0:
                pseudo_only_total += int(pseudo_indices.size)
                pseudo_only_normal_total += int(np.sum(labels[pseudo_indices] == 0))
            if selected_indices:
                selected_total += len(selected_indices)
                selected_normal_total += int(np.sum(labels[np.asarray(selected_indices, dtype=np.int64)] == 0))
                steps, _ = adapt_paper_label_batch(
                    model,
                    optimizer,
                    images,
                    labels,
                    active_indices=query_indices,
                    pseudo_normal_indices=pseudo_indices,
                    pseudo_target_labels=pseudo_labels[pseudo_indices],
                    entropy_indices=entropy_indices,
                    args=args,
                    state=paper_state,
                    method=method,
                )
                optimizer_steps += int(steps)
                if use_score_ema:
                    update_ema_model(score_model, model, decay=float(args.tta_score_ema_decay))
            with torch.no_grad():
                tta_maps = anomaly_maps_np(model, images, image_size=int(args.crop_size))
            labels_all.append(labels)
            scores_all.append(scores_from_maps(tta_maps))
            continue

        if method == "eatta_paper":
            diff = feature_perturbation_diffs(model=score_model, images=images, original_scores=scores, args=args)
            pseudo_labels = binary_pseudo_labels_from_scores(scores)
            query_indices = eatta_history_select_indices(
                diff=diff,
                pseudo_labels=pseudo_labels,
                state=paper_state,
                args=args,
            )
            entropy_indices = paper_low_entropy_indices(
                entropy=paper_entropy,
                excluded_indices=query_indices,
                args=args,
                entropy_q=float(args.eatta_pseudo_normal_entropy_q),
                max_fraction=float(args.eatta_pseudo_normal_max_fraction),
            )
            if entropy_indices.size > 0:
                pseudo_only_total += int(entropy_indices.size)
                pseudo_only_normal_total += int(np.sum(labels[entropy_indices] == 0))
            active_total += int(query_indices.size)
            active_normal += int(np.sum(labels[query_indices] == 0)) if query_indices.size > 0 else 0
            active_anomaly += int(np.sum(labels[query_indices] == 1)) if query_indices.size > 0 else 0
            selected_indices = sorted(set(query_indices.tolist()) | set(entropy_indices.tolist()))
            if selected_indices:
                selected_total += len(selected_indices)
                selected_normal_total += int(np.sum(labels[np.asarray(selected_indices, dtype=np.int64)] == 0))
                steps, _ = adapt_paper_label_batch(
                    model,
                    optimizer,
                    images,
                    labels,
                    active_indices=query_indices,
                    pseudo_normal_indices=np.asarray([], dtype=np.int64),
                    pseudo_target_labels=None,
                    entropy_indices=entropy_indices,
                    args=args,
                    state=paper_state,
                    method=method,
                )
                optimizer_steps += int(steps)
                if use_score_ema:
                    update_ema_model(score_model, model, decay=float(args.tta_score_ema_decay))
            with torch.no_grad():
                tta_maps = anomaly_maps_np(model, images, image_size=int(args.crop_size))
            labels_all.append(labels)
            scores_all.append(scores_from_maps(tta_maps))
            continue

        if method == "eatta" and str(args.eatta_query_mode) == "feature_perturb_sensitivity":
            query = feature_perturbation_query(score_model, images, scores, args, batch_index=batch_index)
        else:
            query = median_score_query(scores)
        query_label = int(labels[query])
        active_total += 1
        active_normal += int(query_label == 0)
        active_anomaly += int(query_label == 1)

        positive_indices: list[int] = []
        pseudo_indices_for_stats: list[int] = []
        if query_label == 0:
            positive_indices.append(int(query))
        if method == "eatta":
            entropy = map_entropy
            generator = torch.Generator(device=images.device)
            generator.manual_seed(int(args.stream_seed) + 10_003 * int(batch_index))
            noisy = images + torch.randn(
                images.shape,
                generator=generator,
                device=images.device,
                dtype=images.dtype,
            ) * float(args.eatta_noise_std)
            with torch.no_grad():
                noisy_maps = anomaly_maps_np(score_model, noisy, image_size=int(args.crop_size))
            stability = np.abs(scores_from_maps(noisy_maps) - scores)
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
        pseudo_indices_for_stats = sorted(set(pseudo_indices_for_stats))
        if pseudo_indices_for_stats:
            pseudo_only_total += len(pseudo_indices_for_stats)
            pseudo_only_normal_total += int(np.sum(labels[np.asarray(pseudo_indices_for_stats, dtype=np.int64)] == 0))
        if positive_indices:
            selected_total += len(positive_indices)
            selected_normal_total += int(np.sum(labels[np.asarray(positive_indices, dtype=np.int64)] == 0))
            for index in positive_indices:
                steps, _ = adapt_one_sample_signed(
                    model,
                    optimizer,
                    images[index : index + 1],
                    args,
                    loss_sign=1.0,
                )
                optimizer_steps += int(steps)
            if use_score_ema:
                update_ema_model(score_model, model, decay=float(args.tta_score_ema_decay))

        with torch.no_grad():
            tta_maps = anomaly_maps_np(model, images, image_size=int(args.crop_size))
        labels_all.append(labels)
        scores_all.append(scores_from_maps(tta_maps))

    return np.concatenate(labels_all), np.concatenate(scores_all), {
        "selected_total": float(selected_total),
        "selected_normal_total": float(selected_normal_total),
        "pseudo_only_total": float(pseudo_only_total),
        "pseudo_only_normal_total": float(pseudo_only_normal_total),
        "optimizer_steps": float(optimizer_steps),
        "active_label_total": float(active_total),
        "active_label_normal_total": float(active_normal),
        "active_label_anomaly_total": float(active_anomaly),
        "paper_logit_center": float(paper_logit_center),
        "paper_logit_scale": float(paper_logit_scale),
    }


def metric_row(
    category: str,
    split: str,
    method: str,
    boundary_model: str,
    metrics: dict[str, float],
    args: argparse.Namespace,
    n_train_normal: int,
    labels: np.ndarray,
    checkpoint: Path,
    stats: dict[str, float] | None = None,
) -> dict[str, Any]:
    stats = stats or {}
    selected_total = int(stats.get("selected_total", 0))
    selected_normal_total = int(stats.get("selected_normal_total", 0))
    pseudo_only_total = int(stats.get("pseudo_only_total", 0))
    pseudo_only_normal_total = int(stats.get("pseudo_only_normal_total", 0))
    tail_total = int(stats.get("tail_total", 0))
    tail_correct = int(stats.get("tail_correct_total", 0))
    return {
        "category": category,
        "split": split,
        "domain": domain_name(split),
        "method": method,
        "boundary_model": boundary_model,
        "feature_family": str(args.feature_family),
        "feature_dim": int(stats.get("feature_dim", 0)),
        "stream_order": str(args.stream_order),
        "seed": int(args.seed),
        "stream_seed": int(args.stream_seed),
        "n_train_normal": int(n_train_normal),
        "n_images": int(labels.shape[0]),
        "n_normal": int(np.sum(labels == 0)),
        "n_anomaly": int(np.sum(labels == 1)),
        "image_auroc": metrics["auroc"],
        "image_ap": metrics["ap"],
        "image_f1_max": metrics["f1_max"],
        "image_fpr95": metrics["fpr95"],
        "selected_pseudo_normal_count": selected_total,
        "selected_pseudo_normal_purity": float(selected_normal_total / selected_total) if selected_total > 0 else float("nan"),
        "pseudo_only_count": pseudo_only_total,
        "pseudo_only_purity": float(pseudo_only_normal_total / pseudo_only_total) if pseudo_only_total > 0 else float("nan"),
        "optimizer_steps": int(stats.get("optimizer_steps", 0)),
        "active_label_count": int(stats.get("active_label_total", 0)),
        "active_label_normal_count": int(stats.get("active_label_normal_total", 0)),
        "active_label_anomaly_count": int(stats.get("active_label_anomaly_total", 0)),
        "active_tail_pseudo_label_count": tail_total,
        "active_tail_pseudo_label_normal_count": int(stats.get("tail_normal_total", 0)),
        "active_tail_pseudo_label_anomaly_count": int(stats.get("tail_anomaly_total", 0)),
        "active_tail_pseudo_label_accuracy": float(tail_correct / tail_total) if tail_total > 0 else float("nan"),
        "active_svm_confidence_threshold": float(args.active_svm_confidence_threshold),
        "active_svm_query_side_mode": str(args.active_svm_query_side_mode),
        "active_svm_boundary_ema_decay": float(args.active_svm_boundary_ema_decay),
        "active_svm_tail_start_mode": str(args.active_svm_tail_start_mode),
        "active_svm_tail_pseudo_label_fraction": float(args.active_svm_tail_pseudo_label_fraction),
        "active_svm_tail_pseudo_label_mode": str(args.active_svm_tail_pseudo_label_mode),
        "active_svm_tail_confidence_margin": float(args.active_svm_tail_confidence_margin),
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
        "active_label_ce_joint_scale_mode": str(args.active_label_ce_joint_scale_mode),
        "active_label_ce_pseudo_weight_mode": str(args.active_label_ce_pseudo_weight_mode),
        "active_label_ce_pseudo_weight_mean": float(stats.get("active_label_ce_pseudo_weight_mean", float("nan"))),
        "active_label_ce_pseudo_weight_min": float(stats.get("active_label_ce_pseudo_weight_min", float("nan"))),
        "active_label_ce_pseudo_weight_max": float(stats.get("active_label_ce_pseudo_weight_max", float("nan"))),
        "active_label_ce_steps": int(stats.get("active_label_ce_steps", 0)),
        "active_extra_loss_mode": str(args.active_extra_loss_mode),
        "active_extra_loss_weight": float(args.active_extra_loss_weight),
        "tta_lr": float(args.tta_lr),
        "tta_steps": int(args.tta_steps),
        "tta_param_scope": str(args.tta_param_scope),
        "tta_score_source": str(args.tta_score_source),
        "tta_score_ema_decay": float(args.tta_score_ema_decay),
        "tta_distill_l2_weight": float(args.tta_distill_l2_weight),
        "tta_bn_anchor_weight": float(args.tta_bn_anchor_weight),
        "paper_oracle_num": int(args.paper_oracle_num),
        "paper_score_center_q": float(args.paper_score_center_q),
        "paper_logit_scale_mult": float(args.paper_logit_scale_mult),
        "paper_logit_center": float(stats.get("paper_logit_center", float("nan"))),
        "paper_logit_scale": float(stats.get("paper_logit_scale", float("nan"))),
        "paper_memory_size": int(args.paper_memory_size),
        "paper_replay_batch_size": int(args.paper_replay_batch_size),
        "paper_memory_replay_mode": str(args.paper_memory_replay_mode),
        "atta_entropy_high_threshold": float(args.atta_entropy_high_threshold),
        "atta_pseudo_normal_entropy_q": float(args.atta_pseudo_normal_entropy_q),
        "atta_pseudo_normal_max_fraction": float(args.atta_pseudo_normal_max_fraction),
        "atta_cluster_increase": int(args.atta_cluster_increase),
        "atta_cluster_budget": int(args.atta_cluster_budget),
        "eatta_pseudo_normal_entropy_q": float(args.eatta_pseudo_normal_entropy_q),
        "eatta_pseudo_normal_max_fraction": float(args.eatta_pseudo_normal_max_fraction),
        "eatta_entropy_margin": float(args.eatta_entropy_margin),
        "eatta_class_history": int(args.eatta_class_history),
        "eatta_gnd_momentum": float(args.eatta_gnd_momentum),
        "eatta_feature_perturb_std": float(args.eatta_feature_perturb_std),
        "checkpoint_path": str(checkpoint),
    }


def tta_variant_suffix(args: argparse.Namespace) -> str:
    suffix = {
        "bn_only": "-BNOnly",
        "bottleneck_decoder_bn_only": "-BottleDecoderBNOnly",
        "bottleneck_bn_only": "-BottleneckBNOnly",
        "decoder_bn_only": "-DecoderBNOnly",
        "bottleneck_full": "-BottleneckFull",
        "decoder_full": "-DecoderFull",
        "view_adapter_decoder": "-ViewAdapterDecoder",
    }[str(args.tta_param_scope)]
    if float(args.tta_distill_l2_weight) > 0.0:
        suffix += f"-DistillL2w{safe_name(f'{float(args.tta_distill_l2_weight):.3g}')}"
    if float(args.tta_bn_anchor_weight) > 0.0:
        suffix += f"-BNAnchorw{safe_name(f'{float(args.tta_bn_anchor_weight):.3g}')}"
    if float(args.active_svm_boundary_ema_decay) > 0.0:
        suffix += f"-SVMEMA{safe_name(f'{float(args.active_svm_boundary_ema_decay):.3g}')}"
    if str(args.active_svm_tail_start_mode) != "always":
        suffix += f"-TailStart{safe_name(str(args.active_svm_tail_start_mode))}"
    if str(args.active_svm_tail_pseudo_label_mode) != "fraction":
        suffix += f"-TailMode{safe_name(str(args.active_svm_tail_pseudo_label_mode))}M{safe_name(f'{float(args.active_svm_tail_confidence_margin):.3g}')}"
    if str(args.active_memory_weight_mode) != "none":
        suffix += f"-ActiveMem{safe_name(str(args.active_memory_weight_mode))}"
    if str(args.active_label_ce_mode) != "none":
        suffix += (
            f"-ActiveCE{safe_name(str(args.active_label_ce_mode))}"
            f"-{safe_name(str(args.active_label_ce_targets))}"
            f"w{safe_name(f'{float(args.active_label_ce_weight):.3g}')}"
            f"-{safe_name(str(args.active_label_ce_update))}"
        )
        if str(args.active_label_ce_update) == "joint" and str(args.active_label_ce_joint_scale_mode) != "mean":
            suffix += f"-JointScale{safe_name(str(args.active_label_ce_joint_scale_mode))}"
        if str(args.active_label_ce_pseudo_weight_mode) != "none":
            suffix += f"-PseudoW{safe_name(str(args.active_label_ce_pseudo_weight_mode))}"
    if str(args.active_extra_loss_mode) != "none" and float(args.active_extra_loss_weight) > 0.0:
        suffix += (
            f"-Extra{safe_name(str(args.active_extra_loss_mode))}"
            f"w{safe_name(f'{float(args.active_extra_loss_weight):.3g}')}"
        )
    return suffix


def paper_faithful_variant_suffix(args: argparse.Namespace) -> str:
    suffix = {
        "bn_only": "-BNOnly",
        "bottleneck_decoder_bn_only": "-BottleDecoderBNOnly",
        "bottleneck_bn_only": "-BottleneckBNOnly",
        "decoder_bn_only": "-DecoderBNOnly",
        "bottleneck_full": "-BottleneckFull",
        "decoder_full": "-DecoderFull",
        "view_adapter_decoder": "-ViewAdapterDecoder",
    }[str(args.tta_param_scope)]
    suffix += f"-SourceQ{safe_name(f'{float(args.paper_score_center_q):.3g}')}"
    suffix += f"-Mem{int(args.paper_memory_size)}"
    suffix += f"-Replay{safe_name(str(args.paper_memory_replay_mode))}{int(args.paper_replay_batch_size)}"
    if float(args.tta_distill_l2_weight) > 0.0:
        suffix += f"-DistillL2w{safe_name(f'{float(args.tta_distill_l2_weight):.3g}')}"
    if float(args.tta_bn_anchor_weight) > 0.0:
        suffix += f"-BNAnchorw{safe_name(f'{float(args.tta_bn_anchor_weight):.3g}')}"
    return suffix


def paper_hybrid_variant_suffix(args: argparse.Namespace) -> str:
    suffix = {
        "bn_only": "-BNOnly",
        "bottleneck_decoder_bn_only": "-BottleDecoderBNOnly",
        "bottleneck_bn_only": "-BottleneckBNOnly",
        "decoder_bn_only": "-DecoderBNOnly",
        "bottleneck_full": "-BottleneckFull",
        "decoder_full": "-DecoderFull",
        "view_adapter_decoder": "-ViewAdapterDecoder",
    }[str(args.tta_param_scope)]
    suffix += f"-BatchMedian-Mem{int(args.paper_memory_size)}"
    suffix += f"-Replay{safe_name(str(args.paper_memory_replay_mode))}{int(args.paper_replay_batch_size)}"
    if float(args.tta_distill_l2_weight) > 0.0:
        suffix += f"-DistillL2w{safe_name(f'{float(args.tta_distill_l2_weight):.3g}')}"
    if float(args.tta_bn_anchor_weight) > 0.0:
        suffix += f"-BNAnchorw{safe_name(f'{float(args.tta_bn_anchor_weight):.3g}')}"
    return suffix


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def requested_methods(args: argparse.Namespace) -> list[str]:
    methods: list[str] = []
    for method in args.methods:
        normalized = "source" if str(method) == "base" else str(method)
        if normalized not in methods:
            methods.append(normalized)
    return methods


def run_category(category: str, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    data_root = Path(args.data_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    methods = requested_methods(args)
    train_dir = train_split_dir(data_root, category)
    train_dataset = RobustADImageDataset(
        train_dir,
        category=category,
        resize_size=int(args.resize_size),
        crop_size=int(args.crop_size),
        label_filter=0,
        stream_order="sequential",
        seed=int(args.seed),
    )
    source_model, checkpoint = load_model(checkpoint_root, category=category, device=device)
    paper_faithful_methods = {"atta_paper_faithful", "eatta_paper_faithful"}
    paper_hybrid_methods = {"atta_paper_hybrid", "eatta_paper_hybrid"}
    source_stats = (
        compute_source_stats(source_model, train_dataset, args, device)
        if (
            ("active_boundary" in methods and feature_family_needs_source_stats(str(args.feature_family)))
            or bool(paper_faithful_methods.intersection(methods))
        )
        else None
    )
    rows: list[dict[str, Any]] = []
    for split_dir in test_split_dirs(data_root, category, args.splits):
        split = split_token(split_dir, category)
        dataset = RobustADImageDataset(
            split_dir,
            category=category,
            resize_size=int(args.resize_size),
            crop_size=int(args.crop_size),
            stream_order=str(args.stream_order),
            seed=int(args.stream_seed),
        )
        loader = make_loader(dataset, batch_size=int(args.batch_size), num_workers=int(args.num_workers), shuffle=False)
        source_labels: np.ndarray | None = None
        source_scores: np.ndarray | None = None
        for method in methods:
            if method == "source":
                if source_labels is None or source_scores is None:
                    source_labels, source_scores = evaluate_source(source_model, loader, args, device)
                metrics = binary_metrics(source_labels, source_scores)
                rows.append(
                    metric_row(
                        category=category,
                        split=split,
                        method="AnomalibRD4AD_source",
                        boundary_model="source",
                        metrics=metrics,
                        args=args,
                        n_train_normal=len(train_dataset),
                        labels=source_labels,
                        checkpoint=checkpoint,
                    ),
                )
                print(f"[source] {category}/{split} auroc={metrics['auroc'] * 100:.2f}", flush=True)
            elif method in {
                "atta",
                "eatta",
                "atta_paper",
                "eatta_paper",
                "atta_paper_faithful",
                "eatta_paper_faithful",
                "atta_paper_hybrid",
                "eatta_paper_hybrid",
            }:
                labels, scores, stats = evaluate_label_tta(
                    source_model,
                    loader,
                    args,
                    device,
                    method=method,
                    source_stats=source_stats,
                )
                metrics = binary_metrics(labels, scores)
                method_label = {
                    "atta": "ATTA-AD",
                    "eatta": "EATTA-AD",
                    "atta_paper": "ATTA-PaperAD",
                    "eatta_paper": "EATTA-PaperAD",
                    "atta_paper_faithful": "ATTA-PaperAD-Faithful",
                    "eatta_paper_faithful": "EATTA-PaperAD-Faithful",
                    "atta_paper_hybrid": "ATTA-PaperAD-Hybrid",
                    "eatta_paper_hybrid": "EATTA-PaperAD-Hybrid",
                }[method]
                if method in paper_faithful_methods:
                    suffix = paper_faithful_variant_suffix(args)
                elif method in paper_hybrid_methods:
                    suffix = paper_hybrid_variant_suffix(args)
                else:
                    suffix = tta_variant_suffix(args)
                rows.append(
                    metric_row(
                        category=category,
                        split=split,
                        method=f"AnomalibRD4AD_{method_label}{suffix}",
                        boundary_model=method,
                        metrics=metrics,
                        args=args,
                        n_train_normal=len(train_dataset),
                        labels=labels,
                        checkpoint=checkpoint,
                        stats=stats,
                    ),
                )
                print(
                    f"[{method}] {category}/{split} auroc={metrics['auroc'] * 100:.2f} "
                    f"selected={int(stats['selected_total'])}",
                    flush=True,
                )
            elif method == "active_boundary":
                if feature_family_needs_source_stats(str(args.feature_family)) and source_stats is None:
                    source_stats = compute_source_stats(source_model, train_dataset, args, device)
                labels, scores, stats = evaluate_tta(source_model, loader, source_stats or {}, args, device)
                metrics = binary_metrics(labels, scores)
                rows.append(
                    metric_row(
                        category=category,
                        split=split,
                        method=(
                            f"AnomalibRD4AD_ActiveBoundary-{args.boundary_model}-"
                            f"{safe_name(str(args.feature_family))}-SPTail1{tta_variant_suffix(args)}"
                        ),
                        boundary_model=str(args.boundary_model),
                        metrics=metrics,
                        args=args,
                        n_train_normal=len(train_dataset),
                        labels=labels,
                        checkpoint=checkpoint,
                        stats=stats,
                    ),
                )
                print(
                    f"[{args.boundary_model}] {category}/{split} feature={args.feature_family} "
                    f"auroc={metrics['auroc'] * 100:.2f} selected={int(stats['selected_total'])}",
                    flush=True,
                )
            else:
                raise ValueError(f"Unsupported method: {method}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare linear SVM and SVDD active boundaries on RobustAD RD4AD BTTA.")
    parser.add_argument("--data-root", default="data/robustad")
    parser.add_argument("--checkpoint-root", default="pretrained/rd4ad_robustad_anomalib_table2_seed0_20260506/rd4ad_checkpoints")
    parser.add_argument("--output-root", default="outputs/robustad_svdd_boundary_compare")
    parser.add_argument("--categories", nargs="+", default=["MetalParts", "PCB", "PiledBags"])
    parser.add_argument("--splits", nargs="*", default=[])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "source",
            "base",
            "atta",
            "eatta",
            "atta_paper",
            "eatta_paper",
            "atta_paper_faithful",
            "eatta_paper_faithful",
            "atta_paper_hybrid",
            "eatta_paper_hybrid",
            "active_boundary",
        ),
        default=["source", "active_boundary"],
    )
    parser.add_argument("--boundary-model", choices=("linear_svm", "svdd"), default="svdd")
    parser.add_argument(
        "--feature-family",
        choices=(
            "base5",
            "score1d",
            "final_raw3",
            "multiscale",
            "multiscale_nosource",
            "final_multiscale_raw9",
            "spatial",
            "frequency",
            "frequency_nosource",
            "multiscale_frequency",
            "multiscale_frequency_nosource",
            "current_12d",
            "current_scale1freq_6d",
            "score_freq_4d",
            "score_mean_freq_5d",
            "score_std_freq_5d",
            "scale1_no_low_5d",
            "current_no_low_11d",
            "current_edge_13d",
            "current_scale8_15d",
            "current_msfreq_18d",
            "encoder_raw9",
            "encoder_raw9_finalfreq3",
            "hybrid_21d",
            "msfreq_no_low",
            "compact_msfreq",
            "edge_msfreq",
            "all",
        ),
        default="base5",
        help=(
            "Boundary feature family: base5; multiscale=3 scales x 5 stats; "
            "score1d=max anomaly-map score only; "
            "final_raw3=final map max/mean/std only; "
            "spatial=base5 plus connected-component features; "
            "frequency=base5 plus DCT energy features; "
            "multiscale_frequency=multiscale plus DCT energy features; "
            "multiscale_nosource=scale max/mean/std only, without source-normal stats; "
            "final_multiscale_raw9=alias for multiscale_nosource; "
            "frequency_nosource=DCT low/high/centroid only, without source-normal stats; "
            "multiscale_frequency_nosource=scale max/mean/std plus DCT, without source-normal stats; "
            "current_12d=alias for multiscale_frequency_nosource; "
            "current_scale1freq_6d=scale1 max/mean/std plus final DCT; "
            "score_freq_4d=score max plus final DCT; "
            "score_mean_freq_5d=score max/mean plus final DCT; "
            "score_std_freq_5d=score max/std plus final DCT; "
            "scale1_no_low_5d=scale1 max/mean/std plus high/centroid DCT; "
            "current_no_low_11d=current_12d without low-frequency DCT; "
            "current_edge_13d=current_12d plus high-low DCT contrast; "
            "current_scale8_15d=scale 1/2/4/8 raw stats plus final DCT; "
            "current_msfreq_18d=scale 1/2/4 raw stats plus per-scale DCT; "
            "encoder_raw9=encoder layer residual maps x max/mean/std; "
            "encoder_raw9_finalfreq3=encoder_raw9 plus final-map DCT; "
            "hybrid_21d=current_12d plus encoder_raw9; "
            "msfreq_no_low drops low-frequency DCT; compact_msfreq keeps base5, scale max/area/z, and high/centroid DCT; "
            "edge_msfreq keeps scale std and high-low DCT contrast; "
            "all=multiscale plus spatial/frequency."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stream-seed", type=int, default=0)
    parser.add_argument("--stream-order", choices=("random", "sequential"), default="random")
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
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
    parser.add_argument("--tta-distill-l2-weight", type=float, default=0.0)
    parser.add_argument("--tta-bn-anchor-weight", type=float, default=0.0)
    parser.add_argument("--tta-score-source", choices=("frozen", "adapted", "adapted_ema"), default="adapted_ema")
    parser.add_argument("--tta-score-ema-decay", type=float, default=0.95)
    parser.add_argument("--active-svm-source-pixel-q", type=float, default=0.99)
    parser.add_argument("--active-svm-confidence-threshold", type=float, default=0.10)
    parser.add_argument(
        "--active-svm-query-side-mode",
        choices=("boundary_nearest", "below_nearest", "above_nearest", "alternate_nearest"),
        default="boundary_nearest",
    )
    parser.add_argument("--active-svm-boundary-ema-decay", type=float, default=0.0)
    parser.add_argument(
        "--active-svm-tail-start-mode",
        choices=("always", "after_confident_pseudo_normal"),
        default="always",
    )
    parser.add_argument("--active-svm-tail-pseudo-label-fraction", type=float, default=0.01)
    parser.add_argument(
        "--active-svm-tail-pseudo-label-mode",
        choices=("fraction", "svm_confidence"),
        default="fraction",
    )
    parser.add_argument("--active-svm-tail-confidence-margin", type=float, default=0.5)
    parser.add_argument("--active-svm-lower-tail-pseudo-normal-weight", type=float, default=0.5)
    parser.add_argument("--active-svm-upper-tail-pseudo-anomaly-weight", type=float, default=0.2)
    parser.add_argument("--active-memory-weight-mode", choices=("none", "normal_nearest"), default="none")
    parser.add_argument("--active-memory-max-size", type=int, default=64)
    parser.add_argument("--active-memory-min-size", type=int, default=3)
    parser.add_argument("--active-memory-distance-scale", type=float, default=1.0)
    parser.add_argument("--active-memory-weight-min", type=float, default=0.25)
    parser.add_argument("--active-memory-weight-max", type=float, default=1.5)
    parser.add_argument("--active-label-ce-mode", choices=("none", "anomaly_only", "all"), default="none")
    parser.add_argument("--active-label-ce-targets", choices=("active", "pseudo", "active_pseudo"), default="active")
    parser.add_argument("--active-label-ce-weight", type=float, default=1.0)
    parser.add_argument("--active-label-ce-update", choices=("separate", "joint"), default="separate")
    parser.add_argument("--active-label-ce-joint-scale-mode", choices=("mean", "full"), default="mean")
    parser.add_argument("--active-label-ce-pseudo-weight-mode", choices=("none", "svm_margin"), default="none")
    parser.add_argument(
        "--active-extra-loss-mode",
        choices=("none", "score_mean", "score_max", "map_entropy", "map_tv"),
        default="none",
        help="Optional extra adaptation loss applied to selected active-boundary samples.",
    )
    parser.add_argument("--active-extra-loss-weight", type=float, default=0.0)
    parser.add_argument("--eatta-noise-std", type=float, default=0.01)
    parser.add_argument(
        "--eatta-query-mode",
        choices=("median_score", "feature_perturb_sensitivity"),
        default="median_score",
    )
    parser.add_argument("--eatta-feature-perturb-std", type=float, default=0.01)
    parser.add_argument("--eatta-pseudo-normal-score-q", type=float, default=0.25)
    parser.add_argument("--eatta-pseudo-normal-entropy-q", type=float, default=0.4)
    parser.add_argument("--eatta-pseudo-normal-stability-q", type=float, default=0.5)
    parser.add_argument("--eatta-pseudo-normal-max-fraction", type=float, default=0.4)
    parser.add_argument("--paper-oracle-num", type=int, default=1)
    parser.add_argument("--paper-score-center-q", type=float, default=0.99)
    parser.add_argument("--paper-logit-scale-mult", type=float, default=1.0)
    parser.add_argument("--paper-memory-size", type=int, default=512)
    parser.add_argument("--paper-replay-batch-size", type=int, default=4)
    parser.add_argument(
        "--paper-memory-replay-mode",
        choices=("recent", "full"),
        default="recent",
        help="recent replays the latest bounded memory subset; full accumulates gradients over all retained ATTA memory.",
    )
    parser.add_argument("--atta-entropy-high-threshold", type=float, default=0.01)
    parser.add_argument("--atta-pseudo-normal-entropy-q", type=float, default=0.4)
    parser.add_argument("--atta-pseudo-normal-max-fraction", type=float, default=0.4)
    parser.add_argument("--atta-cluster-increase", type=int, default=1)
    parser.add_argument("--atta-cluster-budget", type=int, default=300)
    parser.add_argument("--eatta-entropy-margin", type=float, default=0.2772588722239781)
    parser.add_argument("--eatta-class-history", type=int, default=1)
    parser.add_argument("--eatta-gnd-momentum", type=float, default=0.8)
    args = parser.parse_args()
    if args.stream_order != "random":
        print("[warn] project default is random stream; non-random stream requested explicitly", file=sys.stderr)
    if args.eatta_noise_std < 0.0:
        parser.error("--eatta-noise-std must be >= 0")
    if args.eatta_feature_perturb_std < 0.0:
        parser.error("--eatta-feature-perturb-std must be >= 0")
    if not 0.0 <= float(args.active_svm_boundary_ema_decay) < 1.0:
        parser.error("--active-svm-boundary-ema-decay must be in [0, 1)")
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
    if args.active_extra_loss_weight < 0.0:
        parser.error("--active-extra-loss-weight must be >= 0")
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
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: list[dict[str, Any]] = []
    for category in args.categories:
        rows.extend(run_category(category, args, device))
        write_csv(output_root / "robustad_detailed.csv", rows)
    write_csv(output_root / "robustad_detailed.csv", rows)
    target_rows = [row for row in rows if row["split"] != "test0"]
    summary_parts = []
    for method in sorted({str(row["method"]) for row in target_rows}):
        method_rows = [row for row in target_rows if row["method"] == method]
        summary_parts.append(f"{method}={np.nanmean([r['image_auroc'] for r in method_rows]) * 100:.3f}")
    print(
        f"[done] boundary={args.boundary_model} target_mean " + " ".join(summary_parts) + " "
        f"elapsed={time.time() - started:.1f}s output={output_root / 'robustad_detailed.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()

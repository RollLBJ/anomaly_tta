from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.special import gammaln
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models.resnet import BasicBlock, Bottleneck
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_robustad_rd4ad_bt_recsvm import (  # noqa: E402
    DETAIL_COLUMNS,
    RobustADImageDataset,
    ScoreOnlyBTRecSVMSelector,
    binary_metrics,
    domain_name,
    json_safe,
    make_loader,
    read_existing,
    resolve_path,
    safe_name,
    select_svm_reliability_replay_score_only,
    set_seed,
    split_token,
    test_split_dirs,
    train_split_dir,
    trim_history_batches,
    write_csv,
)

DETAIL_COLUMNS = tuple(
    dict.fromkeys(
        (
            *DETAIL_COLUMNS,
            "tta_method",
            "tta_param_scope",
            "entropy_margin",
            "atta_oracle_num",
            "atta_query_strategy",
            "eatta_oracle_num",
            "eatta_noise_std",
            "eatta_weight_momentum",
            "eatta_class_history",
            "eatta_pseudo_normal_score_q",
            "eatta_pseudo_normal_entropy_q",
            "eatta_pseudo_normal_stability_q",
            "eatta_pseudo_normal_max_fraction",
            "sar_rho",
            "tta_pos_scale_aug",
            "tta_pos_scale_translate_frac",
            "tta_pos_scale_scale_low",
            "tta_pos_scale_scale_high",
            "tta_pos_scale_infer_agg",
            "tta_score_source",
            "tta_score_ema_decay",
            "selection_score_mean",
            "active_svm_feature_mode",
            "active_svm_include_encoder3_feature",
            "active_svm_encoder_feature_layers",
            "active_boundary_model",
            "active_query_mode",
            "active_query_perturb_mode",
            "active_query_perturb_candidate_count",
            "active_query_perturb_std",
            "active_svm_source_pixel_q",
            "active_svm_confidence_threshold",
            "active_svm_tail_scope",
            "active_svm_tail_pseudo_label_fraction",
            "active_svm_lower_tail_pseudo_normal_weight",
            "active_svm_upper_tail_pseudo_anomaly_weight",
            "active_anomaly_reverse_lr",
            "active_anomaly_target_mode",
            "active_label_count",
            "active_label_normal_count",
            "active_label_anomaly_count",
            "active_label_anomaly_reverse_count",
            "active_tail_pseudo_label_count",
            "active_tail_pseudo_label_normal_count",
            "active_tail_pseudo_label_anomaly_count",
            "active_tail_pseudo_label_accuracy",
        ),
    ),
)


TABLE2_RD4AD_AUROC = {
    ("MetalParts", "test0"): 89.59,
    ("MetalParts", "test1"): 96.77,
    ("MetalParts", "test2"): 87.43,
    ("MetalParts", "test3"): 90.47,
    ("MetalParts", "test4"): 80.05,
    ("MetalParts", "test5"): 71.99,
    ("MetalParts", "test6"): 67.16,
    ("PiledBags", "test0"): 93.01,
    ("PiledBags", "test1"): 85.41,
    ("PiledBags", "test2"): 50.34,
    ("PiledBags", "test3"): 74.85,
    ("PiledBags", "test4"): 55.13,
    ("PiledBags", "test5"): 76.33,
    ("PCB", "test0"): 99.74,
    ("PCB", "test1"): 57.36,
    ("PCB", "test2"): 66.58,
    ("PCB", "test3"): 54.65,
    ("PCB", "test4"): 47.98,
    ("PCB", "test5"): 70.56,
}

ACTIVE_SVM_MAP_FEATURE_NAMES = (
    "score_max",
    "score_top1_mean",
    "score_top5_mean",
    "score_mean",
    "score_std",
    "area_ratio",
    "source_z_max",
    "source_z_top5",
)

ACTIVE_SVM_ENCODER_LAYER_IDS = (1, 2, 3)
ACTIVE_SVM_ENCODER_LAYER_FEATURE_NAMES = {
    layer: f"encoder{layer}_z_l2" for layer in ACTIVE_SVM_ENCODER_LAYER_IDS
}
ACTIVE_SVM_ENCODER3_FEATURE_NAMES = (ACTIVE_SVM_ENCODER_LAYER_FEATURE_NAMES[3],)


def active_svm_encoder_feature_layers(args: argparse.Namespace) -> tuple[int, ...]:
    layers = {int(layer) for layer in getattr(args, "active_svm_encoder_feature_layers", [])}
    if bool(getattr(args, "active_svm_include_encoder3_feature", False)):
        layers.add(3)
    return tuple(layer for layer in ACTIVE_SVM_ENCODER_LAYER_IDS if layer in layers)


def active_svm_all_feature_names(args: argparse.Namespace) -> list[str]:
    names = list(ACTIVE_SVM_MAP_FEATURE_NAMES)
    for layer in active_svm_encoder_feature_layers(args):
        names.append(ACTIVE_SVM_ENCODER_LAYER_FEATURE_NAMES[layer])
    return names


def active_svm_feature_names(args: argparse.Namespace) -> list[str]:
    if str(args.active_svm_feature_mode) != "map_stats":
        return ["score_max"]
    drop_features = set(str(name) for name in getattr(args, "active_svm_drop_features", []))
    return [name for name in active_svm_all_feature_names(args) if name not in drop_features]


def active_svm_feature_indices(args: argparse.Namespace) -> list[int]:
    selected = active_svm_feature_names(args)
    index_by_name = {name: index for index, name in enumerate(active_svm_all_feature_names(args))}
    return [index_by_name[name] for name in selected]


def active_tail_weight_token(value: float) -> str:
    return safe_name(f"{float(value):g}".replace("-", "m").replace(".", "p"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Anomalib-equivalent Reverse Distillation RD4AD on RobustAD, optionally with BT-RecSVM TTA.",
    )
    parser.add_argument("--data-root", default="data/robustad")
    parser.add_argument("--output-root", default="outputs/robustad_anomalib_rd4ad")
    parser.add_argument("--categories", nargs="+", default=["MetalParts", "PCB", "PiledBags"])
    parser.add_argument("--splits", nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stream-seed",
        type=int,
        default=None,
        help="Optional seed for target stream ordering and active query randomness; RD4AD checkpoint matching still uses --seed.",
    )
    parser.add_argument("--stream-order", choices=("random", "sequential"), default="random")
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--layers", nargs="+", default=["layer1", "layer2", "layer3"])
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cache-train-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rd4ad-epochs", type=int, default=200)
    parser.add_argument("--rd4ad-lr", type=float, default=0.005)
    parser.add_argument("--rd4ad-beta1", type=float, default=0.5)
    parser.add_argument("--rd4ad-beta2", type=float, default=0.99)
    parser.add_argument("--rd4ad-grad-accum-steps", type=int, default=1)
    parser.add_argument("--rd4ad-max-train-batches", type=int, default=0)
    parser.add_argument("--force-rd4ad-train", action="store_true")
    parser.add_argument("--run-tta", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--tta-method",
        choices=("bt_recsvm", "tent", "sar", "atta", "eatta", "active_svm_boundary"),
        default="bt_recsvm",
    )
    parser.add_argument("--tta-lr", type=float, default=3e-3)
    parser.add_argument("--tta-steps", type=int, default=10)
    parser.add_argument("--tta-grad-clip", type=float, default=1.0)
    parser.add_argument("--tta-adapt-batch-size", type=int, default=8)
    parser.add_argument("--tta-score-source", choices=("frozen", "adapted", "adapted_ema"), default="frozen")
    parser.add_argument("--tta-score-ema-decay", type=float, default=0.95)
    parser.add_argument("--active-svm-feature-mode", choices=("score", "map_stats"), default="score")
    parser.add_argument(
        "--active-boundary-model",
        choices=("linear_svm", "student_t", "gmm", "gda", "qda", "kde", "logistic", "isotonic_logistic"),
        default="linear_svm",
    )
    parser.add_argument("--active-svm-source-pixel-q", type=float, default=0.99)
    parser.add_argument("--active-svm-drop-features", nargs="*", default=[])
    parser.add_argument("--active-svm-include-encoder3-feature", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--active-svm-encoder-feature-layers", nargs="*", choices=("1", "2", "3"), default=[])
    parser.add_argument(
        "--active-query-mode",
        choices=("boundary", "boundary_perturb_high"),
        default="boundary",
        help=(
            "Active query policy. boundary keeps the existing nearest-boundary query; "
            "boundary_perturb_high first keeps boundary-near candidates, then queries the "
            "candidate with the largest weak-perturbation anomaly score."
        ),
    )
    parser.add_argument("--active-query-perturb-candidate-count", type=int, default=4)
    parser.add_argument(
        "--active-query-perturb-mode",
        choices=("gaussian_noise", "brightness", "contrast", "blur"),
        default="gaussian_noise",
    )
    parser.add_argument("--active-query-perturb-std", type=float, default=0.05)
    parser.add_argument("--active-svm-confidence-threshold", type=float, default=0.25)
    parser.add_argument("--active-svm-tail-scope", choices=("batch", "stream_past"), default="batch")
    parser.add_argument("--active-svm-tail-pseudo-label-fraction", type=float, default=0.0)
    parser.add_argument("--active-svm-lower-tail-pseudo-normal-weight", type=float, default=1.0)
    parser.add_argument("--active-svm-upper-tail-pseudo-anomaly-weight", type=float, default=1.0)
    parser.add_argument("--active-anomaly-reverse-lr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--active-anomaly-target-mode",
        choices=("none", "labeled_defect_score_median"),
        default="none",
    )
    parser.add_argument(
        "--tta-param-scope",
        choices=("full", "bn_only", "mid_bn_only", "bottleneck_full"),
        default="full",
    )
    parser.add_argument("--entropy-margin", type=float, default=0.6)
    parser.add_argument("--eatta-oracle-num", type=int, default=1)
    parser.add_argument("--eatta-noise-std", type=float, default=0.01)
    parser.add_argument("--eatta-weight-momentum", type=float, default=0.8)
    parser.add_argument("--eatta-class-history", type=int, default=1)
    parser.add_argument("--tta-pos-scale-aug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tta-pos-scale-translate-frac", type=float, default=0.06)
    parser.add_argument("--tta-pos-scale-scales", nargs=2, type=float, default=[0.90, 1.10])
    parser.add_argument("--tta-pos-scale-infer-agg", choices=("mean", "max"), default="mean")
    parser.add_argument("--atta-oracle-num", type=int, default=1)
    parser.add_argument(
        "--atta-query-strategy",
        choices=("median_score", "max_score", "min_score", "random"),
        default="median_score",
    )
    parser.add_argument("--eatta-pseudo-normal-score-q", type=float, default=0.25)
    parser.add_argument("--eatta-pseudo-normal-entropy-q", type=float, default=0.50)
    parser.add_argument("--eatta-pseudo-normal-stability-q", type=float, default=0.50)
    parser.add_argument("--eatta-pseudo-normal-max-fraction", type=float, default=0.50)
    parser.add_argument("--sar-rho", type=float, default=0.05)
    parser.add_argument("--selector-mode", choices=("expblend", "svm_reliability_replay"), default="expblend")
    parser.add_argument("--selector-q", type=float, default=0.10)
    parser.add_argument("--selector-tau", type=float, default=16.0)
    parser.add_argument("--selector-min-fraction", type=float, default=0.0)
    parser.add_argument("--selector-min-history", type=int, default=10)
    parser.add_argument("--selector-max-history", type=int, default=4096)
    parser.add_argument("--selector-svm-fit-history", type=int, default=256)
    parser.add_argument("--selector-svm-normal-core-q", type=float, default=0.20)
    parser.add_argument("--selector-svm-min-core-samples", type=int, default=5)
    parser.add_argument("--selector-svm-inlier-q", type=float, default=0.90)
    parser.add_argument("--selector-svm-max-q", type=float, default=0.30)
    parser.add_argument("--selector-svm-nu", type=float, default=-1.0)
    parser.add_argument("--svm-expblend-warmup-batches", type=int, default=1)
    parser.add_argument("--svm-expblend-tau-batches", type=float, default=3.0)
    parser.add_argument("--svm-expblend-lambda-scale", type=float, default=1.0)
    parser.add_argument("--svm-expblend-lambda-cap", type=float, default=1.0)
    parser.add_argument("--svm-reliability-method", choices=("bootstrap", "density", "margin"), default="density")
    parser.add_argument("--svm-reliability-threshold", type=float, default=0.60)
    parser.add_argument("--svm-reliability-eps-ratio", type=float, default=0.05)
    parser.add_argument("--svm-reliability-bootstrap-n", type=int, default=30)
    parser.add_argument("--svm-reliability-bootstrap-seed", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.resize_size < 32 or args.crop_size < 32:
        parser.error("--resize-size/--crop-size must be >= 32")
    if args.batch_size < 1 or args.train_batch_size < 1 or args.tta_adapt_batch_size < 1:
        parser.error("batch sizes must be >= 1")
    if args.rd4ad_epochs < 0:
        parser.error("--rd4ad-epochs must be >= 0")
    if args.rd4ad_max_train_batches < 0:
        parser.error("--rd4ad-max-train-batches must be >= 0")
    if args.rd4ad_grad_accum_steps < 1:
        parser.error("--rd4ad-grad-accum-steps must be >= 1")
    if args.tta_steps < 0:
        parser.error("--tta-steps must be >= 0")
    if args.tta_grad_clip < 0.0:
        parser.error("--tta-grad-clip must be >= 0")
    if not 0.0 <= args.tta_score_ema_decay < 1.0:
        parser.error("--tta-score-ema-decay must be in [0, 1)")
    if not 0.0 < args.active_svm_source_pixel_q < 1.0:
        parser.error("--active-svm-source-pixel-q must be in (0, 1)")
    valid_drop_features = set(ACTIVE_SVM_MAP_FEATURE_NAMES) | set(ACTIVE_SVM_ENCODER_LAYER_FEATURE_NAMES.values())
    invalid_drop_features = sorted(set(args.active_svm_drop_features) - valid_drop_features)
    if invalid_drop_features:
        parser.error(f"--active-svm-drop-features contains unknown names: {invalid_drop_features}")
    if args.active_svm_feature_mode != "map_stats" and args.active_svm_drop_features:
        parser.error("--active-svm-drop-features is only valid with --active-svm-feature-mode map_stats")
    if args.active_svm_feature_mode != "map_stats" and active_svm_encoder_feature_layers(args):
        parser.error("encoder SVM features are only valid with --active-svm-feature-mode map_stats")
    if args.active_svm_feature_mode == "map_stats" and not active_svm_feature_names(args):
        parser.error("--active-svm-drop-features removed every map_stats feature")
    if args.active_query_perturb_candidate_count < 1:
        parser.error("--active-query-perturb-candidate-count must be >= 1")
    if args.active_query_perturb_std < 0.0:
        parser.error("--active-query-perturb-std must be >= 0")
    if args.active_svm_confidence_threshold < 0.0:
        parser.error("--active-svm-confidence-threshold must be >= 0")
    if not 0.0 <= args.active_svm_tail_pseudo_label_fraction <= 0.5:
        parser.error("--active-svm-tail-pseudo-label-fraction must be in [0, 0.5]")
    if args.active_svm_lower_tail_pseudo_normal_weight < 0.0:
        parser.error("--active-svm-lower-tail-pseudo-normal-weight must be >= 0")
    if args.active_svm_upper_tail_pseudo_anomaly_weight < 0.0:
        parser.error("--active-svm-upper-tail-pseudo-anomaly-weight must be >= 0")
    if bool(args.active_anomaly_reverse_lr) and str(args.active_anomaly_target_mode) != "none":
        parser.error("--active-anomaly-reverse-lr and --active-anomaly-target-mode cannot be enabled together")
    if args.entropy_margin < 0.0:
        parser.error("--entropy-margin must be >= 0")
    if args.sar_rho < 0.0:
        parser.error("--sar-rho must be >= 0")
    if args.eatta_oracle_num < 1:
        parser.error("--eatta-oracle-num must be >= 1")
    if args.eatta_noise_std < 0.0:
        parser.error("--eatta-noise-std must be >= 0")
    if not 0.0 <= args.eatta_weight_momentum < 1.0:
        parser.error("--eatta-weight-momentum must be in [0, 1)")
    if args.eatta_class_history < 0:
        parser.error("--eatta-class-history must be >= 0")
    if args.atta_oracle_num < 1:
        parser.error("--atta-oracle-num must be >= 1")
    for name in (
        "eatta_pseudo_normal_score_q",
        "eatta_pseudo_normal_entropy_q",
        "eatta_pseudo_normal_stability_q",
        "eatta_pseudo_normal_max_fraction",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
        parser.error("--svm-reliability-eps-ratio must be >= 0")
    if args.svm_reliability_bootstrap_n < 1:
        parser.error("--svm-reliability-bootstrap-n must be >= 1")
    return args


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
    def __init__(self, backbone: str, layers: Sequence[str], pre_trained: bool = True) -> None:
        super().__init__()
        self.backbone = str(backbone)
        self.layers = list(layers)
        probe = timm.create_model(self.backbone, pretrained=False, features_only=True, exportable=True)
        layer_names = [info["module"] for info in probe.feature_info.info]
        missing = [layer for layer in self.layers if layer not in layer_names]
        if missing:
            raise ValueError(f"Missing timm layers for {self.backbone}: {missing}; available={layer_names}")
        idx = [layer_names.index(layer) for layer in self.layers]
        self.feature_extractor = timm.create_model(
            self.backbone,
            pretrained=pre_trained,
            pretrained_cfg=None,
            features_only=True,
            exportable=True,
            out_indices=idx,
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
        blocks: int,
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
        self.conv1 = conv1x1(256 * block.expansion, 512 * block.expansion)
        self.bn1 = norm_layer(512 * block.expansion)
        self.conv2 = conv3x3(512 * block.expansion, 512 * block.expansion, 2)
        self.bn2 = norm_layer(512 * block.expansion)
        self.conv3 = conv1x1(512 * block.expansion, 512 * block.expansion)
        self.bn3 = norm_layer(512 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.bn_layer = self._make_layer(block, 512, blocks)

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


class DecoderBasicBlock(nn.Module):
    expansion = 1

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
        if groups != 1 or base_width != 64:
            raise ValueError("DecoderBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 is not supported in DecoderBasicBlock")
        self.conv1 = (
            nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=stride, groups=groups, bias=False, dilation=dilation)
            if stride == 2
            else conv3x3(inplanes, planes, stride)
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        identity = batch
        out = self.relu(self.bn1(self.conv1(batch)))
        out = self.bn2(self.conv2(out))
        if self.upsample is not None:
            identity = self.upsample(batch)
        return self.relu(out + identity)


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
            identity = self.upsample(batch)
        return self.relu(out + identity)


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
        block: type[DecoderBasicBlock | DecoderBottleneck],
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
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(
        self,
        block: type[DecoderBasicBlock | DecoderBottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
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


def get_bottleneck_layer(backbone: str) -> OCBE:
    return OCBE(BasicBlock, 2) if backbone in {"resnet18", "resnet34"} else OCBE(Bottleneck, 3)


def get_decoder(backbone: str) -> ResNetDecoder:
    decoder_map = {
        "resnet18": lambda: ResNetDecoder(DecoderBasicBlock, [2, 2, 2, 2]),
        "resnet34": lambda: ResNetDecoder(DecoderBasicBlock, [3, 4, 6, 3]),
        "resnet50": lambda: ResNetDecoder(DecoderBottleneck, [3, 4, 6, 3]),
        "resnet101": lambda: ResNetDecoder(DecoderBottleneck, [3, 4, 23, 3]),
        "resnet152": lambda: ResNetDecoder(DecoderBottleneck, [3, 8, 36, 3]),
        "resnext50_32x4d": lambda: ResNetDecoder(DecoderBottleneck, [3, 4, 6, 3], groups=32, width_per_group=4),
        "resnext101_32x8d": lambda: ResNetDecoder(DecoderBottleneck, [3, 4, 23, 3], groups=32, width_per_group=8),
        "wide_resnet50_2": lambda: ResNetDecoder(DecoderBottleneck, [3, 4, 6, 3], width_per_group=128),
        "wide_resnet101_2": lambda: ResNetDecoder(DecoderBottleneck, [3, 4, 23, 3], width_per_group=128),
    }
    if backbone not in decoder_map:
        raise ValueError(f"Decoder with architecture {backbone} is not supported")
    return decoder_map[backbone]()


class AnomalibReverseDistillationModel(nn.Module):
    def __init__(self, backbone: str, layers: Sequence[str], image_size: int) -> None:
        super().__init__()
        self.encoder = TimmFeatureExtractor(backbone=backbone, layers=layers, pre_trained=True)
        self.bottleneck = get_bottleneck_layer(backbone)
        self.decoder = get_decoder(backbone)
        self.image_size = (int(image_size), int(image_size))

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.decoder.parameters()) + list(self.bottleneck.parameters())

    def forward_features(self, images: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        encoder_features = self.encoder(images)
        decoder_features = self.decoder(self.bottleneck(encoder_features))
        return encoder_features, decoder_features

    def reconstruction_loss(self, images: torch.Tensor) -> torch.Tensor:
        encoder_features, decoder_features = self.forward_features(images)
        cos_loss = nn.CosineSimilarity(dim=1)
        loss_sum = images.new_tensor(0.0)
        for encoder_feature, decoder_feature in zip(encoder_features, decoder_features, strict=True):
            loss_sum = loss_sum + torch.mean(
                1.0
                - cos_loss(
                    encoder_feature.view(encoder_feature.shape[0], -1),
                    decoder_feature.view(decoder_feature.shape[0], -1),
                ),
            )
        return loss_sum

    def anomaly_maps(self, images: torch.Tensor) -> torch.Tensor:
        encoder_features, decoder_features = self.forward_features(images)
        anomaly_map = images.new_zeros((images.shape[0], 1, *self.image_size))
        for encoder_feature, decoder_feature in zip(encoder_features, decoder_features, strict=True):
            distance_map = 1.0 - F.cosine_similarity(encoder_feature, decoder_feature, dim=1)
            distance_map = F.interpolate(
                distance_map.unsqueeze(1),
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )
            anomaly_map = anomaly_map + distance_map
        return anomaly_map.squeeze(1)


def checkpoint_path(output_root: Path, category: str) -> Path:
    return output_root / "rd4ad_checkpoints" / safe_name(category) / "anomalib_reverse_distillation.pt"


def stream_seed(args: argparse.Namespace) -> int:
    return int(args.seed if args.stream_seed is None else args.stream_seed)


def checkpoint_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backbone": str(args.backbone),
        "layers": list(args.layers),
        "resize_size": int(args.resize_size),
        "crop_size": int(args.crop_size),
        "rd4ad_epochs": int(args.rd4ad_epochs),
        "rd4ad_lr": float(args.rd4ad_lr),
        "rd4ad_beta1": float(args.rd4ad_beta1),
        "rd4ad_beta2": float(args.rd4ad_beta2),
        "rd4ad_grad_accum_steps": int(args.rd4ad_grad_accum_steps),
        "rd4ad_max_train_batches": int(args.rd4ad_max_train_batches),
        "seed": int(args.seed),
    }


def checkpoint_config_matches(found: dict[str, Any], expected: dict[str, Any]) -> bool:
    if found == expected:
        return True
    if "rd4ad_grad_accum_steps" not in found:
        expected_without_accum = dict(expected)
        expected_without_accum.pop("rd4ad_grad_accum_steps", None)
        return found == expected_without_accum
    return False
    def anomaly_maps(self, images: torch.Tensor) -> torch.Tensor:
        encoder_features, decoder_features = self.forward_features(images)
        anomaly_map = images.new_zeros((images.shape[0], 1, *self.image_size))
        for encoder_feature, decoder_feature in zip(encoder_features, decoder_features, strict=True):
            distance_map = 1.0 - F.cosine_similarity(encoder_feature, decoder_feature, dim=1)
            distance_map = F.interpolate(
                distance_map.unsqueeze(1),
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )
            anomaly_map = anomaly_map + distance_map
        return anomaly_map.squeeze(1)


def checkpoint_path(output_root: Path, category: str) -> Path:
    return output_root / "rd4ad_checkpoints" / safe_name(category) / "anomalib_reverse_distillation.pt"


def stream_seed(args: argparse.Namespace) -> int:
    return int(args.seed if args.stream_seed is None else args.stream_seed)


def checkpoint_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backbone": str(args.backbone),
        "layers": list(args.layers),
        "resize_size": int(args.resize_size),
        "crop_size": int(args.crop_size),
        "rd4ad_epochs": int(args.rd4ad_epochs),
        "rd4ad_lr": float(args.rd4ad_lr),
        "rd4ad_beta1": float(args.rd4ad_beta1),
        "rd4ad_beta2": float(args.rd4ad_beta2),
        "rd4ad_grad_accum_steps": int(args.rd4ad_grad_accum_steps),
        "rd4ad_max_train_batches": int(args.rd4ad_max_train_batches),
        "seed": int(args.seed),
    }


def checkpoint_config_matches(found: dict[str, Any], expected: dict[str, Any]) -> bool:
    if found == expected:
        return True
    if "rd4ad_grad_accum_steps" not in found:
        expected_without_accum = dict(expected)
        expected_without_accum.pop("rd4ad_grad_accum_steps", None)
        return found == expected_without_accum
    return False


def build_model(args: argparse.Namespace, device: torch.device) -> AnomalibReverseDistillationModel:
    model = AnomalibReverseDistillationModel(
        backbone=str(args.backbone),
        layers=list(args.layers),
        image_size=int(args.crop_size),
    )
    return model.to(device)


def is_batch_norm(module: nn.Module) -> bool:
    return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))


def build_model(args: argparse.Namespace, device: torch.device) -> AnomalibReverseDistillationModel:
    model = AnomalibReverseDistillationModel(
        backbone=str(args.backbone),
        layers=list(args.layers),
        image_size=int(args.crop_size),
    )
    return model.to(device)


def is_batch_norm(module: nn.Module) -> bool:
    return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))


def tta_trainable_parameters(model: AnomalibReverseDistillationModel, args: argparse.Namespace) -> list[nn.Parameter]:
    scope = str(args.tta_param_scope)
    if scope == "full":
        return model.trainable_parameters()
    if scope == "bottleneck_full":
        return list(model.bottleneck.parameters())
    modules = model.bottleneck.modules() if scope == "mid_bn_only" else (*model.bottleneck.modules(), *model.decoder.modules())
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        if is_batch_norm(module):
            for param in module.parameters():
                if id(param) not in seen:
                    params.append(param)
                    seen.add(id(param))
    return params


def configure_tta_parameters(model: AnomalibReverseDistillationModel, args: argparse.Namespace) -> list[nn.Parameter]:
    params = tta_trainable_parameters(model=model, args=args)
    selected_param_ids = {id(param) for param in params}
    for param in model.parameters():
        param.requires_grad_(id(param) in selected_param_ids)
    if not params:
        raise RuntimeError(f"No TTA parameters selected for scope={args.tta_param_scope}")
    return params


def set_tta_train_mode(model: AnomalibReverseDistillationModel, args: argparse.Namespace) -> None:
    scope = str(args.tta_param_scope)
    if scope == "full":
        model.train()
        return
    model.eval()
    if scope == "bottleneck_full":
        model.bottleneck.train()
        return
    modules = model.bottleneck.modules() if scope == "mid_bn_only" else (*model.bottleneck.modules(), *model.decoder.modules())
    for module in modules:
        if is_batch_norm(module):
            module.train()


def tta_method_name(args: argparse.Namespace) -> str:
    if str(args.tta_method) == "tent":
        base = "AnomalibRD4AD_TENT"
    elif str(args.tta_method) == "sar":
        base = f"AnomalibRD4AD_SAR-AD-rho{active_tail_weight_token(float(args.sar_rho))}"
    elif str(args.tta_method) == "atta":
        base = "AnomalibRD4AD_ATTA-AD"
    elif str(args.tta_method) == "eatta":
        base = "AnomalibRD4AD_EATTA-AD"
    elif str(args.tta_method) == "active_svm_boundary":
        base = "AnomalibRD4AD_ActiveSVMBoundary"
        if str(args.active_svm_feature_mode) == "map_stats":
            drop_features = set(str(name) for name in getattr(args, "active_svm_drop_features", []))
            if drop_features == {"source_z_max", "source_z_top5"}:
                base = f"{base}-MapStatsNoSourceZSVM"
            elif drop_features:
                drop_suffix = safe_name("_".join(sorted(drop_features)))
                base = f"{base}-MapStatsDrop{drop_suffix}SVM"
            else:
                base = f"{base}-MapStatsSVM"
            encoder_layers = active_svm_encoder_feature_layers(args)
            if encoder_layers == (3,):
                base = f"{base}-Encoder3"
            elif encoder_layers:
                layer_suffix = "".join(str(layer) for layer in encoder_layers)
                base = f"{base}-EncoderL{layer_suffix}"
        if str(args.active_boundary_model) != "linear_svm":
            model_suffix = {
                "student_t": "StudentT",
                "gmm": "GMM",
                "gda": "GDA",
                "qda": "QDA",
                "kde": "KDE",
                "logistic": "Logistic",
                "isotonic_logistic": "IsoLogistic",
            }[str(args.active_boundary_model)]
            base = f"{base}-{model_suffix}"
        if float(args.active_svm_tail_pseudo_label_fraction) > 0.0:
            tail_pct = int(round(float(args.active_svm_tail_pseudo_label_fraction) * 100.0))
            if str(args.active_svm_tail_scope) == "stream_past":
                base = f"{base}-StreamPastTailPseudo{tail_pct}pctSVM"
            else:
                base = f"{base}-TailPseudo{tail_pct}pctSVM"
            lower_weight = float(args.active_svm_lower_tail_pseudo_normal_weight)
            upper_weight = float(args.active_svm_upper_tail_pseudo_anomaly_weight)
            if not (math.isclose(lower_weight, 1.0) and math.isclose(upper_weight, 1.0)):
                base = (
                    f"{base}-LPNw{active_tail_weight_token(lower_weight)}"
                    f"-UPAw{active_tail_weight_token(upper_weight)}"
                )
        if str(args.tta_score_source) == "adapted":
            base = f"{base}-AdaptScore"
        elif str(args.tta_score_source) == "adapted_ema":
            decay_pct = int(round(float(args.tta_score_ema_decay) * 100.0))
            base = f"{base}-AdaptEMA{decay_pct}Score"
        if str(getattr(args, "active_query_mode", "boundary")) == "boundary_perturb_high":
            perturb_mode = safe_name(str(args.active_query_perturb_mode))
            base = f"{base}-PerturbQueryK{int(args.active_query_perturb_candidate_count)}-{perturb_mode}"
        if bool(args.active_anomaly_reverse_lr):
            base = f"{base}-RevAnomLR"
        if str(getattr(args, "active_anomaly_target_mode", "none")) != "none":
            base = f"{base}-AnomTargetMedian"
    else:
        base = (
            "AnomalibRD4AD_BT-RecSVM"
            if str(args.selector_mode) == "expblend"
            else "AnomalibRD4AD_BT-RecSVM-SVMReplay"
        )
    suffix = {
        "full": "" if str(args.tta_method) == "bt_recsvm" else "-Full",
        "bn_only": "-BNOnly",
        "mid_bn_only": "-MidBNOnly",
        "bottleneck_full": "-BottleneckOnly",
        payload = torch.load(path, map_location=device, weights_only=False)
        if checkpoint_config_matches(payload.get("config", {}), expected_config):
            model.load_state_dict(payload["model_state"])
            model.eval()
            return model, path

    if int(args.rd4ad_epochs) > 0:
        loader = make_loader(train_dataset, args=args, batch_size=int(args.train_batch_size), shuffle=True)
        optimizer = torch.optim.Adam(
            model.trainable_parameters(),
            lr=float(args.rd4ad_lr),
            betas=(float(args.rd4ad_beta1), float(args.rd4ad_beta2)),
        )
        for epoch in range(1, int(args.rd4ad_epochs) + 1):
            model.train()
            progress = tqdm(
                loader,
                desc=f"Anomalib RD4AD train {category} epoch {epoch}",
                leave=False,
                disable=not sys.stderr.isatty(),
    maps_np = smooth_maps(maps.detach().cpu().numpy(), sigma=4.0)
    return maps_np.reshape(maps_np.shape[0], maps_np.shape[-2], maps_np.shape[-1]).max(axis=(1, 2)).astype(np.float64)


def top_fraction_mean(flat_maps: np.ndarray, fraction: float) -> np.ndarray:
    k = max(1, int(math.ceil(float(flat_maps.shape[1]) * float(fraction))))
    k = min(k, int(flat_maps.shape[1]))
    top = np.partition(flat_maps, kth=flat_maps.shape[1] - k, axis=1)[:, -k:]
    return top.mean(axis=1).astype(np.float64)


def map_stat_features_from_maps(
    maps_np: np.ndarray,
    source_feature_stats: dict[str, float],
) -> np.ndarray:
    flat = maps_np.reshape(maps_np.shape[0], -1).astype(np.float64, copy=False)
    score_max = flat.max(axis=1)
    score_top1_mean = top_fraction_mean(flat, 0.01)
    score_top5_mean = top_fraction_mean(flat, 0.05)
def top_fraction_mean(flat_maps: np.ndarray, fraction: float) -> np.ndarray:
    k = max(1, int(math.ceil(float(flat_maps.shape[1]) * float(fraction))))
    k = min(k, int(flat_maps.shape[1]))
    top = np.partition(flat_maps, kth=flat_maps.shape[1] - k, axis=1)[:, -k:]
    return top.mean(axis=1).astype(np.float64)


def map_stat_features_from_maps(
    maps_np: np.ndarray,
    source_feature_stats: dict[str, float],
) -> np.ndarray:
    flat = maps_np.reshape(maps_np.shape[0], -1).astype(np.float64, copy=False)
    score_max = flat.max(axis=1)
    score_top1_mean = top_fraction_mean(flat, 0.01)
    score_top5_mean = top_fraction_mean(flat, 0.05)
    score_mean = flat.mean(axis=1)
    score_std = flat.std(axis=1)
    pixel_threshold = float(source_feature_stats["pixel_threshold"])
    area_ratio = (flat > pixel_threshold).mean(axis=1).astype(np.float64)
    max_std = max(float(source_feature_stats["max_std"]), 1e-8)
    top5_std = max(float(source_feature_stats["top5_std"]), 1e-8)
    source_z_max = (score_max - float(source_feature_stats["max_mean"])) / max_std
    source_z_top5 = (score_top5_mean - float(source_feature_stats["top5_mean"])) / top5_std
    return np.stack(
        (
            score_max,
            score_top1_mean,
            score_top5_mean,
            score_mean,
            score_std,
            area_ratio,
            source_z_max,
            source_z_top5,
        ),
        axis=1,
    ).astype(np.float64)


def encoder3_pooled_features(model: AnomalibReverseDistillationModel, images: torch.Tensor) -> np.ndarray:
    encoder_features = model.encoder(images)
    encoder3 = encoder_features[-1]
    pooled = encoder3.mean(dim=(2, 3)).detach().cpu().numpy()
    return pooled.astype(np.float64)


def encoder3_z_l2_feature(
    pooled: np.ndarray,
    source_feature_stats: dict[str, float],
) -> np.ndarray:
    mean = np.asarray(source_feature_stats["encoder3_mean"], dtype=np.float64).reshape(1, -1)
    std = np.asarray(source_feature_stats["encoder3_std"], dtype=np.float64).reshape(1, -1)
    z = (np.asarray(pooled, dtype=np.float64) - mean) / np.maximum(std, 1e-6)
    return np.sqrt(np.mean(z * z, axis=1)).reshape(-1, 1).astype(np.float64)


@torch.no_grad()
    return -0.5 * np.sum(z * z + np.log(2.0 * math.pi * scale.reshape(1, -1) ** 2), axis=1)


def log_diag_student_t_density(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, df: float) -> np.ndarray:
    scale = np.maximum(scale, 1e-6)
    z = (x - mean.reshape(1, -1)) / scale.reshape(1, -1)
    const = (
        gammaln((float(df) + 1.0) / 2.0)
        - gammaln(float(df) / 2.0)
        - 0.5 * math.log(float(df) * math.pi)
        - np.log(scale.reshape(1, -1))
    )
    return np.sum(const - ((float(df) + 1.0) / 2.0) * np.log1p((z * z) / float(df)), axis=1)


def class_diag_stats(train_x: np.ndarray, labels: np.ndarray, class_label: int) -> tuple[np.ndarray, np.ndarray]:
    values = train_x[labels == int(class_label)]
    mean = np.median(values, axis=0).astype(np.float64)
    scale = np.std(values, axis=0).astype(np.float64)
    global_scale = np.std(train_x, axis=0).astype(np.float64)
    scale = np.where(scale > 1e-6, scale, global_scale)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return mean, scale


def kde_bandwidth(values: np.ndarray) -> float:
    if values.shape[0] <= 1:
        return 1.0
    dim = max(1, int(values.shape[1]))
    sigma = float(np.nanmedian(np.std(values, axis=0)))
    if not math.isfinite(sigma) or sigma <= 1e-6:
        sigma = 1.0
    bw = sigma * float(values.shape[0]) ** (-1.0 / float(dim + 4))
    return float(np.clip(bw, 0.1, 2.0))

            raw = -raw
        prob = fit["isotonic_model"].predict(raw).astype(np.float64)
        return prob - 0.5
    if model_type == "student_t":
        normal = fit["class_params"][0]
        anomaly = fit["class_params"][1]
        return log_diag_student_t_density(
            scaled,
            np.asarray(anomaly["mean"], dtype=np.float64),
            np.asarray(anomaly["scale"], dtype=np.float64),
            float(fit["df"]),
        ) - log_diag_student_t_density(
            scaled,
            np.asarray(normal["mean"], dtype=np.float64),
            np.asarray(normal["scale"], dtype=np.float64),
            float(fit["df"]),
        )
    if model_type == "gmm":
        return fit["models"][1].score_samples(scaled) - fit["models"][0].score_samples(scaled)
    return -0.5 * np.sum(z * z + np.log(2.0 * math.pi * scale.reshape(1, -1) ** 2), axis=1)


def log_diag_student_t_density(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, df: float) -> np.ndarray:
    scale = np.maximum(scale, 1e-6)
    z = (x - mean.reshape(1, -1)) / scale.reshape(1, -1)
    const = (
        gammaln((float(df) + 1.0) / 2.0)
        - gammaln(float(df) / 2.0)
        - 0.5 * math.log(float(df) * math.pi)
        - np.log(scale.reshape(1, -1))
    )
    return np.sum(const - ((float(df) + 1.0) / 2.0) * np.log1p((z * z) / float(df)), axis=1)


def class_diag_stats(train_x: np.ndarray, labels: np.ndarray, class_label: int) -> tuple[np.ndarray, np.ndarray]:
    values = train_x[labels == int(class_label)]
    mean = np.median(values, axis=0).astype(np.float64)
    scale = np.std(values, axis=0).astype(np.float64)
    global_scale = np.std(train_x, axis=0).astype(np.float64)
    scale = np.where(scale > 1e-6, scale, global_scale)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return mean, scale


def kde_bandwidth(values: np.ndarray) -> float:
    if values.shape[0] <= 1:
        return 1.0
    dim = max(1, int(values.shape[1]))
    sigma = float(np.nanmedian(np.std(values, axis=0)))
    if not math.isfinite(sigma) or sigma <= 1e-6:
        sigma = 1.0
    bw = sigma * float(values.shape[0]) ** (-1.0 / float(dim + 4))
    return float(np.clip(bw, 0.1, 2.0))


def active_boundary_raw_decision(fit: dict[str, Any], scaled: np.ndarray) -> np.ndarray:
    model_type = str(fit["model_type"])
    if model_type == "linear_svm":
        model = fit["model"]
        decision = model.decision_function(scaled).reshape(-1).astype(np.float64)
        if int(model.classes_[-1]) != 1:
            decision = -decision
        return decision
    if model_type in {"gda", "qda", "logistic"}:
        model = fit["model"]
        decision = model.decision_function(scaled).reshape(-1).astype(np.float64)
        if int(model.classes_[-1]) != 1:
            decision = -decision
        return decision
    if model_type == "isotonic_logistic":
        base_model = fit["base_model"]
        raw = base_model.decision_function(scaled).reshape(-1).astype(np.float64)
        if int(base_model.classes_[-1]) != 1:
            raw = -raw
        prob = fit["isotonic_model"].predict(raw).astype(np.float64)
        return prob - 0.5
    if model_type == "student_t":
        normal = fit["class_params"][0]
        anomaly = fit["class_params"][1]
        return log_diag_student_t_density(
            scaled,
            np.asarray(anomaly["mean"], dtype=np.float64),
            np.asarray(anomaly["scale"], dtype=np.float64),
            float(fit["df"]),
        ) - log_diag_student_t_density(
            scaled,
            np.asarray(normal["mean"], dtype=np.float64),
            np.asarray(normal["scale"], dtype=np.float64),
            float(fit["df"]),
        )
    if model_type == "gmm":
        return fit["models"][1].score_samples(scaled) - fit["models"][0].score_samples(scaled)
    if model_type == "kde":
        return fit["models"][1].score_samples(scaled) - fit["models"][0].score_samples(scaled)
    raise ValueError(f"Unsupported active boundary model: {model_type}")


def finish_active_boundary_fit(
    fit: dict[str, Any],
    feature_array: np.ndarray,
    train_x: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> dict[str, Any]:
    raw_train = active_boundary_raw_decision(fit=fit, scaled=train_x)
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
    return fit


def fit_active_score_svm(
    features: Sequence[Any],
    labels: Sequence[int],
    sample_weights: Sequence[float] | None = None,
    model_name: str = "linear_svm",
) -> dict[str, Any] | None:
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if label_array.size < 2 or len(features) < 2 or np.unique(label_array).size < 2:
        return None
    weight_array = None
    if sample_weights is not None:
        weight_array = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if weight_array.size != label_array.size:
            raise ValueError("Active boundary sample weights must match labels")
        keep = np.isfinite(weight_array) & (weight_array > 0.0)
        if not np.all(keep):
            feature_list = list(features)
            features = [feature_list[index] for index, use in enumerate(keep) if bool(use)]
            label_array = label_array[keep]
            weight_array = weight_array[keep]
        if label_array.size < 2 or len(features) < 2 or np.unique(label_array).size < 2:
            return None
    if int(np.sum(label_array == 0)) < 1 or int(np.sum(label_array == 1)) < 1:
        return None
    feature_array, center, scale = standardized_active_features(features)
    if feature_array.shape[0] < 2:
        return None

    train_x = (feature_array - center.reshape(1, -1)) / scale.reshape(1, -1)
    model_name = str(model_name)
    if model_name == "linear_svm":
        model = SVC(kernel="linear", C=1.0, class_weight="balanced")
        model.fit(train_x, label_array, sample_weight=weight_array)
        coef = np.asarray(model.coef_[0], dtype=np.float64)
        intercept = float(model.intercept_[0])
        model_fit: dict[str, Any] = {
            "model_type": "linear_svm",
            "model": model,
            "coef": coef.tolist(),
            "coef_norm": float(np.linalg.norm(coef)),
            "coef_score_max": float(coef[0]) if coef.size > 0 else float("nan"),
            "intercept": intercept,
        }
    elif model_name == "student_t":
        normal_mean, normal_scale = class_diag_stats(train_x, label_array, 0)
        anomaly_mean, anomaly_scale = class_diag_stats(train_x, label_array, 1)
        model_fit = {
            "model_type": "student_t",
            "class_params": {
                0: {"mean": normal_mean.tolist(), "scale": normal_scale.tolist()},
                1: {"mean": anomaly_mean.tolist(), "scale": anomaly_scale.tolist()},
            },
            "df": 4.0,
            "coef": [],
            "coef_norm": float("nan"),
            "coef_score_max": float("nan"),
            "intercept": float("nan"),
        }
    elif model_name == "gmm":
        models = {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for class_label in (0, 1):
                    values = train_x[label_array == class_label]
                    n_components = min(2, int(values.shape[0]))
                    models[class_label] = GaussianMixture(
                        n_components=max(1, n_components),
                        covariance_type="full",
                        reg_covar=1e-3,
                        random_state=0,
                    ).fit(values)
        except Exception:
            return None
        model_fit = {
            "model_type": "gmm",
            "models": models,
            "coef": [],
            "coef_norm": float("nan"),
            "coef_score_max": float("nan"),
            "intercept": float("nan"),
        }
    elif model_name == "gda":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])
                model.fit(train_x, label_array)
        except Exception:
            return None
        model_fit = {
            "model_type": "gda",
            "model": model,
            "coef": np.asarray(getattr(model, "coef_", np.asarray([[]]))[0], dtype=np.float64).tolist(),
            "coef_norm": float(np.linalg.norm(getattr(model, "coef_", np.asarray([[0.0]])))),
            "coef_score_max": float(getattr(model, "coef_", np.asarray([[float("nan")]]))[0][0]),
            "intercept": float(np.ravel(getattr(model, "intercept_", [float("nan")]))[0]),
        }
    raise ValueError(f"Unsupported active boundary model: {model_type}")


def finish_active_boundary_fit(
    fit: dict[str, Any],
    feature_array: np.ndarray,
    train_x: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> dict[str, Any]:
    raw_train = active_boundary_raw_decision(fit=fit, scaled=train_x)
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
    return fit


def fit_active_score_svm(
    features: Sequence[Any],
    labels: Sequence[int],
    sample_weights: Sequence[float] | None = None,
    model_name: str = "linear_svm",
) -> dict[str, Any] | None:
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if label_array.size < 2 or len(features) < 2 or np.unique(label_array).size < 2:
        return None
    weight_array = None
    if sample_weights is not None:
        weight_array = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if weight_array.size != label_array.size:
            raise ValueError("Active boundary sample weights must match labels")
        keep = np.isfinite(weight_array) & (weight_array > 0.0)
        if not np.all(keep):
            feature_list = list(features)
            features = [feature_list[index] for index, use in enumerate(keep) if bool(use)]
            label_array = label_array[keep]
            weight_array = weight_array[keep]
        if label_array.size < 2 or len(features) < 2 or np.unique(label_array).size < 2:
            return None
    if int(np.sum(label_array == 0)) < 1 or int(np.sum(label_array == 1)) < 1:
        return None
    feature_array, center, scale = standardized_active_features(features)
    if feature_array.shape[0] < 2:
        return None

    train_x = (feature_array - center.reshape(1, -1)) / scale.reshape(1, -1)
    model_name = str(model_name)
    if model_name == "linear_svm":
        model = SVC(kernel="linear", C=1.0, class_weight="balanced")
        model.fit(train_x, label_array, sample_weight=weight_array)
        coef = np.asarray(model.coef_[0], dtype=np.float64)
        intercept = float(model.intercept_[0])
        model_fit: dict[str, Any] = {
            "model_type": "linear_svm",
            "model": model,
            "coef": coef.tolist(),
            "coef_norm": float(np.linalg.norm(coef)),
            "coef_score_max": float(coef[0]) if coef.size > 0 else float("nan"),
            "intercept": intercept,
        }
    elif model_name == "student_t":
        normal_mean, normal_scale = class_diag_stats(train_x, label_array, 0)
        anomaly_mean, anomaly_scale = class_diag_stats(train_x, label_array, 1)
        model_fit = {
            "model_type": "student_t",
            "class_params": {
                0: {"mean": normal_mean.tolist(), "scale": normal_scale.tolist()},
                1: {"mean": anomaly_mean.tolist(), "scale": anomaly_scale.tolist()},
            },
            "df": 4.0,
            "coef": [],
            "coef_norm": float("nan"),
            "coef_score_max": float("nan"),
            "intercept": float("nan"),
        }
    elif model_name == "gmm":
        models = {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for class_label in (0, 1):
                    values = train_x[label_array == class_label]
                    n_components = min(2, int(values.shape[0]))
                    models[class_label] = GaussianMixture(
                        n_components=max(1, n_components),
                        covariance_type="full",
                        reg_covar=1e-3,
                        random_state=0,
                    ).fit(values)
        except Exception:
            return None
        model_fit = {
            "model_type": "gmm",
            "models": models,
            "coef": [],
            "coef_norm": float("nan"),
            "coef_score_max": float("nan"),
            "intercept": float("nan"),
        }
    elif model_name == "gda":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])
                model.fit(train_x, label_array)
        except Exception:
            return None
        model_fit = {
            "model_type": "gda",
            "model": model,
            "coef": np.asarray(getattr(model, "coef_", np.asarray([[]]))[0], dtype=np.float64).tolist(),
            "coef_norm": float(np.linalg.norm(getattr(model, "coef_", np.asarray([[0.0]])))),
            "coef_score_max": float(getattr(model, "coef_", np.asarray([[float("nan")]]))[0][0]),
            "intercept": float(np.ravel(getattr(model, "intercept_", [float("nan")]))[0]),
        }
    elif model_name == "qda":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = QuadraticDiscriminantAnalysis(priors=[0.5, 0.5], reg_param=0.1)
                model.fit(train_x, label_array)
        except Exception:
            return None
        model_fit = {
            "model_type": "qda",
            "model": model,
            "coef": [],
            "coef_norm": float("nan"),
            "coef_score_max": float("nan"),
            "intercept": float("nan"),
        }
    elif model_name == "kde":
        try:
            models = {
                class_label: KernelDensity(
                    kernel="gaussian",
                    bandwidth=kde_bandwidth(train_x[label_array == class_label]),
                ).fit(train_x[label_array == class_label])
                for class_label in (0, 1)
            }
        except Exception:
            return None
        model_fit = {
            "model_type": "kde",
            "models": models,
            "coef": [],
            "coef_norm": float("nan"),
            "coef_score_max": float("nan"),
            "intercept": float("nan"),
        }
    elif model_name in {"logistic", "isotonic_logistic"}:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs")
                model.fit(train_x, label_array, sample_weight=weight_array)
        except Exception:
            return None
        coef = np.asarray(model.coef_[0], dtype=np.float64)
        if model_name == "logistic":
            model_fit = {
                "model_type": "logistic",
                "model": model,
                "coef": coef.tolist(),
                "coef_norm": float(np.linalg.norm(coef)),
                "coef_score_max": float(coef[0]) if coef.size > 0 else float("nan"),
                "intercept": float(model.intercept_[0]),
            }
        else:
            raw = model.decision_function(train_x).reshape(-1).astype(np.float64)
            if int(model.classes_[-1]) != 1:
                raw = -raw
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(raw, label_array.astype(np.float64))
            model_fit = {
                "model_type": "isotonic_logistic",
                "base_model": model,
                "isotonic_model": iso,
                "coef": coef.tolist(),
                "coef_norm": float(np.linalg.norm(coef)),
                "coef_score_max": float(coef[0]) if coef.size > 0 else float("nan"),
                "intercept": float(model.intercept_[0]),
            }
    else:
        raise ValueError(f"Unsupported active boundary model: {model_name}")

    if weight_array is not None:
        model_fit["sample_weight_min"] = float(np.min(weight_array))
        model_fit["sample_weight_max"] = float(np.max(weight_array))
        model_fit["sample_weight_mean"] = float(np.mean(weight_array))
    else:
        model_fit["sample_weight_min"] = float("nan")
        model_fit["sample_weight_max"] = float("nan")
        model_fit["sample_weight_mean"] = float("nan")
    if feature_array.shape[1] == 1 and model_name == "linear_svm":
        coef = np.asarray(model_fit["coef"], dtype=np.float64)
        intercept = float(model_fit["intercept"])
        if abs(float(coef[0])) > 1e-12:
            threshold = float(center[0] + float(-intercept / coef[0]) * scale[0])
        else:
            threshold = float(center[0])
    else:
        threshold = float("nan")
    model_fit["threshold"] = float(threshold)
    return finish_active_boundary_fit(
        fit=model_fit,
        feature_array=feature_array,
        train_x=train_x,
        center=center,
        scale=scale,
    )


def active_svm_decision(fit: dict[str, Any], features: np.ndarray) -> np.ndarray:
    feature_array = np.asarray(features, dtype=np.float64)
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(-1, 1)
    center = np.asarray(fit["center"], dtype=np.float64).reshape(1, -1)
    scale = np.asarray(fit["scale"], dtype=np.float64).reshape(1, -1)
    scaled = (feature_array - center) / scale
    raw_decision = active_boundary_raw_decision(fit=fit, scaled=scaled)
    return raw_decision / max(float(fit.get("decision_scale", 1.0)), 1e-8)


def finite_or_none(values: np.ndarray | None, size: int) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < int(size):
        return None
    if not np.any(np.isfinite(array)):
        return None
    return array


def choose_boundary_perturb_high_candidate(
    candidates: np.ndarray,
    boundary_distance: np.ndarray,
    perturb_scores: np.ndarray | None,
def anomaly_map_sample_entropy(anomaly_map: torch.Tensor) -> torch.Tensor:
    return entropy_from_probs(anomaly_map_to_prob(anomaly_map)).mean(dim=(-1, -2))


def binary_logits_from_scores(
    scores: torch.Tensor,
    center: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if center is None:
        center = scores.detach().mean()
    if scale is None:
        scale = scores.detach().std(unbiased=False)
    scale = scale.clamp_min(1e-3)
    anomaly_logit = ((scores - center) / scale).clamp(-10.0, 10.0)
    return torch.stack((-anomaly_logit, anomaly_logit), dim=-1), center, scale


def gradient_norm(loss: torch.Tensor, params: Sequence[nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    norms = [grad.norm() for grad in grads if grad is not None]
    if not norms:
        return loss.new_zeros(())
    return torch.norm(torch.stack(norms))


def no_stat_update_anomaly_maps(
    model: AnomalibReverseDistillationModel,
    images: torch.Tensor,
) -> torch.Tensor:
    modules = list(model.modules())
    training_modes = [module.training for module in modules]
    model.eval()
    try:
        with torch.no_grad():
            return model.anomaly_maps(images)
    finally:
        for module, training in zip(modules, training_modes, strict=True):
            module.train(training)


def eatta_select_indices(
    state: dict[str, Any],
    pseudo_labels: torch.Tensor,
    sorted_indices: torch.Tensor,
    diff: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    oracle_num = min(int(args.eatta_oracle_num), int(sorted_indices.numel()))
    if oracle_num <= 0:
        return sorted_indices[:0]
    recent_labels = state.setdefault("recent_labels", [])
    class_diff = state.setdefault("class_diff", {})
    recent = set(recent_labels[-int(args.eatta_class_history) :]) if int(args.eatta_class_history) > 0 else set()
    selected: list[int] = []
    for index_tensor in sorted_indices:
    selected_labels = labels[selected_indices].detach().long().clamp(0, 1) if selected_indices.numel() > 0 else labels[:0]
    loss = float(state["entropy_weight_ema"]) * loss_ent + float(state["supervised_weight_ema"]) * loss_ce
    return loss, int(selected_indices.numel()), int(torch.sum(selected_labels == 0).item())


def robust_zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return array
    center = float(np.median(array))
def adapt_on_batch_baseline(
    model: AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    if scale <= 1e-8:
        scale = 1.0
    return (array - center) / scale


def select_atta_query_indices(
    scores: np.ndarray,
    k: int,
    strategy: str,
    seed: int,
    batch_index: int,
) -> np.ndarray:
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if score_array.size == 0 or int(k) <= 0:
        return np.asarray([], dtype=np.int64)
    k = min(int(k), int(score_array.size))
    strategy = str(strategy)
    if strategy == "median_score":
        target = float(np.quantile(score_array, 0.50))
        order = np.argsort(np.abs(score_array - target))
    elif strategy == "max_score":
        order = np.argsort(-score_array)
    elif strategy == "min_score":
        order = np.argsort(score_array)
    elif strategy == "random":
        rng = np.random.default_rng(int(seed) + 100_003 * int(batch_index))
        order = rng.permutation(score_array.size)
    else:
        raise ValueError(f"Unsupported ATTA query strategy: {strategy}")
    return np.asarray(order[:k], dtype=np.int64)
            candidate_count=int(perturb_candidate_count),
            mode="threshold_nearest_fallback",
        )
    return int(np.argmin(distances)), "threshold_nearest_fallback"


def select_score_tail_pseudo_labels(
    scores: np.ndarray,
    fraction: float,
    exclude_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if score_array.size == 0 or float(fraction) <= 0.0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    k = max(1, int(math.ceil(float(score_array.size) * float(fraction))))
    k = min(k, int(score_array.size))
    order = np.argsort(score_array)
    candidates = [(int(index), 0) for index in order[:k]]
    candidates.extend((int(index), 1) for index in order[-k:])
    selected: list[int] = []
    pseudo_labels: list[int] = []
    seen: set[int] = set()
    for index, pseudo_label in candidates:
        if exclude_index is not None and index == int(exclude_index):
            continue
        if index in seen:
            continue
        selected.append(index)
        pseudo_labels.append(int(pseudo_label))
        seen.add(index)
    return np.asarray(selected, dtype=np.int64), np.asarray(pseudo_labels, dtype=np.int64)


def select_stream_tail_pseudo_labels(
    scores: Sequence[float],
    fraction: float,
    excluded_ids: set[int],
    selected_ids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if score_array.size == 0 or float(fraction) <= 0.0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    k = max(1, int(math.ceil(float(score_array.size) * float(fraction))))
    k = min(k, int(score_array.size))
    order = np.argsort(score_array)
    candidates = [(int(index), 0) for index in order[:k]]
    candidates.extend((int(index), 1) for index in order[-k:])
    selected: list[int] = []
    pseudo_labels: list[int] = []
    seen: set[int] = set()
    for global_id, pseudo_label in candidates:
        if global_id in excluded_ids or global_id in selected_ids or global_id in seen:
            continue
        selected.append(global_id)
        pseudo_labels.append(int(pseudo_label))
        seen.add(global_id)
    return np.asarray(selected, dtype=np.int64), np.asarray(pseudo_labels, dtype=np.int64)


def entropy_from_probs(prob: torch.Tensor) -> torch.Tensor:
    prob = prob.clamp(1e-6, 1.0 - 1e-6)
    return -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())


def anomaly_map_to_prob(anomaly_map: torch.Tensor) -> torch.Tensor:
    mean = anomaly_map.mean(dim=(-1, -2), keepdim=True)
    strategy = str(strategy)
    if strategy == "median_score":
        target = float(np.quantile(score_array, 0.50))
        order = np.argsort(np.abs(score_array - target))
    elif strategy == "max_score":
        order = np.argsort(-score_array)
    elif strategy == "min_score":
        order = np.argsort(score_array)
    elif strategy == "random":
        rng = np.random.default_rng(int(seed) + 100_003 * int(batch_index))
        order = rng.permutation(score_array.size)
    else:
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if center is None:
        center = scores.detach().mean()
    if scale is None:
        scale = scores.detach().std(unbiased=False)
    scale = scale.clamp_min(1e-3)
    anomaly_logit = ((scores - center) / scale).clamp(-10.0, 10.0)
    return torch.stack((-anomaly_logit, anomaly_logit), dim=-1), center, scale


def gradient_norm(loss: torch.Tensor, params: Sequence[nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    norms = [grad.norm() for grad in grads if grad is not None]
    if not norms:
        return loss.new_zeros(())
    return torch.norm(torch.stack(norms))


def no_stat_update_anomaly_maps(
    model: AnomalibReverseDistillationModel,
    images: torch.Tensor,
) -> torch.Tensor:
    modules = list(model.modules())
    training_modes = [module.training for module in modules]
    model.eval()
    try:
        with torch.no_grad():
            return model.anomaly_maps(images)
    finally:
        for module, training in zip(modules, training_modes, strict=True):
            module.train(training)


def query_perturbation_scores(
    model: AnomalibReverseDistillationModel,
    images: torch.Tensor,
    args: argparse.Namespace,
    batch_index: int,
) -> np.ndarray:
    magnitude = float(args.active_query_perturb_std)
    if magnitude <= 0.0:
        maps = no_stat_update_anomaly_maps(model=model, images=images)
        return maps.amax(dim=(-1, -2)).detach().cpu().numpy().astype(np.float64)
    perturb_mode = str(args.active_query_perturb_mode)
    if perturb_mode == "gaussian_noise":
        generator = torch.Generator(device=images.device)
        generator.manual_seed(int(stream_seed(args)) + 1_000_003 * int(batch_index))
        noise = torch.randn(
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for name, ema_value in ema_state.items():
        model_value = model_state[name].detach()
    return steps, last_loss


def adapt_on_selected_sequential(
    model: AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    args: argparse.Namespace,
    loss_sign: float = 1.0,
) -> tuple[int, float]:
    if images.numel() == 0:
        return 0, float("nan")
    total_steps = 0
    last_loss = float("nan")
    for index in range(int(images.shape[0])):
        steps, last_loss = adapt_on_selected(
            model=model,
            optimizer=optimizer,
            images=images[index : index + 1],
            args=args,
            loss_sign=loss_sign,
        )
        total_steps += int(steps)
    return total_steps, last_loss


def adapt_on_batch_anomaly_active(
    model: AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: np.ndarray,
    source_scores: np.ndarray,
    state: dict[str, Any],
    args: argparse.Namespace,
    batch_index: int,
) -> tuple[int, float, int, int, int, int, int, dict[str, Any]]:
    if images.numel() == 0 or int(args.tta_steps) == 0:
        return 0, float("nan"), 0, 0, 0, 0, 0, {}

    method = str(args.tta_method)
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    score_array = np.asarray(source_scores, dtype=np.float64).reshape(-1)
    if method == "atta":
        query_indices = select_atta_query_indices(
            scores=score_array,
            k=int(args.atta_oracle_num),
            strategy=str(args.atta_query_strategy),
            seed=int(args.seed),
        rank_score = robust_zscore(score_array) + robust_zscore(entropy_array) + robust_zscore(instability_array)
        selected = candidates[np.argsort(rank_score[candidates])[:max_count]]
        mask[selected] = True
    return mask, {
        "eatta_score_threshold": score_threshold,
        "eatta_entropy_threshold": entropy_threshold,
        "eatta_stability_threshold": stability_threshold,
    }


def adapt_on_selected(
    model: AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    args: argparse.Namespace,
    loss_sign: float = 1.0,
) -> tuple[int, float]:
    if images.numel() == 0 or int(args.tta_steps) == 0:
        return 0, float("nan")
    set_tta_train_mode(model=model, args=args)
    last_loss = float("nan")
    steps = 0
    adapt_batch_size = max(1, int(args.tta_adapt_batch_size))
    trainable_params = tta_trainable_parameters(model=model, args=args)
    for step in range(1, int(args.tta_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        adapt_images = position_scale_augmented_images(images, args=args, include_original=True)
        chunks = list(adapt_images.split(adapt_batch_size, dim=0))
        total_loss_value = 0.0
        total_items = 0
        for chunk in chunks:
            loss = model.reconstruction_loss(chunk)
            scaled_loss = float(loss_sign) * loss / float(len(chunks))
            scaled_loss.backward()
            total_loss_value += float(loss.detach().item()) * int(chunk.shape[0])
            total_items += int(chunk.shape[0])
        if args.tta_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.tta_grad_clip))
        optimizer.step()
        last_loss = total_loss_value / max(1, total_items)
        steps = step
    model.eval()
    return steps, last_loss


def adapt_on_selected_sequential(
    model: AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    args: argparse.Namespace,
    loss_sign: float = 1.0,
) -> tuple[int, float]:
    if images.numel() == 0:
        return 0, float("nan")
    total_steps = 0
    last_loss = float("nan")
    for index in range(int(images.shape[0])):
        steps, last_loss = adapt_on_selected(
            model=model,
            optimizer=optimizer,
            images=images[index : index + 1],
            args=args,
            loss_sign=loss_sign,
        )
        total_steps += int(steps)
    return total_steps, last_loss


def adapt_on_batch_anomaly_active(
    model: AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
            args=args,
        )
    reverse_count = int(reverse_mask.sum())
    if reverse_count > 0:
        reverse_tensor = torch.from_numpy(reverse_mask).to(device=images.device, dtype=torch.bool)
        reverse_steps, reverse_loss = adapt_on_selected_sequential(
            model=model,
            optimizer=optimizer,
            images=images[reverse_tensor],
            args=args,
            loss_sign=-1.0,
        )
        steps += int(reverse_steps)
        last_loss = reverse_loss

    pseudo_count = int(pseudo_mask.sum())
    pseudo_normal_count = int(np.sum(label_array[pseudo_mask] == 0)) if pseudo_count > 0 else 0
    query_labels = label_array[query_indices] if query_indices.size > 0 else np.asarray([], dtype=np.int64)
    trace = {
        "ad_active_method": method,
        "ad_active_query_mode": query_mode,
        "ad_active_query_indices": [int(index) for index in query_indices.tolist()],
        "ad_active_query_labels": [int(label) for label in query_labels.tolist()],
        "ad_active_query_scores": [float(score_array[index]) for index in query_indices.tolist()],
        "ad_active_query_entropy": [
            float(entropy[index]) if np.isfinite(entropy[index]) else float("nan") for index in query_indices.tolist()
        ],
        "ad_active_query_instability": [
            float(instability[index]) if np.isfinite(instability[index]) else float("nan") for index in query_indices.tolist()
        ],
        "ad_active_label_count": int(query_indices.size),
        "ad_active_label_normal_count": int(np.sum(query_labels == 0)) if query_labels.size > 0 else 0,
        "ad_active_label_anomaly_count": int(np.sum(query_labels == 1)) if query_labels.size > 0 else 0,
        "ad_active_reverse_anomaly_count": int(reverse_count),
        "ad_pseudo_normal_count": int(pseudo_count),
        "ad_pseudo_normal_purity": float(pseudo_normal_count / pseudo_count) if pseudo_count > 0 else float("nan"),
        "ad_update_mode": "per_sample_sequential",
        "atta_oracle_num": int(args.atta_oracle_num),
        "atta_query_strategy": str(args.atta_query_strategy),
        "eatta_oracle_num": int(args.eatta_oracle_num),
        "eatta_pseudo_normal_score_q": float(args.eatta_pseudo_normal_score_q),
        "eatta_pseudo_normal_entropy_q": float(args.eatta_pseudo_normal_entropy_q),
        "eatta_pseudo_normal_stability_q": float(args.eatta_pseudo_normal_stability_q),
        "eatta_pseudo_normal_max_fraction": float(args.eatta_pseudo_normal_max_fraction),
        "eatta_recent_pseudo_labels": list(state.get("eatta_recent_pseudo_labels", [])),
    }
    trace.update(pseudo_diag)
    active_label_count = int(query_indices.size)
    active_normal_count = int(np.sum(query_labels == 0)) if query_labels.size > 0 else 0
    active_anomaly_count = int(np.sum(query_labels == 1)) if query_labels.size > 0 else 0
    state["recent_query_labels"] = (
        list(state.get("recent_query_labels", [])) + [int(label) for label in query_labels.tolist()]
    )[-max(1, int(args.eatta_class_history)) :]
    return (
        int(steps),
        last_loss,
        selected_count,
        selected_normal_count,
        active_label_count,
        active_normal_count,
        active_anomaly_count,
            target_score=target_score,
        )
        total_steps += int(steps)
    return total_steps, last_loss


def labeled_defect_score_median(
    features: Sequence[np.ndarray],
    labels: Sequence[int],
    weights: Sequence[float],
) -> float | None:
    scores = [
        float(np.asarray(feature, dtype=np.float64).reshape(-1)[0])
        for feature, label, weight in zip(features, labels, weights, strict=True)
        if int(label) == 1 and float(weight) >= 1.0
    ]
    if not scores:
        return None
    return float(np.median(np.asarray(scores, dtype=np.float64)))

        }
    score_threshold = float(np.quantile(score_array, float(args.eatta_pseudo_normal_score_q)))
    entropy_threshold = float(np.quantile(entropy_array, float(args.eatta_pseudo_normal_entropy_q)))
    stability_threshold = float(np.quantile(instability_array, float(args.eatta_pseudo_normal_stability_q)))
    candidate_mask = (
        (score_array <= score_threshold)
        & (entropy_array <= entropy_threshold)
        & (instability_array <= stability_threshold)
    )
    if query_indices.size > 0:
        candidate_mask[np.asarray(query_indices, dtype=np.int64)] = False
    candidates = np.flatnonzero(candidate_mask)
    max_count = max(1, int(math.ceil(float(score_array.size) * float(args.eatta_pseudo_normal_max_fraction))))
    max_count = min(max_count, int(candidates.size))
    if max_count > 0:
        rank_score = robust_zscore(score_array) + robust_zscore(entropy_array) + robust_zscore(instability_array)
        selected = candidates[np.argsort(rank_score[candidates])[:max_count]]
        mask[selected] = True
    return mask, {
        "eatta_score_threshold": score_threshold,
        "eatta_entropy_threshold": entropy_threshold,
        "eatta_stability_threshold": stability_threshold,
    }


def adapt_on_selected(
    model: AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    args: argparse.Namespace,
    loss_sign: float = 1.0,
) -> tuple[int, float]:
    if images.numel() == 0 or int(args.tta_steps) == 0:
        return 0, float("nan")
    set_tta_train_mode(model=model, args=args)
    last_loss = float("nan")
    steps = 0
    adapt_batch_size = max(1, int(args.tta_adapt_batch_size))
    trainable_params = tta_trainable_parameters(model=model, args=args)
    for step in range(1, int(args.tta_steps) + 1):
    entropy_array = np.asarray(entropy, dtype=np.float64).reshape(-1)
    instability_array = np.asarray(instability, dtype=np.float64).reshape(-1)
    mask = np.zeros_like(score_array, dtype=bool)
    if score_array.size == 0 or float(args.eatta_pseudo_normal_max_fraction) <= 0.0:
        return mask, {
            "eatta_score_threshold": float("nan"),
            "eatta_entropy_threshold": float("nan"),
            "eatta_stability_threshold": float("nan"),
        }
    score_threshold = float(np.quantile(score_array, float(args.eatta_pseudo_normal_score_q)))
    entropy_threshold = float(np.quantile(entropy_array, float(args.eatta_pseudo_normal_entropy_q)))
    stability_threshold = float(np.quantile(instability_array, float(args.eatta_pseudo_normal_stability_q)))
    candidate_mask = (
        (score_array <= score_threshold)
        & (entropy_array <= entropy_threshold)
        & (instability_array <= stability_threshold)
    )
    if query_indices.size > 0:
        candidate_mask[np.asarray(query_indices, dtype=np.int64)] = False
    candidates = np.flatnonzero(candidate_mask)
    max_count = max(1, int(math.ceil(float(score_array.size) * float(args.eatta_pseudo_normal_max_fraction))))
    max_count = min(max_count, int(candidates.size))
    if max_count > 0:
        rank_score = robust_zscore(score_array) + robust_zscore(entropy_array) + robust_zscore(instability_array)
        selected = candidates[np.argsort(rank_score[candidates])[:max_count]]
        mask[selected] = True
    return mask, {
        "eatta_score_threshold": score_threshold,
        "eatta_entropy_threshold": entropy_threshold,
        "eatta_stability_threshold": stability_threshold,
    }


def adapt_on_selected(
    model: AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    args: argparse.Namespace,
    loss_sign: float = 1.0,
) -> tuple[int, float]:
    if images.numel() == 0 or int(args.tta_steps) == 0:
        return 0, float("nan")
    set_tta_train_mode(model=model, args=args)
    last_loss = float("nan")
    steps = 0
    adapt_batch_size = max(1, int(args.tta_adapt_batch_size))
    trainable_params = tta_trainable_parameters(model=model, args=args)
    for step in range(1, int(args.tta_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        adapt_images = position_scale_augmented_images(images, args=args, include_original=True)
        chunks = list(adapt_images.split(adapt_batch_size, dim=0))
        total_loss_value = 0.0
        total_items = 0
        for chunk in chunks:
            loss = model.reconstruction_loss(chunk)
            scaled_loss = float(loss_sign) * loss / float(len(chunks))
            scaled_loss.backward()
            total_loss_value += float(loss.detach().item()) * int(chunk.shape[0])
            total_items += int(chunk.shape[0])
        if args.tta_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.tta_grad_clip))
        optimizer.step()
        last_loss = total_loss_value / max(1, total_items)
        steps = step
    model.eval()
    return steps, last_loss


def adapt_on_selected_sequential(
    model: AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    args: argparse.Namespace,
    loss_sign: float = 1.0,
) -> tuple[int, float]:
    if images.numel() == 0:
        return 0, float("nan")
    total_steps = 0
    last_loss = float("nan")
    for index in range(int(images.shape[0])):
        steps, last_loss = adapt_on_selected(
            model=model,
            optimizer=optimizer,
            images=images[index : index + 1],
            args=args,
            loss_sign=loss_sign,
        )
        total_steps += int(steps)
    return total_steps, last_loss


def adapt_anomaly_score_to_target(
    model: AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    args: argparse.Namespace,
    target_score: float,
) -> tuple[int, float]:
    if images.numel() == 0 or int(args.tta_steps) == 0:
        return 0, float("nan")
    set_tta_train_mode(model=model, args=args)
    last_loss = float("nan")
    steps = 0
    adapt_batch_size = max(1, int(args.tta_adapt_batch_size))
    trainable_params = tta_trainable_parameters(model=model, args=args)
    target = images.new_tensor(float(target_score))
    for step in range(1, int(args.tta_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        adapt_images = position_scale_augmented_images(images, args=args, include_original=True)
        chunks = list(adapt_images.split(adapt_batch_size, dim=0))
        total_loss_value = 0.0
        total_items = 0
        for chunk in chunks:
            maps = model.anomaly_maps(chunk)
            scores = maps.reshape(maps.shape[0], -1).amax(dim=1)
            loss = F.mse_loss(scores, target.expand_as(scores))
            (loss / float(len(chunks))).backward()
            total_loss_value += float(loss.detach().item()) * int(chunk.shape[0])
            total_items += int(chunk.shape[0])
        if args.tta_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.tta_grad_clip))
        optimizer.step()
        last_loss = total_loss_value / max(1, total_items)
        steps = step
    model.eval()
    return steps, last_loss


def adapt_anomaly_score_to_target_sequential(
    model: AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    args: argparse.Namespace,
    target_score: float,
) -> tuple[int, float]:
    if images.numel() == 0:
    train_dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    loader = make_loader(train_dataset, args=args, batch_size=int(args.batch_size), shuffle=False)
    pixel_chunks: list[np.ndarray] = []
    max_values: list[np.ndarray] = []
    top5_values: list[np.ndarray] = []
    encoder3_values: list[np.ndarray] = []
    model.to(device).eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        maps = model.anomaly_maps(images)
        maps_np = smooth_maps(maps.detach().cpu().numpy(), sigma=4.0)
        flat = maps_np.reshape(maps_np.shape[0], -1).astype(np.float64, copy=False)
        pixel_chunks.append(flat.reshape(-1).astype(np.float32, copy=True))
        max_values.append(flat.max(axis=1).astype(np.float64))
        top5_values.append(top_fraction_mean(flat, 0.05))
        if bool(getattr(args, "active_svm_include_encoder3_feature", False)):
            encoder3_values.append(encoder3_pooled_features(model=model, images=images))
    pixels = np.concatenate(pixel_chunks, axis=0)
    max_array = np.concatenate(max_values, axis=0)
    top5_array = np.concatenate(top5_values, axis=0)
    stats: dict[str, Any] = {
        "pixel_threshold": float(np.quantile(pixels, float(args.active_svm_source_pixel_q))),
        "max_mean": float(np.mean(max_array)),
        "max_std": float(np.std(max_array)),
        "top5_mean": float(np.mean(top5_array)),
        "top5_std": float(np.std(top5_array)),
        "pixel_q": float(args.active_svm_source_pixel_q),
    }
    if encoder3_values:
        encoder3_array = np.concatenate(encoder3_values, axis=0).astype(np.float64)
        stats["encoder3_mean"] = np.mean(encoder3_array, axis=0).astype(np.float64).tolist()
        stats["encoder3_std"] = np.std(encoder3_array, axis=0).astype(np.float64).tolist()
    return stats


@torch.no_grad()
def selection_features_from_model(
    model: AnomalibReverseDistillationModel,
    images: torch.Tensor,
    args: argparse.Namespace,
    source_feature_stats: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if str(args.active_svm_feature_mode) == "map_stats":
        if source_feature_stats is None:
            raise RuntimeError("Source feature stats are required for map_stats SVM features")
        features_full = rd4ad_map_stat_features(
            model=model,
            images=images,
            source_feature_stats=source_feature_stats,
            include_encoder3_feature=bool(getattr(args, "active_svm_include_encoder3_feature", False)),
        )
        features = features_full[:, active_svm_feature_indices(args)]
        return features, features_full[:, 0].copy()
    scores = rd4ad_scores(model=model, images=images)
    return scores.reshape(-1, 1).astype(np.float64), scores


            expblend_tau_batches=args.svm_expblend_tau_batches,
            expblend_lambda_scale=args.svm_expblend_lambda_scale,
            expblend_lambda_cap=args.svm_expblend_lambda_cap,
        )
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    trace_rows: list[dict[str, Any]] = []
    selected_total = 0
    selected_normal_total = 0
    optimizer_steps = 0
    active_label_total = 0
    active_label_normal_total = 0
    active_label_anomaly_total = 0
    active_label_anomaly_reverse_total = 0
    active_tail_pseudo_label_total = 0
    active_tail_pseudo_label_normal_total = 0
    active_tail_pseudo_label_anomaly_total = 0
    active_tail_pseudo_label_correct_total = 0
    score_history_batches: list[np.ndarray] = []
    replay_buffer_images: list[torch.Tensor] = []
    replay_buffer_scores: list[np.ndarray] = []
    replay_buffer_labels: list[np.ndarray] = []
    reliability_gate_open = False
    reliability_gate_open_batch: int | None = None
    baseline_state: dict[str, Any] = {}
    active_labeled_features: list[np.ndarray] = []
    active_labeled_labels: list[int] = []
    active_labeled_weights: list[float] = []

    try:
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy().astype(np.int64)
            if score_source == "frozen":
                if source_selection_batches is None:
                    raise RuntimeError("Frozen selection batches were not precomputed")
                selection_features, source_scores = source_selection_batches[batch_index - 1]
            elif score_source == "adapted":
                selection_features, source_scores = selection_features_from_model(
                    model=model,
                    images=images,
                    args=args,
                    source_feature_stats=source_feature_stats,
                )
            elif score_source == "adapted_ema":
                if score_model is None:
                    raise RuntimeError("EMA score model was not initialized")
                selection_features, source_scores = selection_features_from_model(
                    model=score_model,
                    images=images,
                    args=args,
                    source_feature_stats=source_feature_stats,
                )
            else:
                raise ValueError(f"Unsupported TTA score source: {score_source}")
            steps = 0
            adapt_loss = float("nan")
            trace_extra: dict[str, Any] = {}
            batch_selected_count = 0
            batch_selected_normal_count = 0
            batch_selected_purity = float("nan")
            threshold: float | None = None
            if str(args.tta_method) in {"tent", "sar"}:
                label_tensor = torch.from_numpy(labels).to(device=device, dtype=torch.long)
                steps, adapt_loss, batch_selected_count, batch_selected_normal_count = adapt_on_batch_baseline(
                    model=model,
                    optimizer=optimizer,
                    images=images,
                    labels=label_tensor,
                    trainable_params=trainable_params,
                    state=baseline_state,
                    args=args,
                    device=device,
                )
                if batch_selected_count > 0:
                    batch_selected_purity = float(batch_selected_normal_count / batch_selected_count)
                selected_total += int(batch_selected_count)
                selected_normal_total += int(batch_selected_normal_count)
                optimizer_steps += int(steps)
                history_count_after = int(batch_index)
                trace_extra.update(
                    {
                        "baseline_tta_method": str(args.tta_method),
                        "active_label_count": int(batch_selected_count),
                        "active_label_normal_count": int(batch_selected_normal_count),
                    },
                )
            elif str(args.tta_method) in {"atta", "eatta"}:
                (
    trace.update(pseudo_diag)
    active_label_count = int(query_indices.size)
    active_normal_count = int(np.sum(query_labels == 0)) if query_labels.size > 0 else 0
    active_anomaly_count = int(np.sum(query_labels == 1)) if query_labels.size > 0 else 0
    state["recent_query_labels"] = (
        list(state.get("recent_query_labels", [])) + [int(label) for label in query_labels.tolist()]
    )[-max(1, int(args.eatta_class_history)) :]
    return (
        int(steps),
        last_loss,
        selected_count,
        selected_normal_count,
        active_label_count,
        active_normal_count,
        active_anomaly_count,
        trace,
    )


def adapt_on_batch_baseline(
    model: AnomalibReverseDistillationModel,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    trainable_params: Sequence[nn.Parameter],
    state: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, float, int, int]:
    if images.numel() == 0 or int(args.tta_steps) == 0:
        return 0, float("nan"), 0, 0
    last_loss = float("nan")
    steps = 0
    selected_total = 0
    selected_normal_total = 0
    for step in range(1, int(args.tta_steps) + 1):
        set_tta_train_mode(model=model, args=args)
        optimizer.zero_grad(set_to_none=True)
        anomaly_maps = model.anomaly_maps(images)
        if str(args.tta_method) == "tent":
            loss = anomaly_map_sample_entropy(anomaly_maps).mean()
            selected_count = 0
            selected_normal_count = 0
        elif str(args.tta_method) == "sar":
            sample_entropy = anomaly_map_sample_entropy(anomaly_maps)
            reliable_mask = sample_entropy < float(args.entropy_margin)
            selected_count = int(reliable_mask.sum().detach().item())
            selected_normal_count = (
                int((labels[reliable_mask.detach()] == 0).sum().detach().item()) if selected_count > 0 else 0
            )
            if selected_count <= 0:
                last_loss = float(sample_entropy.mean().detach().item())
                continue
            first_loss = sample_entropy[reliable_mask].mean()
            first_loss.backward()
            grad_norms = [param.grad.norm(p=2) for param in trainable_params if param.grad is not None]
            grad_norm = torch.norm(torch.stack(grad_norms)) if grad_norms else first_loss.new_zeros(())
            perturbations: list[tuple[nn.Parameter, torch.Tensor]] = []
            if float(args.sar_rho) > 0.0 and float(grad_norm.detach().item()) > 0.0:
                scale = float(args.sar_rho) / (grad_norm + 1e-12)
                with torch.no_grad():
    model = copy.deepcopy(source_model).to(device).eval()
    score_model = copy.deepcopy(model).to(device).eval() if score_source == "adapted_ema" else None
    trainable_params = configure_tta_parameters(model=model, args=args)
    optimizer = torch.optim.Adam(trainable_params, lr=float(args.tta_lr))
    selector = None
    if str(args.tta_method) == "bt_recsvm" and str(args.selector_mode) == "expblend":
        selector = ScoreOnlyBTRecSVMSelector(
            q=args.selector_q,
            tau=args.selector_tau,
            min_fraction=args.selector_min_fraction,
            min_history=args.selector_min_history,
            max_history=args.selector_max_history,
            svm_fit_history=args.selector_svm_fit_history,
            svm_normal_core_q=args.selector_svm_normal_core_q,
            svm_min_core_samples=args.selector_svm_min_core_samples,
            svm_inlier_q=args.selector_svm_inlier_q,
            svm_max_q=args.selector_svm_max_q,
            expblend_warmup_batches=args.svm_expblend_warmup_batches,
            expblend_tau_batches=args.svm_expblend_tau_batches,
            expblend_lambda_scale=args.svm_expblend_lambda_scale,
            expblend_lambda_cap=args.svm_expblend_lambda_cap,
        )
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    trace_rows: list[dict[str, Any]] = []
    selected_total = 0
    selected_normal_total = 0
    optimizer_steps = 0
    active_label_total = 0
    active_label_normal_total = 0
    active_label_anomaly_total = 0
    active_label_anomaly_reverse_total = 0
    active_tail_pseudo_label_total = 0
    active_tail_pseudo_label_normal_total = 0
    active_tail_pseudo_label_anomaly_total = 0
    active_tail_pseudo_label_correct_total = 0
    score_history_batches: list[np.ndarray] = []
    replay_buffer_images: list[torch.Tensor] = []
    replay_buffer_scores: list[np.ndarray] = []
    replay_buffer_labels: list[np.ndarray] = []
    reliability_gate_open = False
    reliability_gate_open_batch: int | None = None
    baseline_state: dict[str, Any] = {}
    active_labeled_features: list[np.ndarray] = []
    active_labeled_labels: list[int] = []
    active_labeled_weights: list[float] = []
    stream_tail_features: list[np.ndarray] = []
    stream_tail_scores: list[float] = []
    stream_tail_labels: list[int] = []
    stream_tail_selected_ids: set[int] = set()

    try:
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy().astype(np.int64)
            if score_source == "frozen":
                if source_selection_batches is None:
                    raise RuntimeError("Frozen selection batches were not precomputed")
                selection_features, source_scores = source_selection_batches[batch_index - 1]
            elif score_source == "adapted":
                selection_features, source_scores = selection_features_from_model(
                    model=model,
                    images=images,
                    args=args,
                    source_feature_stats=source_feature_stats,
                )
            elif score_source == "adapted_ema":
                if score_model is None:
                    raise RuntimeError("EMA score model was not initialized")
                selection_features, source_scores = selection_features_from_model(
                    model=score_model,
                    images=images,
                    args=args,
                    source_feature_stats=source_feature_stats,
                )
            else:
                raise ValueError(f"Unsupported TTA score source: {score_source}")
            batch_global_ids = np.asarray([], dtype=np.int64)
            if str(args.tta_method) == "active_svm_boundary" and str(args.active_svm_tail_scope) == "stream_past":
                stream_start = len(stream_tail_scores)
                batch_global_ids = np.arange(stream_start, stream_start + int(labels.shape[0]), dtype=np.int64)
                stream_tail_features.extend(feature.copy() for feature in selection_features)
                stream_tail_scores.extend(float(score) for score in source_scores)
                stream_tail_labels.extend(int(label) for label in labels)
            steps = 0
            adapt_loss = float("nan")
            trace_extra: dict[str, Any] = {}
            batch_selected_count = 0
            batch_selected_normal_count = 0
            batch_selected_purity = float("nan")
            threshold: float | None = None
            if str(args.tta_method) in {"tent", "sar"}:
                label_tensor = torch.from_numpy(labels).to(device=device, dtype=torch.long)
                steps, adapt_loss, batch_selected_count, batch_selected_normal_count = adapt_on_batch_baseline(
                    model=model,
                    optimizer=optimizer,
                    images=images,
                    labels=label_tensor,
                    trainable_params=trainable_params,
                    state=baseline_state,
                    args=args,
                    device=device,
                )
                if batch_selected_count > 0:
                    batch_selected_purity = float(batch_selected_normal_count / batch_selected_count)
                selected_total += int(batch_selected_count)
                selected_normal_total += int(batch_selected_normal_count)
                optimizer_steps += int(steps)
                history_count_after = int(batch_index)
                trace_extra.update(
                    {
                        "baseline_tta_method": str(args.tta_method),
                        "active_label_count": int(batch_selected_count),
                        "active_label_normal_count": int(batch_selected_normal_count),
                    },
                )
            elif str(args.tta_method) in {"atta", "eatta"}:
                (
                    steps,
                    adapt_loss,
                    batch_selected_count,
                    batch_selected_normal_count,
                    active_query_count,
                    active_query_normal_count,
                    active_query_anomaly_count,
                    ad_active_trace,
                ) = adapt_on_batch_anomaly_active(
                    model=model,
                    optimizer=optimizer,
                    images=images,
                    labels=labels,
                    source_scores=source_scores,
                    state=baseline_state,
                    args=args,
                    batch_index=int(batch_index),
                )
                if batch_selected_count > 0:
                    batch_selected_purity = float(batch_selected_normal_count / batch_selected_count)
                selected_total += int(batch_selected_count)
                selected_normal_total += int(batch_selected_normal_count)
                optimizer_steps += int(steps)
                active_label_total += int(active_query_count)
                active_label_normal_total += int(active_query_normal_count)
                active_label_anomaly_total += int(active_query_anomaly_count)
                active_label_anomaly_reverse_total += int(ad_active_trace.get("ad_active_reverse_anomaly_count", 0))
                history_count_after = int(batch_index)
                trace_extra.update(ad_active_trace)
                trace_extra.update(
                    {
                        "baseline_tta_method": str(args.tta_method),
                        "active_label_count": int(active_query_count),
                        "active_label_normal_count": int(active_query_normal_count),
                        "active_label_anomaly_count": int(active_query_anomaly_count),
                        "active_anomaly_reverse_lr": bool(args.active_anomaly_reverse_lr),
                        "active_adapt_sample_count": int(batch_selected_count),
                        "active_adapt_normal_count": int(batch_selected_normal_count),
                    },
                )
            elif str(args.tta_method) == "active_svm_boundary":
                fit_before = fit_active_score_svm(
                    active_labeled_features,
                    active_labeled_labels,
                source_feature_stats=source_feature_stats,
            ),
        )
    return batches


@torch.no_grad()
def update_ema_model(
    ema_model: AnomalibReverseDistillationModel,
    model: AnomalibReverseDistillationModel,
    decay: float,
) -> None:
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for name, ema_value in ema_state.items():
        model_value = model_state[name].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(float(decay)).add_(model_value, alpha=1.0 - float(decay))
        else:
            ema_value.copy_(model_value)
    ema_model.eval()


def evaluate_tta(
    source_model: AnomalibReverseDistillationModel,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    trace_path: Path,
    source_feature_stats: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, int, int, int, int, int, int, int]:
    score_source = str(args.tta_score_source)
    source_selection_batches = (
        precompute_source_selection_batches(
            source_model,
            loader=loader,
            args=args,
            device=device,
            source_feature_stats=source_feature_stats,
        )
        if score_source == "frozen"
        else None
    )
    source_model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = copy.deepcopy(source_model).to(device).eval()
    score_model = copy.deepcopy(model).to(device).eval() if score_source == "adapted_ema" else None
    trainable_params = configure_tta_parameters(model=model, args=args)
    optimizer = torch.optim.Adam(trainable_params, lr=float(args.tta_lr))
    selector = None
    if str(args.tta_method) == "bt_recsvm" and str(args.selector_mode) == "expblend":
        selector = ScoreOnlyBTRecSVMSelector(
            q=args.selector_q,
            tau=args.selector_tau,
            min_fraction=args.selector_min_fraction,
            min_history=args.selector_min_history,
            max_history=args.selector_max_history,
            svm_fit_history=args.selector_svm_fit_history,
            svm_normal_core_q=args.selector_svm_normal_core_q,
            svm_min_core_samples=args.selector_svm_min_core_samples,
            svm_inlier_q=args.selector_svm_inlier_q,
            svm_max_q=args.selector_svm_max_q,
            expblend_warmup_batches=args.svm_expblend_warmup_batches,
            expblend_tau_batches=args.svm_expblend_tau_batches,
            expblend_lambda_scale=args.svm_expblend_lambda_scale,
            expblend_lambda_cap=args.svm_expblend_lambda_cap,
        )
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    trace_rows: list[dict[str, Any]] = []
    selected_total = 0
    selected_normal_total = 0
    optimizer_steps = 0
    active_label_total = 0
    active_label_normal_total = 0
    active_label_anomaly_total = 0
    active_label_anomaly_reverse_total = 0
    active_tail_pseudo_label_total = 0
    active_tail_pseudo_label_normal_total = 0
    active_tail_pseudo_label_anomaly_total = 0
    active_tail_pseudo_label_correct_total = 0
    score_history_batches: list[np.ndarray] = []
    replay_buffer_images: list[torch.Tensor] = []
    replay_buffer_scores: list[np.ndarray] = []
    replay_buffer_labels: list[np.ndarray] = []
    reliability_gate_open = False
    reliability_gate_open_batch: int | None = None
    baseline_state: dict[str, Any] = {}
    active_labeled_features: list[np.ndarray] = []
    active_labeled_labels: list[int] = []
    active_labeled_weights: list[float] = []
    stream_tail_features: list[np.ndarray] = []
    stream_tail_scores: list[float] = []
    stream_tail_labels: list[int] = []
    stream_tail_selected_ids: set[int] = set()

    try:
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy().astype(np.int64)
            if score_source == "frozen":
                if source_selection_batches is None:
                    raise RuntimeError("Frozen selection batches were not precomputed")
                selection_features, source_scores = source_selection_batches[batch_index - 1]
            elif score_source == "adapted":
                selection_features, source_scores = selection_features_from_model(
                    model=model,
                    images=images,
                    args=args,
                    source_feature_stats=source_feature_stats,
                )
            elif score_source == "adapted_ema":
                if score_model is None:
                    raise RuntimeError("EMA score model was not initialized")
                selection_features, source_scores = selection_features_from_model(
                    model=score_model,
                    images=images,
                    args=args,
                    source_feature_stats=source_feature_stats,
                )
            else:
                raise ValueError(f"Unsupported TTA score source: {score_source}")
            batch_global_ids = np.asarray([], dtype=np.int64)
            if str(args.tta_method) == "active_svm_boundary" and str(args.active_svm_tail_scope) == "stream_past":
                stream_start = len(stream_tail_scores)
                batch_global_ids = np.arange(stream_start, stream_start + int(labels.shape[0]), dtype=np.int64)
                stream_tail_features.extend(feature.copy() for feature in selection_features)
                stream_tail_scores.extend(float(score) for score in source_scores)
                stream_tail_labels.extend(int(label) for label in labels)
            steps = 0
            adapt_loss = float("nan")
            trace_extra: dict[str, Any] = {}
            batch_selected_count = 0
            batch_selected_normal_count = 0
            batch_selected_purity = float("nan")
            threshold: float | None = None
            if str(args.tta_method) in {"tent", "sar"}:
                label_tensor = torch.from_numpy(labels).to(device=device, dtype=torch.long)
                steps, adapt_loss, batch_selected_count, batch_selected_normal_count = adapt_on_batch_baseline(
                    model=model,
                    optimizer=optimizer,
                    images=images,
                    labels=label_tensor,
                    trainable_params=trainable_params,
                    state=baseline_state,
                    args=args,
                    device=device,
                )
                if batch_selected_count > 0:
                    batch_selected_purity = float(batch_selected_normal_count / batch_selected_count)
                selected_total += int(batch_selected_count)
                selected_normal_total += int(batch_selected_normal_count)
                optimizer_steps += int(steps)
                history_count_after = int(batch_index)
                trace_extra.update(
                    {
                        "baseline_tta_method": str(args.tta_method),
                        "active_label_count": int(batch_selected_count),
                        "active_label_normal_count": int(batch_selected_normal_count),
                    },
                )
            elif str(args.tta_method) in {"atta", "eatta"}:
                (
                    steps,
                    adapt_loss,
                    batch_selected_count,
                    batch_selected_normal_count,
                    active_query_count,
                    active_query_normal_count,
                    active_query_anomaly_count,
                    ad_active_trace,
                ) = adapt_on_batch_anomaly_active(
                    model=model,
                    optimizer=optimizer,
                    images=images,
                    labels=labels,
                    source_scores=source_scores,
                    state=baseline_state,
                    args=args,
                    batch_index=int(batch_index),
                )
                if batch_selected_count > 0:
                    batch_selected_purity = float(batch_selected_normal_count / batch_selected_count)
                selected_total += int(batch_selected_count)
                selected_normal_total += int(batch_selected_normal_count)
                optimizer_steps += int(steps)
        "stream_seed": stream_seed(args),
        "n_train_normal": int(n_train_normal),
        "n_images": int(labels.shape[0]),
        "n_normal": int(np.sum(labels == 0)),
        "n_anomaly": int(np.sum(labels == 1)),
        "image_auroc": metrics["auroc"],
        "image_ap": metrics["ap"],
        "image_f1_max": metrics["f1_max"],
        "image_fpr95": metrics["fpr95"],
        "selected_pseudo_normal_count": int(selected_total),
        "selected_pseudo_normal_purity": purity,
        "optimizer_steps": int(optimizer_steps),
        "tta_lr": float(args.tta_lr),
        "tta_steps": int(args.tta_steps),
                        "active_adapt_normal_count": int(batch_selected_normal_count),
                    },
                )
            elif str(args.tta_method) == "active_svm_boundary":
                fit_before = fit_active_score_svm(
                    active_labeled_features,
                    active_labeled_labels,
                    sample_weights=active_labeled_weights,
                    model_name=str(args.active_boundary_model),
                )
                query_perturb_scores: np.ndarray | None = None
                if str(args.active_query_mode) == "boundary_perturb_high":
                    query_score_model = score_model if score_model is not None else model
                    query_perturb_scores = query_perturbation_scores(
                        model=query_score_model,
                        images=images,
                        args=args,
                        batch_index=int(batch_index),
                    )
                query_index, query_mode = select_active_query_index(
            if str(args.tta_method) == "active_svm_boundary" and str(args.active_svm_tail_scope) == "stream_past":
                stream_start = len(stream_tail_scores)
                batch_global_ids = np.arange(stream_start, stream_start + int(labels.shape[0]), dtype=np.int64)
                stream_tail_features.extend(feature.copy() for feature in selection_features)
                stream_tail_scores.extend(float(score) for score in source_scores)
                stream_tail_labels.extend(int(label) for label in labels)
            steps = 0
            adapt_loss = float("nan")
            trace_extra: dict[str, Any] = {}
            batch_selected_count = 0
            batch_selected_normal_count = 0
            batch_selected_purity = float("nan")
            threshold: float | None = None
            if str(args.tta_method) in {"tent", "sar"}:
                label_tensor = torch.from_numpy(labels).to(device=device, dtype=torch.long)
                steps, adapt_loss, batch_selected_count, batch_selected_normal_count = adapt_on_batch_baseline(
                    model=model,
                    optimizer=optimizer,
                    images=images,
                    labels=label_tensor,
                    trainable_params=trainable_params,
                    state=baseline_state,
                    args=args,
                    device=device,
                )
                if batch_selected_count > 0:
                    batch_selected_purity = float(batch_selected_normal_count / batch_selected_count)
                selected_total += int(batch_selected_count)
                selected_normal_total += int(batch_selected_normal_count)
                optimizer_steps += int(steps)
                history_count_after = int(batch_index)
                trace_extra.update(
                    {
                        "baseline_tta_method": str(args.tta_method),
                        "active_label_count": int(batch_selected_count),
                        "active_label_normal_count": int(batch_selected_normal_count),
                    },
                )
            elif str(args.tta_method) in {"atta", "eatta"}:
                (
                    steps,
                    adapt_loss,
                    batch_selected_count,
                    batch_selected_normal_count,
                    active_query_count,
                    active_query_normal_count,
                    active_query_anomaly_count,
                    ad_active_trace,
                ) = adapt_on_batch_anomaly_active(
                    model=model,
                    optimizer=optimizer,
                    images=images,
                    labels=labels,
                    source_scores=source_scores,
                    state=baseline_state,
                    args=args,
                    batch_index=int(batch_index),
                )
                if batch_selected_count > 0:
                    batch_selected_purity = float(batch_selected_normal_count / batch_selected_count)
                selected_total += int(batch_selected_count)
                selected_normal_total += int(batch_selected_normal_count)
                optimizer_steps += int(steps)
                active_label_total += int(active_query_count)
                active_label_normal_total += int(active_query_normal_count)
                active_label_anomaly_total += int(active_query_anomaly_count)
                active_label_anomaly_reverse_total += int(ad_active_trace.get("ad_active_reverse_anomaly_count", 0))
                history_count_after = int(batch_index)
                trace_extra.update(ad_active_trace)
                trace_extra.update(
                    {
                        "baseline_tta_method": str(args.tta_method),
                        "active_label_count": int(active_query_count),
                        "active_label_normal_count": int(active_query_normal_count),
                        "active_label_anomaly_count": int(active_query_anomaly_count),
                        "active_anomaly_reverse_lr": bool(args.active_anomaly_reverse_lr),
                        "active_adapt_sample_count": int(batch_selected_count),
                        "active_adapt_normal_count": int(batch_selected_normal_count),
                    },
                )
            elif str(args.tta_method) == "active_svm_boundary":
                fit_before = fit_active_score_svm(
                    active_labeled_features,
                    active_labeled_labels,
                    sample_weights=active_labeled_weights,
                    model_name=str(args.active_boundary_model),
                )
                query_perturb_scores: np.ndarray | None = None
                if str(args.active_query_mode) == "boundary_perturb_high":
                    query_score_model = score_model if score_model is not None else model
                    query_perturb_scores = query_perturbation_scores(
                        model=query_score_model,
                        images=images,
                        args=args,
                        batch_index=int(batch_index),
                    )
                query_index, query_mode = select_active_query_index(
                    source_scores,
                    fit=fit_before,
                    features=selection_features,
                    query_mode=str(args.active_query_mode),
                    perturb_scores=query_perturb_scores,
                    perturb_candidate_count=int(args.active_query_perturb_candidate_count),
                )
                query_score = float(source_scores[query_index])
                query_perturb_score = (
                    float(query_perturb_scores[query_index])
                    if query_perturb_scores is not None and int(query_index) < int(query_perturb_scores.shape[0])
                    else float("nan")
                )
                query_label = int(labels[query_index])
                active_labeled_features.append(selection_features[query_index].copy())
                active_labeled_labels.append(query_label)
                active_labeled_weights.append(1.0)
                active_label_total += 1
                active_label_normal_total += int(query_label == 0)
                active_label_anomaly_total += int(query_label == 1)

                tail_global_ids = np.asarray([], dtype=np.int64)
                if str(args.active_svm_tail_scope) == "stream_past":
                    query_global_id = int(batch_global_ids[query_index])
                    raw_tail_indices, raw_tail_pseudo_labels = select_stream_tail_pseudo_labels(
                        stream_tail_scores,
                        fraction=float(args.active_svm_tail_pseudo_label_fraction),
                        excluded_ids={query_global_id},
                        selected_ids=stream_tail_selected_ids,
                    )
                    tail_global_ids = raw_tail_indices
                else:
                    raw_tail_indices, raw_tail_pseudo_labels = select_score_tail_pseudo_labels(
                        source_scores,
                        fraction=float(args.active_svm_tail_pseudo_label_fraction),
                        exclude_index=int(query_index),
                    )
                tail_indices_list: list[int] = []
                tail_global_ids_list: list[int] = []
                tail_pseudo_label_list: list[int] = []
                tail_weight_list: list[float] = []
                lower_tail_weight = float(args.active_svm_lower_tail_pseudo_normal_weight)
                upper_tail_weight = float(args.active_svm_upper_tail_pseudo_anomaly_weight)
                for tail_index, tail_pseudo_label in zip(raw_tail_indices, raw_tail_pseudo_labels, strict=True):
                    tail_weight = lower_tail_weight if int(tail_pseudo_label) == 0 else upper_tail_weight
                    if tail_weight <= 0.0:
                        continue
                    if str(args.active_svm_tail_scope) == "stream_past":
                        tail_global_ids_list.append(int(tail_index))
                        current_positions = np.flatnonzero(batch_global_ids == int(tail_index))
                        if current_positions.size > 0:
                            tail_indices_list.append(int(current_positions[0]))
                    else:
                        tail_indices_list.append(int(tail_index))
                    tail_pseudo_label_list.append(int(tail_pseudo_label))
                    tail_weight_list.append(float(tail_weight))
                tail_indices = np.asarray(tail_indices_list, dtype=np.int64)
                tail_global_ids = np.asarray(tail_global_ids_list, dtype=np.int64)
                tail_pseudo_labels = np.asarray(tail_pseudo_label_list, dtype=np.int64)
                tail_pseudo_label_count = int(tail_indices.size)
                if str(args.active_svm_tail_scope) == "stream_past":
                    tail_pseudo_label_count = int(len(tail_pseudo_label_list))
                    tail_true_labels = np.asarray(
                        [stream_tail_labels[int(global_id)] for global_id in tail_global_ids[:tail_pseudo_label_count]],
                        dtype=np.int64,
                    )
                    tail_features = [stream_tail_features[int(global_id)] for global_id in tail_global_ids[:tail_pseudo_label_count]]
                else:
                    tail_true_labels = labels[tail_indices] if tail_pseudo_label_count > 0 else np.asarray([], dtype=np.int64)
                    tail_features = [selection_features[int(tail_index)].copy() for tail_index in tail_indices]
                self_revise_stats = {
                    "checked_count": 0,
                    "confusing_count": 0,
                    "removed_count": 0,
                    "downweighted_count": 0,
                }
                raw_stream_tail_global_ids = tail_global_ids.copy()
                if str(args.active_svm_self_revise_mode) != "none" and tail_pseudo_label_count > 0:
                    fit_self_revise = fit_active_score_svm(
                        active_labeled_features,
                        active_labeled_labels,
                        sample_weights=active_labeled_weights,
                        model_name=str(args.active_boundary_model),
                    )
                    keep_tail_mask, revised_tail_weight_list, self_revise_stats = revise_tail_pseudo_label_weights(
                        fit=fit_self_revise,
                        features=tail_features,
                        pseudo_labels=tail_pseudo_labels,
                        weights=tail_weight_list,
                        mode=str(args.active_svm_self_revise_mode),
                        margin=float(args.active_svm_self_revise_margin),
                        weight_scale=float(args.active_svm_self_revise_weight_scale),
                    )
                    if not np.all(keep_tail_mask):
                        tail_features = [
                            tail_feature
                            for tail_feature, keep_tail in zip(tail_features, keep_tail_mask, strict=True)
                            if bool(keep_tail)
                        ]
                        tail_pseudo_labels = tail_pseudo_labels[keep_tail_mask]
                        tail_true_labels = tail_true_labels[keep_tail_mask]
                        if str(args.active_svm_tail_scope) == "stream_past":
                            tail_global_ids = tail_global_ids[keep_tail_mask]
                            tail_indices = np.asarray(
                                [
                                    int(current_positions[0])
                                    for global_id in tail_global_ids
                                    for current_positions in [np.flatnonzero(batch_global_ids == int(global_id))]
                                    if current_positions.size > 0
                                ],
                                dtype=np.int64,
                            )
                        else:
                            tail_indices = tail_indices[keep_tail_mask]
                    tail_weight_list = [
                        float(weight)
                        for weight, keep_tail in zip(revised_tail_weight_list, keep_tail_mask, strict=True)
                        if bool(keep_tail)
                    ]
                    tail_pseudo_label_count = int(tail_pseudo_labels.size)
                if str(args.active_svm_tail_scope) == "stream_past":
                    for global_id in raw_stream_tail_global_ids:
                        stream_tail_selected_ids.add(int(global_id))
                tail_pseudo_label_normal_count = int(np.sum(tail_pseudo_labels == 0)) if tail_pseudo_label_count > 0 else 0
                tail_pseudo_label_anomaly_count = int(tail_pseudo_label_count - tail_pseudo_label_normal_count)
                tail_pseudo_label_correct_count = (
                    int(np.sum(tail_true_labels == tail_pseudo_labels)) if tail_pseudo_label_count > 0 else 0
                )
                for tail_feature, tail_pseudo_label, tail_weight in zip(
                    tail_features,
                    tail_pseudo_labels,
                    tail_weight_list,
                    strict=True,
                ):
                    active_labeled_features.append(tail_feature.copy())
                    active_labeled_labels.append(int(tail_pseudo_label))
                    active_labeled_weights.append(float(tail_weight))
                active_tail_pseudo_label_total += tail_pseudo_label_count
                active_tail_pseudo_label_normal_total += tail_pseudo_label_normal_count
                active_tail_pseudo_label_anomaly_total += tail_pseudo_label_anomaly_count
                active_tail_pseudo_label_correct_total += tail_pseudo_label_correct_count
                active_tail_self_revise_checked_total += int(self_revise_stats["checked_count"])
                active_tail_self_revise_confusing_total += int(self_revise_stats["confusing_count"])
                active_tail_self_revise_removed_total += int(self_revise_stats["removed_count"])
                active_tail_self_revise_downweighted_total += int(self_revise_stats["downweighted_count"])

                fit_after = fit_active_score_svm(
                    active_labeled_features,
                    active_labeled_labels,
                    sample_weights=active_labeled_weights,
                    model_name=str(args.active_boundary_model),
                )
                pseudo_mask = np.zeros_like(labels, dtype=bool)
                adapt_mask = np.zeros_like(labels, dtype=bool)
                        "active_svm_pseudo_normal_purity": (
                            float(np.mean(labels[pseudo_mask] == 0)) if bool(pseudo_mask.any()) else float("nan")
                        ),
                        "active_anomaly_reverse_lr": bool(args.active_anomaly_reverse_lr),
                        "active_anomaly_target_mode": str(args.active_anomaly_target_mode),
                        "active_anomaly_target_score": float(anomaly_target_score),
                        "active_query_reverse_anomaly_count": int(batch_reverse_anomaly_count),
                        "active_query_target_anomaly_count": int(batch_target_anomaly_count),
                        "active_adapt_sample_count": int(batch_selected_count),
                        "active_adapt_normal_count": int(batch_selected_normal_count),
                        "active_update_mode": "per_sample_sequential",
                    },
                )
            elif str(args.selector_mode) == "svm_reliability_replay":
                replay_buffered_before = int(sum(chunk.shape[0] for chunk in replay_buffer_scores))
        "selector_tau": float(args.selector_tau),
        "selector_min_history": int(args.selector_min_history),
        "svm_expblend_warmup_batches": int(args.svm_expblend_warmup_batches),
        "svm_expblend_tau_batches": float(args.svm_expblend_tau_batches),
        "svm_expblend_lambda_scale": float(args.svm_expblend_lambda_scale),
        "svm_expblend_lambda_cap": float(args.svm_expblend_lambda_cap),
        "selector_svm_normal_core_q": float(args.selector_svm_normal_core_q),
        "selector_svm_min_core_samples": int(args.selector_svm_min_core_samples),
        "selector_svm_inlier_q": float(args.selector_svm_inlier_q),
        "selector_svm_max_q": float(args.selector_svm_max_q),
        "selector_svm_nu": float(args.selector_svm_nu),
        "svm_reliability_method": str(args.svm_reliability_method),
        "svm_reliability_threshold": float(args.svm_reliability_threshold),
        "checkpoint_path": str(checkpoint),
        "trace_path": "" if trace_path is None else str(trace_path),
    }


def run_split(
    category: str,
    split_dir: Path,
    source_model: AnomalibReverseDistillationModel,
    train_dataset: RobustADImageDataset,
    checkpoint: Path,
    args: argparse.Namespace,
    device: torch.device,
    output_root: Path,
    source_feature_stats: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    split = split_token(split_dir, category=category)
    dataset = RobustADImageDataset(
        split_dir=split_dir,
        resize_size=int(args.resize_size),
        crop_size=int(args.crop_size),
        label_filter=None,
        stream_order=str(args.stream_order),
        seed=stream_seed(args),
    )
    loader = make_loader(dataset, args=args, batch_size=int(args.batch_size), shuffle=False)
    source_labels, source_scores = evaluate_source(model=source_model, loader=loader, device=device)
    source_metrics = binary_metrics(labels=source_labels, scores=source_scores)
    rows = [
        metric_row(
            category=category,
            split=split,
            method="AnomalibRD4AD_source",
            metrics=source_metrics,
            args=args,
            n_train_normal=len(train_dataset),
            labels=source_labels,
            checkpoint=checkpoint,
        ),
    ]
    if bool(args.run_tta):
        tta_method = tta_method_name(args)
        trace_path = output_root / "traces" / tta_method / category / f"{split}.json"
        (
            tta_labels,
            tta_scores,
            selected_total,
            selected_normal_total,
            optimizer_steps,
            active_label_total,
            active_label_normal_total,
            active_label_anomaly_total,
            active_label_anomaly_reverse_total,
            active_tail_pseudo_label_total,
            active_tail_pseudo_label_normal_total,
            active_tail_pseudo_label_anomaly_total,
            active_tail_pseudo_label_correct_total,
        ) = evaluate_tta(
            source_model=source_model,
                    selected_normal_total += batch_selected_normal_count
                    optimizer_steps += int(steps)
                selector.update(source_scores)
                history_count_after = int(len(selector.score_history))
                trace_extra.update(
                    {
                        "svm_boot_threshold": selection.boot_threshold,
                        "svm_score_threshold": selection.svm_score_threshold,
                        "svm_mixed_threshold": selection.svm_mixed_threshold,
                        "svm_blend_lambda": selection.svm_blend_lambda,
                        "svm_lambda_raw": selection.svm_lambda_raw,
                        "svm_active": selection.svm_active,
                        "svm_history_count": selection.svm_history_count,
                        "svm_history_batches": selection.svm_history_batches,
                        "svm_core_count": selection.svm_core_count,
                    },
                )
            if score_source == "adapted_ema" and score_model is not None and int(steps) > 0:
                update_ema_model(ema_model=score_model, model=model, decay=float(args.tta_score_ema_decay))
            scores = rd4ad_scores_with_optional_aug(model=model, images=images, args=args)
            labels_all.append(labels)
            scores_all.append(scores)
            row = {
                active_label_anomaly_reverse_total=active_label_anomaly_reverse_total,
                active_tail_pseudo_label_total=active_tail_pseudo_label_total,
                active_tail_pseudo_label_normal_total=active_tail_pseudo_label_normal_total,
                active_tail_pseudo_label_anomaly_total=active_tail_pseudo_label_anomaly_total,
                active_tail_pseudo_label_correct_total=active_tail_pseudo_label_correct_total,
                trace_path=trace_path,
            ),
        )
    table2_value = TABLE2_RD4AD_AUROC.get((category, split))
    if table2_value is not None:
        print(
            f"[Anomalib-RD4AD] {category}/{split} source={source_metrics['auroc'] * 100:.2f} "
            f"table2={table2_value:.2f} delta={source_metrics['auroc'] * 100 - table2_value:+.2f}",
            flush=True,
        )
    return rows


def run_category(category: str, args: argparse.Namespace, device: torch.device, output_root: Path) -> list[dict[str, Any]]:
    data_root = resolve_path(args.data_root)
    train_dir = train_split_dir(data_root=data_root, category=category)
    train_dataset = RobustADImageDataset(
        split_dir=train_dir,
        resize_size=int(args.resize_size),
        crop_size=int(args.crop_size),
        label_filter=0,
        stream_order="sequential",
        seed=int(args.seed),
    )
    if bool(args.cache_train_images):
        train_dataset = CachedTensorDataset(train_dataset, desc=f"Cache train {category}")
    source_model, checkpoint = train_or_load_rd4ad(
        category=category,
        train_dataset=train_dataset,
        args=args,
        device=device,
        output_root=output_root,
    )
    source_feature_stats = None
    if bool(args.run_tta) and str(args.tta_method) == "active_svm_boundary" and str(args.active_svm_feature_mode) == "map_stats":
        source_feature_stats = compute_source_feature_stats(
            model=source_model,
            train_dataset=train_dataset,
            args=args,
            device=device,
        )
    rows: list[dict[str, Any]] = []
    for split_dir in test_split_dirs(data_root=data_root, category=category, splits=args.splits):
        rows.extend(
            run_split(
                category=category,
                split_dir=split_dir,
                source_model=source_model,
                train_dataset=train_dataset,
    selected_normal_total: int = 0,
    optimizer_steps: int = 0,
    active_label_total: int = 0,
    active_label_normal_total: int = 0,
    active_label_anomaly_total: int = 0,
    active_label_anomaly_reverse_total: int = 0,
    active_tail_pseudo_label_total: int = 0,
    active_tail_pseudo_label_normal_total: int = 0,
    active_tail_pseudo_label_anomaly_total: int = 0,
    active_tail_pseudo_label_correct_total: int = 0,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    purity = float(selected_normal_total / selected_total) if selected_total > 0 else float("nan")
    tail_pseudo_label_accuracy = (
        float(active_tail_pseudo_label_correct_total / active_tail_pseudo_label_total)
        if active_tail_pseudo_label_total > 0
        else float("nan")
    )
    return {
        "category": category,
        "split": split,
        "domain": domain_name(category, split),
        "method": method,
        "backbone": args.backbone,
        "image_size": int(args.crop_size),
        "stream_order": args.stream_order,
        "seed": int(args.seed),
        "stream_seed": stream_seed(args),
        "n_train_normal": int(n_train_normal),
        "n_images": int(labels.shape[0]),
        "n_normal": int(np.sum(labels == 0)),
        "n_anomaly": int(np.sum(labels == 1)),
        "image_auroc": metrics["auroc"],
        "image_ap": metrics["ap"],
        "image_f1_max": metrics["f1_max"],
        "image_fpr95": metrics["fpr95"],
        "selected_pseudo_normal_count": int(selected_total),
        "selected_pseudo_normal_purity": purity,
        "optimizer_steps": int(optimizer_steps),
        "tta_lr": float(args.tta_lr),
        "tta_steps": int(args.tta_steps),
        "tta_method": str(args.tta_method),
        "tta_param_scope": str(args.tta_param_scope),
        "entropy_margin": float(args.entropy_margin),
        "atta_oracle_num": int(args.atta_oracle_num),
        "atta_query_strategy": str(args.atta_query_strategy),
        "eatta_oracle_num": int(args.eatta_oracle_num),
        "eatta_noise_std": float(args.eatta_noise_std),
        "eatta_weight_momentum": float(args.eatta_weight_momentum),
        "eatta_class_history": int(args.eatta_class_history),
        "eatta_pseudo_normal_score_q": float(args.eatta_pseudo_normal_score_q),
        "eatta_pseudo_normal_entropy_q": float(args.eatta_pseudo_normal_entropy_q),
        "eatta_pseudo_normal_stability_q": float(args.eatta_pseudo_normal_stability_q),
        "eatta_pseudo_normal_max_fraction": float(args.eatta_pseudo_normal_max_fraction),
        "sar_rho": float(args.sar_rho),
        "tta_pos_scale_aug": bool(args.tta_pos_scale_aug),
        "tta_pos_scale_translate_frac": float(args.tta_pos_scale_translate_frac),
        "tta_pos_scale_scale_low": float(sorted(args.tta_pos_scale_scales)[0]),
        "tta_pos_scale_scale_high": float(sorted(args.tta_pos_scale_scales)[1]),
        "tta_pos_scale_infer_agg": str(args.tta_pos_scale_infer_agg),
        "tta_score_source": str(args.tta_score_source),
        "tta_score_ema_decay": float(args.tta_score_ema_decay),
        "active_svm_feature_mode": str(args.active_svm_feature_mode),
        "active_svm_include_encoder3_feature": bool(getattr(args, "active_svm_include_encoder3_feature", False)),
        "active_svm_encoder_feature_layers": "-".join(
            str(layer) for layer in active_svm_encoder_feature_layers(args)
        ),
        "active_boundary_model": str(args.active_boundary_model),
        "active_query_mode": str(args.active_query_mode),
        "active_query_perturb_mode": str(args.active_query_perturb_mode),
        "active_query_perturb_candidate_count": int(args.active_query_perturb_candidate_count),
        "active_query_perturb_std": float(args.active_query_perturb_std),
        "active_svm_source_pixel_q": float(args.active_svm_source_pixel_q),
        "active_svm_confidence_threshold": float(args.active_svm_confidence_threshold),
        "active_svm_tail_scope": str(args.active_svm_tail_scope),
        "active_svm_tail_pseudo_label_fraction": float(args.active_svm_tail_pseudo_label_fraction),
        "active_svm_lower_tail_pseudo_normal_weight": float(args.active_svm_lower_tail_pseudo_normal_weight),
        "active_svm_upper_tail_pseudo_anomaly_weight": float(args.active_svm_upper_tail_pseudo_anomaly_weight),
        "active_anomaly_reverse_lr": bool(args.active_anomaly_reverse_lr),
        "active_anomaly_target_mode": str(args.active_anomaly_target_mode),
        "active_label_count": int(active_label_total),
        "active_label_normal_count": int(active_label_normal_total),
        "active_label_anomaly_count": int(active_label_anomaly_total),
        "active_label_anomaly_reverse_count": int(active_label_anomaly_reverse_total),
        "active_tail_pseudo_label_count": int(active_tail_pseudo_label_total),
        "active_tail_pseudo_label_normal_count": int(active_tail_pseudo_label_normal_total),
        "active_tail_pseudo_label_anomaly_count": int(active_tail_pseudo_label_anomaly_total),
        "active_tail_pseudo_label_accuracy": tail_pseudo_label_accuracy,
        "selector_mode": str(args.selector_mode),
        "selector_q": float(args.selector_q),
        "selector_tau": float(args.selector_tau),
        "selector_min_history": int(args.selector_min_history),
        "svm_expblend_warmup_batches": int(args.svm_expblend_warmup_batches),
        "svm_expblend_tau_batches": float(args.svm_expblend_tau_batches),
                category=category,
                split=split,
                method=tta_method,
                metrics=tta_metrics,
                args=args,
                n_train_normal=len(train_dataset),
                labels=tta_labels,
                checkpoint=checkpoint,
                selected_total=selected_total,
                selected_normal_total=selected_normal_total,
                optimizer_steps=optimizer_steps,
                active_label_total=active_label_total,
                active_label_normal_total=active_label_normal_total,
                active_label_anomaly_total=active_label_anomaly_total,
                active_label_anomaly_reverse_total=active_label_anomaly_reverse_total,
                active_tail_pseudo_label_total=active_tail_pseudo_label_total,
                active_tail_pseudo_label_normal_total=active_tail_pseudo_label_normal_total,

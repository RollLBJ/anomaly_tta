"""Deep analysis: test2-5 vs test6 for MetalParts (the only category with test6).

Extracts per-split boundary stats to understand why test6 under-adapts.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/qilab/byeongju_lee/anomaly_tta")

CONDITIONS = {
    "conf0.25": ROOT / "outputs/seed012_btta_sptail0.01_bs32_decoder_bn_only_gpu0",
    "conf0.15": ROOT / "outputs/seed01_btta_sptail0.01_conf0.15_bs32_decoder_bn_only_gpu0",
    "conf0.20": ROOT / "outputs/seed01_btta_sptail0.01_conf0.20_bs32_decoder_bn_only_gpu0",
}

SEEDS = [0, 1, 2]

# Key columns
COLS = [
    "category", "split", "method", "stream_seed",
    "n_images", "n_normal", "n_anomaly",
    "image_auroc",
    "selected_pseudo_normal_count", "selected_pseudo_normal_purity",
    "optimizer_steps",
    "active_label_count",
    "active_tail_pseudo_label_count", "active_tail_pseudo_label_accuracy",
]


def load_robustad(cond_dir, seeds):
    frames = []
    for s in seeds:
        p = cond_dir / f"robustad_streamseed{s}/btta_12d_mean/robustad_detailed.csv"
        if p.exists():
            df = pd.read_csv(p, usecols=COLS)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def fmt(mean, std):
    return f"${mean:.2f} \\pm {std:.2f}$"


def analyze_metalparts(cond_name, df):
    """Analyze MetalParts per-split for BTTA method."""
    mp = df[df["category"] == "MetalParts"].copy()
    btta = mp[mp["method"].str.contains("ActiveBoundary")]
    source = mp[~mp["method"].str.contains("ActiveBoundary")]

    print(f"\n#### {cond_name} — MetalParts per-split (BTTA)")
    print("| Split | Source AUROC | BTTA AUROC | Δ AUROC | PseudoN Count | PseudoN Purity | Optim Steps | Tail PL Count | Tail PL Acc |")
    print("|:---|:---|:---|:---|:---|:---|:---|:---|:---|")

    for split in ["test1", "test2", "test3", "test4", "test5", "test6"]:
        s_sub = source[source["split"] == split]
        b_sub = btta[btta["split"] == split]
        if s_sub.empty or b_sub.empty:
            continue

        s_auroc = s_sub["image_auroc"].mean() * 100
        b_auroc_m = b_sub["image_auroc"].mean() * 100
        b_auroc_s = b_sub["image_auroc"].std(ddof=1) * 100 if len(b_sub) > 1 else 0

        delta = b_auroc_m - s_auroc

        count_m = b_sub["selected_pseudo_normal_count"].mean()
        count_s = b_sub["selected_pseudo_normal_count"].std(ddof=1) if len(b_sub) > 1 else 0

        purity_m = b_sub["selected_pseudo_normal_purity"].mean() * 100
        purity_s = b_sub["selected_pseudo_normal_purity"].std(ddof=1) * 100 if len(b_sub) > 1 else 0

        steps_m = b_sub["optimizer_steps"].mean()
        steps_s = b_sub["optimizer_steps"].std(ddof=1) if len(b_sub) > 1 else 0

        tail_count_m = b_sub["active_tail_pseudo_label_count"].mean()
        tail_count_s = b_sub["active_tail_pseudo_label_count"].std(ddof=1) if len(b_sub) > 1 else 0

        tail_acc_m = b_sub["active_tail_pseudo_label_accuracy"].mean() * 100
        tail_acc_s = b_sub["active_tail_pseudo_label_accuracy"].std(ddof=1) * 100 if len(b_sub) > 1 else 0

        sign = "+" if delta > 0 else ""
        print(f"| {split} | {s_auroc:.2f} | {fmt(b_auroc_m, b_auroc_s)} | {sign}{delta:.2f} | {fmt(count_m, count_s)} | {fmt(purity_m, purity_s)} | {fmt(steps_m, steps_s)} | {fmt(tail_count_m, tail_count_s)} | {fmt(tail_acc_m, tail_acc_s)} |")


def analyze_all_categories_test6_vs_rest(cond_name, df):
    """Compare test6 vs test2-5 aggregated across all categories."""
    btta = df[df["method"].str.contains("ActiveBoundary")].copy()
    source = df[~df["method"].str.contains("ActiveBoundary")].copy()

    # test2-5 group vs test6 group
    groups = {
        "test2-5": ["test2", "test3", "test4", "test5"],
        "test6": ["test6"],
    }

    print(f"\n#### {cond_name} — test2-5 vs test6 (all categories, BTTA)")
    print("| Group | Source AUROC | BTTA AUROC | Δ | PseudoN Count | PseudoN Purity | Anomaly Ratio |")
    print("|:---|:---|:---|:---|:---|:---|:---|")

    for gname, splits in groups.items():
        s_sub = source[source["split"].isin(splits)]
        b_sub = btta[btta["split"].isin(splits)]
        if s_sub.empty or b_sub.empty:
            continue

        # per-seed average, then mean/std
        s_auroc_per_seed = s_sub.groupby("stream_seed")["image_auroc"].mean() * 100
        b_auroc_per_seed = b_sub.groupby("stream_seed")["image_auroc"].mean() * 100
        delta_per_seed = b_auroc_per_seed - s_auroc_per_seed

        count_per_seed = b_sub.groupby("stream_seed")["selected_pseudo_normal_count"].mean()
        purity_per_seed = b_sub.groupby("stream_seed")["selected_pseudo_normal_purity"].mean() * 100

        # anomaly ratio
        anom_ratio = b_sub["n_anomaly"].sum() / b_sub["n_images"].sum() * 100

        print(f"| {gname} | {fmt(s_auroc_per_seed.mean(), s_auroc_per_seed.std(ddof=1) if len(s_auroc_per_seed)>1 else 0)} | {fmt(b_auroc_per_seed.mean(), b_auroc_per_seed.std(ddof=1) if len(b_auroc_per_seed)>1 else 0)} | {fmt(delta_per_seed.mean(), delta_per_seed.std(ddof=1) if len(delta_per_seed)>1 else 0)} | {fmt(count_per_seed.mean(), count_per_seed.std(ddof=1) if len(count_per_seed)>1 else 0)} | {fmt(purity_per_seed.mean(), purity_per_seed.std(ddof=1) if len(purity_per_seed)>1 else 0)} | {anom_ratio:.1f}% |")


def main():
    for cond_name, cond_dir in CONDITIONS.items():
        df = load_robustad(cond_dir, SEEDS)
        if df is None:
            continue
        analyze_metalparts(cond_name, df)
        analyze_all_categories_test6_vs_rest(cond_name, df)


if __name__ == "__main__":
    main()

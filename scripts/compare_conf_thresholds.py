"""Compare SVM confidence threshold conditions (0.25, 0.15, 0.20).

Aggregates image_auroc across seeds 0,1,2 for RobustAD and AeBAD.
Outputs tables in $mean \\pm std$ format.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path("/home/qilab/byeongju_lee/anomaly_tta")

CONDITIONS = {
    "conf0.25": ROOT / "outputs/seed012_btta_sptail0.01_bs32_decoder_bn_only_gpu0",
    "conf0.15": ROOT / "outputs/seed01_btta_sptail0.01_conf0.15_bs32_decoder_bn_only_gpu0",
    "conf0.20": ROOT / "outputs/seed01_btta_sptail0.01_conf0.20_bs32_decoder_bn_only_gpu0",
}

SEEDS = [0, 1, 2]

DATASETS = {
    "RobustAD": {
        "pattern": "robustad_streamseed{seed}/btta_12d_mean/robustad_detailed.csv",
    },
    "AeBAD": {
        "pattern": "aebad_streamseed{seed}/btta_12d_mean/robustad_detailed.csv",
    },
}


def load_all(cond_dir, dataset_key, seeds):
    """Load and concat CSVs for all seeds."""
    pattern = DATASETS[dataset_key]["pattern"]
    frames = []
    for s in seeds:
        p = cond_dir / pattern.format(seed=s)
        if not p.exists():
            print(f"WARNING: missing {p}", file=sys.stderr)
            continue
        df = pd.read_csv(p)
        df["stream_seed"] = s
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def is_btta(method_str):
    return "ActiveBoundary" in str(method_str)


def fmt(mean, std):
    return f"${mean:.2f} \\pm {std:.2f}$"


def summarize_per_split(df):
    """Per-split AUROC: average across categories within each seed, then mean±std across seeds."""
    df = df.copy()
    df["is_btta"] = df["method"].apply(is_btta)

    results = {}
    for split in sorted(df["split"].unique()):
        split_df = df[df["split"] == split]
        for label, btta_flag in [("Source", False), ("BTTA", True)]:
            sub = split_df[split_df["is_btta"] == btta_flag]
            if sub.empty:
                continue
            # Average across categories per seed, then mean/std across seeds
            per_seed = sub.groupby("stream_seed")["image_auroc"].mean() * 100
            results[(split, label)] = (per_seed.mean(), per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0)

    # Overall (all splits)
    for label, btta_flag in [("Source", False), ("BTTA", True)]:
        sub = df[df["is_btta"] == btta_flag]
        if sub.empty:
            continue
        per_seed = sub.groupby("stream_seed")["image_auroc"].mean() * 100
        results[("Overall", label)] = (per_seed.mean(), per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0)

    return results


def summarize_pseudo_stats(df):
    """Pseudo-normal count and purity stats for BTTA rows."""
    btta = df[df["method"].apply(is_btta)].copy()
    if btta.empty:
        return {}

    results = {}
    for split in sorted(btta["split"].unique()):
        sub = btta[btta["split"] == split]
        # Per seed: average count and purity across categories
        count_per_seed = sub.groupby("stream_seed")["selected_pseudo_normal_count"].mean()
        purity_per_seed = sub.groupby("stream_seed")["selected_pseudo_normal_purity"].mean() * 100
        results[split] = {
            "count": (count_per_seed.mean(), count_per_seed.std(ddof=1) if len(count_per_seed) > 1 else 0.0),
            "purity": (purity_per_seed.mean(), purity_per_seed.std(ddof=1) if len(purity_per_seed) > 1 else 0.0),
        }

    # Overall
    count_per_seed = btta.groupby("stream_seed")["selected_pseudo_normal_count"].mean()
    purity_per_seed = btta.groupby("stream_seed")["selected_pseudo_normal_purity"].mean() * 100
    results["Overall"] = {
        "count": (count_per_seed.mean(), count_per_seed.std(ddof=1) if len(count_per_seed) > 1 else 0.0),
        "purity": (purity_per_seed.mean(), purity_per_seed.std(ddof=1) if len(purity_per_seed) > 1 else 0.0),
    }
    return results


def print_auroc_table(dataset_name, all_results):
    """Print AUROC comparison table across conditions."""
    # Collect all splits
    splits = set()
    for cond, res in all_results.items():
        for (split, _) in res:
            splits.add(split)
    splits = sorted([s for s in splits if s != "Overall"]) + ["Overall"]

    # Header
    cond_names = list(all_results.keys())
    print(f"\n### {dataset_name} — Image AUROC (%) ↑")
    header = "| Split | Source |"
    for c in cond_names:
        header += f" BTTA ({c}) |"
    print(header)
    sep = "|:---|:---|"
    for _ in cond_names:
        sep += ":---|"
    print(sep)

    for split in splits:
        # Source is same across conditions, take from first
        src = all_results[cond_names[0]].get((split, "Source"))
        src_str = fmt(*src) if src else "-"
        row = f"| {split} | {src_str} |"
        for c in cond_names:
            val = all_results[c].get((split, "BTTA"))
            row += f" {fmt(*val) if val else '-'} |"
        print(row)


def print_pseudo_table(dataset_name, all_pseudo):
    """Print pseudo-normal count & purity comparison."""
    splits = set()
    for cond, res in all_pseudo.items():
        splits.update(res.keys())
    splits = sorted([s for s in splits if s != "Overall"]) + ["Overall"]

    cond_names = list(all_pseudo.keys())
    print(f"\n### {dataset_name} — Pseudo-Normal Count / Purity (%)")
    header = "| Split |"
    for c in cond_names:
        header += f" {c} Count | {c} Purity |"
    print(header)
    sep = "|:---|"
    for _ in cond_names:
        sep += ":---|:---|"
    print(sep)

    for split in splits:
        row = f"| {split} |"
        for c in cond_names:
            stats = all_pseudo[c].get(split)
            if stats:
                row += f" {fmt(*stats['count'])} | {fmt(*stats['purity'])} |"
            else:
                row += " - | - |"
        print(row)


def main():
    for ds_name in ["RobustAD", "AeBAD"]:
        all_auroc = {}
        all_pseudo = {}

        for cond_name, cond_dir in CONDITIONS.items():
            df = load_all(cond_dir, ds_name, SEEDS)
            if df is None:
                print(f"WARNING: no data for {cond_name}/{ds_name}", file=sys.stderr)
                continue
            all_auroc[cond_name] = summarize_per_split(df)
            all_pseudo[cond_name] = summarize_pseudo_stats(df)

        if all_auroc:
            print_auroc_table(ds_name, all_auroc)
        if all_pseudo:
            print_pseudo_table(ds_name, all_pseudo)


if __name__ == "__main__":
    main()

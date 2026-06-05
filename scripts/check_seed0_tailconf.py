import pandas as pd
from pathlib import Path

def print_stats(name, path, seed="0"):
    merged_csv = Path(path) / f"robustad_streamseed{seed}" / "btta_12d_mean" / "robustad_detailed.csv"
    if merged_csv.exists():
        csv_paths = [merged_csv]
    else:
        csv_paths = list((Path(path) / f"robustad_streamseed{seed}" / "btta_12d_mean").glob("*/robustad_detailed.csv"))
        if not csv_paths:
            print(f"[{name}] No csv files found in {path}.")
            return
    
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    # Filter only target tests for a quick look
    df = df[df["split"].isin(["test2", "test3", "test4", "test5", "test6", "video1", "video2"])]
    print(f"\n--- {name} ---")
    for category in df["category"].unique():
        cat_df = df[df["category"] == category]
        for _, row in cat_df.iterrows():
            split = row["split"]
            method = row["method"]
            if "ActiveBoundary" not in method:
                continue
            auroc = row["image_auroc"]
            tail_count = row.get("active_tail_pseudo_label_count", -1)
            tail_acc = row.get("active_tail_pseudo_label_accuracy", -1)
            pseudo_normal = row.get("active_adapt_normal_count", row.get("active_label_normal_count", -1))
            pseudo_purity = row.get("pseudo_only_purity", -1)
            print(f"[{category}/{split}] AUROC: {auroc*100:.2f} | Tail Count: {tail_count} (Acc: {tail_acc:.2f}) | Adapt Normal: {pseudo_normal} (Purity: {pseudo_purity:.2f})")

base_old = "/home/qilab/byeongju_lee/anomaly_tta/outputs/seed01_btta_sptail0.01_conf0.15_bs32_decoder_bn_only_gpu0"
base_new = "/home/qilab/byeongju_lee/anomaly_tta/outputs/seed0_btta_tailconf0.8_uncapped_conf0.15_bs32_decoder_bn_only_gpu1"

print_stats("Old (1% Tail, conf0.15)", base_old)
print_stats("New (Confidence Tail, conf0.15)", base_new)

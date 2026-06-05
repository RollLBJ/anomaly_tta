from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


CURRENT_METHOD = "AnomalibRD4AD_ActiveSVMBoundary-MapStatsSVM-TailPseudo5pctSVM-AdaptEMA95Score-BNOnly"
ATTA_METHOD = "AnomalibRD4AD_ATTA-AD-BNOnly"
SOURCE_METHOD = "AnomalibRD4AD_source"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def view_from_dir(path: Path) -> str:
    return path.name.removeprefix("view_")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/m2ad_rd4ad_btta_full_randomstream_seed0")
    by_key: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for detail_path in sorted(root.glob("view_*/robustad_detailed.csv")):
        view = view_from_dir(detail_path.parent)
        for row in read_csv(detail_path):
            row = dict(row)
            row["view"] = view
            by_key[(view, str(row["category"]))][str(row["method"])] = row

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for (view, category), methods in sorted(by_key.items()):
        source = methods.get(SOURCE_METHOD)
        atta = methods.get(ATTA_METHOD)
        current = methods.get(CURRENT_METHOD)
        if source is None or atta is None or current is None:
            missing.append(f"{view}/{category}")
            continue
        source_auroc = to_float(source["image_auroc"])
        atta_auroc = to_float(atta["image_auroc"])
        current_auroc = to_float(current["image_auroc"])
        rows.append(
            {
                "view": view,
                "category": category,
                "source_auroc": source_auroc,
                "atta_label_only_auroc": atta_auroc,
                "current_svm_pseudo_auroc": current_auroc,
                "atta_minus_source_auroc": atta_auroc - source_auroc,
                "current_minus_source_auroc": current_auroc - source_auroc,
                "current_minus_atta_auroc": current_auroc - atta_auroc,
                "source_ap": to_float(source["image_ap"]),
                "atta_label_only_ap": to_float(atta["image_ap"]),
                "current_svm_pseudo_ap": to_float(current["image_ap"]),
                "current_minus_atta_ap": to_float(current["image_ap"]) - to_float(atta["image_ap"]),
                "atta_active_label_count": atta.get("active_label_count", ""),
                "atta_selected_count": atta.get("selected_pseudo_normal_count", ""),
                "atta_optimizer_steps": atta.get("optimizer_steps", ""),
                "current_active_label_count": current.get("active_label_count", ""),
                "current_selected_count": current.get("selected_pseudo_normal_count", ""),
                "current_tail_pseudo_label_count": current.get("active_tail_pseudo_label_count", ""),
                "current_optimizer_steps": current.get("optimizer_steps", ""),
            },
        )

    fieldnames = [
        "view",
        "category",
        "source_auroc",
        "atta_label_only_auroc",
        "current_svm_pseudo_auroc",
        "atta_minus_source_auroc",
        "current_minus_source_auroc",
        "current_minus_atta_auroc",
        "source_ap",
        "atta_label_only_ap",
        "current_svm_pseudo_ap",
        "current_minus_atta_ap",
        "atta_active_label_count",
        "atta_selected_count",
        "atta_optimizer_steps",
        "current_active_label_count",
        "current_selected_count",
        "current_tail_pseudo_label_count",
        "current_optimizer_steps",
    ]
    write_csv(root / "m2ad_atta_label_only_vs_current_svm_pseudo.csv", rows, fieldnames)

    summary_rows = [
        {
            "runs": len(rows),
            "mean_source_auroc": mean([row["source_auroc"] for row in rows]),
            "mean_atta_label_only_auroc": mean([row["atta_label_only_auroc"] for row in rows]),
            "mean_current_svm_pseudo_auroc": mean([row["current_svm_pseudo_auroc"] for row in rows]),
            "mean_atta_minus_source_auroc": mean([row["atta_minus_source_auroc"] for row in rows]),
            "mean_current_minus_source_auroc": mean([row["current_minus_source_auroc"] for row in rows]),
            "mean_current_minus_atta_auroc": mean([row["current_minus_atta_auroc"] for row in rows]),
            "mean_source_ap": mean([row["source_ap"] for row in rows]),
            "mean_atta_label_only_ap": mean([row["atta_label_only_ap"] for row in rows]),
            "mean_current_svm_pseudo_ap": mean([row["current_svm_pseudo_ap"] for row in rows]),
            "mean_current_minus_atta_ap": mean([row["current_minus_atta_ap"] for row in rows]),
            "current_better_than_atta": sum(1 for row in rows if row["current_minus_atta_auroc"] > 0),
            "atta_better_than_current": sum(1 for row in rows if row["current_minus_atta_auroc"] < 0),
            "ties": sum(1 for row in rows if row["current_minus_atta_auroc"] == 0),
            "mean_atta_active_label_count": mean([to_float(row["atta_active_label_count"]) for row in rows]),
            "mean_atta_selected_count": mean([to_float(row["atta_selected_count"]) for row in rows]),
            "mean_current_active_label_count": mean([to_float(row["current_active_label_count"]) for row in rows]),
            "mean_current_selected_count": mean([to_float(row["current_selected_count"]) for row in rows]),
            "mean_current_tail_pseudo_label_count": mean(
                [to_float(row["current_tail_pseudo_label_count"]) for row in rows],
            ),
            "mean_atta_optimizer_steps": mean([to_float(row["atta_optimizer_steps"]) for row in rows]),
            "mean_current_optimizer_steps": mean([to_float(row["current_optimizer_steps"]) for row in rows]),
        },
    ]
    write_csv(
        root / "m2ad_atta_label_only_vs_current_svm_pseudo_summary.csv",
        summary_rows,
        [
            "runs",
            "mean_source_auroc",
            "mean_atta_label_only_auroc",
            "mean_current_svm_pseudo_auroc",
            "mean_atta_minus_source_auroc",
            "mean_current_minus_source_auroc",
            "mean_current_minus_atta_auroc",
            "mean_source_ap",
            "mean_atta_label_only_ap",
            "mean_current_svm_pseudo_ap",
            "mean_current_minus_atta_ap",
            "current_better_than_atta",
            "atta_better_than_current",
            "ties",
            "mean_atta_active_label_count",
            "mean_atta_selected_count",
            "mean_current_active_label_count",
            "mean_current_selected_count",
            "mean_current_tail_pseudo_label_count",
            "mean_atta_optimizer_steps",
            "mean_current_optimizer_steps",
        ],
    )

    for group_name, group_key in (
        ("view", lambda row: str(row["view"])),
        ("category", lambda row: str(row["category"])),
    ):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[group_key(row)].append(row)
        group_rows = []
        for key, group in sorted(grouped.items()):
            group_rows.append(
                {
                    group_name: key,
                    "runs": len(group),
                    "mean_source_auroc": mean([row["source_auroc"] for row in group]),
                    "mean_atta_label_only_auroc": mean([row["atta_label_only_auroc"] for row in group]),
                    "mean_current_svm_pseudo_auroc": mean([row["current_svm_pseudo_auroc"] for row in group]),
                    "mean_current_minus_atta_auroc": mean([row["current_minus_atta_auroc"] for row in group]),
                    "current_better_than_atta": sum(1 for row in group if row["current_minus_atta_auroc"] > 0),
                    "atta_better_than_current": sum(1 for row in group if row["current_minus_atta_auroc"] < 0),
                },
            )
        write_csv(
            root / f"m2ad_atta_label_only_vs_current_svm_pseudo_by_{group_name}.csv",
            group_rows,
            [
                group_name,
                "runs",
                "mean_source_auroc",
                "mean_atta_label_only_auroc",
                "mean_current_svm_pseudo_auroc",
                "mean_current_minus_atta_auroc",
                "current_better_than_atta",
                "atta_better_than_current",
            ],
        )
    print(f"[m2ad-compare] rows={len(rows)} missing={len(missing)} root={root}")
    if missing:
        print("[m2ad-compare] missing=" + ",".join(missing[:20]))


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
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
    name = path.name
    return name.removeprefix("view_")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/m2ad_rd4ad_btta_full_randomstream_seed0")
    rows: list[dict[str, Any]] = []
    for detail_path in sorted(root.glob("view_*/robustad_detailed.csv")):
        view = view_from_dir(detail_path.parent)
        for row in read_csv(detail_path):
            row = dict(row)
            row["view"] = view
            rows.append(row)

    if not rows:
        raise SystemExit(f"No detail rows found under {root}")

    detail_field_tail: list[str] = []
    seen_fields = {"view"}
    for row in rows:
        for name in row.keys():
            if name not in seen_fields:
                detail_field_tail.append(name)
                seen_fields.add(name)
    detail_fields = ["view", *detail_field_tail]
    write_csv(root / "m2ad_detailed.csv", rows, detail_fields)

    delta_rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(str(row["view"]), str(row["category"]))][str(row["method"])] = row
    for (view, category), method_rows in sorted(by_key.items()):
        source = method_rows.get("AnomalibRD4AD_source")
        for method, row in sorted(method_rows.items()):
            if method == "AnomalibRD4AD_source":
                continue
            source_auroc = to_float(source["image_auroc"]) if source is not None else float("nan")
            tta_auroc = to_float(row["image_auroc"])
            delta_rows.append(
                {
                    "view": view,
                    "category": category,
                    "method": method,
                    "source_auroc": source_auroc,
                    "btta_auroc": tta_auroc,
                    "delta_auroc": tta_auroc - source_auroc,
                    "source_ap": to_float(source["image_ap"]) if source is not None else float("nan"),
                    "btta_ap": to_float(row["image_ap"]),
                    "selected_pseudo_normal_count": row.get("selected_pseudo_normal_count", ""),
                    "selected_pseudo_normal_purity": row.get("selected_pseudo_normal_purity", ""),
                    "optimizer_steps": row.get("optimizer_steps", ""),
                },
            )
    write_csv(
        root / "m2ad_btta_delta_by_view_class.csv",
        delta_rows,
        [
            "view",
            "category",
            "method",
            "source_auroc",
            "btta_auroc",
            "delta_auroc",
            "source_ap",
            "btta_ap",
            "selected_pseudo_normal_count",
            "selected_pseudo_normal_purity",
            "optimizer_steps",
        ],
    )

    summary_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["view"]), str(row["method"]))].append(row)
    for (view, method), group_rows in sorted(groups.items()):
        summary_rows.append(
            {
                "view": view,
                "method": method,
                "runs": len(group_rows),
                "mean_auroc": mean([to_float(row["image_auroc"]) for row in group_rows]),
                "mean_ap": mean([to_float(row["image_ap"]) for row in group_rows]),
                "mean_f1_max": mean([to_float(row["image_f1_max"]) for row in group_rows]),
                "mean_selected_pseudo_normal_count": mean(
                    [to_float(row.get("selected_pseudo_normal_count")) for row in group_rows],
                ),
                "mean_optimizer_steps": mean([to_float(row.get("optimizer_steps")) for row in group_rows]),
            },
        )
    write_csv(
        root / "m2ad_summary_by_view_method.csv",
        summary_rows,
        [
            "view",
            "method",
            "runs",
            "mean_auroc",
            "mean_ap",
            "mean_f1_max",
            "mean_selected_pseudo_normal_count",
            "mean_optimizer_steps",
        ],
    )

    overall_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        overall_groups[str(row["method"])].append(row)
    overall_rows = []
    for method, group_rows in sorted(overall_groups.items()):
        overall_rows.append(
            {
                "method": method,
                "runs": len(group_rows),
                "mean_auroc": mean([to_float(row["image_auroc"]) for row in group_rows]),
                "mean_ap": mean([to_float(row["image_ap"]) for row in group_rows]),
                "mean_f1_max": mean([to_float(row["image_f1_max"]) for row in group_rows]),
                "mean_selected_pseudo_normal_count": mean(
                    [to_float(row.get("selected_pseudo_normal_count")) for row in group_rows],
                ),
                "mean_optimizer_steps": mean([to_float(row.get("optimizer_steps")) for row in group_rows]),
            },
        )
    write_csv(
        root / "m2ad_summary_overall.csv",
        overall_rows,
        [
            "method",
            "runs",
            "mean_auroc",
            "mean_ap",
            "mean_f1_max",
            "mean_selected_pseudo_normal_count",
            "mean_optimizer_steps",
        ],
    )
    print(f"[m2ad-merge] rows={len(rows)} root={root}")


if __name__ == "__main__":
    main()

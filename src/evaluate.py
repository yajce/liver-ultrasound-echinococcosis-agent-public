"""离线评测：只读 experiments/<run>/results.csv，计算二分类指标并写出 metrics.json。

统计口径：
- positive = echinococcosis，其余 true_label 一律归为 negative 类；
- image_valid=false 或 predicted_diagnosis=not_applicable 的样本单独统计，
  不进入混淆矩阵（不静默忽略）；
- parse_ok=False 的样本单独列出数量与 image_id（不静默忽略）；
- true_label 为空的样本（无标注）单独计数，不计入指标。

用法：
    python -m src.evaluate experiments/<run>/results.csv
"""

import argparse
import csv
import json
from pathlib import Path

POSITIVE_LABEL = "echinococcosis"


def load_results(results_csv: Path) -> list[dict]:
    with open(results_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _parse_bool(value: str) -> bool | None:
    """results.csv 中的 image_valid 以 'True'/'False'/'' 存储。"""
    value = (value or "").strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def compute_metrics(
    records: list[dict], positive_label: str = POSITIVE_LABEL
) -> dict:
    tp = fp = tn = fn = 0
    parse_failure_ids: list[str] = []
    not_applicable_ids: list[str] = []
    n_without_label = 0

    for row in records:
        image_id = (row.get("image_id") or "").strip()
        true_label = (row.get("true_label") or "").strip().lower()
        predicted = (row.get("predicted_diagnosis") or "").strip().lower()
        parse_ok = (row.get("parse_ok") or "").strip().lower() == "true"
        image_valid = _parse_bool(row.get("image_valid") or "")

        if not true_label:
            n_without_label += 1
            continue
        if not parse_ok:
            parse_failure_ids.append(image_id)
            continue
        if image_valid is False or predicted == "not_applicable":
            not_applicable_ids.append(image_id)
            continue

        true_pos = true_label == positive_label
        pred_pos = predicted == positive_label
        if true_pos and pred_pos:
            tp += 1
        elif true_pos and not pred_pos:
            fn += 1
        elif not true_pos and pred_pos:
            fp += 1
        else:
            tn += 1

    n_evaluated = tp + fp + tn + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)                      # Recall = Sensitivity
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, n_evaluated)
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    return {
        "positive_label": positive_label,
        "n_total": len(records),
        "n_evaluated": n_evaluated,
        "n_parse_failures": len(parse_failure_ids),
        "parse_failure_ids": parse_failure_ids,
        "n_not_applicable": len(not_applicable_ids),
        "not_applicable_ids": not_applicable_ids,
        "n_without_true_label": n_without_label,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="离线评测 results.csv")
    parser.add_argument("results_csv", type=Path, help="results.csv 路径")
    args = parser.parse_args()

    metrics = compute_metrics(load_results(args.results_csv))
    out_path = args.results_csv.parent / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    cm = metrics["confusion_matrix"]
    print(f"results_csv : {args.results_csv}")
    print(f"total rows  : {metrics['n_total']}")
    print(f"evaluated   : {metrics['n_evaluated']}")
    print(f"no label    : {metrics['n_without_true_label']}")
    print(f"parse failed: {metrics['n_parse_failures']} "
          f"{metrics['parse_failure_ids']}")
    print(f"not applic. : {metrics['n_not_applicable']} "
          f"{metrics['not_applicable_ids']}")
    print(f"confusion matrix (positive = {metrics['positive_label']}):")
    print("              pred +   pred -")
    print(f"  true +   {cm['tp']:>8} {cm['fn']:>8}")
    print(f"  true -   {cm['fp']:>8} {cm['tn']:>8}")
    for key in ("accuracy", "precision", "recall_sensitivity",
                "specificity", "f1"):
        value = metrics[key]
        shown = f"{value:.4f}" if value is not None else "n/a"
        print(f"{key:>18} : {shown}")
    print(f"metrics.json: {out_path}")


if __name__ == "__main__":
    main()

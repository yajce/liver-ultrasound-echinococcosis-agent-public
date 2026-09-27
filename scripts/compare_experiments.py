#!/usr/bin/env python3
"""Compare two completed experiments on the fixed benchmark split."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "splits" / "all_split.csv"
POSITIVE_LABEL = "echinococcosis"
NEGATIVE_LABEL = "negative"
NOT_APPLICABLE_LABEL = "not_applicable"
PARSE_FAILED_LABEL = "parse_failed"
ALLOWED_LABELS = {POSITIVE_LABEL, NEGATIVE_LABEL}
ALLOWED_PARSED_PREDICTIONS = ALLOWED_LABELS | {NOT_APPLICABLE_LABEL}
PREDICTION_STATES = ALLOWED_PARSED_PREDICTIONS | {PARSE_FAILED_LABEL}
RESULT_REQUIRED_COLUMNS = {
    "image_id",
    "true_label",
    "predicted_diagnosis",
    "parse_ok",
    "image_valid",
}
CHANGE_CATEGORIES = (
    "wrong -> correct",
    "correct -> wrong",
    "unchanged correct",
    "unchanged wrong",
)
METRIC_ORDER = (
    "N",
    "evaluated_binary_n",
    "not_applicable_n",
    "parse_failed_n",
    "TP",
    "FN",
    "FP",
    "TN",
    "Accuracy",
    "Precision",
    "Recall_Sensitivity",
    "Specificity",
    "F1",
    "strict_accuracy",
)


def read_csv_required(
    path: Path, required_columns: set[str], source_name: str
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{source_name} not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = sorted(required_columns - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(f"{source_name} is missing columns: {missing}")
        return list(reader)


def index_unique(
    rows: list[dict[str, str]], source_name: str
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, 2):
        image_id = (row.get("image_id") or "").strip()
        if not image_id:
            raise RuntimeError(f"{source_name} row {row_number} has no image_id")
        if image_id in indexed:
            raise RuntimeError(f"{source_name} has duplicate image_id: {image_id}")
        indexed[image_id] = row
    return indexed


def parse_boolean(value: str, field_name: str, image_id: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized not in {"true", "false"}:
        raise RuntimeError(
            f"{field_name} must be True or False for comparison; "
            f"{image_id} has {value!r}"
        )
    return normalized == "true"


def load_split(path: Path, selected_split: str) -> dict[str, str]:
    rows = read_csv_required(
        path, {"image_id", "true_label", "split"}, "all_split.csv"
    )
    if len(rows) != 166:
        raise RuntimeError(f"all_split.csv must contain 166 rows; found {len(rows)}")
    indexed = index_unique(rows, "all_split.csv")
    selected: dict[str, str] = {}
    for image_id, row in indexed.items():
        split = (row.get("split") or "").strip()
        true_label = (row.get("true_label") or "").strip()
        if split not in {"dev", "test"}:
            raise RuntimeError(f"Unknown split for {image_id}: {split!r}")
        if true_label not in ALLOWED_LABELS:
            raise RuntimeError(f"Unknown true_label for {image_id}: {true_label!r}")
        if selected_split == "overall" or split == selected_split:
            selected[image_id] = true_label

    expected = {"dev": 100, "test": 66, "overall": 166}[selected_split]
    if len(selected) != expected:
        raise RuntimeError(
            f"Split {selected_split!r} must contain {expected} rows; "
            f"found {len(selected)}"
        )
    return selected


def load_predictions(
    path: Path, source_name: str, target_labels: dict[str, str]
) -> dict[str, str]:
    predictions: dict[str, str] = {}
    if not path.is_file():
        raise FileNotFoundError(f"{source_name} not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = sorted(
            RESULT_REQUIRED_COLUMNS - set(reader.fieldnames or [])
        )
        if missing_columns:
            raise RuntimeError(
                f"{source_name} is missing columns: {missing_columns}"
            )
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if image_id not in target_labels:
                continue
            if image_id in predictions:
                raise RuntimeError(
                    f"{source_name} has duplicate selected image_id: {image_id}"
                )
            expected_label = target_labels[image_id]
            true_label = (row.get("true_label") or "").strip()
            if true_label != expected_label:
                raise RuntimeError(
                    f"Label mismatch in {source_name} for {image_id}: "
                    f"{true_label!r} != {expected_label!r}"
                )
            parse_ok = parse_boolean(
                row.get("parse_ok") or "", "parse_ok", image_id
            )
            recorded_prediction = (row.get("predicted_diagnosis") or "").strip()
            if not parse_ok:
                predicted = PARSE_FAILED_LABEL
            elif recorded_prediction in ALLOWED_PARSED_PREDICTIONS:
                predicted = recorded_prediction
            else:
                raise RuntimeError(
                    f"Unknown prediction in {source_name} for "
                    f"{image_id}: {recorded_prediction!r}"
                )
            if parse_ok:
                # A model may abstain on this valid benchmark by returning
                # image_valid=False with diagnosis=not_applicable. Preserve that
                # output for strict correctness comparison instead of rejecting it.
                parse_boolean(
                    row.get("image_valid") or "", "image_valid", image_id
                )
            predictions[image_id] = predicted
    missing = sorted(set(target_labels) - set(predictions))
    if missing:
        raise RuntimeError(
            f"{source_name} is missing {len(missing)} selected image_id values: {missing}"
        )
    return predictions


def safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute_metrics(
    labels: dict[str, str], predictions: dict[str, str]
) -> dict[str, int | float | None]:
    tp = fn = fp = tn = not_applicable_n = parse_failed_n = 0
    for image_id, true_label in labels.items():
        predicted = predictions[image_id]
        if predicted not in PREDICTION_STATES:
            raise RuntimeError(
                f"Unknown normalized prediction state for {image_id}: {predicted!r}"
            )
        if predicted == NOT_APPLICABLE_LABEL:
            not_applicable_n += 1
            continue
        if predicted == PARSE_FAILED_LABEL:
            parse_failed_n += 1
            continue
        if true_label == POSITIVE_LABEL and predicted == POSITIVE_LABEL:
            tp += 1
        elif true_label == POSITIVE_LABEL:
            fn += 1
        elif predicted == POSITIVE_LABEL:
            fp += 1
        else:
            tn += 1
    evaluated_binary_n = tp + fn + fp + tn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, evaluated_binary_n)
    strict_accuracy = safe_div(tp + tn, len(labels))
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else None
    )
    return {
        "N": len(labels),
        "evaluated_binary_n": evaluated_binary_n,
        "not_applicable_n": not_applicable_n,
        "parse_failed_n": parse_failed_n,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall_Sensitivity": recall,
        "Specificity": specificity,
        "F1": f1,
        "strict_accuracy": strict_accuracy,
    }


def write_csv_atomic(
    destination: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(destination)


def compare(
    baseline_path: Path,
    candidate_path: Path,
    splits_path: Path,
    selected_split: str,
    output_dir: Path,
) -> tuple[dict[str, int | float | None], dict[str, int | float | None]]:
    labels = load_split(splits_path, selected_split)
    baseline_predictions = load_predictions(
        baseline_path, "baseline results.csv", labels
    )
    candidate_predictions = load_predictions(
        candidate_path, "candidate results.csv", labels
    )
    baseline_metrics = compute_metrics(labels, baseline_predictions)
    candidate_metrics = compute_metrics(labels, candidate_predictions)
    baseline_not_applicable_ids = sorted(
        image_id
        for image_id, prediction in baseline_predictions.items()
        if prediction == NOT_APPLICABLE_LABEL
    )
    candidate_not_applicable_ids = sorted(
        image_id
        for image_id, prediction in candidate_predictions.items()
        if prediction == NOT_APPLICABLE_LABEL
    )
    baseline_parse_failed_ids = sorted(
        image_id
        for image_id, prediction in baseline_predictions.items()
        if prediction == PARSE_FAILED_LABEL
    )
    candidate_parse_failed_ids = sorted(
        image_id
        for image_id, prediction in candidate_predictions.items()
        if prediction == PARSE_FAILED_LABEL
    )

    metric_rows: list[dict[str, object]] = []
    for metric in METRIC_ORDER:
        baseline_value = baseline_metrics[metric]
        candidate_value = candidate_metrics[metric]
        delta = (
            None
            if baseline_value is None or candidate_value is None
            else candidate_value - baseline_value
        )
        metric_rows.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": delta,
            }
        )
    write_csv_atomic(
        output_dir / "metrics_comparison.csv",
        ["metric", "baseline", "candidate", "delta"],
        metric_rows,
    )
    write_csv_atomic(
        output_dir / "not_applicable_summary.csv",
        [
            "baseline_not_applicable",
            "baseline_not_applicable_image_ids",
            "candidate_not_applicable",
            "candidate_not_applicable_image_ids",
            "baseline_parse_failed",
            "baseline_parse_failed_image_ids",
            "candidate_parse_failed",
            "candidate_parse_failed_image_ids",
        ],
        [
            {
                "baseline_not_applicable": len(baseline_not_applicable_ids),
                "baseline_not_applicable_image_ids": ";".join(
                    baseline_not_applicable_ids
                ),
                "candidate_not_applicable": len(candidate_not_applicable_ids),
                "candidate_not_applicable_image_ids": ";".join(
                    candidate_not_applicable_ids
                ),
                "baseline_parse_failed": len(baseline_parse_failed_ids),
                "baseline_parse_failed_image_ids": ";".join(
                    baseline_parse_failed_ids
                ),
                "candidate_parse_failed": len(candidate_parse_failed_ids),
                "candidate_parse_failed_image_ids": ";".join(
                    candidate_parse_failed_ids
                ),
            }
        ],
    )

    category_counts: Counter[str] = Counter()
    changed_rows: list[dict[str, object]] = []
    for image_id in sorted(labels):
        true_label = labels[image_id]
        baseline_prediction = baseline_predictions[image_id]
        candidate_prediction = candidate_predictions[image_id]
        baseline_correct = baseline_prediction == true_label
        candidate_correct = candidate_prediction == true_label
        if not baseline_correct and candidate_correct:
            category = "wrong -> correct"
        elif baseline_correct and not candidate_correct:
            category = "correct -> wrong"
        elif baseline_correct:
            category = "unchanged correct"
        else:
            category = "unchanged wrong"
        category_counts[category] += 1

        if baseline_prediction != candidate_prediction:
            changed_rows.append(
                {
                    "image_id": image_id,
                    "split": selected_split,
                    "true_label": true_label,
                    "baseline_prediction": baseline_prediction,
                    "candidate_prediction": candidate_prediction,
                    "prediction_transition": (
                        f"{baseline_prediction} -> {candidate_prediction}"
                    ),
                    "baseline_correct": baseline_correct,
                    "candidate_correct": candidate_correct,
                    "change_category": category,
                }
            )

    summary_rows = [
        {"change_category": category, "count": category_counts[category]}
        for category in CHANGE_CATEGORIES
    ]
    write_csv_atomic(
        output_dir / "change_summary.csv",
        ["change_category", "count"],
        summary_rows,
    )
    write_csv_atomic(
        output_dir / "changed_predictions.csv",
        [
            "image_id",
            "split",
            "true_label",
            "baseline_prediction",
            "candidate_prediction",
            "prediction_transition",
            "baseline_correct",
            "candidate_correct",
            "change_category",
        ],
        changed_rows,
    )

    print(f"split: {selected_split} ({len(labels)} images)")
    print("metric                 baseline    candidate        delta")
    for row in metric_rows:
        print(
            f"{str(row['metric']):<22} "
            f"{str(row['baseline']):>11} "
            f"{str(row['candidate']):>12} "
            f"{str(row['delta']):>12}"
        )
    for row in summary_rows:
        print(f"{row['change_category']}: {row['count']}")
    print(
        "baseline_not_applicable: "
        f"{len(baseline_not_applicable_ids)} {baseline_not_applicable_ids}"
    )
    print(
        "candidate_not_applicable: "
        f"{len(candidate_not_applicable_ids)} {candidate_not_applicable_ids}"
    )
    print(
        "baseline_parse_failed: "
        f"{len(baseline_parse_failed_ids)} {baseline_parse_failed_ids}"
    )
    print(
        "candidate_parse_failed: "
        f"{len(candidate_parse_failed_ids)} {candidate_parse_failed_ids}"
    )
    changed_image_ids = [str(row["image_id"]) for row in changed_rows]
    print(f"changed image_id rows: {len(changed_image_ids)}")
    print(f"changed image_ids: {changed_image_ids}")
    print(f"output: {output_dir}")
    return baseline_metrics, candidate_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two experiment results on the fixed benchmark split."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("dev", "test", "overall"), default="dev"
    )
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    baseline_path = args.baseline.resolve()
    candidate_path = args.candidate.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            candidate_path.parent
            / "comparison"
            / f"vs_{baseline_path.parent.name}_{args.split}"
        )
    compare(
        baseline_path,
        candidate_path,
        args.splits.resolve(),
        args.split,
        output_dir.resolve(),
    )


if __name__ == "__main__":
    main()

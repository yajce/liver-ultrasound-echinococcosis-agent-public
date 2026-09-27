#!/usr/bin/env python3
"""Analyze DeepSeek false positives by negative-source subtype on dev only."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = PROJECT_ROOT / "data" / "benchmark_metadata.csv"
DEFAULT_DEV_LABELS = PROJECT_ROOT / "data" / "splits" / "dev.csv"
DEFAULT_RESULTS = (
    PROJECT_ROOT / "experiments" / "deepseek_baseline_dev" / "results.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "deepseek_baseline_dev" / "analysis"
)

NEGATIVE_LABEL = "negative"
POSITIVE_LABEL = "echinococcosis"
SUBTYPE_ORDER = ("Normal", "Benign", "Malignant")
ALLOWED_TRUE_LABELS = {NEGATIVE_LABEL, POSITIVE_LABEL}
METRICS_COLUMNS = ("subtype", "N", "TN", "FP", "specificity", "FP_rate")
FALSE_POSITIVE_COLUMNS = (
    "image_id",
    "source_class",
    "predicted_diagnosis",
    "confidence",
    "findings",
)


def read_csv(path: Path, required: set[str], source_name: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{source_name} not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(f"{source_name} is missing columns: {missing}")
        return list(reader)


def index_selected_rows(
    rows: Iterable[dict[str, str]],
    selected_ids: set[str],
    source_name: str,
) -> dict[str, dict[str, str]]:
    """Retain only dev rows; fields of non-dev rows are never analyzed."""
    selected: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, 2):
        image_id = (row.get("image_id") or "").strip()
        if image_id not in selected_ids:
            continue
        if image_id in selected:
            raise RuntimeError(f"{source_name} has duplicate dev image_id: {image_id}")
        selected[image_id] = row

    missing = sorted(selected_ids - set(selected))
    if missing:
        raise RuntimeError(
            f"{source_name} is missing {len(missing)} dev image_id values: {missing}"
        )
    return selected


def load_dev_labels(path: Path) -> dict[str, str]:
    rows = read_csv(path, {"image_id", "true_label"}, "dev.csv")
    labels: dict[str, str] = {}
    for row_number, row in enumerate(rows, 2):
        image_id = (row.get("image_id") or "").strip()
        true_label = (row.get("true_label") or "").strip()
        if not image_id:
            raise RuntimeError(f"dev.csv row {row_number} has no image_id")
        if image_id in labels:
            raise RuntimeError(f"dev.csv has duplicate image_id: {image_id}")
        if true_label not in ALLOWED_TRUE_LABELS:
            raise RuntimeError(
                f"dev.csv has unknown true_label for {image_id}: {true_label!r}"
            )
        labels[image_id] = true_label
    if not labels:
        raise RuntimeError("dev.csv contains no samples")
    return labels


def parse_boolean(value: str, field_name: str, image_id: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized not in {"true", "false"}:
        raise RuntimeError(
            f"{field_name} must be True or False for {image_id}; found {value!r}"
        )
    return normalized == "true"


def validate_confidence(value: str, image_id: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        return ""
    try:
        confidence = float(normalized)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid confidence for {image_id}: {normalized!r}"
        ) from exc
    if not 0 <= confidence <= 1:
        raise RuntimeError(f"Confidence outside [0, 1] for {image_id}: {confidence}")
    return normalized


def extract_findings(raw_response: str, image_id: str) -> str:
    """Extract findings using the project's first-'{' to last-'}' convention."""
    try:
        start = raw_response.index("{")
        end = raw_response.rindex("}")
        payload = json.loads(raw_response[start : end + 1])
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot parse raw_response for {image_id}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"raw_response JSON is not an object for {image_id}")
    findings = payload.get("findings")
    return findings if isinstance(findings, str) else ""


def write_csv(path: Path, columns: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percent(value: float) -> str:
    return f"{value:.2%}"


def build_report(
    metrics: list[dict[str, object]],
    false_positives: list[dict[str, object]],
    dev_n: int,
    negative_n: int,
) -> str:
    fp_counts = {
        subtype: sum(row["source_class"] == subtype for row in false_positives)
        for subtype in SUBTYPE_ORDER
    }
    ranked_rates = sorted(
        metrics,
        key=lambda row: (-float(row["FP_rate"]), SUBTYPE_ORDER.index(str(row["subtype"]))),
    )
    highest = ranked_rates[0]
    largest_count_subtype = max(
        SUBTYPE_ORDER, key=lambda subtype: (fp_counts[subtype], -SUBTYPE_ORDER.index(subtype))
    )

    lines = [
        "# DeepSeek Baseline Negative Subtype Analysis",
        "",
        "## Scope",
        "",
        f"- Dev samples analyzed: {dev_n}",
        f"- Dev negative samples analyzed: {negative_n}",
        "- Source classes: Normal, Benign, Malignant",
        "- Held-out test samples were not selected or analyzed.",
        "",
        "## Negative subtype metrics",
        "",
        "| Subtype | N | TN | FP | Specificity | FP rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['subtype']} | {row['N']} | {row['TN']} | {row['FP']} | "
            f"{percent(float(row['specificity']))} | {percent(float(row['FP_rate']))} |"
        )

    total_fp = len(false_positives)
    lines.extend(
        [
            "",
            "## False-positive distribution",
            "",
            f"- Total false positives: {total_fp}",
            *[
                f"- {subtype}: {fp_counts[subtype]} of {total_fp} false positives"
                for subtype in SUBTYPE_ORDER
            ],
            "",
            "## Interpretation",
            "",
            f"- {highest['subtype']} has the highest observed FP rate "
            f"({percent(float(highest['FP_rate']))}) in this dev subset.",
            f"- {largest_count_subtype} contributes the largest FP count "
            f"({fp_counts[largest_count_subtype]}), but counts are interpreted alongside "
            "the number of samples in each source class.",
            "- FP count and FP rate answer different questions: the count reflects both "
            "class size and error frequency, while FP rate normalizes by the subtype's N.",
            "- These are descriptive dev-set results. No statistical significance or "
            "new medical interpretation is inferred from the findings text.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    metadata_path: Path,
    dev_labels_path: Path,
    results_path: Path,
    output_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    dev_labels = load_dev_labels(dev_labels_path)
    dev_ids = set(dev_labels)
    negative_ids = {
        image_id for image_id, label in dev_labels.items() if label == NEGATIVE_LABEL
    }

    metadata_rows = read_csv(
        metadata_path,
        {"image_id", "true_label", "source_class"},
        "benchmark_metadata.csv",
    )
    result_rows = read_csv(
        results_path,
        {
            "image_id",
            "true_label",
            "predicted_diagnosis",
            "confidence",
            "parse_ok",
            "raw_response",
        },
        "results.csv",
    )
    metadata = index_selected_rows(metadata_rows, dev_ids, "benchmark_metadata.csv")
    results = index_selected_rows(result_rows, dev_ids, "results.csv")

    grouped: dict[str, list[str]] = {subtype: [] for subtype in SUBTYPE_ORDER}
    false_positives: list[dict[str, object]] = []
    for image_id in sorted(negative_ids):
        metadata_row = metadata[image_id]
        result_row = results[image_id]
        for source_name, row in (
            ("benchmark_metadata.csv", metadata_row),
            ("results.csv", result_row),
        ):
            recorded_label = (row.get("true_label") or "").strip()
            if recorded_label != NEGATIVE_LABEL:
                raise RuntimeError(
                    f"Label mismatch in {source_name} for {image_id}: "
                    f"{recorded_label!r}"
                )

        source_class = (metadata_row.get("source_class") or "").strip()
        if source_class not in grouped:
            raise RuntimeError(
                f"Unknown negative source_class for {image_id}: {source_class!r}"
            )
        grouped[source_class].append(image_id)

        parse_ok = parse_boolean(result_row.get("parse_ok") or "", "parse_ok", image_id)
        if not parse_ok:
            raise RuntimeError(
                f"Negative dev sample {image_id} is parse_failed; TN/FP metrics require "
                "a binary prediction"
            )
        prediction = (result_row.get("predicted_diagnosis") or "").strip()
        if prediction not in {NEGATIVE_LABEL, POSITIVE_LABEL}:
            raise RuntimeError(
                f"Negative dev sample {image_id} has non-binary prediction: {prediction!r}"
            )
        confidence = validate_confidence(result_row.get("confidence") or "", image_id)
        if prediction == POSITIVE_LABEL:
            false_positives.append(
                {
                    "image_id": image_id,
                    "source_class": source_class,
                    "predicted_diagnosis": prediction,
                    "confidence": confidence,
                    "findings": extract_findings(
                        result_row.get("raw_response") or "", image_id
                    ),
                }
            )

    subtype_rank = {subtype: index for index, subtype in enumerate(SUBTYPE_ORDER)}
    false_positives.sort(
        key=lambda row: (subtype_rank[str(row["source_class"])], str(row["image_id"]))
    )

    fp_ids = {str(row["image_id"]) for row in false_positives}
    metrics: list[dict[str, object]] = []
    for subtype in SUBTYPE_ORDER:
        subtype_ids = grouped[subtype]
        n = len(subtype_ids)
        fp = sum(image_id in fp_ids for image_id in subtype_ids)
        tn = n - fp
        if not n:
            raise RuntimeError(f"No dev negative samples found for subtype {subtype}")
        metrics.append(
            {
                "subtype": subtype,
                "N": n,
                "TN": tn,
                "FP": fp,
                "specificity": tn / n,
                "FP_rate": fp / n,
            }
        )

    if sum(int(row["N"]) for row in metrics) != len(negative_ids):
        raise RuntimeError("Negative subtype totals do not reconcile to dev negatives")
    if sum(int(row["FP"]) for row in metrics) != len(false_positives):
        raise RuntimeError("Subtype FP totals do not reconcile to the FP export")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "negative_subtype_metrics.csv", METRICS_COLUMNS, metrics)
    write_csv(
        output_dir / "false_positives_by_subtype.csv",
        FALSE_POSITIVE_COLUMNS,
        false_positives,
    )
    report = build_report(metrics, false_positives, len(dev_labels), len(negative_ids))
    (output_dir / "negative_subtype_analysis.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    return metrics, false_positives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze DeepSeek dev false positives by negative-source subtype."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--labels", type=Path, default=DEFAULT_DEV_LABELS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, false_positives = analyze(
        args.metadata, args.labels, args.results, args.output_dir
    )
    print(f"dev negative N : {sum(int(row['N']) for row in metrics)}")
    print(f"false positives: {len(false_positives)}")
    for row in metrics:
        print(
            f"{row['subtype']:<9} N={row['N']:>2} TN={row['TN']:>2} "
            f"FP={row['FP']:>2} specificity={float(row['specificity']):.4f} "
            f"FP_rate={float(row['FP_rate']):.4f}"
        )


if __name__ == "__main__":
    main()

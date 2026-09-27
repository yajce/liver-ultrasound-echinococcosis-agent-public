#!/usr/bin/env python3
"""Offline baseline analysis with strict joins and reproducible CSV outputs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "experiments" / "glm_baseline_v02" / "results.csv"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "benchmark_metadata.csv"
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "splits" / "all_split.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "glm_baseline_v02" / "analysis"
)
EXPECTED_TOTAL = 166
POSITIVE_LABEL = "echinococcosis"
NEGATIVE_LABEL = "negative"
ALLOWED_LABELS = {POSITIVE_LABEL, NEGATIVE_LABEL}
NEGATIVE_SOURCE_CLASSES = ("Benign", "Malignant", "Normal")
OUTCOME_ORDER = ("TP", "TN", "FP", "FN")
SPLIT_ORDER = {"dev": 0, "test": 1}

RESULT_REQUIRED_COLUMNS = {
    "image_id",
    "true_label",
    "image_valid",
    "predicted_diagnosis",
    "predicted_subtype",
    "confidence",
    "parse_ok",
    "model",
    "skill_enabled",
    "prompt_version",
    "latency",
    "raw_response",
}
METADATA_REQUIRED_COLUMNS = {
    "image_id",
    "true_label",
    "source_dataset",
    "source_class",
    "source_filename",
}
SPLIT_REQUIRED_COLUMNS = {"image_id", "true_label", "split"}
ERROR_COLUMNS = [
    "image_id",
    "split",
    "true_label",
    "predicted_diagnosis",
    "confidence",
    "image_valid",
    "predicted_subtype",
    "latency",
    "source_dataset",
    "source_class",
    "source_filename",
    "raw_response",
]


def read_csv_required(
    path: Path, required_columns: set[str], source_name: str
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{source_name} not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(required_columns - fieldnames)
        if missing_columns:
            raise RuntimeError(
                f"{source_name} is missing required columns: {missing_columns}"
            )
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
            raise RuntimeError(
                f"{source_name} contains duplicate image_id: {image_id}"
            )
        indexed[image_id] = row
    return indexed


def require_same_ids(
    reference: dict[str, dict[str, str]],
    other: dict[str, dict[str, str]],
    other_name: str,
) -> None:
    missing = sorted(set(reference) - set(other))
    unexpected = sorted(set(other) - set(reference))
    if missing or unexpected:
        raise RuntimeError(
            f"{other_name} image_id join is incomplete; "
            f"missing={missing}, unexpected={unexpected}"
        )


def parse_bool(value: str, field_name: str, image_id: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(
        f"Invalid or missing {field_name} for {image_id}: {value!r}"
    )


def parse_optional_float(
    value: str, field_name: str, image_id: str
) -> float | None:
    normalized = (value or "").strip()
    if not normalized:
        return None
    try:
        parsed = float(normalized)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid {field_name} for {image_id}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise RuntimeError(f"Non-finite {field_name} for {image_id}: {value!r}")
    return parsed


def load_joined_records(
    results_path: Path, metadata_path: Path, splits_path: Path
) -> list[dict[str, object]]:
    results = read_csv_required(
        results_path, RESULT_REQUIRED_COLUMNS, "baseline results.csv"
    )
    metadata = read_csv_required(
        metadata_path, METADATA_REQUIRED_COLUMNS, "benchmark_metadata.csv"
    )
    splits = read_csv_required(splits_path, SPLIT_REQUIRED_COLUMNS, "all_split.csv")

    if len(results) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"baseline results.csv must contain exactly {EXPECTED_TOTAL} rows; "
            f"found {len(results)}"
        )
    if len(metadata) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"benchmark_metadata.csv must contain exactly {EXPECTED_TOTAL} rows; "
            f"found {len(metadata)}"
        )
    if len(splits) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"all_split.csv must contain exactly {EXPECTED_TOTAL} rows; "
            f"found {len(splits)}"
        )

    results_by_id = index_unique(results, "baseline results.csv")
    metadata_by_id = index_unique(metadata, "benchmark_metadata.csv")
    splits_by_id = index_unique(splits, "all_split.csv")
    require_same_ids(results_by_id, metadata_by_id, "benchmark_metadata.csv")
    require_same_ids(results_by_id, splits_by_id, "all_split.csv")

    split_counts: Counter[tuple[str, str]] = Counter()
    joined: list[dict[str, object]] = []
    for image_id, result in results_by_id.items():
        metadata_row = metadata_by_id[image_id]
        split_row = splits_by_id[image_id]
        result_label = (result.get("true_label") or "").strip()
        metadata_label = (metadata_row.get("true_label") or "").strip()
        split_label = (split_row.get("true_label") or "").strip()
        if not result_label:
            raise RuntimeError(f"Missing true_label in results for {image_id}")
        if result_label not in ALLOWED_LABELS:
            raise RuntimeError(
                f"Unknown true_label in results for {image_id}: {result_label!r}"
            )
        if not (result_label == metadata_label == split_label):
            raise RuntimeError(
                f"Label mismatch for {image_id}: results={result_label!r}, "
                f"metadata={metadata_label!r}, split={split_label!r}"
            )

        predicted = (result.get("predicted_diagnosis") or "").strip()
        if predicted not in ALLOWED_LABELS:
            raise RuntimeError(
                f"Unknown or non-binary predicted_diagnosis for {image_id}: "
                f"{predicted!r}"
            )
        if not parse_bool(result.get("parse_ok") or "", "parse_ok", image_id):
            raise RuntimeError(f"parse_ok=False for {image_id}")
        if not parse_bool(
            result.get("image_valid") or "", "image_valid", image_id
        ):
            raise RuntimeError(f"image_valid=False for {image_id}")

        split = (split_row.get("split") or "").strip()
        if split not in SPLIT_ORDER:
            raise RuntimeError(f"Unknown or missing split for {image_id}: {split!r}")
        split_counts[(split, result_label)] += 1

        confidence = parse_optional_float(
            result.get("confidence") or "", "confidence", image_id
        )
        if confidence is not None and not 0 <= confidence <= 1:
            raise RuntimeError(
                f"confidence outside [0, 1] for {image_id}: {confidence}"
            )
        latency = parse_optional_float(
            result.get("latency") or "", "latency", image_id
        )
        if latency is not None and latency < 0:
            raise RuntimeError(f"Negative latency for {image_id}: {latency}")

        combined: dict[str, object] = dict(result)
        combined.update(
            {
                "image_id": image_id,
                "true_label": result_label,
                "predicted_diagnosis": predicted,
                "split": split,
                "source_dataset": (
                    metadata_row.get("source_dataset") or ""
                ).strip(),
                "source_class": (metadata_row.get("source_class") or "").strip(),
                "source_filename": (
                    metadata_row.get("source_filename") or ""
                ).strip(),
                "_confidence": confidence,
                "_latency": latency,
            }
        )
        joined.append(combined)

    expected_split_counts = {
        ("dev", POSITIVE_LABEL): 50,
        ("dev", NEGATIVE_LABEL): 50,
        ("test", POSITIVE_LABEL): 33,
        ("test", NEGATIVE_LABEL): 33,
    }
    if split_counts != expected_split_counts:
        raise RuntimeError(
            f"Split is incomplete or unbalanced: {dict(split_counts)}; "
            f"expected {expected_split_counts}"
        )
    return joined


def classify_outcome(row: dict[str, object]) -> str:
    true_positive = row["true_label"] == POSITIVE_LABEL
    predicted_positive = row["predicted_diagnosis"] == POSITIVE_LABEL
    if true_positive and predicted_positive:
        return "TP"
    if true_positive:
        return "FN"
    if predicted_positive:
        return "FP"
    return "TN"


def safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute_metrics(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    selected = list(rows)
    counts = Counter(classify_outcome(row) for row in selected)
    tp = counts["TP"]
    fn = counts["FN"]
    fp = counts["FP"]
    tn = counts["TN"]
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, len(selected))
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else None
    )
    return {
        "N": len(selected),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall_Sensitivity": recall,
        "Specificity": specificity,
        "F1": f1,
    }


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def numeric_summary(values: list[float | None]) -> dict[str, object]:
    present = [value for value in values if value is not None]
    return {
        "count": len(present),
        "missing": len(values) - len(present),
        "mean": statistics.mean(present) if present else None,
        "median": statistics.median(present) if present else None,
        "min": min(present) if present else None,
        "max": max(present) if present else None,
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


def write_text_atomic(destination: Path, text: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(destination)


def error_export_row(row: dict[str, object]) -> dict[str, object]:
    return {column: row.get(column, "") for column in ERROR_COLUMNS}


def format_rate(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def format_number(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def metrics_markdown(metrics: dict[str, object]) -> str:
    return "\n".join(
        [
            "| Metric | Value |",
            "|---|---:|",
            f"| N | {metrics['N']} |",
            f"| TP | {metrics['TP']} |",
            f"| FN | {metrics['FN']} |",
            f"| FP | {metrics['FP']} |",
            f"| TN | {metrics['TN']} |",
            f"| Accuracy | {format_rate(metrics['Accuracy'])} |",
            f"| Precision | {format_rate(metrics['Precision'])} |",
            "| Recall / Sensitivity | "
            f"{format_rate(metrics['Recall_Sensitivity'])} |",
            f"| Specificity | {format_rate(metrics['Specificity'])} |",
            f"| F1 | {format_rate(metrics['F1'])} |",
        ]
    )


def build_report(
    joined: list[dict[str, object]],
    metrics_by_scope: dict[str, dict[str, object]],
    negative_breakdown: list[dict[str, object]],
    confidence_rows: list[dict[str, object]],
    latency_row: dict[str, object],
    false_negatives: list[dict[str, object]],
    false_positives: list[dict[str, object]],
    results_path: Path,
) -> str:
    models = sorted({str(row.get("model", "")) for row in joined})
    prompt_versions = sorted(
        {str(row.get("prompt_version", "")) for row in joined}
    )
    skill_values = sorted({str(row.get("skill_enabled", "")) for row in joined})
    error_counts = Counter(
        (str(row["split"]), classify_outcome(row)) for row in joined
    )
    dataset_counts = Counter(
        (str(row["split"]), str(row["true_label"])) for row in joined
    )
    try:
        displayed_results_path = results_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        displayed_results_path = results_path.name

    negative_lines = [
        "| Source class | N | TN | FP | Specificity | Mean self-reported confidence | Missing confidence |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in negative_breakdown:
        negative_lines.append(
            f"| {row['source_class']} | {row['N']} | {row['TN']} | {row['FP']} "
            f"| {format_rate(row['Specificity'])} | "
            f"{format_number(row['mean_confidence'])} | "
            f"{row['missing_confidence']} |"
        )

    confidence_lines = [
        "| Outcome | Count | Confidence count | Missing | Mean | Median | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in confidence_rows:
        confidence_lines.append(
            f"| {row['outcome']} | {row['count']} | {row['confidence_count']} "
            f"| {row['missing_confidence']} | {format_number(row['mean_confidence'])} "
            f"| {format_number(row['median_confidence'])} "
            f"| {format_number(row['min_confidence'])} "
            f"| {format_number(row['max_confidence'])} |"
        )

    lines = [
        "# GLM-5.3-Flash Baseline Analysis",
        "",
        "## 1. Experiment setup",
        "",
        f"- Source results: `{displayed_results_path}`",
        f"- Records: {len(joined)}",
        f"- Model values: {', '.join(models)}",
        f"- Prompt version values: {', '.join(prompt_versions)}",
        f"- Skill enabled values: {', '.join(skill_values)}",
        "- This report uses existing predictions only. It does not invoke a model.",
        "",
        "## 2. Dataset split",
        "",
        "| Split | N | Echinococcosis | Negative |",
        "|---|---:|---:|---:|",
        f"| dev | {dataset_counts[('dev', POSITIVE_LABEL)] + dataset_counts[('dev', NEGATIVE_LABEL)]} "
        f"| {dataset_counts[('dev', POSITIVE_LABEL)]} "
        f"| {dataset_counts[('dev', NEGATIVE_LABEL)]} |",
        f"| test | {dataset_counts[('test', POSITIVE_LABEL)] + dataset_counts[('test', NEGATIVE_LABEL)]} "
        f"| {dataset_counts[('test', POSITIVE_LABEL)]} "
        f"| {dataset_counts[('test', NEGATIVE_LABEL)]} |",
        "",
        "The test split is held out from Skill development and rule tuning.",
        "",
        "## 3. Overall performance",
        "",
        metrics_markdown(metrics_by_scope["overall"]),
        "",
        "## 4. Dev performance",
        "",
        metrics_markdown(metrics_by_scope["dev"]),
        "",
        "## 5. Held-out test performance",
        "",
        metrics_markdown(metrics_by_scope["test"]),
        "",
        "Test errors are counted here but are not interpreted case by case.",
        "",
        "## 6. Error distribution",
        "",
        "| Split | False negatives | False positives |",
        "|---|---:|---:|",
        f"| dev | {error_counts[('dev', 'FN')]} | {error_counts[('dev', 'FP')]} |",
        f"| test | {error_counts[('test', 'FN')]} | {error_counts[('test', 'FP')]} |",
        f"| overall | {len(false_negatives)} | {len(false_positives)} |",
        "",
        "Detailed rows are saved in `false_negatives.csv` and `false_positives.csv`. "
        "Only dev errors may guide Skill iteration.",
        "",
        "## 7. Negative source breakdown",
        "",
        *negative_lines,
        "",
        "## 8. Confidence analysis",
        "",
        "Confidence values below are model self-reported confidence, not calibrated probabilities.",
        "",
        *confidence_lines,
        "",
        "## 9. Latency",
        "",
        "Percentiles use linear interpolation between adjacent ordered observations.",
        "",
        "| Count | Missing | Mean | Median | Min | Max | p25 | p75 | p90 | p95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {latency_row['count']} | {latency_row['missing_latency']} "
        f"| {format_number(latency_row['mean'])} "
        f"| {format_number(latency_row['median'])} "
        f"| {format_number(latency_row['min'])} "
        f"| {format_number(latency_row['max'])} "
        f"| {format_number(latency_row['p25'])} "
        f"| {format_number(latency_row['p75'])} "
        f"| {format_number(latency_row['p90'])} "
        f"| {format_number(latency_row['p95'])} |",
        "",
        "All latency values are in seconds.",
        "",
        "## 10. Guidance for Skill iteration",
        "",
        "- Develop and compare Skill v1/v2 using the dev split only.",
        "- Do not use the test split for rule tuning before selecting the final Skill version.",
        "- Compare false negatives and false positives on dev when prioritizing revisions.",
        "- Keep the model, benchmark, baseline prompt, parser, and evaluation logic fixed.",
        "",
        "## 11. Limitations",
        "",
        "- Positive and negative images come from different public data sources, creating a risk of dataset-source bias.",
        "- This benchmark is exploratory and should be treated as a pilot benchmark.",
        "- Model self-reported confidence is not a calibrated probability.",
        "- A single static ultrasound image is not equivalent to a complete clinical ultrasound examination.",
        "",
    ]
    return "\n".join(lines)


def analyze(
    results_path: Path,
    metadata_path: Path,
    splits_path: Path,
    output_dir: Path,
) -> dict[str, dict[str, object]]:
    joined = load_joined_records(results_path, metadata_path, splits_path)

    metrics_by_scope = {
        "overall": compute_metrics(joined),
        "dev": compute_metrics(row for row in joined if row["split"] == "dev"),
        "test": compute_metrics(row for row in joined if row["split"] == "test"),
    }
    metrics_rows = [
        {"scope": scope, **metrics_by_scope[scope]}
        for scope in ("overall", "dev", "test")
    ]
    write_csv_atomic(
        output_dir / "metrics_by_split.csv",
        [
            "scope",
            "N",
            "TP",
            "FN",
            "FP",
            "TN",
            "Accuracy",
            "Precision",
            "Recall_Sensitivity",
            "Specificity",
            "F1",
        ],
        metrics_rows,
    )

    ordered = sorted(
        joined,
        key=lambda row: (SPLIT_ORDER[str(row["split"])], str(row["image_id"])),
    )
    false_negatives = [row for row in ordered if classify_outcome(row) == "FN"]
    false_positives = [row for row in ordered if classify_outcome(row) == "FP"]
    write_csv_atomic(
        output_dir / "false_negatives.csv",
        ERROR_COLUMNS,
        [error_export_row(row) for row in false_negatives],
    )
    write_csv_atomic(
        output_dir / "false_positives.csv",
        ERROR_COLUMNS,
        [error_export_row(row) for row in false_positives],
    )

    negatives = [row for row in joined if row["true_label"] == NEGATIVE_LABEL]
    found_source_classes = {str(row["source_class"]) for row in negatives}
    if found_source_classes != set(NEGATIVE_SOURCE_CLASSES):
        raise RuntimeError(
            "Unexpected or missing negative source_class values: "
            f"{sorted(found_source_classes)}"
        )
    negative_breakdown: list[dict[str, object]] = []
    for source_class in NEGATIVE_SOURCE_CLASSES:
        group = [row for row in negatives if row["source_class"] == source_class]
        outcome_counts = Counter(classify_outcome(row) for row in group)
        confidence_summary = numeric_summary(
            [row["_confidence"] for row in group]  # type: ignore[list-item]
        )
        negative_breakdown.append(
            {
                "source_class": source_class,
                "N": len(group),
                "TN": outcome_counts["TN"],
                "FP": outcome_counts["FP"],
                "Specificity": safe_div(
                    outcome_counts["TN"],
                    outcome_counts["TN"] + outcome_counts["FP"],
                ),
                "mean_confidence": confidence_summary["mean"],
                "missing_confidence": confidence_summary["missing"],
            }
        )
    write_csv_atomic(
        output_dir / "negative_source_breakdown.csv",
        [
            "source_class",
            "N",
            "TN",
            "FP",
            "Specificity",
            "mean_confidence",
            "missing_confidence",
        ],
        negative_breakdown,
    )

    confidence_rows: list[dict[str, object]] = []
    for outcome_name in OUTCOME_ORDER:
        group = [row for row in joined if classify_outcome(row) == outcome_name]
        summary = numeric_summary(
            [row["_confidence"] for row in group]  # type: ignore[list-item]
        )
        confidence_rows.append(
            {
                "outcome": outcome_name,
                "count": len(group),
                "confidence_count": summary["count"],
                "missing_confidence": summary["missing"],
                "mean_confidence": summary["mean"],
                "median_confidence": summary["median"],
                "min_confidence": summary["min"],
                "max_confidence": summary["max"],
            }
        )
    write_csv_atomic(
        output_dir / "confidence_by_outcome.csv",
        [
            "outcome",
            "count",
            "confidence_count",
            "missing_confidence",
            "mean_confidence",
            "median_confidence",
            "min_confidence",
            "max_confidence",
        ],
        confidence_rows,
    )

    latency_values = [row["_latency"] for row in joined]
    latency_summary = numeric_summary(latency_values)  # type: ignore[arg-type]
    present_latencies = [
        float(value) for value in latency_values if value is not None
    ]
    latency_row = {
        "scope": "overall",
        "count": latency_summary["count"],
        "missing_latency": latency_summary["missing"],
        "mean": latency_summary["mean"],
        "median": latency_summary["median"],
        "min": latency_summary["min"],
        "max": latency_summary["max"],
        "p25": percentile(present_latencies, 0.25),
        "p75": percentile(present_latencies, 0.75),
        "p90": percentile(present_latencies, 0.90),
        "p95": percentile(present_latencies, 0.95),
    }
    write_csv_atomic(
        output_dir / "latency_summary.csv",
        [
            "scope",
            "count",
            "missing_latency",
            "mean",
            "median",
            "min",
            "max",
            "p25",
            "p75",
            "p90",
            "p95",
        ],
        [latency_row],
    )

    report = build_report(
        joined,
        metrics_by_scope,
        negative_breakdown,
        confidence_rows,
        latency_row,
        false_negatives,
        false_positives,
        results_path,
    )
    write_text_atomic(output_dir / "baseline_analysis.md", report)
    return metrics_by_scope


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the completed GLM baseline entirely offline."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    metrics_by_scope = analyze(
        args.results.resolve(),
        args.metadata.resolve(),
        args.splits.resolve(),
        args.output_dir.resolve(),
    )
    for scope in ("overall", "dev", "test"):
        metrics = metrics_by_scope[scope]
        print(
            f"{scope}: N={metrics['N']} TP={metrics['TP']} FN={metrics['FN']} "
            f"FP={metrics['FP']} TN={metrics['TN']} "
            f"Accuracy={format_rate(metrics['Accuracy'])} "
            f"F1={format_rate(metrics['F1'])}"
        )
    print(f"analysis: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

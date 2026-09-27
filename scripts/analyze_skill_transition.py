#!/usr/bin/env python3
"""Analyze paired baseline-to-Skill-v1 transitions on the dev split only."""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "all_split.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "benchmark_metadata.csv"
BASELINE_PATH = PROJECT_ROOT / "experiments" / "glm_baseline_v02" / "results.csv"
SKILL_PATH = PROJECT_ROOT / "experiments" / "glm_skill_v1_dev" / "results.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "glm_skill_v1_dev" / "analysis"

POSITIVE_LABEL = "echinococcosis"
NEGATIVE_LABEL = "negative"
NOT_APPLICABLE = "not_applicable"
ALLOWED_TRUE_LABELS = {POSITIVE_LABEL, NEGATIVE_LABEL}
ALLOWED_PREDICTIONS = {POSITIVE_LABEL, NEGATIVE_LABEL, NOT_APPLICABLE}
PREDICTION_ORDER = (POSITIVE_LABEL, NEGATIVE_LABEL, NOT_APPLICABLE)
TRANSITION_ORDER = (
    "wrong_to_correct",
    "correct_to_wrong",
    "unchanged_correct",
    "unchanged_wrong",
)
EXPECTED_TRANSITIONS = {
    "wrong_to_correct": 5,
    "correct_to_wrong": 12,
    "unchanged_correct": 73,
    "unchanged_wrong": 10,
}
EXPECTED_DEV_N = 100
EXPECTED_CHANGED_N = 18
EXPECTED_SKILL_NOT_APPLICABLE_N = 2

RESULT_REQUIRED_COLUMNS = {
    "image_id",
    "true_label",
    "predicted_diagnosis",
    "predicted_subtype",
    "confidence",
    "parse_ok",
    "raw_response",
}
METADATA_REQUIRED_COLUMNS = {
    "image_id",
    "true_label",
    "source_dataset",
    "source_class",
}
TRANSITION_COLUMNS = [
    "image_id",
    "true_label",
    "source_dataset",
    "source_class",
    "baseline_prediction",
    "baseline_correct",
    "baseline_confidence",
    "baseline_subtype",
    "baseline_findings",
    "baseline_raw_response",
    "skill_prediction",
    "skill_correct",
    "skill_confidence",
    "skill_subtype",
    "skill_findings",
    "skill_raw_response",
    "transition",
]

PHRASE_PATTERNS = {
    "daughter cyst": r"\bdaughter cysts?\b",
    "double wall": r"\bdouble[- ]wall\b",
    "membrane": r"\bmembranes?\b",
    "water lily": r"\bwater[- ]lily\b",
    "hydatid sand": r"\bhydatid sand\b",
    "calcification": r"\bcalcifi(?:cation|cations|ed)\b",
    "acoustic shadow": r"\b(?:posterior\s+)?acoustic shadow(?:ing)?\b",
    "simple cyst": r"\bsimple(?: liver)? cysts?\b",
    "nonspecific": r"\bnon[- ]?specific\b",
    "characteristic": r"\bcharacteristic\b",
    "not sufficient": r"\bnot sufficient\b",
    "cannot exclude": r"\bcannot exclude\b",
    "not clearly seen": r"\bnot clearly seen\b",
}


def open_reader(
    path: Path, required_columns: set[str], source_name: str
) -> tuple[object, csv.DictReader]:
    if not path.is_file():
        raise FileNotFoundError(f"{source_name} not found: {path}")
    csv_file = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(csv_file)
    missing = sorted(required_columns - set(reader.fieldnames or []))
    if missing:
        csv_file.close()
        raise RuntimeError(f"{source_name} is missing columns: {missing}")
    return csv_file, reader


def load_dev_split(path: Path) -> dict[str, str]:
    csv_file, reader = open_reader(
        path, {"image_id", "true_label", "split"}, "all_split.csv"
    )
    dev_labels: dict[str, str] = {}
    seen_ids: set[str] = set()
    try:
        for row_number, row in enumerate(reader, 2):
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                raise RuntimeError(f"all_split.csv row {row_number} has no image_id")
            if image_id in seen_ids:
                raise RuntimeError(f"all_split.csv has duplicate image_id: {image_id}")
            seen_ids.add(image_id)
            if (row.get("split") or "").strip() != "dev":
                continue
            true_label = (row.get("true_label") or "").strip()
            if true_label not in ALLOWED_TRUE_LABELS:
                raise RuntimeError(
                    f"Unknown dev true_label for {image_id}: {true_label!r}"
                )
            dev_labels[image_id] = true_label
    finally:
        csv_file.close()
    if len(dev_labels) != EXPECTED_DEV_N:
        raise RuntimeError(
            f"dev split must contain {EXPECTED_DEV_N} rows; found {len(dev_labels)}"
        )
    return dev_labels


def parse_findings(raw_response: str, source_name: str, image_id: str) -> str:
    try:
        start = raw_response.index("{")
        end = raw_response.rindex("}")
        payload = json.loads(raw_response[start : end + 1])
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot parse raw_response JSON in {source_name} for {image_id}"
        ) from exc
    findings = payload.get("findings") if isinstance(payload, dict) else None
    return findings if isinstance(findings, str) else ""


def load_dev_results(
    path: Path, source_name: str, dev_labels: dict[str, str]
) -> dict[str, dict[str, str]]:
    """Retain dev rows only; non-dev rows are skipped before result fields are read."""
    csv_file, reader = open_reader(path, RESULT_REQUIRED_COLUMNS, source_name)
    selected: dict[str, dict[str, str]] = {}
    try:
        for row_number, row in enumerate(reader, 2):
            image_id = (row.get("image_id") or "").strip()
            if image_id not in dev_labels:
                continue
            if image_id in selected:
                raise RuntimeError(f"{source_name} has duplicate dev image_id: {image_id}")
            true_label = (row.get("true_label") or "").strip()
            if true_label != dev_labels[image_id]:
                raise RuntimeError(
                    f"Label mismatch in {source_name} for {image_id}: "
                    f"{true_label!r} != {dev_labels[image_id]!r}"
                )
            if (row.get("parse_ok") or "").strip().lower() != "true":
                raise RuntimeError(f"{source_name} has parse_ok=False for {image_id}")
            prediction = (row.get("predicted_diagnosis") or "").strip()
            if prediction not in ALLOWED_PREDICTIONS:
                raise RuntimeError(
                    f"Unknown prediction in {source_name} for {image_id}: "
                    f"{prediction!r}"
                )
            raw_response = row.get("raw_response") or ""
            selected[image_id] = {
                "prediction": prediction,
                "confidence": (row.get("confidence") or "").strip(),
                "subtype": (row.get("predicted_subtype") or "").strip(),
                "findings": parse_findings(raw_response, source_name, image_id),
                "raw_response": raw_response,
            }
    finally:
        csv_file.close()
    missing = sorted(set(dev_labels) - set(selected))
    if missing:
        raise RuntimeError(
            f"{source_name} is missing {len(missing)} dev image_id values: {missing}"
        )
    return selected


def load_dev_metadata(
    path: Path, dev_labels: dict[str, str]
) -> dict[str, dict[str, str]]:
    csv_file, reader = open_reader(
        path, METADATA_REQUIRED_COLUMNS, "benchmark_metadata.csv"
    )
    selected: dict[str, dict[str, str]] = {}
    try:
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if image_id not in dev_labels:
                continue
            if image_id in selected:
                raise RuntimeError(
                    f"benchmark_metadata.csv has duplicate dev image_id: {image_id}"
                )
            true_label = (row.get("true_label") or "").strip()
            if true_label != dev_labels[image_id]:
                raise RuntimeError(
                    f"Metadata label mismatch for {image_id}: "
                    f"{true_label!r} != {dev_labels[image_id]!r}"
                )
            selected[image_id] = {
                "source_dataset": (row.get("source_dataset") or "").strip(),
                "source_class": (row.get("source_class") or "").strip(),
            }
    finally:
        csv_file.close()
    missing = sorted(set(dev_labels) - set(selected))
    if missing:
        raise RuntimeError(
            "benchmark_metadata.csv is missing "
            f"{len(missing)} dev image_id values: {missing}"
        )
    return selected


def classify_transition(baseline_correct: bool, skill_correct: bool) -> str:
    if not baseline_correct and skill_correct:
        return "wrong_to_correct"
    if baseline_correct and not skill_correct:
        return "correct_to_wrong"
    if baseline_correct:
        return "unchanged_correct"
    return "unchanged_wrong"


def build_transition_rows(
    dev_labels: dict[str, str],
    metadata: dict[str, dict[str, str]],
    baseline: dict[str, dict[str, str]],
    skill: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for image_id in sorted(dev_labels):
        true_label = dev_labels[image_id]
        baseline_correct = baseline[image_id]["prediction"] == true_label
        skill_correct = skill[image_id]["prediction"] == true_label
        rows.append(
            {
                "image_id": image_id,
                "true_label": true_label,
                "source_dataset": metadata[image_id]["source_dataset"],
                "source_class": metadata[image_id]["source_class"],
                "baseline_prediction": baseline[image_id]["prediction"],
                "baseline_correct": baseline_correct,
                "baseline_confidence": baseline[image_id]["confidence"],
                "baseline_subtype": baseline[image_id]["subtype"],
                "baseline_findings": baseline[image_id]["findings"],
                "baseline_raw_response": baseline[image_id]["raw_response"],
                "skill_prediction": skill[image_id]["prediction"],
                "skill_correct": skill_correct,
                "skill_confidence": skill[image_id]["confidence"],
                "skill_subtype": skill[image_id]["subtype"],
                "skill_findings": skill[image_id]["findings"],
                "skill_raw_response": skill[image_id]["raw_response"],
                "transition": classify_transition(baseline_correct, skill_correct),
            }
        )
    return rows


def parse_confidence(value: object, image_id: str, field_name: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid {field_name} for {image_id}: {value!r}"
        ) from exc
    if not 0 <= parsed <= 1:
        raise RuntimeError(f"Out-of-range {field_name} for {image_id}: {parsed}")
    return parsed


def format_number(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def build_confidence_summary(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for transition in TRANSITION_ORDER:
        selected = [row for row in rows if row["transition"] == transition]
        baseline_values: list[float] = []
        skill_values: list[float] = []
        paired_deltas: list[float] = []
        for row in selected:
            image_id = str(row["image_id"])
            baseline_value = parse_confidence(
                row["baseline_confidence"], image_id, "baseline_confidence"
            )
            skill_value = parse_confidence(
                row["skill_confidence"], image_id, "skill_confidence"
            )
            if baseline_value is not None:
                baseline_values.append(baseline_value)
            if skill_value is not None:
                skill_values.append(skill_value)
            if baseline_value is not None and skill_value is not None:
                paired_deltas.append(skill_value - baseline_value)
        summary.append(
            {
                "transition": transition,
                "count": len(selected),
                "baseline_mean_confidence": format_number(
                    statistics.mean(baseline_values) if baseline_values else None
                ),
                "skill_mean_confidence": format_number(
                    statistics.mean(skill_values) if skill_values else None
                ),
                "mean_delta": format_number(
                    statistics.mean(paired_deltas) if paired_deltas else None
                ),
                "median_delta": format_number(
                    statistics.median(paired_deltas) if paired_deltas else None
                ),
                "baseline_missing_confidence": len(selected) - len(baseline_values),
                "skill_missing_confidence": len(selected) - len(skill_values),
                "paired_confidence_n": len(paired_deltas),
            }
        )
    return summary


def build_prediction_transition_counts(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    changed = [
        row
        for row in rows
        if row["baseline_prediction"] != row["skill_prediction"]
    ]
    output: list[dict[str, object]] = []
    for true_label in ("all", POSITIVE_LABEL, NEGATIVE_LABEL):
        scoped = (
            changed
            if true_label == "all"
            else [row for row in changed if row["true_label"] == true_label]
        )
        for baseline_prediction in PREDICTION_ORDER:
            for skill_prediction in PREDICTION_ORDER:
                if baseline_prediction == skill_prediction:
                    continue
                count = sum(
                    row["baseline_prediction"] == baseline_prediction
                    and row["skill_prediction"] == skill_prediction
                    for row in scoped
                )
                output.append(
                    {
                        "true_label": true_label,
                        "baseline_prediction": baseline_prediction,
                        "skill_prediction": skill_prediction,
                        "prediction_transition": (
                            f"{baseline_prediction} -> {skill_prediction}"
                        ),
                        "count": count,
                    }
                )
    return output


def combined_text(row: dict[str, object], prefix: str) -> str:
    return (
        f"{row[f'{prefix}_findings']}\n{row[f'{prefix}_raw_response']}"
    ).lower()


def build_phrase_frequency_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    compiled = {
        phrase: re.compile(pattern, re.IGNORECASE)
        for phrase, pattern in PHRASE_PATTERNS.items()
    }
    output: list[dict[str, object]] = []
    for transition in ("wrong_to_correct", "correct_to_wrong"):
        selected = [row for row in rows if row["transition"] == transition]
        for phrase, pattern in compiled.items():
            baseline_count = sum(
                bool(pattern.search(combined_text(row, "baseline")))
                for row in selected
            )
            skill_count = sum(
                bool(pattern.search(combined_text(row, "skill")))
                for row in selected
            )
            baseline_rate = baseline_count / len(selected) if selected else 0.0
            skill_rate = skill_count / len(selected) if selected else 0.0
            output.append(
                {
                    "transition": transition,
                    "phrase": phrase,
                    "group_n": len(selected),
                    "baseline_image_count": baseline_count,
                    "baseline_rate": format_number(baseline_rate),
                    "skill_image_count": skill_count,
                    "skill_rate": format_number(skill_rate),
                    "count_delta": skill_count - baseline_count,
                    "rate_delta": format_number(skill_rate - baseline_rate),
                }
            )
    return sorted(
        output,
        key=lambda row: (
            TRANSITION_ORDER.index(str(row["transition"])),
            -abs(float(str(row["rate_delta"]))),
            str(row["phrase"]),
        ),
    )


def write_csv_atomic(
    destination: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, object]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames, extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def markdown_escape(value: object) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")


def markdown_case_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| image_id | true_label | baseline | Skill v1 | transition |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(row[field])
                for field in (
                    "image_id",
                    "true_label",
                    "baseline_prediction",
                    "skill_prediction",
                    "transition",
                )
            )
            + " |"
        )
    return lines


def build_markdown_report(
    rows: list[dict[str, object]],
    transition_counts: Counter[str],
    confidence_rows: list[dict[str, object]],
    phrase_rows: list[dict[str, object]],
) -> str:
    wrong_to_correct = [
        row for row in rows if row["transition"] == "wrong_to_correct"
    ]
    correct_to_wrong = [
        row for row in rows if row["transition"] == "correct_to_wrong"
    ]
    not_applicable = [
        row for row in rows if row["skill_prediction"] == NOT_APPLICABLE
    ]
    changed = [
        row
        for row in rows
        if row["baseline_prediction"] != row["skill_prediction"]
    ]
    direction_counts = Counter(
        f"{row['baseline_prediction']} -> {row['skill_prediction']}"
        for row in changed
    )
    toward_conservative = sum(
        row["baseline_prediction"] == POSITIVE_LABEL
        and row["skill_prediction"] in {NEGATIVE_LABEL, NOT_APPLICABLE}
        for row in changed
    )
    toward_positive = sum(
        row["baseline_prediction"] in {NEGATIVE_LABEL, NOT_APPLICABLE}
        and row["skill_prediction"] == POSITIVE_LABEL
        for row in changed
    )
    corrected_false_positives = sum(
        row["true_label"] == NEGATIVE_LABEL for row in wrong_to_correct
    )
    corrected_false_negatives = sum(
        row["true_label"] == POSITIVE_LABEL for row in wrong_to_correct
    )
    introduced_false_negatives = sum(
        row["true_label"] == POSITIVE_LABEL
        and row["skill_prediction"] == NEGATIVE_LABEL
        for row in correct_to_wrong
    )
    introduced_false_positives = sum(
        row["true_label"] == NEGATIVE_LABEL
        and row["skill_prediction"] == POSITIVE_LABEL
        for row in correct_to_wrong
    )
    introduced_positive_abstentions = sum(
        row["true_label"] == POSITIVE_LABEL
        and row["skill_prediction"] == NOT_APPLICABLE
        for row in correct_to_wrong
    )
    negative_to_abstention = sum(
        row["baseline_prediction"] == NEGATIVE_LABEL
        and row["skill_prediction"] == NOT_APPLICABLE
        for row in changed
    )

    lines = [
        "# Skill v1 Transition Analysis",
        "",
        "## 1. Setup",
        "",
        "- Scope: locked dev split only.",
        f"- Paired dev images: {len(rows)}.",
        "- Correctness rule: `predicted_diagnosis == true_label`.",
        "- `not_applicable` is retained as an abstention and is incorrect on this benchmark.",
        "- Confidence values are model self-reported confidence, not calibrated probabilities.",
        "",
        "## 2. Overall transition counts",
        "",
        "| transition | count |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {transition} | {transition_counts[transition]} |"
        for transition in TRANSITION_ORDER
    )
    lines.extend(
        [
            "",
            f"Changed predictions: {len(changed)}.",
            "",
            "Changed image IDs: "
            + ", ".join(str(row["image_id"]) for row in changed)
            + ".",
            "",
            "## 3. Wrong -> Correct",
            "",
        ]
    )
    lines.extend(markdown_case_table(wrong_to_correct))
    lines.extend(["", "## 4. Correct -> Wrong", ""])
    lines.extend(markdown_case_table(correct_to_wrong))
    lines.extend(["", "## 5. Not-applicable cases", ""])
    lines.extend(markdown_case_table(not_applicable))
    lines.extend(["", "Skill v1 findings for these abstentions:", ""])
    for row in not_applicable:
        lines.append(
            f"- `{row['image_id']}` ({row['true_label']}): "
            f"{markdown_escape(row['skill_findings'])}"
        )
    lines.extend(
        [
            "",
            "## 6. Prediction direction changes",
            "",
            "| direction | count |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| {direction} | {count} |"
        for direction, count in sorted(direction_counts.items())
    )
    lines.extend(
        [
            "",
            f"Changes away from `echinococcosis`: {toward_conservative}.",
            f"Changes from `negative` to `not_applicable`: {negative_to_abstention}.",
            f"Changes toward `echinococcosis`: {toward_positive}.",
            "",
            "## 7. Confidence changes",
            "",
            "All values below are self-reported confidence summaries.",
            "",
            "| transition | count | baseline mean | Skill v1 mean | mean delta | median delta | baseline missing | Skill v1 missing | paired n |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in confidence_rows:
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(row[field])
                for field in (
                    "transition",
                    "count",
                    "baseline_mean_confidence",
                    "skill_mean_confidence",
                    "mean_delta",
                    "median_delta",
                    "baseline_missing_confidence",
                    "skill_missing_confidence",
                    "paired_confidence_n",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 8. Textual pattern changes",
            "",
            "Phrase frequency is the number of images whose findings/raw response contains the phrase or its listed regex variant. It is descriptive only.",
            "",
            "| transition | phrase | baseline images | Skill v1 images | delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in phrase_rows:
        if int(row["count_delta"]) > 0:
            lines.append(
                f"| {row['transition']} | {row['phrase']} | "
                f"{row['baseline_image_count']} | {row['skill_image_count']} | "
                f"+{row['count_delta']} |"
            )
    lines.extend(
        [
            "",
            "## 9. Implications for Skill v2",
            "",
            f"- On dev, Skill v1 corrected {corrected_false_positives} baseline false positives and {corrected_false_negatives} baseline false negatives.",
            f"- It introduced {introduced_false_negatives} binary false negatives, {introduced_false_positives} binary false positives, and {introduced_positive_abstentions} incorrect abstention among previously correct positive cases.",
            f"- Among changed predictions, {toward_conservative} moved away from `echinococcosis` and {negative_to_abstention} moved from `negative` to abstention, compared with {toward_positive} toward `echinococcosis`. This is consistent with a more conservative decision tendency on dev.",
            "- A Skill v2 experiment should preserve the observed reduction in false positives while testing ways to reduce new false negatives and abstentions.",
            "- Change one prompt element at a time and rerun the same locked dev transition analysis so the effect remains attributable.",
            "- These are experiment-design implications from dev behavior, not new medical rules.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_expected_counts(
    rows: list[dict[str, object]], transition_counts: Counter[str]
) -> None:
    observed = {name: transition_counts[name] for name in TRANSITION_ORDER}
    if observed != EXPECTED_TRANSITIONS:
        raise RuntimeError(
            f"Transition counts differ from compare_experiments: "
            f"expected {EXPECTED_TRANSITIONS}, found {observed}"
        )
    changed_n = sum(
        row["baseline_prediction"] != row["skill_prediction"] for row in rows
    )
    if changed_n != EXPECTED_CHANGED_N:
        raise RuntimeError(
            f"Changed row count must be {EXPECTED_CHANGED_N}; found {changed_n}"
        )
    not_applicable_n = sum(
        row["skill_prediction"] == NOT_APPLICABLE for row in rows
    )
    if not_applicable_n != EXPECTED_SKILL_NOT_APPLICABLE_N:
        raise RuntimeError(
            "Skill not_applicable count must be "
            f"{EXPECTED_SKILL_NOT_APPLICABLE_N}; found {not_applicable_n}"
        )


def main() -> None:
    dev_labels = load_dev_split(SPLIT_PATH)
    baseline = load_dev_results(BASELINE_PATH, "baseline results.csv", dev_labels)
    skill = load_dev_results(SKILL_PATH, "Skill v1 results.csv", dev_labels)
    metadata = load_dev_metadata(METADATA_PATH, dev_labels)
    rows = build_transition_rows(dev_labels, metadata, baseline, skill)

    transition_counts: Counter[str] = Counter(
        str(row["transition"]) for row in rows
    )
    validate_expected_counts(rows, transition_counts)

    subsets = {
        "wrong_to_correct.csv": [
            row for row in rows if row["transition"] == "wrong_to_correct"
        ],
        "correct_to_wrong.csv": [
            row for row in rows if row["transition"] == "correct_to_wrong"
        ],
        "unchanged_wrong.csv": [
            row for row in rows if row["transition"] == "unchanged_wrong"
        ],
        "not_applicable.csv": [
            row for row in rows if row["skill_prediction"] == NOT_APPLICABLE
        ],
    }
    prediction_counts = build_prediction_transition_counts(rows)
    confidence_rows = build_confidence_summary(rows)
    phrase_rows = build_phrase_frequency_rows(rows)
    report = build_markdown_report(
        rows, transition_counts, confidence_rows, phrase_rows
    )

    write_csv_atomic(
        OUTPUT_DIR / "skill_v1_transitions.csv", TRANSITION_COLUMNS, rows
    )
    for filename, subset_rows in subsets.items():
        write_csv_atomic(OUTPUT_DIR / filename, TRANSITION_COLUMNS, subset_rows)
    write_csv_atomic(
        OUTPUT_DIR / "prediction_transition_counts.csv",
        [
            "true_label",
            "baseline_prediction",
            "skill_prediction",
            "prediction_transition",
            "count",
        ],
        prediction_counts,
    )
    write_csv_atomic(
        OUTPUT_DIR / "confidence_transition_summary.csv",
        [
            "transition",
            "count",
            "baseline_mean_confidence",
            "skill_mean_confidence",
            "mean_delta",
            "median_delta",
            "baseline_missing_confidence",
            "skill_missing_confidence",
            "paired_confidence_n",
        ],
        confidence_rows,
    )
    write_csv_atomic(
        OUTPUT_DIR / "phrase_frequency_comparison.csv",
        [
            "transition",
            "phrase",
            "group_n",
            "baseline_image_count",
            "baseline_rate",
            "skill_image_count",
            "skill_rate",
            "count_delta",
            "rate_delta",
        ],
        phrase_rows,
    )
    report_path = OUTPUT_DIR / "skill_v1_transition_analysis.md"
    temporary_report = report_path.with_name(f".{report_path.name}.tmp")
    temporary_report.write_text(report, encoding="utf-8", newline="\n")
    temporary_report.replace(report_path)

    changed_ids = [
        str(row["image_id"])
        for row in rows
        if row["baseline_prediction"] != row["skill_prediction"]
    ]
    print(f"dev total: {len(rows)}")
    for transition in TRANSITION_ORDER:
        print(f"{transition}: {transition_counts[transition]}")
    print(f"not_applicable: {len(subsets['not_applicable.csv'])}")
    print(f"changed rows: {len(changed_ids)}")
    print(f"changed image_ids: {changed_ids}")
    print(f"output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

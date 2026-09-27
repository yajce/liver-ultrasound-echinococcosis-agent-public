#!/usr/bin/env python3
"""Analyze baseline-to-Skill-v3 prediction changes on the dev split only."""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "dev.csv"
BASELINE_PATH = PROJECT_ROOT / "experiments" / "glm_baseline_v02" / "results.csv"
V3_PATH = PROJECT_ROOT / "experiments" / "glm_skill_v3_dev" / "results.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "glm_skill_v3_dev" / "analysis"

POSITIVE = "echinococcosis"
NEGATIVE = "negative"
NOT_APPLICABLE = "not_applicable"
PARSE_FAILED = "parse_failed"
TRUE_LABELS = {POSITIVE, NEGATIVE}
PARSED_PREDICTIONS = {POSITIVE, NEGATIVE, NOT_APPLICABLE}
TRANSITION_ORDER = (
    "wrong_to_correct",
    "correct_to_wrong",
    "unchanged_correct",
    "unchanged_wrong",
)
EXPECTED_COUNTS = {
    "wrong_to_correct": 8,
    "correct_to_wrong": 8,
    "unchanged_correct": 77,
    "unchanged_wrong": 7,
}
EXPECTED_DEV_N = 100
EXPECTED_CHANGED_N = 16

RESULT_COLUMNS = {
    "image_id",
    "true_label",
    "image_valid",
    "predicted_diagnosis",
    "predicted_subtype",
    "confidence",
    "parse_ok",
    "raw_response",
}
OUTPUT_COLUMNS = [
    "image_id",
    "true_label",
    "baseline_prediction",
    "v3_prediction",
    "prediction_direction",
    "transition_type",
    "baseline_correct",
    "v3_correct",
    "baseline_confidence",
    "v3_confidence",
    "confidence_delta",
    "baseline_subtype",
    "v3_subtype",
    "baseline_image_valid",
    "v3_image_valid",
    "baseline_parse_ok",
    "v3_parse_ok",
    "baseline_special_state",
    "v3_special_state",
    "baseline_findings",
    "v3_findings",
]

MORPHOLOGY_RE = re.compile(
    r"\b(?:morpholog\w*|lesion\w*|cystic|solid|mixed|boundary|wall|"
    r"architecture|internal|echo(?:es|genic)?|structure\w*|parenchyma|"
    r"surrounding liver)\b",
    re.IGNORECASE,
)
DIAGNOSTIC_LANGUAGE_RE = re.compile(
    r"\b(?:echinococcosis|hydatid|diagnos\w*|suggestive|consistent with|"
    r"compatible with|support(?:s|ing|ed)?|favor(?:s|ed)?)\b",
    re.IGNORECASE,
)

TEXT_PATTERNS = {
    "morphology_vocabulary": MORPHOLOGY_RE,
    "observation_verbs": re.compile(
        r"\b(?:shows?|showing|demonstrates?|visible|observed|appears?|"
        r"appearance|noted|identified|seen)\b",
        re.IGNORECASE,
    ),
    "negation_language": re.compile(
        r"\b(?:no|not|without|absence|absent|lack|lacking)\b",
        re.IGNORECASE,
    ),
    "absence_based_exclusion_language": re.compile(
        r"(?:\b(?:no|without|absence|lack(?:ing)?|not (?:seen|identified|"
        r"visualized|visible|present))\b.{0,100}\b(?:characteristic|typical|"
        r"specific|feature|sign|evidence|morphology)\b)|"
        r"(?:\b(?:characteristic|typical|specific|feature|sign|evidence|"
        r"morphology)\b.{0,100}\b(?:absent|not (?:seen|identified|visualized|"
        r"visible|present))\b)",
        re.IGNORECASE,
    ),
    "conservative_language": re.compile(
        r"\b(?:nonspecific|not sufficient|does not support|no evidence|"
        r"not diagnostic|cannot confirm|insufficient|unlikely)\b",
        re.IGNORECASE,
    ),
    "generic_cystic_language": re.compile(
        r"\b(?:simple cyst|generic cyst|nonspecific cystic|"
        r"uncomplicated cyst|cystic lesion)\b",
        re.IGNORECASE,
    ),
    "uncertainty_language": re.compile(
        r"\b(?:possible|possibly|probable|likely|may|might|could|"
        r"cannot exclude|uncertain|suspicious)\b",
        re.IGNORECASE,
    ),
    "positive_support_language": re.compile(
        r"\b(?:support(?:s|ing|ed)?|suggestive|consistent with|"
        r"compatible with|favor(?:s|ed)?)\b",
        re.IGNORECASE,
    ),
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


def load_dev_labels(path: Path) -> dict[str, str]:
    csv_file, reader = open_reader(
        path, {"image_id", "true_label"}, "dev.csv"
    )
    labels: dict[str, str] = {}
    try:
        for row_number, row in enumerate(reader, 2):
            image_id = (row.get("image_id") or "").strip()
            true_label = (row.get("true_label") or "").strip()
            if not image_id:
                raise RuntimeError(f"dev.csv row {row_number} has no image_id")
            if image_id in labels:
                raise RuntimeError(f"dev.csv has duplicate image_id: {image_id}")
            if true_label not in TRUE_LABELS:
                raise RuntimeError(
                    f"Unknown true_label for {image_id}: {true_label!r}"
                )
            labels[image_id] = true_label
    finally:
        csv_file.close()
    if len(labels) != EXPECTED_DEV_N:
        raise RuntimeError(
            f"dev.csv must contain {EXPECTED_DEV_N} rows; found {len(labels)}"
        )
    return labels


def parse_bool(value: str, field_name: str, image_id: str) -> bool | None:
    normalized = (value or "").strip().lower()
    if not normalized:
        return None
    if normalized not in {"true", "false"}:
        raise RuntimeError(
            f"Invalid {field_name} for {image_id}: {value!r}"
        )
    return normalized == "true"


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


def parse_confidence(value: str, source_name: str, image_id: str) -> float | None:
    normalized = (value or "").strip()
    if not normalized:
        return None
    try:
        confidence = float(normalized)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid confidence in {source_name} for {image_id}: {value!r}"
        ) from exc
    if not 0 <= confidence <= 1:
        raise RuntimeError(
            f"Out-of-range confidence in {source_name} for {image_id}: "
            f"{confidence}"
        )
    return confidence


def load_dev_results(
    path: Path, source_name: str, dev_labels: dict[str, str]
) -> dict[str, dict[str, object]]:
    """Skip non-dev IDs before accessing any result or response fields."""
    csv_file, reader = open_reader(path, RESULT_COLUMNS, source_name)
    selected: dict[str, dict[str, object]] = {}
    try:
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if image_id not in dev_labels:
                continue
            if image_id in selected:
                raise RuntimeError(
                    f"{source_name} has duplicate dev image_id: {image_id}"
                )
            true_label = (row.get("true_label") or "").strip()
            if true_label != dev_labels[image_id]:
                raise RuntimeError(
                    f"Label mismatch in {source_name} for {image_id}: "
                    f"{true_label!r} != {dev_labels[image_id]!r}"
                )
            parse_ok = parse_bool(
                row.get("parse_ok") or "", "parse_ok", image_id
            )
            if parse_ok is None:
                raise RuntimeError(
                    f"Missing parse_ok in {source_name} for {image_id}"
                )
            recorded_prediction = (row.get("predicted_diagnosis") or "").strip()
            if not parse_ok:
                prediction = PARSE_FAILED
                findings = ""
            elif recorded_prediction in PARSED_PREDICTIONS:
                prediction = recorded_prediction
                findings = parse_findings(
                    row.get("raw_response") or "", source_name, image_id
                )
            else:
                raise RuntimeError(
                    f"Unknown prediction in {source_name} for {image_id}: "
                    f"{recorded_prediction!r}"
                )
            selected[image_id] = {
                "prediction": prediction,
                "confidence": parse_confidence(
                    row.get("confidence") or "", source_name, image_id
                ),
                "subtype": (row.get("predicted_subtype") or "").strip(),
                "image_valid": parse_bool(
                    row.get("image_valid") or "", "image_valid", image_id
                ),
                "parse_ok": parse_ok,
                "findings": findings,
            }
    finally:
        csv_file.close()
    missing = sorted(set(dev_labels) - set(selected))
    if missing:
        raise RuntimeError(
            f"{source_name} is missing {len(missing)} dev image_id values: {missing}"
        )
    return selected


def transition_type(baseline_correct: bool, v3_correct: bool) -> str:
    if not baseline_correct and v3_correct:
        return "wrong_to_correct"
    if baseline_correct and not v3_correct:
        return "correct_to_wrong"
    if baseline_correct:
        return "unchanged_correct"
    return "unchanged_wrong"


def special_state(prediction: str) -> str:
    return prediction if prediction in {NOT_APPLICABLE, PARSE_FAILED} else ""


def format_number(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def build_all_rows(
    labels: dict[str, str],
    baseline: dict[str, dict[str, object]],
    v3: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for image_id in sorted(labels):
        true_label = labels[image_id]
        baseline_prediction = str(baseline[image_id]["prediction"])
        v3_prediction = str(v3[image_id]["prediction"])
        baseline_correct = baseline_prediction == true_label
        v3_correct = v3_prediction == true_label
        baseline_confidence = baseline[image_id]["confidence"]
        v3_confidence = v3[image_id]["confidence"]
        confidence_delta = (
            float(v3_confidence) - float(baseline_confidence)
            if baseline_confidence is not None and v3_confidence is not None
            else None
        )
        rows.append(
            {
                "image_id": image_id,
                "true_label": true_label,
                "baseline_prediction": baseline_prediction,
                "v3_prediction": v3_prediction,
                "prediction_direction": (
                    f"{baseline_prediction} -> {v3_prediction}"
                ),
                "transition_type": transition_type(
                    baseline_correct, v3_correct
                ),
                "baseline_correct": baseline_correct,
                "v3_correct": v3_correct,
                "baseline_confidence": format_number(
                    baseline_confidence  # type: ignore[arg-type]
                ),
                "v3_confidence": format_number(
                    v3_confidence  # type: ignore[arg-type]
                ),
                "confidence_delta": format_number(confidence_delta),
                "baseline_subtype": baseline[image_id]["subtype"],
                "v3_subtype": v3[image_id]["subtype"],
                "baseline_image_valid": baseline[image_id]["image_valid"],
                "v3_image_valid": v3[image_id]["image_valid"],
                "baseline_parse_ok": baseline[image_id]["parse_ok"],
                "v3_parse_ok": v3[image_id]["parse_ok"],
                "baseline_special_state": special_state(baseline_prediction),
                "v3_special_state": special_state(v3_prediction),
                "baseline_findings": baseline[image_id]["findings"],
                "v3_findings": v3[image_id]["findings"],
            }
        )
    return rows


def morphology_before_diagnosis(text: str) -> bool:
    morphology = MORPHOLOGY_RE.search(text)
    if morphology is None:
        return False
    diagnostic = DIAGNOSTIC_LANGUAGE_RE.search(text)
    return diagnostic is None or morphology.start() < diagnostic.start()


def text_behavior_value(text: str, behavior: str) -> bool:
    if behavior == "morphology_before_diagnostic_language":
        return morphology_before_diagnosis(text)
    return bool(TEXT_PATTERNS[behavior].search(text))


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def build_summary_rows(
    all_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    transition_counts = Counter(str(row["transition_type"]) for row in all_rows)
    for transition in TRANSITION_ORDER:
        output.append(
            {
                "section": "transition_count",
                "group": transition,
                "metric": "count",
                "baseline_value": "",
                "v3_value": transition_counts[transition],
                "delta": "",
                "notes": "paired correctness category",
            }
        )

    changed_rows = [
        row
        for row in all_rows
        if row["baseline_prediction"] != row["v3_prediction"]
    ]
    direction_counts = Counter(
        str(row["prediction_direction"]) for row in changed_rows
    )
    for direction in sorted(direction_counts):
        output.append(
            {
                "section": "prediction_direction",
                "group": "changed",
                "metric": direction,
                "baseline_value": "",
                "v3_value": direction_counts[direction],
                "delta": "",
                "notes": "changed predictions only",
            }
        )

    for transition in ("wrong_to_correct", "correct_to_wrong"):
        selected = [
            row for row in all_rows if row["transition_type"] == transition
        ]
        baseline_confidences: list[float] = []
        v3_confidences: list[float] = []
        paired_deltas: list[float] = []
        for row in selected:
            baseline_text = str(row["baseline_confidence"])
            v3_text = str(row["v3_confidence"])
            baseline_confidence = float(baseline_text) if baseline_text else None
            v3_confidence = float(v3_text) if v3_text else None
            if baseline_confidence is not None:
                baseline_confidences.append(baseline_confidence)
            if v3_confidence is not None:
                v3_confidences.append(v3_confidence)
            if baseline_confidence is not None and v3_confidence is not None:
                paired_deltas.append(v3_confidence - baseline_confidence)
        confidence_metrics = {
            "mean_confidence": (
                mean_or_none(baseline_confidences),
                mean_or_none(v3_confidences),
            ),
            "mean_paired_confidence_delta": (
                None,
                mean_or_none(paired_deltas),
            ),
            "median_paired_confidence_delta": (
                None,
                median_or_none(paired_deltas),
            ),
            "missing_confidence": (
                float(len(selected) - len(baseline_confidences)),
                float(len(selected) - len(v3_confidences)),
            ),
        }
        for metric, (baseline_value, v3_value) in confidence_metrics.items():
            delta = (
                v3_value - baseline_value
                if baseline_value is not None and v3_value is not None
                else None
            )
            output.append(
                {
                    "section": "confidence",
                    "group": transition,
                    "metric": metric,
                    "baseline_value": format_number(baseline_value),
                    "v3_value": format_number(v3_value),
                    "delta": format_number(delta),
                    "notes": "self-reported confidence",
                }
            )

        behaviors = [
            "morphology_before_diagnostic_language",
            *TEXT_PATTERNS.keys(),
        ]
        for behavior in behaviors:
            baseline_count = sum(
                text_behavior_value(str(row["baseline_findings"]), behavior)
                for row in selected
            )
            v3_count = sum(
                text_behavior_value(str(row["v3_findings"]), behavior)
                for row in selected
            )
            output.append(
                {
                    "section": "text_behavior",
                    "group": transition,
                    "metric": behavior,
                    "baseline_value": baseline_count,
                    "v3_value": v3_count,
                    "delta": v3_count - baseline_count,
                    "notes": f"image count out of {len(selected)}",
                }
            )

        for state in (NOT_APPLICABLE, PARSE_FAILED):
            baseline_count = sum(
                row["baseline_prediction"] == state for row in selected
            )
            v3_count = sum(row["v3_prediction"] == state for row in selected)
            output.append(
                {
                    "section": "special_state",
                    "group": transition,
                    "metric": state,
                    "baseline_value": baseline_count,
                    "v3_value": v3_count,
                    "delta": v3_count - baseline_count,
                    "notes": "prediction state count",
                }
            )

        record_metrics = {
            "subtype_present": (
                sum(bool(str(row["baseline_subtype"]).strip()) for row in selected),
                sum(bool(str(row["v3_subtype"]).strip()) for row in selected),
            ),
            "image_valid_false": (
                sum(row["baseline_image_valid"] is False for row in selected),
                sum(row["v3_image_valid"] is False for row in selected),
            ),
            "image_valid_missing": (
                sum(row["baseline_image_valid"] is None for row in selected),
                sum(row["v3_image_valid"] is None for row in selected),
            ),
            "parse_ok_false": (
                sum(row["baseline_parse_ok"] is False for row in selected),
                sum(row["v3_parse_ok"] is False for row in selected),
            ),
        }
        for metric, (baseline_count, v3_count) in record_metrics.items():
            output.append(
                {
                    "section": "record_state",
                    "group": transition,
                    "metric": metric,
                    "baseline_value": baseline_count,
                    "v3_value": v3_count,
                    "delta": v3_count - baseline_count,
                    "notes": f"image count out of {len(selected)}",
                }
            )
    return output


def markdown_escape(value: object) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")


def summary_lookup(
    summary_rows: list[dict[str, object]], section: str, group: str, metric: str
) -> dict[str, object]:
    for row in summary_rows:
        if (
            row["section"] == section
            and row["group"] == group
            and row["metric"] == metric
        ):
            return row
    raise KeyError((section, group, metric))


def build_report(
    all_rows: list[dict[str, object]],
    changed_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> str:
    transition_counts = Counter(str(row["transition_type"]) for row in all_rows)
    directions = Counter(str(row["prediction_direction"]) for row in changed_rows)
    wrong_to_correct = [
        row for row in changed_rows if row["transition_type"] == "wrong_to_correct"
    ]
    correct_to_wrong = [
        row for row in changed_rows if row["transition_type"] == "correct_to_wrong"
    ]
    corrected_false_negatives = sum(
        row["true_label"] == POSITIVE for row in wrong_to_correct
    )
    corrected_false_positives = sum(
        row["true_label"] == NEGATIVE for row in wrong_to_correct
    )
    new_binary_false_negatives = sum(
        row["true_label"] == POSITIVE and row["v3_prediction"] == NEGATIVE
        for row in correct_to_wrong
    )
    new_positive_abstentions = sum(
        row["true_label"] == POSITIVE
        and row["v3_prediction"] == NOT_APPLICABLE
        for row in correct_to_wrong
    )
    new_false_positives = sum(
        row["true_label"] == NEGATIVE and row["v3_prediction"] == POSITIVE
        for row in correct_to_wrong
    )
    lines = [
        "# Baseline vs Skill v3 Transition Analysis",
        "",
        "## Scope",
        "",
        "This report uses the locked dev split only. It compares stored model outputs and does not interpret image content. Keyword counts describe model language, including negated language; they are not evidence that a finding is present in an image.",
        "",
        "## Overall paired transitions",
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
            f"Changed predictions: {len(changed_rows)}.",
            "",
            "## Changed samples",
            "",
            "| image_id | true_label | baseline | v3 | transition | confidence delta |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for row in changed_rows:
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(row[field])
                for field in (
                    "image_id",
                    "true_label",
                    "baseline_prediction",
                    "v3_prediction",
                    "transition_type",
                    "confidence_delta",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Prediction directions",
            "",
            "| direction | count |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| {direction} | {count} |"
        for direction, count in sorted(directions.items())
    )
    lines.extend(
        [
            "",
            "## Confidence changes",
            "",
            "Confidence means self-reported confidence, not calibrated probability.",
            "",
            "| transition | baseline mean | v3 mean | paired mean delta | paired median delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for transition in ("wrong_to_correct", "correct_to_wrong"):
        mean_row = summary_lookup(
            summary_rows, "confidence", transition, "mean_confidence"
        )
        delta_row = summary_lookup(
            summary_rows,
            "confidence",
            transition,
            "mean_paired_confidence_delta",
        )
        median_row = summary_lookup(
            summary_rows,
            "confidence",
            transition,
            "median_paired_confidence_delta",
        )
        lines.append(
            f"| {transition} | {mean_row['baseline_value']} | "
            f"{mean_row['v3_value']} | {delta_row['v3_value']} | "
            f"{median_row['v3_value']} |"
        )
    lines.extend(
        [
            "",
            "## Subtype, validity, and parse status",
            "",
            "| transition | state | baseline | v3 | delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        if row["section"] in {"special_state", "record_state"}:
            lines.append(
                f"| {row['group']} | {row['metric']} | "
                f"{row['baseline_value']} | {row['v3_value']} | "
                f"{row['delta']} |"
            )
    lines.extend(
        [
            "",
            "## Text behavior counts",
            "",
            "Each value is the number of images in the transition group whose findings contain the lexical pattern.",
            "",
            "| transition | behavior | baseline | v3 | delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        if row["section"] == "text_behavior":
            lines.append(
                f"| {row['group']} | {row['metric']} | "
                f"{row['baseline_value']} | {row['v3_value']} | "
                f"{row['delta']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "### Behavior that helped",
            "",
            f"- V3 corrected {corrected_false_negatives} baseline false negatives and {corrected_false_positives} baseline false positives.",
            "- In wrong-to-correct cases, absence-based exclusion language decreased from 4/8 to 1/8, while positive-support language increased from 3/8 to 5/8.",
            "- This textual shift is consistent with using visible positive support instead of letting absent expected features determine the answer. It is a description of model language, not an image-level medical conclusion.",
            "",
            "### Behavior that harmed",
            "",
            f"- V3 introduced {new_binary_false_negatives} binary false negatives, {new_positive_abstentions} incorrect positive abstention, and {new_false_positives} false positives.",
            "- In correct-to-wrong cases, negation language increased from 5/8 to 7/8 and absence-based exclusion language increased from 2/8 to 3/8; positive-support language decreased from 4/8 to 3/8.",
            "- The two new false positives used positive-support language for heterogeneous or complex morphology while retaining differential uncertainty. The same broad evidence-integration behavior that recovered positives also created these errors.",
            "",
            "### Cross-group findings",
            "",
            "- Morphology appeared before diagnostic language in 8/8 baseline and 8/8 v3 findings in both transition groups. The structured observation order therefore did not distinguish v3 from baseline on these changed cases.",
            "- Generic cystic language did not increase: it remained 3/8 in wrong-to-correct cases and decreased from 4/8 to 3/8 in correct-to-wrong cases. The data do not support generic cyst wording as a single dominant failure mechanism.",
            "- Self-reported confidence decreased slightly in both helpful and harmful transitions, so confidence change does not separate improvements from regressions.",
            "- Keyword matches may occur in positive, negative, or differential statements and must not be treated as evidence that a finding is present in an image.",
            "",
            "## V4 decision",
            "",
            "No sufficiently clear, asymmetric, image-ID-independent issue emerged for a defensible single-variable Skill v4 change. The same relaxation of exclusion behavior helped some cases and harmed others, while total strict accuracy was unchanged. Further Skill tuning on this dev set should stop rather than continue fitting prompt wording to these 16 transitions.",
            "",
        ]
    )
    return "\n".join(lines)


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


def validate_counts(
    all_rows: list[dict[str, object]], changed_rows: list[dict[str, object]]
) -> None:
    transition_counts = Counter(str(row["transition_type"]) for row in all_rows)
    observed = {name: transition_counts[name] for name in TRANSITION_ORDER}
    if observed != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Unexpected transition counts: expected {EXPECTED_COUNTS}, "
            f"found {observed}"
        )
    if len(changed_rows) != EXPECTED_CHANGED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_CHANGED_N} changed rows; found {len(changed_rows)}"
        )


def main() -> None:
    dev_labels = load_dev_labels(DEV_SPLIT_PATH)
    baseline = load_dev_results(
        BASELINE_PATH, "baseline results.csv", dev_labels
    )
    v3 = load_dev_results(V3_PATH, "Skill v3 results.csv", dev_labels)
    all_rows = build_all_rows(dev_labels, baseline, v3)
    changed_rows = [
        row
        for row in all_rows
        if row["baseline_prediction"] != row["v3_prediction"]
    ]
    changed_rows.sort(
        key=lambda row: (
            TRANSITION_ORDER.index(str(row["transition_type"])),
            str(row["image_id"]),
        )
    )
    validate_counts(all_rows, changed_rows)
    summary_rows = build_summary_rows(all_rows)
    report = build_report(all_rows, changed_rows, summary_rows)

    write_csv_atomic(
        OUTPUT_DIR / "baseline_vs_v3_transitions.csv",
        OUTPUT_COLUMNS,
        changed_rows,
    )
    write_csv_atomic(
        OUTPUT_DIR / "baseline_vs_v3_transition_summary.csv",
        [
            "section",
            "group",
            "metric",
            "baseline_value",
            "v3_value",
            "delta",
            "notes",
        ],
        summary_rows,
    )
    report_path = OUTPUT_DIR / "baseline_vs_v3_transition_analysis.md"
    temporary_report = report_path.with_name(f".{report_path.name}.tmp")
    temporary_report.write_text(report, encoding="utf-8", newline="\n")
    temporary_report.replace(report_path)

    counts = Counter(str(row["transition_type"]) for row in all_rows)
    print(f"dev total: {len(all_rows)}")
    for transition in TRANSITION_ORDER:
        print(f"{transition}: {counts[transition]}")
    print(f"changed rows: {len(changed_rows)}")
    print(f"output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

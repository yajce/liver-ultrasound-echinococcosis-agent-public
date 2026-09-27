#!/usr/bin/env python3
"""Analyze DeepSeek baseline-to-Skill-v3 prediction transitions on dev only."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = (
    PROJECT_ROOT / "experiments" / "deepseek_baseline_dev" / "results.csv"
)
DEFAULT_V3 = (
    PROJECT_ROOT / "experiments" / "deepseek_skill_v3_dev" / "results.csv"
)
DEFAULT_DEV = PROJECT_ROOT / "data" / "splits" / "dev.csv"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "benchmark_metadata.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "deepseek_skill_v3_dev" / "analysis"
)

POSITIVE = "echinococcosis"
NEGATIVE = "negative"
NOT_APPLICABLE = "not_applicable"
PARSE_FAILED = "parse_failed"
TRUE_LABELS = {POSITIVE, NEGATIVE}
PARSED_PREDICTIONS = TRUE_LABELS | {NOT_APPLICABLE}
ABSTENTION_STATES = {NOT_APPLICABLE, PARSE_FAILED}
SUBTYPE_ORDER = ("Normal", "Benign", "Malignant")
TRANSITION_ORDER = (
    "wrong_to_correct",
    "correct_to_wrong",
    "unchanged_correct",
    "unchanged_wrong",
)
EXPECTED_TRANSITIONS = {
    "wrong_to_correct": 15,
    "correct_to_wrong": 8,
    "unchanged_correct": 55,
    "unchanged_wrong": 22,
}
EXPECTED_DEV_N = 100

TRANSITION_COLUMNS = (
    "image_id",
    "true_label",
    "source_class",
    "baseline_prediction",
    "v3_prediction",
    "transition_type",
    "baseline_correct",
    "v3_correct",
    "baseline_parse_ok",
    "v3_parse_ok",
    "baseline_confidence",
    "v3_confidence",
    "confidence_delta",
    "baseline_findings",
    "v3_findings",
)
SUMMARY_COLUMNS = (
    "section",
    "category",
    "source_class",
    "true_label",
    "baseline_prediction",
    "v3_prediction",
    "count",
    "baseline_value",
    "v3_value",
    "delta",
)

# Document-frequency patterns only. They describe model wording and do not encode
# or infer diagnostic rules.
TEXT_PATTERNS = {
    "explicit negative wording": r"\bnegative\b",
    "nonspecific wording": r"\bnon[- ]?specific\b",
    "literal insufficient-evidence wording": (
        r"\b(?:insufficient|limited)\s+(?:evidence|support|findings?|features?)\b"
        r"|\bnot sufficient\b"
    ),
    "absence-of-support wording": (
        r"\b(?:no|without|does not (?:show|demonstrate))\b[^.!;]{0,80}"
        r"\b(?:definite|convincing|specific|characteristic|evidence|features?|morphology)\b"
    ),
    "any explicit 'no' wording": r"\bno\b",
    "suspicion wording": r"\b(?:suspicious|suspicion|suggestive|raises? suspicion)\b",
    "cannot-exclude wording": r"\bcannot (?:be )?exclude(?:d)?\b",
    "systematic observation vocabulary": (
        r"\b(?:parenchyma|boundary|wall|internal|surrounding)\b"
    ),
}


def read_csv(path: Path, required: set[str], source_name: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{source_name} not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(f"{source_name} is missing columns: {missing}")
        return list(reader)


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
        if true_label not in TRUE_LABELS:
            raise RuntimeError(f"Unknown dev true_label for {image_id}: {true_label!r}")
        labels[image_id] = true_label
    if len(labels) != EXPECTED_DEV_N:
        raise RuntimeError(
            f"dev.csv must contain {EXPECTED_DEV_N} rows; found {len(labels)}"
        )
    return labels


def parse_boolean(value: str, field: str, image_id: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized not in {"true", "false"}:
        raise RuntimeError(f"{field} must be True or False for {image_id}: {value!r}")
    return normalized == "true"


def parse_confidence(value: str, image_id: str) -> float | None:
    normalized = (value or "").strip()
    if not normalized:
        return None
    try:
        confidence = float(normalized)
    except ValueError as exc:
        raise RuntimeError(f"Invalid confidence for {image_id}: {value!r}") from exc
    if not 0 <= confidence <= 1:
        raise RuntimeError(f"Confidence outside [0, 1] for {image_id}: {confidence}")
    return confidence


def extract_findings(raw_response: str, image_id: str, source_name: str) -> str:
    try:
        start = raw_response.index("{")
        end = raw_response.rindex("}")
        payload = json.loads(raw_response[start : end + 1])
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot parse raw_response in {source_name} for {image_id}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"raw_response is not a JSON object for {image_id}")
    findings = payload.get("findings")
    return findings if isinstance(findings, str) else ""


def load_results(
    path: Path,
    source_name: str,
    dev_labels: dict[str, str],
) -> dict[str, dict[str, object]]:
    required = {
        "image_id",
        "true_label",
        "predicted_diagnosis",
        "confidence",
        "parse_ok",
        "raw_response",
        "model",
        "prompt_version",
        "skill_enabled",
        "skill_version",
    }
    rows = read_csv(path, required, source_name)
    selected: dict[str, dict[str, object]] = {}
    for row in rows:
        image_id = (row.get("image_id") or "").strip()
        if image_id not in dev_labels:
            continue
        if image_id in selected:
            raise RuntimeError(f"{source_name} has duplicate dev image_id: {image_id}")
        true_label = (row.get("true_label") or "").strip()
        if true_label != dev_labels[image_id]:
            raise RuntimeError(
                f"Label mismatch in {source_name} for {image_id}: {true_label!r}"
            )
        parse_ok = parse_boolean(row.get("parse_ok") or "", "parse_ok", image_id)
        recorded_prediction = (row.get("predicted_diagnosis") or "").strip()
        if parse_ok:
            if recorded_prediction not in PARSED_PREDICTIONS:
                raise RuntimeError(
                    f"Unknown prediction in {source_name} for {image_id}: "
                    f"{recorded_prediction!r}"
                )
            prediction = recorded_prediction
            findings = extract_findings(
                row.get("raw_response") or "", image_id, source_name
            )
        else:
            prediction = PARSE_FAILED
            findings = ""
        selected[image_id] = {
            "prediction": prediction,
            "parse_ok": parse_ok,
            "confidence": parse_confidence(row.get("confidence") or "", image_id),
            "findings": findings,
            "model": (row.get("model") or "").strip(),
            "prompt_version": (row.get("prompt_version") or "").strip(),
            "skill_enabled": (row.get("skill_enabled") or "").strip(),
            "skill_version": (row.get("skill_version") or "").strip(),
        }
    missing = sorted(set(dev_labels) - set(selected))
    if missing:
        raise RuntimeError(
            f"{source_name} is missing {len(missing)} dev image_id values: {missing}"
        )
    return selected


def load_metadata(
    path: Path, dev_labels: dict[str, str]
) -> dict[str, dict[str, str]]:
    rows = read_csv(
        path,
        {"image_id", "true_label", "source_class"},
        "benchmark_metadata.csv",
    )
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
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
                f"Metadata label mismatch for {image_id}: {true_label!r}"
            )
        source_class = (row.get("source_class") or "").strip()
        if true_label == NEGATIVE and source_class not in SUBTYPE_ORDER:
            raise RuntimeError(
                f"Unknown negative source_class for {image_id}: {source_class!r}"
            )
        selected[image_id] = row
    missing = sorted(set(dev_labels) - set(selected))
    if missing:
        raise RuntimeError(
            "benchmark_metadata.csv is missing dev image_id values: " + str(missing)
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


def classify_improvement(row: dict[str, object]) -> str:
    true_label = row["true_label"]
    baseline = row["baseline_prediction"]
    v3 = row["v3_prediction"]
    if true_label == NEGATIVE and baseline == POSITIVE and v3 == NEGATIVE:
        return "FP_to_TN"
    if true_label == POSITIVE and baseline == NEGATIVE and v3 == POSITIVE:
        return "FN_to_TP"
    if baseline in ABSTENTION_STATES and v3 == true_label:
        return "abstention_to_correct"
    return "other"


def classify_harm(row: dict[str, object]) -> str:
    true_label = row["true_label"]
    baseline = row["baseline_prediction"]
    v3 = row["v3_prediction"]
    if true_label == POSITIVE and baseline == POSITIVE and v3 == NEGATIVE:
        return "TP_to_FN"
    if true_label == NEGATIVE and baseline == NEGATIVE and v3 == POSITIVE:
        return "TN_to_FP"
    if baseline == true_label and v3 in ABSTENTION_STATES:
        return "correct_to_abstention"
    return "other"


def format_optional(value: float | None) -> str | float:
    return "" if value is None else value


def build_transition_rows(
    dev_labels: dict[str, str],
    metadata: dict[str, dict[str, str]],
    baseline: dict[str, dict[str, object]],
    v3: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for image_id in sorted(dev_labels):
        true_label = dev_labels[image_id]
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
                "source_class": (
                    (metadata[image_id].get("source_class") or "").strip()
                    if true_label == NEGATIVE
                    else ""
                ),
                "baseline_prediction": baseline_prediction,
                "v3_prediction": v3_prediction,
                "transition_type": transition_type(baseline_correct, v3_correct),
                "baseline_correct": baseline_correct,
                "v3_correct": v3_correct,
                "baseline_parse_ok": baseline[image_id]["parse_ok"],
                "v3_parse_ok": v3[image_id]["parse_ok"],
                "baseline_confidence": format_optional(baseline_confidence),
                "v3_confidence": format_optional(v3_confidence),
                "confidence_delta": format_optional(confidence_delta),
                "baseline_findings": baseline[image_id]["findings"],
                "v3_findings": v3[image_id]["findings"],
            }
        )
    return rows


def phrase_counts(
    rows: list[dict[str, object]], transition: str
) -> dict[str, tuple[int, int]]:
    selected = [row for row in rows if row["transition_type"] == transition]
    counts: dict[str, tuple[int, int]] = {}
    for label, pattern in TEXT_PATTERNS.items():
        baseline_count = sum(
            bool(re.search(pattern, str(row["baseline_findings"]), flags=re.I))
            for row in selected
        )
        v3_count = sum(
            bool(re.search(pattern, str(row["v3_findings"]), flags=re.I))
            for row in selected
        )
        counts[label] = (baseline_count, v3_count)
    return counts


def make_summary_rows(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    summary: list[dict[str, object]] = []
    transition_counts = Counter(str(row["transition_type"]) for row in rows)
    for transition in TRANSITION_ORDER:
        summary.append(
            {
                "section": "overall_correctness_transition",
                "category": transition,
                "count": transition_counts[transition],
            }
        )

    improvements = [row for row in rows if row["transition_type"] == "wrong_to_correct"]
    improvement_counts = Counter(classify_improvement(row) for row in improvements)
    for category in ("FP_to_TN", "FN_to_TP", "abstention_to_correct", "other"):
        summary.append(
            {
                "section": "wrong_to_correct_composition",
                "category": category,
                "count": improvement_counts[category],
            }
        )

    harms = [row for row in rows if row["transition_type"] == "correct_to_wrong"]
    harm_counts = Counter(classify_harm(row) for row in harms)
    for category in ("TP_to_FN", "TN_to_FP", "correct_to_abstention", "other"):
        summary.append(
            {
                "section": "correct_to_wrong_composition",
                "category": category,
                "count": harm_counts[category],
            }
        )

    direction_counts = Counter(
        (
            str(row["true_label"]),
            str(row["baseline_prediction"]),
            str(row["v3_prediction"]),
        )
        for row in rows
        if row["baseline_prediction"] != row["v3_prediction"]
    )
    for (true_label, baseline_prediction, v3_prediction), count in sorted(
        direction_counts.items()
    ):
        summary.append(
            {
                "section": "prediction_direction",
                "category": "prediction_changed",
                "true_label": true_label,
                "baseline_prediction": baseline_prediction,
                "v3_prediction": v3_prediction,
                "count": count,
            }
        )

    subtype_summary: dict[str, dict[str, int]] = {}
    for subtype in SUBTYPE_ORDER:
        subtype_rows = [
            row
            for row in rows
            if row["true_label"] == NEGATIVE and row["source_class"] == subtype
        ]
        baseline_fp = sum(row["baseline_prediction"] == POSITIVE for row in subtype_rows)
        v3_fp = sum(row["v3_prediction"] == POSITIVE for row in subtype_rows)
        fp_to_tn = sum(classify_improvement(row) == "FP_to_TN" for row in subtype_rows)
        tn_to_fp = sum(classify_harm(row) == "TN_to_FP" for row in subtype_rows)
        subtype_summary[subtype] = {
            "N": len(subtype_rows),
            "baseline_FP": baseline_fp,
            "v3_FP": v3_fp,
            "FP_to_TN": fp_to_tn,
            "TN_to_FP": tn_to_fp,
        }
        summary.append(
            {
                "section": "negative_subtype",
                "category": "FP_count",
                "source_class": subtype,
                "count": len(subtype_rows),
                "baseline_value": baseline_fp,
                "v3_value": v3_fp,
                "delta": v3_fp - baseline_fp,
            }
        )
        summary.append(
            {
                "section": "negative_subtype",
                "category": "FP_to_TN",
                "source_class": subtype,
                "count": fp_to_tn,
            }
        )
        summary.append(
            {
                "section": "negative_subtype",
                "category": "TN_to_FP",
                "source_class": subtype,
                "count": tn_to_fp,
            }
        )

    for transition in ("wrong_to_correct", "correct_to_wrong"):
        for phrase, (baseline_count, v3_count) in phrase_counts(rows, transition).items():
            summary.append(
                {
                    "section": f"text_phrase_{transition}",
                    "category": phrase,
                    "count": sum(row["transition_type"] == transition for row in rows),
                    "baseline_value": baseline_count,
                    "v3_value": v3_count,
                    "delta": v3_count - baseline_count,
                }
            )

    for row in summary:
        for column in SUMMARY_COLUMNS:
            row.setdefault(column, "")
    return summary, subtype_summary


def confidence_summary(
    rows: list[dict[str, object]], transition: str
) -> tuple[int, float | None, float | None, float | None, float | None]:
    pairs = [
        (float(row["baseline_confidence"]), float(row["v3_confidence"]))
        for row in rows
        if row["transition_type"] == transition
        and row["baseline_confidence"] != ""
        and row["v3_confidence"] != ""
    ]
    if not pairs:
        return 0, None, None, None, None
    deltas = [v3 - baseline for baseline, v3 in pairs]
    return (
        len(pairs),
        statistics.mean(baseline for baseline, _ in pairs),
        statistics.mean(v3 for _, v3 in pairs),
        statistics.mean(deltas),
        statistics.median(deltas),
    )


def percent(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.2%}" if denominator else "NA"


def build_report(
    rows: list[dict[str, object]],
    subtype_summary: dict[str, dict[str, int]],
) -> str:
    transitions = Counter(str(row["transition_type"]) for row in rows)
    improvements = Counter(
        classify_improvement(row)
        for row in rows
        if row["transition_type"] == "wrong_to_correct"
    )
    harms = Counter(
        classify_harm(row)
        for row in rows
        if row["transition_type"] == "correct_to_wrong"
    )
    baseline_predictions = Counter(str(row["baseline_prediction"]) for row in rows)
    v3_predictions = Counter(str(row["v3_prediction"]) for row in rows)
    baseline_correct = sum(bool(row["baseline_correct"]) for row in rows)
    v3_correct = sum(bool(row["v3_correct"]) for row in rows)
    helpful_phrases = phrase_counts(rows, "wrong_to_correct")
    harmful_phrases = phrase_counts(rows, "correct_to_wrong")
    helpful_confidence = confidence_summary(rows, "wrong_to_correct")
    harmful_confidence = confidence_summary(rows, "correct_to_wrong")

    lines = [
        "# DeepSeek Baseline vs Skill v3 Transition Analysis",
        "",
        "## 1. Scope",
        "",
        "- Paired analysis uses the 100 image IDs listed in `data/splits/dev.csv`.",
        "- Model recorded in both result files: `deepseek/DeepSeek-V4-Flash-Vision-Exp`.",
        "- Prompt version in both result files: `baseline-v0.2`.",
        "- Baseline has Skill disabled; the candidate uses frozen Skill v3.",
        "- No held-out test rows, images, or model calls are used.",
        "",
        "## 2. Overall paired transitions",
        "",
        "| Transition | Count |",
        "|---|---:|",
        *[f"| {name} | {transitions[name]} |" for name in TRANSITION_ORDER],
        "",
        f"Baseline strict accuracy: {baseline_correct}/{len(rows)} = {baseline_correct / len(rows):.2%}.",
        f"Skill v3 strict accuracy: {v3_correct}/{len(rows)} = {v3_correct / len(rows):.2%}.",
        "",
        "Prediction counts shifted from "
        f"{baseline_predictions[POSITIVE]} echinococcosis, "
        f"{baseline_predictions[NEGATIVE]} negative, and "
        f"{baseline_predictions[NOT_APPLICABLE]} not_applicable at baseline to "
        f"{v3_predictions[POSITIVE]} echinococcosis, "
        f"{v3_predictions[NEGATIVE]} negative, and "
        f"{v3_predictions[NOT_APPLICABLE]} not_applicable with v3.",
        "",
        "## 3. Wrong to correct",
        "",
        "| Composition | Count |",
        "|---|---:|",
        f"| FP to TN | {improvements['FP_to_TN']} |",
        f"| FN to TP | {improvements['FN_to_TP']} |",
        f"| Abstention to correct | {improvements['abstention_to_correct']} |",
        f"| Other | {improvements['other']} |",
        "",
        "The improvements are dominated by false-positive corrections: "
        f"{improvements['FP_to_TN']} of {transitions['wrong_to_correct']} "
        "wrong-to-correct transitions changed from echinococcosis to negative "
        "for true-negative samples.",
        "",
        "## 4. Correct to wrong",
        "",
        "| Composition | Count |",
        "|---|---:|",
        f"| TP to FN | {harms['TP_to_FN']} |",
        f"| TN to FP | {harms['TN_to_FP']} |",
        f"| Correct to abstention | {harms['correct_to_abstention']} |",
        f"| Other | {harms['other']} |",
        "",
        "The largest harmful direction is TP to FN, showing that the shift toward "
        "negative predictions also removed some correct positive decisions.",
        "",
        "## 5. Negative-source subtypes",
        "",
        "| Source class | N | Baseline FP | v3 FP | FP to TN | TN to FP | Baseline FP rate | v3 FP rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subtype in SUBTYPE_ORDER:
        values = subtype_summary[subtype]
        lines.append(
            f"| {subtype} | {values['N']} | {values['baseline_FP']} | "
            f"{values['v3_FP']} | {values['FP_to_TN']} | {values['TN_to_FP']} | "
            f"{percent(values['baseline_FP'], values['N'])} | "
            f"{percent(values['v3_FP'], values['N'])} |"
        )
    lines.extend(
        [
            "",
            "FP-to-TN corrections occur in all three source classes, so the benefit is "
            "not confined to one subtype. The net reduction is strongest for Normal and "
            "Benign. Malignant also has five FP-to-TN corrections, but three new TN-to-FP "
            "errors leave a smaller net improvement.",
            "",
            "## 6. Findings wording",
            "",
            "Counts below are the number of samples whose findings contain each phrase "
            "pattern. They describe wording only.",
            "",
            "| Wording pattern | Helpful baseline | Helpful v3 | Harmful baseline | Harmful v3 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for phrase in TEXT_PATTERNS:
        lines.append(
            f"| {phrase} | {helpful_phrases[phrase][0]} | "
            f"{helpful_phrases[phrase][1]} | {harmful_phrases[phrase][0]} | "
            f"{harmful_phrases[phrase][1]} |"
        )
    lines.extend(
        [
            "",
            "- In helpful transitions, suspicion wording falls from "
            f"{helpful_phrases['suspicion wording'][0]} to "
            f"{helpful_phrases['suspicion wording'][1]}, while absence-of-support wording "
            f"rises from {helpful_phrases['absence-of-support wording'][0]} to "
            f"{helpful_phrases['absence-of-support wording'][1]}.",
            "- The same suppression of suspicion wording also appears in harmful "
            f"transitions ({harmful_phrases['suspicion wording'][0]} to "
            f"{harmful_phrases['suspicion wording'][1]}). This behavior is not selective: "
            "it helps many negatives but also contributes to some positive-to-negative errors.",
            "- Literal `negative` and `insufficient evidence` phrases are uncommon. The "
            "more frequent behavioral signal is wording such as no definite, convincing, "
            "specific, or characteristic support.",
            "- Systematic observation vocabulary is already common in both baseline and v3 "
            "findings, so it does not clearly separate helpful from harmful transitions.",
            "",
            "## 7. Self-reported confidence",
            "",
            "| Transition | Paired N | Baseline mean | v3 mean | Mean delta | Median delta |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Wrong to correct | {helpful_confidence[0]} | "
            f"{helpful_confidence[1]:.4f} | {helpful_confidence[2]:.4f} | "
            f"{helpful_confidence[3]:+.4f} | {helpful_confidence[4]:+.4f} |",
            f"| Correct to wrong | {harmful_confidence[0]} | "
            f"{harmful_confidence[1]:.4f} | {harmful_confidence[2]:.4f} | "
            f"{harmful_confidence[3]:+.4f} | {harmful_confidence[4]:+.4f} |",
            "",
            "These values are self-reported confidence, not calibrated probabilities.",
            "",
            "## 8. Conclusion",
            "",
            "The paired dev results support the proposed behavioral interpretation: Skill "
            "v3 mainly acts as a conservative decision modifier for "
            "DeepSeek-V4-Flash-Vision-Exp. It substantially reduces false positives "
            f"({improvements['FP_to_TN']} corrected versus {harms['TN_to_FP']} introduced) "
            "while sacrificing some sensitivity "
            f"({harms['TP_to_FN']} TP-to-FN versus {improvements['FN_to_TP']} FN-to-TP). "
            "This describes prediction behavior and should not be interpreted as probability "
            "calibration or as a new medical rule.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, columns: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_experiment_metadata(
    baseline: dict[str, dict[str, object]],
    v3: dict[str, dict[str, object]],
) -> None:
    baseline_models = {str(row["model"]) for row in baseline.values()}
    v3_models = {str(row["model"]) for row in v3.values()}
    if baseline_models != {"deepseek/DeepSeek-V4-Flash-Vision-Exp"}:
        raise RuntimeError(f"Unexpected baseline model values: {baseline_models}")
    if v3_models != baseline_models:
        raise RuntimeError(f"Model mismatch between baseline and v3: {v3_models}")
    if {str(row["prompt_version"]) for row in baseline.values()} != {"baseline-v0.2"}:
        raise RuntimeError("Unexpected baseline prompt_version")
    if {str(row["prompt_version"]) for row in v3.values()} != {"baseline-v0.2"}:
        raise RuntimeError("Unexpected v3 prompt_version")
    if {str(row["skill_enabled"]) for row in baseline.values()} != {"False"}:
        raise RuntimeError("Baseline result does not consistently disable Skill")
    if {
        (str(row["skill_enabled"]), str(row["skill_version"])) for row in v3.values()
    } != {("True", "v3")}:
        raise RuntimeError("Candidate result is not consistently Skill v3")


def analyze(
    baseline_path: Path,
    v3_path: Path,
    dev_path: Path,
    metadata_path: Path,
    output_dir: Path,
) -> list[dict[str, object]]:
    dev_labels = load_dev_labels(dev_path)
    baseline = load_results(baseline_path, "DeepSeek baseline results", dev_labels)
    v3 = load_results(v3_path, "DeepSeek Skill v3 results", dev_labels)
    metadata = load_metadata(metadata_path, dev_labels)
    validate_experiment_metadata(baseline, v3)

    rows = build_transition_rows(dev_labels, metadata, baseline, v3)
    actual_transitions = Counter(str(row["transition_type"]) for row in rows)
    if actual_transitions != Counter(EXPECTED_TRANSITIONS):
        raise RuntimeError(
            f"Transition counts differ from the paired reference: {dict(actual_transitions)}"
        )
    if sum(row["baseline_prediction"] != row["v3_prediction"] for row in rows) != 23:
        raise RuntimeError("Expected exactly 23 changed predictions")

    summary_rows, subtype_summary = make_summary_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "baseline_vs_v3_transitions.csv", TRANSITION_COLUMNS, rows)
    write_csv(
        output_dir / "baseline_vs_v3_transition_summary.csv",
        SUMMARY_COLUMNS,
        summary_rows,
    )
    report = build_report(rows, subtype_summary)
    (output_dir / "baseline_vs_v3_transition_analysis.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze paired DeepSeek baseline-to-Skill-v3 dev transitions."
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_V3)
    parser.add_argument("--labels", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyze(
        args.baseline,
        args.candidate,
        args.labels,
        args.metadata,
        args.output_dir,
    )
    counts = Counter(str(row["transition_type"]) for row in rows)
    for transition in TRANSITION_ORDER:
        print(f"{transition:<18}: {counts[transition]}")
    print(
        "prediction_changed : "
        + str(sum(row["baseline_prediction"] != row["v3_prediction"] for row in rows))
    )


if __name__ == "__main__":
    main()

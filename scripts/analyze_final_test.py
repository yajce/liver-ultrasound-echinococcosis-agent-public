#!/usr/bin/env python3
"""Offline analysis for the frozen six-condition held-out test matrix.

This script only reads completed result files, the fixed test split, benchmark
metadata, and the frozen development summary. It never invokes a model API or
modifies any prediction.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "test.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "benchmark_metadata.csv"
DEV_METRICS_PATH = PROJECT_ROOT / "experiments" / "v4_dev_eval_metrics.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "final_test_analysis"
METRICS_OUTPUT = OUTPUT_DIR / "final_test_metrics.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "final_test_summary.md"

POSITIVE = "echinococcosis"
NEGATIVE = "negative"
NOT_APPLICABLE = "not_applicable"
PARSE_FAILED = "parse_failed"
BINARY_STATES = {POSITIVE, NEGATIVE}
PREDICTION_STATES = BINARY_STATES | {NOT_APPLICABLE, PARSE_FAILED}
EXPECTED_TEST_N = 66
EXPECTED_POSITIVE_N = 33
EXPECTED_NEGATIVE_N = 33
SUBTYPE_ORDER = ("Normal", "Benign", "Malignant")


@dataclass(frozen=True)
class Experiment:
    key: str
    model: str
    condition: str
    results_path: Path
    allow_non_test_rows: bool = False

    @property
    def display_name(self) -> str:
        return f"{self.model} {self.condition}"


@dataclass(frozen=True)
class LoadedExperiment:
    experiment: Experiment
    predictions: dict[str, str]
    source_rows: int
    selected_rows: int
    missing_ids: tuple[str, ...]
    duplicate_ids: tuple[str, ...]
    extra_ids: tuple[str, ...]


@dataclass(frozen=True)
class PairedResult:
    model: str
    reference: str
    candidate: str
    wrong_to_correct: int
    correct_to_wrong: int
    unchanged_correct: int
    unchanged_wrong: int
    fp_to_tn: int
    tn_to_fp: int
    fn_to_tp: int
    tp_to_fn: int
    abstention_transitions: Counter[str]
    parse_failure_transitions: Counter[str]
    other_transitions: Counter[str]
    mcnemar_p: float

    @property
    def comparison(self) -> str:
        return f"{self.reference} vs {self.candidate}"


EXPERIMENTS = (
    Experiment(
        key="glm_baseline",
        model="GLM",
        condition="baseline",
        results_path=PROJECT_ROOT / "experiments" / "glm_baseline_v02" / "results.csv",
        allow_non_test_rows=True,
    ),
    Experiment(
        key="glm_v3",
        model="GLM",
        condition="Skill v3",
        results_path=PROJECT_ROOT / "experiments" / "glm_skill_v3_test" / "results.csv",
    ),
    Experiment(
        key="glm_v4",
        model="GLM",
        condition="Skill v4",
        results_path=PROJECT_ROOT / "experiments" / "glm_skill_v4_test" / "results.csv",
    ),
    Experiment(
        key="deepseek_baseline",
        model="DeepSeek",
        condition="baseline",
        results_path=PROJECT_ROOT / "experiments" / "deepseek_baseline_test" / "results.csv",
    ),
    Experiment(
        key="deepseek_v3",
        model="DeepSeek",
        condition="Skill v3",
        results_path=PROJECT_ROOT / "experiments" / "deepseek_skill_v3_test" / "results.csv",
    ),
    Experiment(
        key="deepseek_v4",
        model="DeepSeek",
        condition="Skill v4",
        results_path=PROJECT_ROOT / "experiments" / "deepseek_skill_v4_test" / "results.csv",
    ),
)

PAIR_DEFINITIONS = (
    ("GLM", "glm_baseline", "glm_v3"),
    ("GLM", "glm_baseline", "glm_v4"),
    ("GLM", "glm_v3", "glm_v4"),
    ("DeepSeek", "deepseek_baseline", "deepseek_v3"),
    ("DeepSeek", "deepseek_baseline", "deepseek_v4"),
    ("DeepSeek", "deepseek_v3", "deepseek_v4"),
)

REQUIRED_RESULT_COLUMNS = {
    "image_id",
    "true_label",
    "image_valid",
    "predicted_diagnosis",
    "parse_ok",
}

METRICS_COLUMNS = [
    "Model",
    "Condition",
    "N",
    "evaluated_binary_n",
    "TP",
    "FN",
    "FP",
    "TN",
    "not_applicable_n",
    "parse_failed_n",
    "binary_accuracy",
    "precision",
    "recall_sensitivity",
    "strict_sensitivity",
    "specificity",
    "f1",
    "strict_accuracy",
]


def read_csv_required(
    path: Path, required_columns: set[str], source_name: str
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{source_name} not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = sorted(required_columns - set(reader.fieldnames or []))
        if missing_columns:
            raise RuntimeError(
                f"{source_name} is missing columns: {', '.join(missing_columns)}"
            )
        return list(reader)


def index_unique(
    rows: list[dict[str, str]], source_name: str
) -> tuple[dict[str, dict[str, str]], tuple[str, ...]]:
    indexed: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        image_id = (row.get("image_id") or "").strip()
        if not image_id:
            raise RuntimeError(f"{source_name} row {row_number} has empty image_id")
        if image_id in indexed:
            duplicates.append(image_id)
            continue
        indexed[image_id] = row
    return indexed, tuple(sorted(set(duplicates)))


def parse_boolean(value: str | None, field_name: str, image_id: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized not in {"true", "false"}:
        raise RuntimeError(
            f"{field_name} must be True or False for {image_id}; got {value!r}"
        )
    return normalized == "true"


def load_test_labels() -> dict[str, str]:
    rows = read_csv_required(
        TEST_SPLIT_PATH, {"image_id", "true_label"}, "test.csv"
    )
    indexed, duplicates = index_unique(rows, "test.csv")
    if duplicates:
        raise RuntimeError(f"test.csv has duplicate image_id values: {duplicates}")
    labels = {
        image_id: (row.get("true_label") or "").strip()
        for image_id, row in indexed.items()
    }
    unknown_labels = sorted(set(labels.values()) - BINARY_STATES)
    if unknown_labels:
        raise RuntimeError(f"test.csv has unknown true labels: {unknown_labels}")
    label_counts = Counter(labels.values())
    if len(labels) != EXPECTED_TEST_N:
        raise RuntimeError(
            f"test.csv must contain {EXPECTED_TEST_N} samples; found {len(labels)}"
        )
    if label_counts != Counter(
        {POSITIVE: EXPECTED_POSITIVE_N, NEGATIVE: EXPECTED_NEGATIVE_N}
    ):
        raise RuntimeError(f"unexpected test label distribution: {dict(label_counts)}")
    return labels


def normalize_prediction(row: dict[str, str], image_id: str) -> str:
    parse_ok = parse_boolean(row.get("parse_ok"), "parse_ok", image_id)
    if not parse_ok:
        return PARSE_FAILED

    image_valid = parse_boolean(row.get("image_valid"), "image_valid", image_id)
    predicted = (row.get("predicted_diagnosis") or "").strip().lower()
    if not image_valid:
        return NOT_APPLICABLE
    if predicted not in BINARY_STATES | {NOT_APPLICABLE}:
        raise RuntimeError(
            f"unknown parsed prediction for {image_id}: {predicted!r}"
        )
    return predicted


def load_experiment(
    experiment: Experiment, labels: dict[str, str]
) -> LoadedExperiment:
    source_name = experiment.display_name
    rows = read_csv_required(
        experiment.results_path, REQUIRED_RESULT_COLUMNS, source_name
    )
    indexed, duplicates = index_unique(rows, source_name)
    test_ids = set(labels)
    source_ids = set(indexed)
    missing_ids = tuple(sorted(test_ids - source_ids))
    extra_ids = tuple(sorted(source_ids - test_ids))

    if duplicates:
        raise RuntimeError(f"{source_name} has duplicate image_id values: {duplicates}")
    if missing_ids:
        raise RuntimeError(f"{source_name} is missing test image_id values: {missing_ids}")
    if extra_ids and not experiment.allow_non_test_rows:
        raise RuntimeError(f"{source_name} has extra image_id values: {extra_ids}")
    if experiment.allow_non_test_rows and len(rows) != 166:
        raise RuntimeError(
            f"{source_name} is expected to be the complete 166-row result; "
            f"found {len(rows)}"
        )

    predictions: dict[str, str] = {}
    for image_id, expected_label in labels.items():
        row = indexed[image_id]
        recorded_label = (row.get("true_label") or "").strip()
        if recorded_label != expected_label:
            raise RuntimeError(
                f"label mismatch for {source_name} {image_id}: "
                f"{recorded_label!r} != {expected_label!r}"
            )
        predictions[image_id] = normalize_prediction(row, image_id)

    return LoadedExperiment(
        experiment=experiment,
        predictions=predictions,
        source_rows=len(rows),
        selected_rows=len(predictions),
        missing_ids=missing_ids,
        duplicate_ids=duplicates,
        extra_ids=extra_ids,
    )


def safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute_metrics(
    experiment: Experiment,
    labels: dict[str, str],
    predictions: dict[str, str],
) -> dict[str, object]:
    tp = fn = fp = tn = not_applicable_n = parse_failed_n = 0
    for image_id, true_label in labels.items():
        predicted = predictions[image_id]
        if predicted == NOT_APPLICABLE:
            not_applicable_n += 1
        elif predicted == PARSE_FAILED:
            parse_failed_n += 1
        elif true_label == POSITIVE and predicted == POSITIVE:
            tp += 1
        elif true_label == POSITIVE and predicted == NEGATIVE:
            fn += 1
        elif true_label == NEGATIVE and predicted == POSITIVE:
            fp += 1
        elif true_label == NEGATIVE and predicted == NEGATIVE:
            tn += 1
        else:
            raise RuntimeError(
                f"invalid label/prediction state for {image_id}: "
                f"{true_label!r}/{predicted!r}"
            )

    evaluated_binary_n = tp + fn + fp + tn
    if evaluated_binary_n + not_applicable_n + parse_failed_n != len(labels):
        raise AssertionError("prediction-state totals do not reconcile")

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    binary_accuracy = safe_div(tp + tn, evaluated_binary_n)
    strict_accuracy = safe_div(tp + tn, len(labels))
    strict_sensitivity = safe_div(tp, EXPECTED_POSITIVE_N)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)

    return {
        "Model": experiment.model,
        "Condition": experiment.condition,
        "N": len(labels),
        "evaluated_binary_n": evaluated_binary_n,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "not_applicable_n": not_applicable_n,
        "parse_failed_n": parse_failed_n,
        "binary_accuracy": binary_accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "strict_sensitivity": strict_sensitivity,
        "specificity": specificity,
        "f1": f1,
        "strict_accuracy": strict_accuracy,
    }


def exact_mcnemar_p(correct_to_wrong: int, wrong_to_correct: int) -> float:
    discordant = correct_to_wrong + wrong_to_correct
    if discordant == 0:
        return 1.0
    smaller = min(correct_to_wrong, wrong_to_correct)
    lower_tail = sum(
        math.comb(discordant, k) for k in range(smaller + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def compute_paired(
    model: str,
    reference: LoadedExperiment,
    candidate: LoadedExperiment,
    labels: dict[str, str],
) -> PairedResult:
    paired_counts: Counter[str] = Counter()
    main_transitions: Counter[str] = Counter()
    abstention_transitions: Counter[str] = Counter()
    parse_failure_transitions: Counter[str] = Counter()
    other_transitions: Counter[str] = Counter()

    for image_id, true_label in labels.items():
        before = reference.predictions[image_id]
        after = candidate.predictions[image_id]
        before_correct = before == true_label
        after_correct = after == true_label

        if not before_correct and after_correct:
            paired_counts["wrong_to_correct"] += 1
        elif before_correct and not after_correct:
            paired_counts["correct_to_wrong"] += 1
        elif before_correct:
            paired_counts["unchanged_correct"] += 1
        else:
            paired_counts["unchanged_wrong"] += 1

        if before == after:
            continue

        direction = f"{before} -> {after}"
        if NOT_APPLICABLE in {before, after}:
            abstention_transitions[direction] += 1
        if PARSE_FAILED in {before, after}:
            parse_failure_transitions[direction] += 1

        if true_label == NEGATIVE and before == POSITIVE and after == NEGATIVE:
            main_transitions["FP_to_TN"] += 1
        elif true_label == NEGATIVE and before == NEGATIVE and after == POSITIVE:
            main_transitions["TN_to_FP"] += 1
        elif true_label == POSITIVE and before == NEGATIVE and after == POSITIVE:
            main_transitions["FN_to_TP"] += 1
        elif true_label == POSITIVE and before == POSITIVE and after == NEGATIVE:
            main_transitions["TP_to_FN"] += 1
        elif NOT_APPLICABLE not in {before, after} and PARSE_FAILED not in {
            before,
            after,
        }:
            other_transitions[direction] += 1

    if sum(paired_counts.values()) != len(labels):
        raise AssertionError("paired correctness counts do not reconcile")

    return PairedResult(
        model=model,
        reference=reference.experiment.condition,
        candidate=candidate.experiment.condition,
        wrong_to_correct=paired_counts["wrong_to_correct"],
        correct_to_wrong=paired_counts["correct_to_wrong"],
        unchanged_correct=paired_counts["unchanged_correct"],
        unchanged_wrong=paired_counts["unchanged_wrong"],
        fp_to_tn=main_transitions["FP_to_TN"],
        tn_to_fp=main_transitions["TN_to_FP"],
        fn_to_tp=main_transitions["FN_to_TP"],
        tp_to_fn=main_transitions["TP_to_FN"],
        abstention_transitions=abstention_transitions,
        parse_failure_transitions=parse_failure_transitions,
        other_transitions=other_transitions,
        mcnemar_p=exact_mcnemar_p(
            paired_counts["correct_to_wrong"],
            paired_counts["wrong_to_correct"],
        ),
    )


def load_negative_subtypes(labels: dict[str, str]) -> dict[str, str]:
    rows = read_csv_required(
        METADATA_PATH,
        {"image_id", "true_label", "source_class"},
        "benchmark_metadata.csv",
    )
    indexed, duplicates = index_unique(rows, "benchmark_metadata.csv")
    if duplicates:
        raise RuntimeError(f"metadata has duplicate image_id values: {duplicates}")

    negative_subtypes: dict[str, str] = {}
    for image_id, true_label in labels.items():
        if image_id not in indexed:
            raise RuntimeError(f"metadata is missing test image_id: {image_id}")
        metadata_label = (indexed[image_id].get("true_label") or "").strip()
        if metadata_label != true_label:
            raise RuntimeError(
                f"metadata label mismatch for {image_id}: "
                f"{metadata_label!r} != {true_label!r}"
            )
        if true_label != NEGATIVE:
            continue
        source_class = (indexed[image_id].get("source_class") or "").strip()
        if source_class not in SUBTYPE_ORDER:
            raise RuntimeError(
                f"unexpected negative source_class for {image_id}: {source_class!r}"
            )
        negative_subtypes[image_id] = source_class

    if len(negative_subtypes) != EXPECTED_NEGATIVE_N:
        raise RuntimeError(
            f"expected {EXPECTED_NEGATIVE_N} negative metadata rows; "
            f"found {len(negative_subtypes)}"
        )
    return negative_subtypes


def compute_subtype_rows(
    loaded: Iterable[LoadedExperiment], negative_subtypes: dict[str, str]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in loaded:
        for subtype in SUBTYPE_ORDER:
            subtype_ids = [
                image_id
                for image_id, source_class in negative_subtypes.items()
                if source_class == subtype
            ]
            states = [item.predictions[image_id] for image_id in subtype_ids]
            tn = sum(state == NEGATIVE for state in states)
            fp = sum(state == POSITIVE for state in states)
            not_applicable_n = sum(state == NOT_APPLICABLE for state in states)
            parse_failed_n = sum(state == PARSE_FAILED for state in states)
            n = len(subtype_ids)
            if tn + fp + not_applicable_n + parse_failed_n != n:
                raise AssertionError("negative subtype totals do not reconcile")
            rows.append(
                {
                    "Model": item.experiment.model,
                    "Condition": item.experiment.condition,
                    "Subtype": subtype,
                    "N": n,
                    "TN": tn,
                    "FP": fp,
                    "not_applicable_n": not_applicable_n,
                    "parse_failed_n": parse_failed_n,
                    "specificity": safe_div(tn, n),
                    "FP_rate": safe_div(fp, n),
                }
            )
    return rows


def load_dev_metrics() -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv_required(
        DEV_METRICS_PATH,
        {
            "model",
            "condition",
            "N",
            "TP",
            "FN",
            "FP",
            "TN",
            "strict_accuracy",
            "recall_sensitivity",
            "specificity",
            "f1",
        },
        "v4_dev_eval_metrics.csv",
    )
    return {
        ((row.get("model") or "").strip(), (row.get("condition") or "").strip()): row
        for row in rows
    }


def write_csv_atomic(
    destination: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
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


def format_rate(value: object, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.{digits}f}%"


def format_float(value: object, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def format_pp(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.2f} pp"


def counter_text(counter: Counter[str]) -> str:
    if not counter:
        return "0"
    return "; ".join(f"{key}: {counter[key]}" for key in sorted(counter))


def markdown_metrics_table(metrics_rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Model | Condition | TP | FN | FP | TN | NA | PF | Strict accuracy | Precision | Strict sensitivity | Specificity | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['Model']} | {row['Condition']} | {row['TP']} | "
            f"{row['FN']} | {row['FP']} | {row['TN']} | "
            f"{row['not_applicable_n']} | {row['parse_failed_n']} | "
            f"{format_rate(row['strict_accuracy'])} | "
            f"{format_rate(row['precision'])} | "
            f"{format_rate(row['strict_sensitivity'])} | "
            f"{format_rate(row['specificity'])} | {format_rate(row['f1'])} |"
        )
    return lines


def markdown_paired_table(paired: Iterable[PairedResult]) -> list[str]:
    lines = [
        "| Model | Comparison | Wrong to correct | Correct to wrong | Unchanged correct | Unchanged wrong | FP to TN | TN to FP | FN to TP | TP to FN | Abstention transitions | Parse-failure transitions |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in paired:
        lines.append(
            f"| {row.model} | {row.comparison} | {row.wrong_to_correct} | "
            f"{row.correct_to_wrong} | {row.unchanged_correct} | "
            f"{row.unchanged_wrong} | {row.fp_to_tn} | {row.tn_to_fp} | "
            f"{row.fn_to_tp} | {row.tp_to_fn} | "
            f"{counter_text(row.abstention_transitions)} | "
            f"{counter_text(row.parse_failure_transitions)} |"
        )
    return lines


def markdown_mcnemar_table(paired: Iterable[PairedResult]) -> list[str]:
    lines = [
        "| Model | Comparison | b: correct to wrong | c: wrong to correct | Exact two-sided p |",
        "|---|---|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row.model} | {row.comparison} | {row.correct_to_wrong} | "
            f"{row.wrong_to_correct} | {format_float(row.mcnemar_p)} |"
        )
    return lines


def markdown_subtype_table(subtype_rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Model | Condition | Subtype | N | TN | FP | NA | PF | Specificity (TN/N) | FP rate (FP/N) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in subtype_rows:
        lines.append(
            f"| {row['Model']} | {row['Condition']} | {row['Subtype']} | "
            f"{row['N']} | {row['TN']} | {row['FP']} | "
            f"{row['not_applicable_n']} | {row['parse_failed_n']} | "
            f"{format_rate(row['specificity'])} | "
            f"{format_rate(row['FP_rate'])} |"
        )
    return lines


def metric_lookup(
    metrics_rows: list[dict[str, object]], model: str, condition: str
) -> dict[str, object]:
    matches = [
        row
        for row in metrics_rows
        if row["Model"] == model and row["Condition"] == condition
    ]
    if len(matches) != 1:
        raise AssertionError(f"metric row not unique: {model}/{condition}")
    return matches[0]


def subtype_lookup(
    subtype_rows: list[dict[str, object]],
    model: str,
    condition: str,
    subtype: str,
) -> dict[str, object]:
    matches = [
        row
        for row in subtype_rows
        if row["Model"] == model
        and row["Condition"] == condition
        and row["Subtype"] == subtype
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"subtype row not unique: {model}/{condition}/{subtype}"
        )
    return matches[0]


def build_summary(
    labels: dict[str, str],
    loaded: list[LoadedExperiment],
    metrics_rows: list[dict[str, object]],
    paired_rows: list[PairedResult],
    subtype_rows: list[dict[str, object]],
    negative_subtypes: dict[str, str],
    dev_metrics: dict[tuple[str, str], dict[str, str]],
) -> str:
    lines: list[str] = [
        "# Final Held-out Test Analysis",
        "",
        "## 1. Scope",
        "",
        "This report evaluates the frozen six-condition matrix on the fixed 66-image held-out test split (33 echinococcosis, 33 negative). GLM baseline was filtered from its complete 166-row result by exact image_id membership in test.csv. No model inference, tuning, prediction editing, or test rerun was performed.",
        "",
        "## 2. Completeness checks",
        "",
        "| Model | Condition | Source rows | Selected test rows | Missing | Duplicates | Non-test source rows | Not applicable | Parse failed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metric_by_key = {
        (str(row["Model"]), str(row["Condition"])): row for row in metrics_rows
    }
    for item in loaded:
        exp = item.experiment
        metric = metric_by_key[(exp.model, exp.condition)]
        lines.append(
            f"| {exp.model} | {exp.condition} | {item.source_rows} | "
            f"{item.selected_rows} | {len(item.missing_ids)} | "
            f"{len(item.duplicate_ids)} | {len(item.extra_ids)} | "
            f"{metric['not_applicable_n']} | {metric['parse_failed_n']} |"
        )
    lines.extend(
        [
            "",
            "All six conditions contain exactly the same 66 test image_id values after filtering. The 100 non-test rows in GLM baseline are expected because that source file contains all 166 benchmark predictions; they were excluded before every calculation.",
            "",
            "## 3. Final test metrics",
            "",
            *markdown_metrics_table(metrics_rows),
            "",
            "Binary metrics use only echinococcosis and negative predictions. Strict accuracy uses all 66 samples. Strict sensitivity is TP/33, so positive abstentions or parse failures cannot reduce its denominator.",
            "",
            "## 4. Paired comparisons",
            "",
            *markdown_paired_table(paired_rows),
            "",
            "Correctness is defined as predicted_diagnosis == true_label. Not-applicable and parse-failed states are incorrect. Transition counts retain those states rather than converting them to a binary diagnosis.",
            "",
            "## 5. McNemar exact tests",
            "",
            *markdown_mcnemar_table(paired_rows),
            "",
            "The two-sided exact p-value uses only discordant correctness pairs: b is correct to wrong and c is wrong to correct. These p-values are supplementary statistical evidence; they do not measure clinical importance, and a non-significant value does not establish equivalence.",
            "",
            "## 6. Negative subtype results",
            "",
            f"The 33 test negatives comprise Normal={sum(v == 'Normal' for v in negative_subtypes.values())}, Benign={sum(v == 'Benign' for v in negative_subtypes.values())}, and Malignant={sum(v == 'Malignant' for v in negative_subtypes.values())}.",
            "",
            *markdown_subtype_table(subtype_rows),
            "",
            "Specificity and FP rate in this section use the full subtype N as the denominator. No medical interpretation of subtype appearance is introduced.",
            "",
            "## 7. DEV FINDINGS",
            "",
            "The frozen 99-image v4 development-evaluation subset previously showed:",
            "",
            "| Model | Condition | N | TP | FN | FP | TN | Strict accuracy | Sensitivity | Specificity | F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in ("GLM", "DeepSeek"):
        for condition in ("baseline", "v3", "v4"):
            row = dev_metrics[(model, condition)]
            lines.append(
                f"| {model} | {condition} | {row['N']} | {row['TP']} | "
                f"{row['FN']} | {row['FP']} | {row['TN']} | "
                f"{format_rate(row['strict_accuracy'])} | "
                f"{format_rate(row['recall_sensitivity'])} | "
                f"{format_rate(row['specificity'])} | {format_rate(row['f1'])} |"
            )

    glm_base = metric_lookup(metrics_rows, "GLM", "baseline")
    glm_v3 = metric_lookup(metrics_rows, "GLM", "Skill v3")
    glm_v4 = metric_lookup(metrics_rows, "GLM", "Skill v4")
    ds_base = metric_lookup(metrics_rows, "DeepSeek", "baseline")
    ds_v3 = metric_lookup(metrics_rows, "DeepSeek", "Skill v3")
    ds_v4 = metric_lookup(metrics_rows, "DeepSeek", "Skill v4")

    def delta(after: dict[str, object], before: dict[str, object], key: str) -> float:
        return float(after[key]) - float(before[key])

    lines.extend(
        [
            "",
            "On development data, GLM v3 matched baseline strict accuracy, while GLM v4 was lower. DeepSeek v3 and v4 both improved strict accuracy over DeepSeek baseline, mainly alongside lower false-positive counts, with reduced sensitivity.",
            "",
            "## 8. HELD-OUT TEST FINDINGS",
            "",
            f"1. Baselines: GLM baseline achieved {format_rate(glm_base['strict_accuracy'])} strict accuracy, {format_rate(glm_base['strict_sensitivity'])} strict sensitivity, and {format_rate(glm_base['specificity'])} specificity. DeepSeek baseline achieved {format_rate(ds_base['strict_accuracy'])}, {format_rate(ds_base['strict_sensitivity'])}, and {format_rate(ds_base['specificity'])}, respectively.",
            f"2. GLM v3: strict accuracy changed by {format_pp(delta(glm_v3, glm_base, 'strict_accuracy'))}, strict sensitivity by {format_pp(delta(glm_v3, glm_base, 'strict_sensitivity'))}, and specificity by {format_pp(delta(glm_v3, glm_base, 'specificity'))} relative to baseline.",
            f"3. GLM v4: strict accuracy changed by {format_pp(delta(glm_v4, glm_base, 'strict_accuracy'))}, strict sensitivity by {format_pp(delta(glm_v4, glm_base, 'strict_sensitivity'))}, and specificity by {format_pp(delta(glm_v4, glm_base, 'specificity'))} relative to baseline.",
            f"4. DeepSeek v3: strict accuracy changed by {format_pp(delta(ds_v3, ds_base, 'strict_accuracy'))}; FP changed from {ds_base['FP']} to {ds_v3['FP']}, and strict sensitivity changed by {format_pp(delta(ds_v3, ds_base, 'strict_sensitivity'))}. This {'replicates' if delta(ds_v3, ds_base, 'strict_accuracy') > 0 and int(ds_v3['FP']) < int(ds_base['FP']) else 'does not replicate'} the development direction of higher strict accuracy with fewer false positives.",
            f"5. DeepSeek v4: strict accuracy changed by {format_pp(delta(ds_v4, ds_base, 'strict_accuracy'))}; FP changed from {ds_base['FP']} to {ds_v4['FP']}, and strict sensitivity changed by {format_pp(delta(ds_v4, ds_base, 'strict_sensitivity'))}. This {'replicates' if delta(ds_v4, ds_base, 'strict_accuracy') > 0 and int(ds_v4['FP']) < int(ds_base['FP']) else 'does not replicate'} the development direction of higher strict accuracy with fewer false positives.",
        ]
    )

    glm_v4_vs_v3 = delta(glm_v4, glm_v3, "strict_accuracy")
    ds_v4_vs_v3 = delta(ds_v4, ds_v3, "strict_accuracy")
    same_direction = (
        glm_v4_vs_v3 == 0
        or ds_v4_vs_v3 == 0
        or (glm_v4_vs_v3 > 0) == (ds_v4_vs_v3 > 0)
    )
    lines.extend(
        [
            f"6. Worked-example consistency: v4 versus v3 changed strict accuracy by {format_pp(glm_v4_vs_v3)} for GLM and {format_pp(ds_v4_vs_v3)} for DeepSeek. The effect is {'directionally consistent across the two models' if same_direction else 'not directionally consistent across the two models'} on this test set.",
            "7. Model dependence: the absolute metrics and Skill-induced sensitivity/specificity trade-offs differ between GLM and DeepSeek. Skill effects should therefore be reported per model rather than pooled into a single model-independent claim.",
            "",
            "### Negative subtype observations",
            "",
        ]
    )
    for model in ("GLM", "DeepSeek"):
        parts: list[str] = []
        for subtype in SUBTYPE_ORDER:
            baseline = subtype_lookup(
                subtype_rows, model, "baseline", subtype
            )
            v3 = subtype_lookup(subtype_rows, model, "Skill v3", subtype)
            v4 = subtype_lookup(subtype_rows, model, "Skill v4", subtype)
            parts.append(
                f"{subtype} FP {baseline['FP']} -> {v3['FP']} -> {v4['FP']} "
                "(baseline, v3, v4)"
            )
        lines.append(f"- {model}: " + "; ".join(parts) + ".")
    lines.extend(
        [
            "",
            "These subtype results distinguish absolute FP counts from FP rates because subtype sample sizes differ. They describe model behavior only and do not support new medical claims.",
            "",
            "## 9. Analysis boundary",
            "",
            "This analysis used only frozen predictions and fixed labels. It did not call a model API, modify a Skill or prompt, rerun test inference, delete results, create v5, or tune any experimental component from test outcomes.",
        ]
    )
    return "\n".join(lines) + "\n"


def print_metrics_table(metrics_rows: list[dict[str, object]]) -> None:
    print("A. Six-group held-out test metrics")
    header = (
        f"{'Model':<9} {'Condition':<10} {'TP':>3} {'FN':>3} {'FP':>3} "
        f"{'TN':>3} {'NA':>3} {'PF':>3} {'StrictAcc':>10} "
        f"{'Precision':>10} {'StrictSens':>10} {'Specificity':>11} {'F1':>8}"
    )
    print(header)
    for row in metrics_rows:
        print(
            f"{str(row['Model']):<9} {str(row['Condition']):<10} "
            f"{int(row['TP']):>3} {int(row['FN']):>3} {int(row['FP']):>3} "
            f"{int(row['TN']):>3} {int(row['not_applicable_n']):>3} "
            f"{int(row['parse_failed_n']):>3} "
            f"{format_rate(row['strict_accuracy']):>10} "
            f"{format_rate(row['precision']):>10} "
            f"{format_rate(row['strict_sensitivity']):>10} "
            f"{format_rate(row['specificity']):>11} "
            f"{format_rate(row['f1']):>8}"
        )


def print_paired_section(title: str, rows: list[PairedResult]) -> None:
    print(f"\n{title}")
    print(
        f"{'Comparison':<25} {'W->C':>4} {'C->W':>4} {'UC':>4} {'UW':>4} "
        f"{'FP->TN':>6} {'TN->FP':>6} {'FN->TP':>6} {'TP->FN':>6} "
        f"{'Abst':>5} {'PF trans':>8}"
    )
    for row in rows:
        print(
            f"{row.comparison:<25} {row.wrong_to_correct:>4} "
            f"{row.correct_to_wrong:>4} {row.unchanged_correct:>4} "
            f"{row.unchanged_wrong:>4} {row.fp_to_tn:>6} {row.tn_to_fp:>6} "
            f"{row.fn_to_tp:>6} {row.tp_to_fn:>6} "
            f"{sum(row.abstention_transitions.values()):>5} "
            f"{sum(row.parse_failure_transitions.values()):>8}"
        )


def print_mcnemar(paired_rows: list[PairedResult]) -> None:
    print("\nD. McNemar exact two-sided p-values")
    print(f"{'Model':<9} {'Comparison':<25} {'b':>3} {'c':>3} {'p':>10}")
    for row in paired_rows:
        print(
            f"{row.model:<9} {row.comparison:<25} "
            f"{row.correct_to_wrong:>3} {row.wrong_to_correct:>3} "
            f"{row.mcnemar_p:>10.6f}"
        )


def print_subtypes(subtype_rows: list[dict[str, object]]) -> None:
    print("\nE. Negative subtype test breakdown")
    print(
        f"{'Model':<9} {'Condition':<10} {'Subtype':<10} {'N':>3} "
        f"{'TN':>3} {'FP':>3} {'Spec':>9} {'FP rate':>9}"
    )
    for row in subtype_rows:
        print(
            f"{str(row['Model']):<9} {str(row['Condition']):<10} "
            f"{str(row['Subtype']):<10} {int(row['N']):>3} "
            f"{int(row['TN']):>3} {int(row['FP']):>3} "
            f"{format_rate(row['specificity']):>9} "
            f"{format_rate(row['FP_rate']):>9}"
        )


def print_completeness(loaded: list[LoadedExperiment]) -> None:
    print("\nCompleteness checks")
    for item in loaded:
        states = Counter(item.predictions.values())
        print(
            f"- {item.experiment.display_name}: selected={item.selected_rows}, "
            f"missing={len(item.missing_ids)}, duplicates={len(item.duplicate_ids)}, "
            f"non_test_source_rows={len(item.extra_ids)}, "
            f"not_applicable={states[NOT_APPLICABLE]}, "
            f"parse_failed={states[PARSE_FAILED]}"
        )


def main() -> None:
    labels = load_test_labels()
    loaded = [load_experiment(experiment, labels) for experiment in EXPERIMENTS]
    loaded_by_key = {item.experiment.key: item for item in loaded}

    metrics_rows = [
        compute_metrics(item.experiment, labels, item.predictions) for item in loaded
    ]
    paired_rows = [
        compute_paired(
            model,
            loaded_by_key[reference_key],
            loaded_by_key[candidate_key],
            labels,
        )
        for model, reference_key, candidate_key in PAIR_DEFINITIONS
    ]
    negative_subtypes = load_negative_subtypes(labels)
    subtype_rows = compute_subtype_rows(loaded, negative_subtypes)
    dev_metrics = load_dev_metrics()

    write_csv_atomic(METRICS_OUTPUT, METRICS_COLUMNS, metrics_rows)
    report = build_summary(
        labels,
        loaded,
        metrics_rows,
        paired_rows,
        subtype_rows,
        negative_subtypes,
        dev_metrics,
    )
    write_text_atomic(SUMMARY_OUTPUT, report)

    print_metrics_table(metrics_rows)
    print_paired_section(
        "B. GLM paired comparisons",
        [row for row in paired_rows if row.model == "GLM"],
    )
    print_paired_section(
        "C. DeepSeek paired comparisons",
        [row for row in paired_rows if row.model == "DeepSeek"],
    )
    print_mcnemar(paired_rows)
    print_subtypes(subtype_rows)
    print_completeness(loaded)
    print(f"\nSaved metrics: {METRICS_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"Saved report : {SUMMARY_OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

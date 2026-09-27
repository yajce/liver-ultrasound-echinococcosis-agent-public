#!/usr/bin/env python3
"""Generate frozen final tables, figures, and reporting notes.

This is an offline reporting script. It reads completed predictions and fixed
splits only. It does not import or invoke the runner, a model backend, or an
API client.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR = EXPERIMENTS_DIR / "final_test_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"

DEV_SPLIT = DATA_DIR / "splits" / "dev.csv"
DEV_V4_SPLIT = DATA_DIR / "splits" / "dev_v4_eval.csv"
TEST_SPLIT = DATA_DIR / "splits" / "test.csv"
METADATA_PATH = DATA_DIR / "benchmark_metadata.csv"
FINAL_TEST_METRICS = OUTPUT_DIR / "final_test_metrics.csv"
DEV_V4_REFERENCE_METRICS = EXPERIMENTS_DIR / "v4_dev_eval_metrics.csv"

FINAL_MAIN_TABLE = OUTPUT_DIR / "final_test_main_table.csv"
DEV_ITERATION_TABLE = OUTPUT_DIR / "dev_skill_iteration_table.csv"
FINAL_NOTES = OUTPUT_DIR / "final_results_notes.md"
FIGURE_1 = FIGURES_DIR / "figure1_test_performance.png"
FIGURE_2 = FIGURES_DIR / "figure2_test_error_counts.png"
FIGURE_3 = FIGURES_DIR / "figure3_deepseek_negative_subtype_fp_rate.png"

POSITIVE = "echinococcosis"
NEGATIVE = "negative"
NOT_APPLICABLE = "not_applicable"
PARSE_FAILED = "parse_failed"
BINARY_STATES = {POSITIVE, NEGATIVE}

TEST_RESULT_FILES = {
    ("GLM", "baseline"): EXPERIMENTS_DIR / "glm_baseline_v02" / "results.csv",
    ("GLM", "Skill v3"): EXPERIMENTS_DIR / "glm_skill_v3_test" / "results.csv",
    ("GLM", "Skill v4"): EXPERIMENTS_DIR / "glm_skill_v4_test" / "results.csv",
    ("DeepSeek", "baseline"): EXPERIMENTS_DIR
    / "deepseek_baseline_test"
    / "results.csv",
    ("DeepSeek", "Skill v3"): EXPERIMENTS_DIR
    / "deepseek_skill_v3_test"
    / "results.csv",
    ("DeepSeek", "Skill v4"): EXPERIMENTS_DIR
    / "deepseek_skill_v4_test"
    / "results.csv",
}

TEST_ORDER = (
    ("GLM", "baseline"),
    ("GLM", "Skill v3"),
    ("GLM", "Skill v4"),
    ("DeepSeek", "baseline"),
    ("DeepSeek", "Skill v3"),
    ("DeepSeek", "Skill v4"),
)

DEV_CONDITIONS = (
    (
        "GLM",
        "baseline",
        EXPERIMENTS_DIR / "glm_baseline_v02" / "results.csv",
        DEV_SPLIT,
        "dev100",
        "Original dev set: 50 positive / 50 negative",
        True,
    ),
    (
        "GLM",
        "Skill v1",
        EXPERIMENTS_DIR / "glm_skill_v1_dev" / "results.csv",
        DEV_SPLIT,
        "dev100",
        "Original dev set: 50 positive / 50 negative",
        False,
    ),
    (
        "GLM",
        "Skill v2",
        EXPERIMENTS_DIR / "glm_skill_v2_dev" / "results.csv",
        DEV_SPLIT,
        "dev100",
        "Original dev set: 50 positive / 50 negative",
        False,
    ),
    (
        "GLM",
        "Skill v3",
        EXPERIMENTS_DIR / "glm_skill_v3_dev" / "results.csv",
        DEV_SPLIT,
        "dev100",
        "Original dev set: 50 positive / 50 negative",
        False,
    ),
    (
        "GLM",
        "Skill v4",
        EXPERIMENTS_DIR / "glm_skill_v4_dev" / "results.csv",
        DEV_V4_SPLIT,
        "dev_v4_eval99",
        "Fair v4 set: img_0020.jpg excluded; 49 positive / 50 negative",
        False,
    ),
    (
        "DeepSeek",
        "baseline",
        EXPERIMENTS_DIR / "deepseek_baseline_dev" / "results.csv",
        DEV_SPLIT,
        "dev100",
        "Original dev set: 50 positive / 50 negative",
        False,
    ),
    (
        "DeepSeek",
        "Skill v3",
        EXPERIMENTS_DIR / "deepseek_skill_v3_dev" / "results.csv",
        DEV_SPLIT,
        "dev100",
        "Original dev set: 50 positive / 50 negative",
        False,
    ),
    (
        "DeepSeek",
        "Skill v4",
        EXPERIMENTS_DIR / "deepseek_skill_v4_dev" / "results.csv",
        DEV_V4_SPLIT,
        "dev_v4_eval99",
        "Fair v4 set: img_0020.jpg excluded; 49 positive / 50 negative",
        False,
    ),
)

RESULT_REQUIRED_COLUMNS = {
    "image_id",
    "true_label",
    "image_valid",
    "predicted_diagnosis",
    "parse_ok",
}

FINAL_MAIN_COLUMNS = [
    "Model",
    "Condition",
    "TP",
    "FN",
    "FP",
    "TN",
    "Strict Accuracy",
    "Precision",
    "Strict Sensitivity",
    "Specificity",
    "F1",
]

DEV_TABLE_COLUMNS = [
    "Model",
    "Condition",
    "Evaluation Subset",
    "N",
    "Positive N",
    "Negative N",
    "evaluated_binary_n",
    "TP",
    "FN",
    "FP",
    "TN",
    "not_applicable_n",
    "parse_failed_n",
    "Strict Accuracy",
    "Precision",
    "Strict Sensitivity",
    "Specificity",
    "F1",
    "Scope Note",
]


def read_csv_required(
    path: Path, required_columns: set[str], source_name: str
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{source_name} not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = sorted(required_columns - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(f"{source_name} is missing columns: {missing}")
        return list(reader)


def index_unique(
    rows: list[dict[str, str]], source_name: str
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        image_id = (row.get("image_id") or "").strip()
        if not image_id:
            raise RuntimeError(f"{source_name} row {row_number} has empty image_id")
        if image_id in indexed:
            raise RuntimeError(f"{source_name} has duplicate image_id: {image_id}")
        indexed[image_id] = row
    return indexed


def load_labels(path: Path) -> dict[str, str]:
    rows = read_csv_required(path, {"image_id", "true_label"}, path.name)
    indexed = index_unique(rows, path.name)
    labels = {
        image_id: (row.get("true_label") or "").strip()
        for image_id, row in indexed.items()
    }
    unknown = sorted(set(labels.values()) - BINARY_STATES)
    if unknown:
        raise RuntimeError(f"{path.name} has unknown labels: {unknown}")
    return labels


def parse_bool(value: str | None, field_name: str, image_id: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized not in {"true", "false"}:
        raise RuntimeError(
            f"{field_name} must be True or False for {image_id}; got {value!r}"
        )
    return normalized == "true"


def normalize_prediction(row: dict[str, str], image_id: str) -> str:
    if not parse_bool(row.get("parse_ok"), "parse_ok", image_id):
        return PARSE_FAILED
    if not parse_bool(row.get("image_valid"), "image_valid", image_id):
        return NOT_APPLICABLE
    prediction = (row.get("predicted_diagnosis") or "").strip().lower()
    if prediction not in BINARY_STATES | {NOT_APPLICABLE}:
        raise RuntimeError(f"unknown prediction for {image_id}: {prediction!r}")
    return prediction


def load_predictions(
    path: Path,
    labels: dict[str, str],
    source_name: str,
    allow_extra_rows: bool,
) -> dict[str, str]:
    rows = read_csv_required(path, RESULT_REQUIRED_COLUMNS, source_name)
    indexed = index_unique(rows, source_name)
    expected_ids = set(labels)
    source_ids = set(indexed)
    missing = sorted(expected_ids - source_ids)
    extra = sorted(source_ids - expected_ids)
    if missing:
        raise RuntimeError(f"{source_name} is missing image_id values: {missing}")
    if extra and not allow_extra_rows:
        raise RuntimeError(f"{source_name} has unexpected extra image_id values: {extra}")

    predictions: dict[str, str] = {}
    for image_id, true_label in labels.items():
        row = indexed[image_id]
        recorded_label = (row.get("true_label") or "").strip()
        if recorded_label != true_label:
            raise RuntimeError(
                f"label mismatch in {source_name} for {image_id}: "
                f"{recorded_label!r} != {true_label!r}"
            )
        predictions[image_id] = normalize_prediction(row, image_id)
    return predictions


def safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute_metrics(
    labels: dict[str, str], predictions: dict[str, str]
) -> dict[str, int | float | None]:
    counts = Counter(labels.values())
    tp = fn = fp = tn = na = pf = 0
    for image_id, true_label in labels.items():
        predicted = predictions[image_id]
        if predicted == NOT_APPLICABLE:
            na += 1
        elif predicted == PARSE_FAILED:
            pf += 1
        elif true_label == POSITIVE and predicted == POSITIVE:
            tp += 1
        elif true_label == POSITIVE and predicted == NEGATIVE:
            fn += 1
        elif true_label == NEGATIVE and predicted == POSITIVE:
            fp += 1
        elif true_label == NEGATIVE and predicted == NEGATIVE:
            tn += 1
        else:
            raise AssertionError(f"unhandled state for {image_id}")

    n = len(labels)
    evaluated = tp + fn + fp + tn
    if evaluated + na + pf != n:
        raise AssertionError("metric states do not reconcile")
    return {
        "N": n,
        "Positive N": counts[POSITIVE],
        "Negative N": counts[NEGATIVE],
        "evaluated_binary_n": evaluated,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "not_applicable_n": na,
        "parse_failed_n": pf,
        "binary_accuracy": safe_div(tp + tn, evaluated),
        "precision": safe_div(tp, tp + fp),
        "recall_sensitivity": safe_div(tp, tp + fn),
        "strict_sensitivity": safe_div(tp, counts[POSITIVE]),
        "specificity": safe_div(tn, tn + fp),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "strict_accuracy": safe_div(tp + tn, n),
    }


def write_csv_atomic(
    destination: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def write_text_atomic(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)


def load_final_test_rows() -> list[dict[str, str]]:
    required = {
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
    }
    rows = read_csv_required(FINAL_TEST_METRICS, required, "final_test_metrics.csv")
    indexed = {
        ((row.get("Model") or "").strip(), (row.get("Condition") or "").strip()): row
        for row in rows
    }
    if set(indexed) != set(TEST_ORDER) or len(rows) != len(TEST_ORDER):
        raise RuntimeError("final_test_metrics.csv does not contain the fixed six rows")
    return [indexed[key] for key in TEST_ORDER]


def verify_final_test_metrics(
    source_rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    test_labels = load_labels(TEST_SPLIT)
    if Counter(test_labels.values()) != Counter({POSITIVE: 33, NEGATIVE: 33}):
        raise RuntimeError("test.csv is not the expected 33/33 split")
    source_by_key = {
        (row["Model"], row["Condition"]): row for row in source_rows
    }
    integer_fields = (
        "N",
        "evaluated_binary_n",
        "TP",
        "FN",
        "FP",
        "TN",
        "not_applicable_n",
        "parse_failed_n",
    )
    float_fields = (
        "binary_accuracy",
        "precision",
        "recall_sensitivity",
        "strict_sensitivity",
        "specificity",
        "f1",
        "strict_accuracy",
    )
    for key in TEST_ORDER:
        model, condition = key
        predictions = load_predictions(
            TEST_RESULT_FILES[key],
            test_labels,
            f"{model} {condition} test results",
            allow_extra_rows=(key == ("GLM", "baseline")),
        )
        computed = compute_metrics(test_labels, predictions)
        source = source_by_key[key]
        for field in integer_fields:
            if int(source[field]) != int(computed[field]):
                raise RuntimeError(
                    f"final_test_metrics mismatch for {model}/{condition}/{field}: "
                    f"{source[field]} != {computed[field]}"
                )
        for field in float_fields:
            if not math.isclose(
                float(source[field]), float(computed[field]), rel_tol=0, abs_tol=1e-12
            ):
                raise RuntimeError(
                    f"final_test_metrics mismatch for {model}/{condition}/{field}: "
                    f"{source[field]} != {computed[field]}"
                )
    return source_by_key


def build_final_main_table(
    source_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for source in source_rows:
        output_rows.append(
            {
                "Model": source["Model"],
                "Condition": source["Condition"],
                "TP": int(source["TP"]),
                "FN": int(source["FN"]),
                "FP": int(source["FP"]),
                "TN": int(source["TN"]),
                "Strict Accuracy": float(source["strict_accuracy"]),
                "Precision": float(source["precision"]),
                "Strict Sensitivity": float(source["strict_sensitivity"]),
                "Specificity": float(source["specificity"]),
                "F1": float(source["f1"]),
            }
        )
    return output_rows


def build_dev_iteration_table() -> list[dict[str, object]]:
    dev_labels = load_labels(DEV_SPLIT)
    dev_v4_labels = load_labels(DEV_V4_SPLIT)
    if len(dev_labels) != 100 or Counter(dev_labels.values()) != Counter(
        {POSITIVE: 50, NEGATIVE: 50}
    ):
        raise RuntimeError("dev.csv is not the expected 50/50 dev100 split")
    if len(dev_v4_labels) != 99 or Counter(dev_v4_labels.values()) != Counter(
        {POSITIVE: 49, NEGATIVE: 50}
    ):
        raise RuntimeError("dev_v4_eval.csv is not the expected 49/50 split")
    if set(dev_labels) - set(dev_v4_labels) != {"img_0020.jpg"}:
        raise RuntimeError("dev_v4_eval.csv does not exclude exactly img_0020.jpg")

    table: list[dict[str, object]] = []
    for (
        model,
        condition,
        results_path,
        split_path,
        subset_name,
        scope_note,
        allow_extra_rows,
    ) in DEV_CONDITIONS:
        labels = dev_labels if split_path == DEV_SPLIT else dev_v4_labels
        predictions = load_predictions(
            results_path,
            labels,
            f"{model} {condition} development results",
            allow_extra_rows=allow_extra_rows,
        )
        metrics = compute_metrics(labels, predictions)
        table.append(
            {
                "Model": model,
                "Condition": condition,
                "Evaluation Subset": subset_name,
                "N": metrics["N"],
                "Positive N": metrics["Positive N"],
                "Negative N": metrics["Negative N"],
                "evaluated_binary_n": metrics["evaluated_binary_n"],
                "TP": metrics["TP"],
                "FN": metrics["FN"],
                "FP": metrics["FP"],
                "TN": metrics["TN"],
                "not_applicable_n": metrics["not_applicable_n"],
                "parse_failed_n": metrics["parse_failed_n"],
                "Strict Accuracy": metrics["strict_accuracy"],
                "Precision": metrics["precision"],
                "Strict Sensitivity": metrics["strict_sensitivity"],
                "Specificity": metrics["specificity"],
                "F1": metrics["f1"],
                "Scope Note": scope_note,
            }
        )
    return table


def load_fair_dev_reference() -> dict[tuple[str, str], dict[str, str]]:
    required = {
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
    }
    rows = read_csv_required(
        DEV_V4_REFERENCE_METRICS, required, "v4_dev_eval_metrics.csv"
    )
    indexed = {
        ((row.get("model") or "").strip(), (row.get("condition") or "").strip()): row
        for row in rows
    }
    expected = {
        (model, condition)
        for model in ("GLM", "DeepSeek")
        for condition in ("baseline", "v3", "v4")
    }
    if set(indexed) != expected:
        raise RuntimeError("v4_dev_eval_metrics.csv does not contain six fair rows")
    return indexed


def load_deepseek_subtype_fp_rates() -> tuple[list[str], dict[str, list[float]], dict[str, int]]:
    test_labels = load_labels(TEST_SPLIT)
    metadata_rows = read_csv_required(
        METADATA_PATH,
        {"image_id", "true_label", "source_class"},
        "benchmark_metadata.csv",
    )
    metadata = index_unique(metadata_rows, "benchmark_metadata.csv")
    subtypes = ["Normal", "Benign", "Malignant"]
    subtype_ids: dict[str, list[str]] = {subtype: [] for subtype in subtypes}
    for image_id, true_label in test_labels.items():
        if image_id not in metadata:
            raise RuntimeError(f"metadata missing test image_id: {image_id}")
        if true_label != NEGATIVE:
            continue
        source_class = (metadata[image_id].get("source_class") or "").strip()
        if source_class not in subtype_ids:
            raise RuntimeError(f"unexpected negative source_class: {source_class!r}")
        subtype_ids[source_class].append(image_id)

    rates: dict[str, list[float]] = {}
    for condition in ("baseline", "Skill v3", "Skill v4"):
        predictions = load_predictions(
            TEST_RESULT_FILES[("DeepSeek", condition)],
            test_labels,
            f"DeepSeek {condition} test results",
            allow_extra_rows=False,
        )
        condition_rates: list[float] = []
        for subtype in subtypes:
            ids = subtype_ids[subtype]
            fp = sum(predictions[image_id] == POSITIVE for image_id in ids)
            condition_rates.append(fp / len(ids))
        rates[condition] = condition_rates
    subtype_n = {subtype: len(ids) for subtype, ids in subtype_ids.items()}
    return subtypes, rates, subtype_n


def _save_figure_atomic(figure: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    figure.savefig(
        temporary,
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    temporary.replace(destination)


def _label_bars(axis: object, bars: Iterable[object], percent: bool) -> None:
    for bar in bars:
        height = bar.get_height()
        label = f"{height * 100:.1f}%" if percent else f"{int(height)}"
        axis.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
        )


def generate_figures(
    final_rows: list[dict[str, object]],
    subtype_names: list[str],
    subtype_rates: dict[str, list[float]],
    subtype_n: dict[str, int],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, PercentFormatter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to generate the requested PNG figures"
        ) from exc

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
        }
    )

    labels = [
        "GLM\nBaseline",
        "GLM\nSkill v3",
        "GLM\nSkill v4",
        "DeepSeek\nBaseline",
        "DeepSeek\nSkill v3",
        "DeepSeek\nSkill v4",
    ]
    x = list(range(len(labels)))

    # Figure 1: grouped rates.
    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    metric_specs = (
        ("Strict Accuracy", "#4C78A8", ""),
        ("Strict Sensitivity", "#E39C45", "//"),
        ("Specificity", "#5B9A68", ".."),
    )
    width = 0.24
    for series_index, (metric, color, hatch) in enumerate(metric_specs):
        offset = (series_index - 1) * width
        positions = [value + offset for value in x]
        values = [float(row[metric]) for row in final_rows]
        bars = ax.bar(
            positions,
            values,
            width,
            label=metric,
            color=color,
            edgecolor="#FFFFFF",
            linewidth=0.7,
            hatch=hatch,
        )
        _label_bars(ax, bars, percent=True)
    ax.set_title("Held-out Test Performance (N = 66)", pad=12)
    ax.set_ylabel("Performance (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.14)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.grid(True, color="#D8D8D8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(2.5, color="#B0B0B0", linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    _save_figure_atomic(fig, FIGURE_1)
    plt.close(fig)

    # Figure 2: FP and FN counts.
    fig, ax = plt.subplots(figsize=(10.8, 5.8), constrained_layout=True)
    width = 0.34
    fp_bars = ax.bar(
        [value - width / 2 for value in x],
        [int(row["FP"]) for row in final_rows],
        width,
        label="False positives",
        color="#6F8FB7",
        edgecolor="#FFFFFF",
        linewidth=0.7,
    )
    fn_bars = ax.bar(
        [value + width / 2 for value in x],
        [int(row["FN"]) for row in final_rows],
        width,
        label="False negatives",
        color="#D99A5B",
        edgecolor="#FFFFFF",
        linewidth=0.7,
        hatch="//",
    )
    _label_bars(ax, fp_bars, percent=False)
    _label_bars(ax, fn_bars, percent=False)
    ax.set_title("Held-out Test Error Counts (N = 66)", pad=12)
    ax.set_ylabel("Number of errors")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(max(int(row["FP"]), int(row["FN"])) for row in final_rows) + 4)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.grid(True, color="#D8D8D8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(2.5, color="#B0B0B0", linewidth=0.8)
    ax.legend(loc="upper left", frameon=False)
    _save_figure_atomic(fig, FIGURE_2)
    plt.close(fig)

    # Figure 3: DeepSeek negative-subtype false-positive rates.
    fig, ax = plt.subplots(figsize=(8.8, 5.7), constrained_layout=True)
    subtype_x = list(range(len(subtype_names)))
    condition_specs = (
        ("baseline", "#6F8FB7", ""),
        ("Skill v3", "#70A17B", "//"),
        ("Skill v4", "#D99A5B", ".."),
    )
    width = 0.24
    for index, (condition, color, hatch) in enumerate(condition_specs):
        offset = (index - 1) * width
        bars = ax.bar(
            [value + offset for value in subtype_x],
            subtype_rates[condition],
            width,
            label=condition.replace("Skill ", "Skill ").title()
            if condition == "baseline"
            else condition,
            color=color,
            edgecolor="#FFFFFF",
            linewidth=0.7,
            hatch=hatch,
        )
        _label_bars(ax, bars, percent=True)
    tick_labels = [f"{name}\n(n = {subtype_n[name]})" for name in subtype_names]
    ax.set_title("DeepSeek False-Positive Rate by Negative Subtype", pad=12)
    ax.set_xlabel("Negative source class")
    ax.set_ylabel("False-positive rate (%)")
    ax.set_xticks(subtype_x, tick_labels)
    ax.set_ylim(0, 0.74)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.grid(True, color="#D8D8D8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=3)
    _save_figure_atomic(fig, FIGURE_3)
    plt.close(fig)


def format_percent(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def markdown_main_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Model | Condition | TP | FN | FP | TN | Strict accuracy | Precision | Strict sensitivity | Specificity | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Model']} | {row['Condition']} | {row['TP']} | {row['FN']} | "
            f"{row['FP']} | {row['TN']} | {format_percent(row['Strict Accuracy'])} | "
            f"{format_percent(row['Precision'])} | "
            f"{format_percent(row['Strict Sensitivity'])} | "
            f"{format_percent(row['Specificity'])} | {format_percent(row['F1'])} |"
        )
    return lines


def markdown_dev_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Model | Condition | Evaluation subset | N | TP | FN | FP | TN | NA | PF | Strict accuracy | Strict sensitivity | Specificity | F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Model']} | {row['Condition']} | {row['Evaluation Subset']} | "
            f"{row['N']} | {row['TP']} | {row['FN']} | {row['FP']} | {row['TN']} | "
            f"{row['not_applicable_n']} | {row['parse_failed_n']} | "
            f"{format_percent(row['Strict Accuracy'])} | "
            f"{format_percent(row['Strict Sensitivity'])} | "
            f"{format_percent(row['Specificity'])} | {format_percent(row['F1'])} |"
        )
    return lines


def exact_mcnemar_p(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    smaller = min(b, c)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (
        2**discordant
    )
    return min(1.0, 2 * lower_tail)


def paired_mcnemar(
    labels: dict[str, str], reference: dict[str, str], candidate: dict[str, str]
) -> tuple[int, int, float]:
    correct_to_wrong = 0
    wrong_to_correct = 0
    for image_id, true_label in labels.items():
        ref_correct = reference[image_id] == true_label
        cand_correct = candidate[image_id] == true_label
        if ref_correct and not cand_correct:
            correct_to_wrong += 1
        elif not ref_correct and cand_correct:
            wrong_to_correct += 1
    return (
        correct_to_wrong,
        wrong_to_correct,
        exact_mcnemar_p(correct_to_wrong, wrong_to_correct),
    )


def build_notes(
    final_rows: list[dict[str, object]],
    dev_rows: list[dict[str, object]],
    fair_dev: dict[tuple[str, str], dict[str, str]],
) -> str:
    final_index = {(row["Model"], row["Condition"]): row for row in final_rows}
    ds_base = final_index[("DeepSeek", "baseline")]
    ds_v3 = final_index[("DeepSeek", "Skill v3")]
    ds_v4 = final_index[("DeepSeek", "Skill v4")]
    glm_base = final_index[("GLM", "baseline")]
    glm_v3 = final_index[("GLM", "Skill v3")]
    glm_v4 = final_index[("GLM", "Skill v4")]

    test_labels = load_labels(TEST_SPLIT)
    ds_predictions = {
        condition: load_predictions(
            TEST_RESULT_FILES[("DeepSeek", condition)],
            test_labels,
            f"DeepSeek {condition} test results",
            allow_extra_rows=False,
        )
        for condition in ("baseline", "Skill v3", "Skill v4")
    }
    b_v3, c_v3, p_v3 = paired_mcnemar(
        test_labels, ds_predictions["baseline"], ds_predictions["Skill v3"]
    )
    b_v4, c_v4, p_v4 = paired_mcnemar(
        test_labels, ds_predictions["baseline"], ds_predictions["Skill v4"]
    )

    lines = [
        "# Final Results Notes",
        "",
        "## Reporting scope",
        "",
        "The held-out test set contains 66 images (33 echinococcosis and 33 negative). All test metrics below come from the frozen final_test_metrics.csv and were independently reconciled against the completed prediction files before table and figure generation.",
        "",
        "The development table uses two explicitly different evaluation subsets. Baseline, v1, v2, and v3 development results use the original dev100 split. Skill v4 uses dev_v4_eval99 because the source morphology description for img_0020.jpg informed the worked example; that image was excluded from v4 development evaluation.",
        "",
        "## Final held-out test main table",
        "",
        *markdown_main_table(final_rows),
        "",
        "![Held-out test performance](figures/figure1_test_performance.png)",
        "",
        "![Held-out test error counts](figures/figure2_test_error_counts.png)",
        "",
        "## Skill iteration development table",
        "",
        *markdown_dev_table(dev_rows),
        "",
        "The v4 rows above are marked dev_v4_eval99 and must not be compared as if they were evaluated on the same 100 images as the earlier rows. For the fair 99-image v4 comparison, the precomputed same-subset strict accuracies were:",
        "",
        "| Model | Baseline | Skill v3 | Skill v4 |",
        "|---|---:|---:|---:|",
        f"| GLM | {format_percent(fair_dev[('GLM', 'baseline')]['strict_accuracy'])} | {format_percent(fair_dev[('GLM', 'v3')]['strict_accuracy'])} | {format_percent(fair_dev[('GLM', 'v4')]['strict_accuracy'])} |",
        f"| DeepSeek | {format_percent(fair_dev[('DeepSeek', 'baseline')]['strict_accuracy'])} | {format_percent(fair_dev[('DeepSeek', 'v3')]['strict_accuracy'])} | {format_percent(fair_dev[('DeepSeek', 'v4')]['strict_accuracy'])} |",
        "",
        "## DEV FINDINGS",
        "",
        "- GLM v1 and v2 reduced development sensitivity and strict accuracy relative to baseline. GLM v3 returned to baseline-level strict accuracy on dev100.",
        "- On the fair 99-image subset, GLM v4 was below both baseline and v3 in strict accuracy.",
        "- DeepSeek v3 reduced FP from 34 to 24 on dev100. On the fair 99-image subset, DeepSeek baseline, v3, and v4 had FP counts of 34, 24, and 20, respectively.",
        "- The DeepSeek FP reduction on development data was accompanied by lower sensitivity, showing a sensitivity-specificity trade-off rather than a uniform improvement in every metric.",
        "",
        "## HELD-OUT TEST FINDINGS",
        "",
        f"1. GLM baseline was the best GLM condition on held-out test: strict accuracy {format_percent(glm_base['Strict Accuracy'])}, compared with {format_percent(glm_v3['Strict Accuracy'])} for v3 and {format_percent(glm_v4['Strict Accuracy'])} for v4.",
        f"2. GLM specificity remained {format_percent(glm_v3['Specificity'])} for v3 and {format_percent(glm_v4['Specificity'])} for v4, while strict sensitivity fell from {format_percent(glm_base['Strict Sensitivity'])} at baseline to {format_percent(glm_v3['Strict Sensitivity'])} and {format_percent(glm_v4['Strict Sensitivity'])}.",
        f"3. DeepSeek baseline showed a high false-positive count: {ds_base['FP']} of 33 negative test images were classified as echinococcosis.",
        f"4. DeepSeek v3 and v4 reduced test FP to {ds_v3['FP']} and {ds_v4['FP']}, consistent with the overall FP-reduction direction seen on development data.",
        f"5. DeepSeek v4 test strict accuracy increased from {format_percent(ds_base['Strict Accuracy'])} at baseline to {format_percent(ds_v4['Strict Accuracy'])}. Strict sensitivity decreased from {format_percent(ds_base['Strict Sensitivity'])} to {format_percent(ds_v4['Strict Sensitivity'])}.",
        "6. Skill effects were model-dependent: the same structured prompts reduced GLM sensitivity without improving its already perfect test specificity, but improved DeepSeek specificity and strict accuracy by reducing FP.",
        f"7. The worked example did not have a consistent cross-model effect. Relative to v3, v4 strict accuracy changed from {format_percent(glm_v3['Strict Accuracy'])} to {format_percent(glm_v4['Strict Accuracy'])} for GLM, but from {format_percent(ds_v3['Strict Accuracy'])} to {format_percent(ds_v4['Strict Accuracy'])} for DeepSeek.",
        "",
        "![DeepSeek false-positive rate by subtype](figures/figure3_deepseek_negative_subtype_fp_rate.png)",
        "",
        "## Statistical context",
        "",
        f"For DeepSeek baseline versus v3, exact McNemar b={b_v3}, c={c_v3}, p={p_v3:.6f}. For baseline versus v4, b={b_v4}, c={c_v4}, p={p_v4:.6f}. Neither p-value is below 0.05. The observed metric changes should not be described as statistically significant, and McNemar p-values do not establish clinical importance.",
        "",
        "## Data limitations",
        "",
        "- Positive and negative images come from different public data sources, creating a risk of dataset-source and domain bias.",
        "- A single static ultrasound image cannot represent a complete dynamic ultrasound examination.",
        "- This is an exploratory pilot benchmark, not a clinical performance validation study.",
        "- The held-out test contains 66 images, so estimates and subtype results have limited precision, especially for Normal (n=5) and Benign (n=9) negative subgroups.",
        "",
        "No new inference, test rerun, prediction editing, Skill modification, or parameter tuning was performed for this report.",
    ]
    return "\n".join(lines) + "\n"


def print_summary(
    final_rows: list[dict[str, object]], dev_rows: list[dict[str, object]]
) -> None:
    print("Final held-out test main table")
    print(
        f"{'Model':<9} {'Condition':<10} {'TP':>3} {'FN':>3} {'FP':>3} "
        f"{'TN':>3} {'StrictAcc':>10} {'Precision':>10} "
        f"{'StrictSens':>10} {'Specificity':>11} {'F1':>8}"
    )
    for row in final_rows:
        print(
            f"{str(row['Model']):<9} {str(row['Condition']):<10} "
            f"{int(row['TP']):>3} {int(row['FN']):>3} "
            f"{int(row['FP']):>3} {int(row['TN']):>3} "
            f"{format_percent(row['Strict Accuracy']):>10} "
            f"{format_percent(row['Precision']):>10} "
            f"{format_percent(row['Strict Sensitivity']):>10} "
            f"{format_percent(row['Specificity']):>11} "
            f"{format_percent(row['F1']):>8}"
        )
    print("\nDevelopment Skill iteration table")
    print(
        f"{'Model':<9} {'Condition':<10} {'Subset':<14} {'N':>3} "
        f"{'TP':>3} {'FN':>3} {'FP':>3} {'TN':>3} {'NA':>3} {'PF':>3} "
        f"{'StrictAcc':>10} {'StrictSens':>10} {'Specificity':>11}"
    )
    for row in dev_rows:
        print(
            f"{str(row['Model']):<9} {str(row['Condition']):<10} "
            f"{str(row['Evaluation Subset']):<14} {int(row['N']):>3} "
            f"{int(row['TP']):>3} {int(row['FN']):>3} "
            f"{int(row['FP']):>3} {int(row['TN']):>3} "
            f"{int(row['not_applicable_n']):>3} "
            f"{int(row['parse_failed_n']):>3} "
            f"{format_percent(row['Strict Accuracy']):>10} "
            f"{format_percent(row['Strict Sensitivity']):>10} "
            f"{format_percent(row['Specificity']):>11}"
        )


def main() -> None:
    source_test_rows = load_final_test_rows()
    verify_final_test_metrics(source_test_rows)
    final_rows = build_final_main_table(source_test_rows)
    dev_rows = build_dev_iteration_table()
    fair_dev = load_fair_dev_reference()
    subtype_names, subtype_rates, subtype_n = load_deepseek_subtype_fp_rates()

    write_csv_atomic(FINAL_MAIN_TABLE, FINAL_MAIN_COLUMNS, final_rows)
    write_csv_atomic(DEV_ITERATION_TABLE, DEV_TABLE_COLUMNS, dev_rows)
    generate_figures(final_rows, subtype_names, subtype_rates, subtype_n)
    write_text_atomic(FINAL_NOTES, build_notes(final_rows, dev_rows, fair_dev))

    print_summary(final_rows, dev_rows)
    print("\nVerified: generated test values exactly match final_test_metrics.csv")
    print(f"Saved: {FINAL_MAIN_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {DEV_ITERATION_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {FIGURE_1.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {FIGURE_2.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {FIGURE_3.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {FINAL_NOTES.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

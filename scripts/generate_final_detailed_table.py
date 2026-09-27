"""Generate the report-ready detailed held-out test performance table.

This script only reads the frozen final test metrics and derives presentation
columns from their confusion-matrix counts. It does not invoke any model,
runner, evaluator, or API.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


SOURCE = Path("experiments/final_test_analysis/final_test_metrics.csv")
OUTPUT_CSV = Path(
    "experiments/final_test_analysis/final_test_detailed_metrics.csv"
)
OUTPUT_MD = Path(
    "experiments/final_test_analysis/final_test_detailed_metrics.md"
)

EXPECTED_ORDER = [
    ("GLM", "baseline"),
    ("GLM", "Skill v3"),
    ("GLM", "Skill v4"),
    ("DeepSeek", "baseline"),
    ("DeepSeek", "Skill v3"),
    ("DeepSeek", "Skill v4"),
]

CSV_FIELDS = [
    "Model",
    "Condition",
    "N",
    "TP",
    "FN",
    "FP",
    "TN",
    "NA",
    "PF",
    "Correct",
    "Wrong",
    "Predicted Positive",
    "Predicted Negative",
    "Overall Accuracy (%)",
    "Strict Accuracy (%)",
    "Precision (%)",
    "Sensitivity / Recall (%)",
    "Specificity (%)",
    "FNR (%)",
    "FPR (%)",
    "F1 (%)",
]

MD_FIELDS = [
    "Model",
    "Condition",
    "N",
    "阳判阳 TP",
    "阳判阴 FN",
    "阴判阳 FP",
    "阴判阴 TN",
    "NA",
    "PF",
    "Correct",
    "Wrong",
    "Pred +",
    "Pred -",
    "Overall Accuracy",
    "Strict Accuracy",
    "Precision",
    "Sensitivity / Recall",
    "Specificity",
    "FNR",
    "FPR",
    "F1",
]


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        raise ValueError("Metric denominator is zero.")
    return numerator / denominator


def assert_close(name: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Source metric mismatch for {name}: "
            f"source={actual!r}, derived={expected!r}"
        )


def percent(value: float) -> str:
    return f"{value * 100:.2f}"


def load_and_validate() -> list[dict[str, object]]:
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    observed_order = [(row["Model"], row["Condition"]) for row in source_rows]
    if observed_order != EXPECTED_ORDER:
        raise ValueError(
            "Unexpected experiment rows or order in final_test_metrics.csv: "
            f"{observed_order!r}"
        )

    detailed_rows: list[dict[str, object]] = []
    for row in source_rows:
        label = f"{row['Model']} {row['Condition']}"
        n = int(row["N"])
        evaluated_binary_n = int(row["evaluated_binary_n"])
        tp = int(row["TP"])
        fn = int(row["FN"])
        fp = int(row["FP"])
        tn = int(row["TN"])
        na = int(row["not_applicable_n"])
        pf = int(row["parse_failed_n"])

        if n != 66:
            raise ValueError(f"{label}: expected held-out test N=66, found {n}.")
        if tp + fn + fp + tn != evaluated_binary_n:
            raise ValueError(f"{label}: binary confusion counts do not sum correctly.")
        if evaluated_binary_n + na + pf != n:
            raise ValueError(f"{label}: binary, NA, and PF counts do not sum to N.")

        correct = tp + tn
        wrong = fn + fp
        predicted_positive = tp + fp
        predicted_negative = tn + fn

        overall_accuracy = safe_divide(correct, evaluated_binary_n)
        strict_accuracy = safe_divide(correct, n)
        precision = safe_divide(tp, predicted_positive)
        sensitivity = safe_divide(tp, tp + fn)
        specificity = safe_divide(tn, tn + fp)
        fnr = safe_divide(fn, tp + fn)
        fpr = safe_divide(fp, tn + fp)
        f1 = safe_divide(2 * tp, 2 * tp + fp + fn)

        assert_close(
            f"{label} binary_accuracy",
            float(row["binary_accuracy"]),
            overall_accuracy,
        )
        assert_close(
            f"{label} strict_accuracy",
            float(row["strict_accuracy"]),
            strict_accuracy,
        )
        assert_close(f"{label} precision", float(row["precision"]), precision)
        assert_close(
            f"{label} recall_sensitivity",
            float(row["recall_sensitivity"]),
            sensitivity,
        )
        assert_close(
            f"{label} strict_sensitivity",
            float(row["strict_sensitivity"]),
            sensitivity,
        )
        assert_close(
            f"{label} specificity", float(row["specificity"]), specificity
        )
        assert_close(f"{label} f1", float(row["f1"]), f1)
        assert_close(f"{label} FNR+Sensitivity", fnr + sensitivity, 1.0)
        assert_close(f"{label} FPR+Specificity", fpr + specificity, 1.0)

        detailed_rows.append(
            {
                "Model": row["Model"],
                "Condition": row["Condition"],
                "N": n,
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
                "NA": na,
                "PF": pf,
                "Correct": correct,
                "Wrong": wrong,
                "Predicted Positive": predicted_positive,
                "Predicted Negative": predicted_negative,
                "Overall Accuracy (%)": percent(overall_accuracy),
                "Strict Accuracy (%)": percent(strict_accuracy),
                "Precision (%)": percent(precision),
                "Sensitivity / Recall (%)": percent(sensitivity),
                "Specificity (%)": percent(specificity),
                "FNR (%)": percent(fnr),
                "FPR (%)": percent(fpr),
                "F1 (%)": percent(f1),
            }
        )

    return detailed_rows


def write_csv(rows: list[dict[str, object]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_CSV.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUTPUT_CSV)


def write_markdown(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Held-out Test Detailed Performance",
        "",
        "All conditions are evaluated on the same frozen held-out test set "
        "(N = 66). Percentages are shown to two decimal places.",
        "",
        "| " + " | ".join(MD_FIELDS) + " |",
        "| " + " | ".join(["---"] * len(MD_FIELDS)) + " |",
    ]

    percentage_fields = [
        "Overall Accuracy (%)",
        "Strict Accuracy (%)",
        "Precision (%)",
        "Sensitivity / Recall (%)",
        "Specificity (%)",
        "FNR (%)",
        "FPR (%)",
        "F1 (%)",
    ]
    for row in rows:
        cells = [
            str(row["Model"]),
            str(row["Condition"]),
            str(row["N"]),
            str(row["TP"]),
            str(row["FN"]),
            str(row["FP"]),
            str(row["TN"]),
            str(row["NA"]),
            str(row["PF"]),
            str(row["Correct"]),
            str(row["Wrong"]),
            str(row["Predicted Positive"]),
            str(row["Predicted Negative"]),
            *[f"{row[field]}%" for field in percentage_fields],
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "- TP（阳判阳）: true positive; FN（阳判阴）: false negative.",
            "- FP（阴判阳）: false positive; TN（阴判阴）: true negative.",
            "- NA: `not_applicable`; PF: `parse_failed`.",
            "- Correct = TP + TN; Wrong = FN + FP.",
            "- Pred + = TP + FP; Pred - = TN + FN.",
            "- Overall Accuracy uses the binary-evaluated denominator; "
            "Strict Accuracy uses all N samples.",
            "- FNR = FN / (TP + FN); FPR = FP / (TN + FP).",
            "- All six conditions have NA = 0 and PF = 0, so Overall "
            "Accuracy equals Strict Accuracy in this table.",
            "",
            "Source: `experiments/final_test_analysis/final_test_metrics.csv`.",
            "",
        ]
    )

    temporary = OUTPUT_MD.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(OUTPUT_MD)


def main() -> None:
    rows = load_and_validate()
    write_csv(rows)
    write_markdown(rows)
    print(f"Generated {OUTPUT_CSV} ({len(rows)} data rows).")
    print(f"Generated {OUTPUT_MD}.")
    print("All derived metrics reconcile with final_test_metrics.csv.")


if __name__ == "__main__":
    main()

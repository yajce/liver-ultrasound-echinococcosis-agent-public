#!/usr/bin/env python3
"""Create the fixed, stratified dev/test split for the 166-image benchmark."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
RANDOM_SEED = 42
POSITIVE_LABEL = "echinococcosis"
NEGATIVE_LABEL = "negative"
EXPECTED_PER_LABEL = 83
DEV_PER_LABEL = 50
TEST_PER_LABEL = 33


def read_labels(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Labels file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != ["image_id", "true_label"]:
            raise RuntimeError(
                f"Unexpected labels.csv columns: {reader.fieldnames}; "
                "expected image_id,true_label"
            )
        rows = list(reader)

    if len(rows) != EXPECTED_PER_LABEL * 2:
        raise RuntimeError(f"Expected 166 label rows, found {len(rows)}")

    image_ids = [(row.get("image_id") or "").strip() for row in rows]
    if any(not image_id for image_id in image_ids):
        raise RuntimeError("labels.csv contains an empty image_id")
    if len(set(image_ids)) != len(image_ids):
        raise RuntimeError("labels.csv contains duplicate image_id values")

    allowed_labels = {POSITIVE_LABEL, NEGATIVE_LABEL}
    normalized: list[dict[str, str]] = []
    for row, image_id in zip(rows, image_ids):
        true_label = (row.get("true_label") or "").strip()
        if true_label not in allowed_labels:
            raise RuntimeError(
                f"Unknown or missing true_label for {image_id}: {true_label!r}"
            )
        normalized.append({"image_id": image_id, "true_label": true_label})

    counts = Counter(row["true_label"] for row in normalized)
    expected_counts = {
        POSITIVE_LABEL: EXPECTED_PER_LABEL,
        NEGATIVE_LABEL: EXPECTED_PER_LABEL,
    }
    if counts != expected_counts:
        raise RuntimeError(
            f"Unexpected class counts: {dict(counts)}; expected {expected_counts}"
        )
    return normalized


def stratified_split(
    labels: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(RANDOM_SEED)
    dev: list[dict[str, str]] = []
    test: list[dict[str, str]] = []

    for true_label in (POSITIVE_LABEL, NEGATIVE_LABEL):
        group = sorted(
            (row.copy() for row in labels if row["true_label"] == true_label),
            key=lambda row: row["image_id"],
        )
        rng.shuffle(group)
        dev.extend(group[:DEV_PER_LABEL])
        test.extend(group[DEV_PER_LABEL:])

    rng.shuffle(dev)
    rng.shuffle(test)
    return dev, test


def validate_split(
    labels: list[dict[str, str]],
    dev: list[dict[str, str]],
    test: list[dict[str, str]],
) -> tuple[int, int]:
    dev_ids = {row["image_id"] for row in dev}
    test_ids = {row["image_id"] for row in test}
    source_ids = {row["image_id"] for row in labels}

    overlap = len(dev_ids & test_ids)
    missing = len(source_ids - (dev_ids | test_ids))
    unexpected = len((dev_ids | test_ids) - source_ids)
    if overlap or missing or unexpected:
        raise RuntimeError(
            f"Invalid split: overlap={overlap}, missing={missing}, "
            f"unexpected={unexpected}"
        )
    if len(dev_ids) != 100 or len(test_ids) != 66:
        raise RuntimeError(
            f"Invalid split sizes: dev={len(dev_ids)}, test={len(test_ids)}"
        )

    dev_counts = Counter(row["true_label"] for row in dev)
    test_counts = Counter(row["true_label"] for row in test)
    if dev_counts != {POSITIVE_LABEL: 50, NEGATIVE_LABEL: 50}:
        raise RuntimeError(f"Invalid dev class counts: {dict(dev_counts)}")
    if test_counts != {POSITIVE_LABEL: 33, NEGATIVE_LABEL: 33}:
        raise RuntimeError(f"Invalid test class counts: {dict(test_counts)}")
    return overlap, missing


def write_csv_atomic(
    destination: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the fixed stratified dev/test benchmark split."
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    labels = read_labels(args.labels.resolve())
    dev, test = stratified_split(labels)
    overlap, missing = validate_split(labels, dev, test)

    output_dir = args.output_dir.resolve()
    write_csv_atomic(output_dir / "dev.csv", ["image_id", "true_label"], dev)
    write_csv_atomic(output_dir / "test.csv", ["image_id", "true_label"], test)
    all_rows = [
        {**row, "split": split_name}
        for split_name, split_rows in (("dev", dev), ("test", test))
        for row in split_rows
    ]
    write_csv_atomic(
        output_dir / "all_split.csv",
        ["image_id", "true_label", "split"],
        all_rows,
    )

    print("dev: 100 (50 positive / 50 negative)")
    print("test: 66 (33 positive / 33 negative)")
    print(f"overlap: {overlap}")
    print(f"missing: {missing}")


if __name__ == "__main__":
    main()

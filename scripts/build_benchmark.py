#!/usr/bin/env python3
"""Build the reproducible, label-hidden 166-image benchmark dataset."""

from __future__ import annotations

import csv
import hashlib
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_IMAGES_DIR = DATA_DIR / "images"
LABELS_PATH = DATA_DIR / "labels.csv"
BENCHMARK_METADATA_PATH = DATA_DIR / "benchmark_metadata.csv"
RANDOM_SEED = 42
EXPECTED_PER_LABEL = 83
EXPECTED_TOTAL = EXPECTED_PER_LABEL * 2
ALLOWED_LABELS = {"echinococcosis", "negative"}
GENERATED_IMAGE_PATTERN = re.compile(r"img_\d{4}\.jpg", re.IGNORECASE)


@dataclass(frozen=True)
class BenchmarkSource:
    path: Path
    true_label: str
    source_dataset: str
    source_class: str
    source_filename: str
    sha256: str


DATASET_CONFIG = (
    (
        DATA_DIR / "raw" / "positive_echinococcosis" / "images",
        "echinococcosis",
        "positive_echinococcosis",
    ),
    (
        DATA_DIR / "raw" / "negative_liver" / "images",
        "negative",
        "negative_liver",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(metadata_path: Path) -> list[dict[str, str]]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_dataset(
    image_dir: Path, true_label: str, source_dataset: str
) -> list[BenchmarkSource]:
    if true_label not in ALLOWED_LABELS:
        raise ValueError(f"Unsupported label: {true_label}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Input image directory not found: {image_dir}")

    image_paths = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".jpg"
        ),
        key=lambda path: path.name,
    )
    if len(image_paths) != EXPECTED_PER_LABEL:
        raise RuntimeError(
            f"{source_dataset}: expected {EXPECTED_PER_LABEL} JPG files, "
            f"found {len(image_paths)}"
        )

    metadata_rows = read_metadata(image_dir / "metadata.csv")
    if len(metadata_rows) != EXPECTED_PER_LABEL:
        raise RuntimeError(
            f"{source_dataset}: expected {EXPECTED_PER_LABEL} metadata rows, "
            f"found {len(metadata_rows)}"
        )

    metadata_by_filename: dict[str, dict[str, str]] = {}
    for row in metadata_rows:
        filename = (row.get("filename") or "").strip()
        if not filename:
            raise RuntimeError(f"{source_dataset}: metadata row has no filename")
        if filename in metadata_by_filename:
            raise RuntimeError(
                f"{source_dataset}: duplicate metadata filename: {filename}"
            )
        metadata_by_filename[filename] = row

    image_names = {path.name for path in image_paths}
    if image_names != set(metadata_by_filename):
        missing_metadata = sorted(image_names - set(metadata_by_filename))
        missing_images = sorted(set(metadata_by_filename) - image_names)
        raise RuntimeError(
            f"{source_dataset}: images and metadata do not match; "
            f"missing metadata={missing_metadata}, missing images={missing_images}"
        )

    sources: list[BenchmarkSource] = []
    for image_path in image_paths:
        row = metadata_by_filename[image_path.name]
        digest = sha256_file(image_path)
        recorded_digest = (row.get("sha256") or "").strip().lower()
        if digest != recorded_digest:
            raise RuntimeError(
                f"{source_dataset}: SHA256 mismatch for {image_path.name}; "
                f"metadata={recorded_digest}, actual={digest}"
            )

        sources.append(
            BenchmarkSource(
                path=image_path,
                true_label=true_label,
                source_dataset=source_dataset,
                source_class=(row.get("source_class") or "").strip(),
                source_filename=(row.get("source_filename") or image_path.name).strip(),
                sha256=digest,
            )
        )

    return sources


def find_duplicate_groups(
    sources: list[BenchmarkSource], *, cross_dataset_only: bool = False
) -> dict[str, list[BenchmarkSource]]:
    grouped: defaultdict[str, list[BenchmarkSource]] = defaultdict(list)
    for source in sources:
        grouped[source.sha256].append(source)

    duplicates: dict[str, list[BenchmarkSource]] = {}
    for digest, group in grouped.items():
        if len(group) < 2:
            continue
        if cross_dataset_only and len({item.source_dataset for item in group}) < 2:
            continue
        duplicates[digest] = group
    return duplicates


def print_duplicate_report(
    label: str, duplicates: dict[str, list[BenchmarkSource]]
) -> None:
    if not duplicates:
        print(f"{label}: none")
        return

    duplicate_count = sum(len(group) - 1 for group in duplicates.values())
    print(
        f"WARNING: {label}: {len(duplicates)} group(s), "
        f"{duplicate_count} duplicate file(s) beyond the first copy"
    )
    for digest, group in sorted(duplicates.items()):
        origins = ", ".join(
            f"{item.source_dataset}/{item.source_filename}" for item in group
        )
        print(f"  {digest}: {origins}")


def shuffled_sources() -> list[BenchmarkSource]:
    sources: list[BenchmarkSource] = []
    for image_dir, true_label, source_dataset in DATASET_CONFIG:
        sources.extend(load_dataset(image_dir, true_label, source_dataset))

    if len(sources) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} total source images, found {len(sources)}"
        )

    labels = Counter(source.true_label for source in sources)
    expected_labels = {
        "echinococcosis": EXPECTED_PER_LABEL,
        "negative": EXPECTED_PER_LABEL,
    }
    if dict(labels) != expected_labels:
        raise RuntimeError(
            f"Unexpected source label distribution: {dict(labels)}; "
            f"expected {expected_labels}"
        )

    all_duplicates = find_duplicate_groups(sources)
    cross_duplicates = find_duplicate_groups(sources, cross_dataset_only=True)
    print_duplicate_report("All-source SHA256 duplicate groups", all_duplicates)
    print_duplicate_report(
        "Cross-positive/negative SHA256 duplicate groups", cross_duplicates
    )

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(sources)
    transitions = sum(
        left.true_label != right.true_label
        for left, right in zip(sources, sources[1:])
    )
    if transitions <= 1:
        raise RuntimeError("Shuffled output is still grouped by label.")
    return sources


def clean_previous_images() -> int:
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    existing_jpgs = [
        path
        for path in OUTPUT_IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    ]
    unmanaged_jpgs = [
        path
        for path in existing_jpgs
        if GENERATED_IMAGE_PATTERN.fullmatch(path.name) is None
    ]
    if unmanaged_jpgs:
        names = ", ".join(sorted(path.name for path in unmanaged_jpgs))
        raise RuntimeError(
            "Output directory contains JPG files not recognized as benchmark "
            f"outputs; refusing to delete or mix them: {names}"
        )

    for path in existing_jpgs:
        path.unlink()
    return len(existing_jpgs)


def write_csv_atomic(
    destination: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    temporary_path = destination.with_name(f".{destination.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(destination)


def write_benchmark(
    sources: list[BenchmarkSource],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    removed_count = clean_previous_images()
    print(f"Previous generated benchmark JPG files removed: {removed_count}")

    labels_rows: list[dict[str, str]] = []
    metadata_rows: list[dict[str, str]] = []
    for index, source in enumerate(sources, 1):
        image_id = f"img_{index:04d}.jpg"
        shutil.copy2(source.path, OUTPUT_IMAGES_DIR / image_id)
        labels_rows.append(
            {"image_id": image_id, "true_label": source.true_label}
        )
        metadata_rows.append(
            {
                "image_id": image_id,
                "true_label": source.true_label,
                "source_dataset": source.source_dataset,
                "source_class": source.source_class,
                "source_filename": source.source_filename,
                "sha256": source.sha256,
            }
        )

    write_csv_atomic(
        LABELS_PATH,
        ["image_id", "true_label"],
        labels_rows,
    )
    write_csv_atomic(
        BENCHMARK_METADATA_PATH,
        [
            "image_id",
            "true_label",
            "source_dataset",
            "source_class",
            "source_filename",
            "sha256",
        ],
        metadata_rows,
    )
    return labels_rows, metadata_rows


def validate_outputs(
    labels_rows: list[dict[str, str]], metadata_rows: list[dict[str, str]]
) -> None:
    expected_ids = [f"img_{index:04d}.jpg" for index in range(1, EXPECTED_TOTAL + 1)]
    output_images = sorted(
        (
            path
            for path in OUTPUT_IMAGES_DIR.iterdir()
            if path.is_file() and path.suffix.lower() == ".jpg"
        ),
        key=lambda path: path.name,
    )
    if [path.name for path in output_images] != expected_ids:
        raise RuntimeError("Output image filenames or count do not match expectations.")
    if [row["image_id"] for row in labels_rows] != expected_ids:
        raise RuntimeError("labels.csv image IDs do not match output filenames.")
    if [row["image_id"] for row in metadata_rows] != expected_ids:
        raise RuntimeError("benchmark_metadata.csv image IDs do not match outputs.")
    if len(labels_rows) != EXPECTED_TOTAL or len(metadata_rows) != EXPECTED_TOTAL:
        raise RuntimeError("Output CSV row count does not match the expected total.")

    label_counts = Counter(row["true_label"] for row in labels_rows)
    if set(label_counts) != ALLOWED_LABELS or any(
        count != EXPECTED_PER_LABEL for count in label_counts.values()
    ):
        raise RuntimeError(f"Invalid output label distribution: {dict(label_counts)}")

    for labels_row, metadata_row in zip(labels_rows, metadata_rows):
        if labels_row != {
            "image_id": metadata_row["image_id"],
            "true_label": metadata_row["true_label"],
        }:
            raise RuntimeError("labels.csv and benchmark_metadata.csv disagree.")
        output_path = OUTPUT_IMAGES_DIR / labels_row["image_id"]
        if sha256_file(output_path) != metadata_row["sha256"]:
            raise RuntimeError(f"Output SHA256 mismatch: {output_path}")

    with LABELS_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        written_label_reader = csv.DictReader(csv_file)
        if written_label_reader.fieldnames != ["image_id", "true_label"]:
            raise RuntimeError("labels.csv has unexpected columns.")
        if list(written_label_reader) != labels_rows:
            raise RuntimeError("Written labels.csv does not match generated labels.")

    with BENCHMARK_METADATA_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as csv_file:
        if list(csv.DictReader(csv_file)) != metadata_rows:
            raise RuntimeError(
                "Written benchmark_metadata.csv does not match generated metadata."
            )


def main() -> None:
    sources = shuffled_sources()
    labels_rows, metadata_rows = write_benchmark(sources)
    validate_outputs(labels_rows, metadata_rows)

    label_counts = Counter(row["true_label"] for row in labels_rows)
    transitions = sum(
        left["true_label"] != right["true_label"]
        for left, right in zip(labels_rows, labels_rows[1:])
    )
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Benchmark JPG files written: {len(metadata_rows)}")
    print(f"labels.csv data rows written: {len(labels_rows)}")
    print(f"benchmark_metadata.csv data rows written: {len(metadata_rows)}")
    print(f"Label counts: {dict(label_counts)}")
    print(f"Label transitions after shuffling: {transitions}")
    print(f"Images directory: {OUTPUT_IMAGES_DIR}")


if __name__ == "__main__":
    main()

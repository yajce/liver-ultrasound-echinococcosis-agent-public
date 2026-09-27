#!/usr/bin/env python3
"""Create a reproducible 83-image negative liver dataset."""

from __future__ import annotations

import csv
import hashlib
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data" / "raw" / "negative_liver" / "source"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "negative_liver" / "images"
RANDOM_SEED = 42
CLASS_CONFIG = (
    ("Benign", 200, 23),
    ("Malignant", 435, 49),
    ("Normal", 100, 11),
)
GENERATED_IMAGE_PATTERN = re.compile(r"negative_\d{3}\.jpg", re.IGNORECASE)


@dataclass(frozen=True)
class SourceImage:
    source_class: str
    path: Path
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_candidates() -> dict[str, list[Path]]:
    candidates_by_class: dict[str, list[Path]] = {}

    for class_name, expected_count, _ in CLASS_CONFIG:
        image_dir = SOURCE_ROOT / class_name / "image"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Source image directory not found: {image_dir}")

        candidates = sorted(
            (
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".jpg"
            ),
            key=lambda path: path.name,
        )
        if len(candidates) != expected_count:
            raise RuntimeError(
                f"{class_name}: expected {expected_count} JPG files, "
                f"found {len(candidates)} in {image_dir}"
            )
        candidates_by_class[class_name] = candidates

    return candidates_by_class


def build_source_images(
    candidates_by_class: dict[str, list[Path]],
) -> tuple[dict[Path, str], list[SourceImage]]:
    hashes_by_path: dict[Path, str] = {}
    all_images: list[SourceImage] = []

    for class_name, _, _ in CLASS_CONFIG:
        for path in candidates_by_class[class_name]:
            digest = sha256_file(path)
            hashes_by_path[path] = digest
            all_images.append(SourceImage(class_name, path, digest))

    return hashes_by_path, all_images


def duplicate_groups(images: list[SourceImage]) -> dict[str, list[SourceImage]]:
    grouped: defaultdict[str, list[SourceImage]] = defaultdict(list)
    for image in images:
        grouped[image.sha256].append(image)
    return {
        digest: group for digest, group in grouped.items() if len(group) > 1
    }


def print_duplicate_report(
    label: str, duplicates: dict[str, list[SourceImage]]
) -> None:
    if not duplicates:
        print(f"{label} SHA256 duplicates: none")
        return

    duplicate_file_count = sum(len(group) - 1 for group in duplicates.values())
    print(
        f"WARNING: {label} SHA256 duplicates: {len(duplicates)} group(s), "
        f"{duplicate_file_count} duplicate file(s) beyond the first copy"
    )
    for digest, group in sorted(duplicates.items()):
        sources = ", ".join(
            f"{image.source_class}/{image.path.name}" for image in group
        )
        print(f"  {digest}: {sources}")


def select_images(
    candidates_by_class: dict[str, list[Path]], hashes_by_path: dict[Path, str]
) -> list[SourceImage]:
    rng = random.Random(RANDOM_SEED)
    selected: list[SourceImage] = []

    for class_name, _, sample_count in CLASS_CONFIG:
        sampled_paths = rng.sample(candidates_by_class[class_name], sample_count)
        for path in sorted(sampled_paths, key=lambda item: item.name):
            selected.append(SourceImage(class_name, path, hashes_by_path[path]))

    return selected


def clean_previous_outputs() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_jpgs = [
        path
        for path in OUTPUT_DIR.iterdir()
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
            "Output directory contains JPG files not recognized as generated "
            f"outputs; refusing to delete or mix them: {names}"
        )

    for path in existing_jpgs:
        path.unlink()
    return len(existing_jpgs)


def write_dataset(selected: list[SourceImage]) -> list[dict[str, str]]:
    cleaned_count = clean_previous_outputs()
    print(f"Previous generated JPG files removed: {cleaned_count}")

    metadata_rows: list[dict[str, str]] = []
    for index, source in enumerate(selected, 1):
        filename = f"negative_{index:03d}.jpg"
        destination = OUTPUT_DIR / filename
        shutil.copy2(source.path, destination)
        metadata_rows.append(
            {
                "filename": filename,
                "source_class": source.source_class,
                "source_filename": source.path.name,
                "source_path": source.path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": source.sha256,
            }
        )

    metadata_path = OUTPUT_DIR / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "filename",
                "source_class",
                "source_filename",
                "source_path",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    return metadata_rows


def validate_outputs(metadata_rows: list[dict[str, str]]) -> None:
    expected_total = sum(sample_count for _, _, sample_count in CLASS_CONFIG)
    expected_names = [
        f"negative_{index:03d}.jpg" for index in range(1, expected_total + 1)
    ]
    output_jpgs = sorted(
        (
            path
            for path in OUTPUT_DIR.iterdir()
            if path.is_file() and path.suffix.lower() == ".jpg"
        ),
        key=lambda path: path.name,
    )

    if [path.name for path in output_jpgs] != expected_names:
        raise RuntimeError("Output JPG filenames or count do not match expectations.")
    if len(metadata_rows) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} metadata rows, found {len(metadata_rows)}."
        )

    for row in metadata_rows:
        output_path = OUTPUT_DIR / row["filename"]
        if sha256_file(output_path) != row["sha256"]:
            raise RuntimeError(f"SHA256 verification failed for {output_path}")

    with (OUTPUT_DIR / "metadata.csv").open(
        "r", encoding="utf-8", newline=""
    ) as csv_file:
        written_rows = list(csv.DictReader(csv_file))
    if written_rows != metadata_rows:
        raise RuntimeError("Written metadata.csv does not match the selected images.")


def main() -> None:
    candidates_by_class = discover_candidates()
    hashes_by_path, all_images = build_source_images(candidates_by_class)
    candidate_duplicates = duplicate_groups(all_images)

    selected = select_images(candidates_by_class, hashes_by_path)
    selected_duplicates = duplicate_groups(selected)

    print(f"Random seed: {RANDOM_SEED}")
    for class_name, _, sample_count in CLASS_CONFIG:
        print(
            f"{class_name}: found {len(candidates_by_class[class_name])}, "
            f"selected {sample_count}"
        )
    print_duplicate_report("All candidate images", candidate_duplicates)
    print_duplicate_report("Selected images", selected_duplicates)

    metadata_rows = write_dataset(selected)
    validate_outputs(metadata_rows)

    print(f"JPG files written: {len(metadata_rows)}")
    print(f"metadata.csv data rows written: {len(metadata_rows)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

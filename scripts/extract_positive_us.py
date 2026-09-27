#!/usr/bin/env python3
"""Extract unique positive ultrasound images embedded in Dataset2_US.pdf.

JPEG image streams are copied without re-encoding. Any non-JPEG embedded image
is converted directly from that image object to JPEG; PDF pages are never
rendered as screenshots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "positive_echinococcosis"
    / "source"
    / "Dataset2_US.pdf"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "raw" / "positive_echinococcosis" / "images"
)
EXCLUDED_FIGURE_LABELS = ("Figure 25", "Figure 85")
CAPTION_GROUP_TOLERANCE_POINTS = 5.0


@dataclass(frozen=True)
class ImageObject:
    pdf_page: int
    image_index_on_page: int
    original_extension: str
    jpeg_bytes: bytes
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract deduplicated liver ultrasound images from Dataset2_US.pdf."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help=f"Source PDF (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def to_jpeg(image_info: dict[str, Any]) -> bytes:
    """Return JPEG bytes while preserving embedded JPEG streams unchanged."""
    source_bytes = image_info["image"]
    extension = image_info["ext"].lower()
    if extension in {"jpg", "jpeg"}:
        return source_bytes

    pixmap = pymupdf.Pixmap(source_bytes)
    if pixmap.colorspace is None or pixmap.colorspace.n not in {1, 3}:
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
    if pixmap.alpha:
        pixmap = pymupdf.Pixmap(pixmap, 0)
    return pixmap.tobytes("jpeg", jpg_quality=95)


def locate_excluded_objects(
    document: pymupdf.Document,
) -> set[tuple[int, int]]:
    """Locate the image object(s) immediately above each excluded caption.

    Figure 25 is a two-panel CT figure, so every image aligned in the closest
    row above its caption is excluded. Figure 85 has one image in that row.
    Returned keys use one-based page and per-page image indices.
    """
    excluded: set[tuple[int, int]] = set()

    for figure_label in EXCLUDED_FIGURE_LABELS:
        caption_locations: list[tuple[int, pymupdf.Rect]] = []
        for page_number, page in enumerate(document, 1):
            caption_locations.extend(
                (page_number, rect) for rect in page.search_for(figure_label)
            )

        if len(caption_locations) != 1:
            raise RuntimeError(
                f"Expected one caption for {figure_label!r}, found "
                f"{len(caption_locations)}. Refusing to guess which images to exclude."
            )

        page_number, caption_rect = caption_locations[0]
        page = document[page_number - 1]
        preceding_images: list[tuple[int, pymupdf.Rect]] = []

        for image_index, image_tuple in enumerate(page.get_images(full=True), 1):
            xref = image_tuple[0]
            for image_rect in page.get_image_rects(xref):
                if image_rect.y1 <= caption_rect.y0 + CAPTION_GROUP_TOLERANCE_POINTS:
                    preceding_images.append((image_index, image_rect))

        if not preceding_images:
            raise RuntimeError(
                f"Could not find an embedded image above {figure_label!r} "
                f"on PDF page {page_number}."
            )

        closest_bottom = max(rect.y1 for _, rect in preceding_images)
        figure_image_indices = {
            image_index
            for image_index, rect in preceding_images
            if closest_bottom - rect.y1 <= CAPTION_GROUP_TOLERANCE_POINTS
        }
        if not figure_image_indices:
            raise RuntimeError(
                f"Could not associate embedded images with {figure_label!r}."
            )

        excluded.update((page_number, index) for index in figure_image_indices)

    return excluded


def collect_unique_images(
    document: pymupdf.Document,
    excluded: set[tuple[int, int]],
) -> tuple[int, int, list[ImageObject]]:
    total_objects = 0
    duplicate_count = 0
    seen_hashes: set[str] = set()
    unique_images: list[ImageObject] = []

    for page_number, page in enumerate(document, 1):
        for image_index, image_tuple in enumerate(page.get_images(full=True), 1):
            total_objects += 1
            if (page_number, image_index) in excluded:
                continue

            image_info = document.extract_image(image_tuple[0])
            jpeg_bytes = to_jpeg(image_info)
            digest = hashlib.sha256(jpeg_bytes).hexdigest()
            if digest in seen_hashes:
                duplicate_count += 1
                continue

            seen_hashes.add(digest)
            unique_images.append(
                ImageObject(
                    pdf_page=page_number,
                    image_index_on_page=image_index,
                    original_extension=image_info["ext"].lower(),
                    jpeg_bytes=jpeg_bytes,
                    sha256=digest,
                )
            )

    return total_objects, duplicate_count, unique_images


def write_outputs(output_dir: Path, images: list[ImageObject]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_filenames = {
        f"positive_{index:03d}.jpg" for index in range(1, len(images) + 1)
    }

    for index, image in enumerate(images, 1):
        filename = f"positive_{index:03d}.jpg"
        (output_dir / filename).write_bytes(image.jpeg_bytes)

    for old_image in output_dir.glob("positive_*.jpg"):
        if old_image.name not in expected_filenames:
            old_image.unlink()

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "filename",
                "pdf_page",
                "image_index_on_page",
                "original_extension",
                "sha256",
            ],
        )
        writer.writeheader()
        for index, image in enumerate(images, 1):
            writer.writerow(
                {
                    "filename": f"positive_{index:03d}.jpg",
                    "pdf_page": image.pdf_page,
                    "image_index_on_page": image.image_index_on_page,
                    "original_extension": image.original_extension,
                    "sha256": image.sha256,
                }
            )


def main() -> None:
    args = parse_args()
    pdf_path = args.pdf.resolve()
    output_dir = args.output_dir.resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"Source PDF not found: {pdf_path}")

    with pymupdf.open(pdf_path) as document:
        excluded = locate_excluded_objects(document)
        total_objects, duplicate_count, unique_images = collect_unique_images(
            document, excluded
        )

    write_outputs(output_dir, unique_images)

    print(f"Source PDF: {pdf_path}")
    print(f"Output directory: {output_dir}")
    print(f"Embedded image objects found: {total_objects}")
    print(f"Non-ultrasound objects excluded: {len(excluded)}")
    print(f"Duplicate images removed (SHA256): {duplicate_count}")
    print(f"Unique ultrasound images written: {len(unique_images)}")


if __name__ == "__main__":
    main()

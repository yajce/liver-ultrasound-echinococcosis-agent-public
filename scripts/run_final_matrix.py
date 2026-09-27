"""Run the fixed final smoke or held-out-test experiment matrix.

This module is deliberately limited to orchestration. Each condition is run
through the existing ``python -m src.runner`` entry point and its dedicated
run directory. It does not evaluate predictions or calculate metrics.
"""

from __future__ import annotations

import argparse
import csv
import locale
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = "data/images"
SMOKE_LABELS = "data/splits/dev_v4_eval.csv"
TEST_LABELS = "data/splits/test.csv"
TEST_EXPECTED_N = 66

DONE_RE = re.compile(r"^\s*done\s*:")
ERROR_RE = re.compile(
    r"^\s*error\s*:\s*.+?\s*->\s*"
    r"(?P<error_type>[A-Za-z_][A-Za-z0-9_.]*):\s*(?P<message>.*)$"
)


class MatrixError(RuntimeError):
    """Raised when a matrix condition cannot be completed safely."""


@dataclass(frozen=True)
class Condition:
    name: str
    backend: str
    skill_version: str | None
    smoke_run_dir: str
    test_run_dir: str
    delay: float

    @property
    def uses_skill(self) -> bool:
        return self.skill_version is not None

    def run_dir(self, mode: str) -> str:
        return self.smoke_run_dir if mode == "smoke" else self.test_run_dir


@dataclass(frozen=True)
class AttemptOutcome:
    returncode: int
    error_types: tuple[str, ...]
    retryable_connection_errors: tuple[bool, ...]

    @property
    def only_api_connection_errors(self) -> bool:
        return bool(self.retryable_connection_errors) and all(
            self.retryable_connection_errors
        )


@dataclass(frozen=True)
class ConditionSummary:
    name: str
    results_rows: int
    succeeded: int
    failed: int
    parse_ok: bool | None


CONDITIONS = (
    Condition(
        name="GLM + Skill v3",
        backend="glm",
        skill_version="v3",
        smoke_run_dir="experiments/final_smoke_glm_v3",
        test_run_dir="experiments/glm_skill_v3_test",
        delay=0.5,
    ),
    Condition(
        name="GLM + Skill v4",
        backend="glm",
        skill_version="v4",
        smoke_run_dir="experiments/final_smoke_glm_v4",
        test_run_dir="experiments/glm_skill_v4_test",
        delay=0.5,
    ),
    Condition(
        name="DeepSeek baseline",
        backend="deepseek",
        skill_version=None,
        smoke_run_dir="experiments/final_smoke_deepseek_baseline",
        test_run_dir="experiments/deepseek_baseline_test",
        delay=1.0,
    ),
    Condition(
        name="DeepSeek + Skill v3",
        backend="deepseek",
        skill_version="v3",
        smoke_run_dir="experiments/final_smoke_deepseek_v3",
        test_run_dir="experiments/deepseek_skill_v3_test",
        delay=1.0,
    ),
    Condition(
        name="DeepSeek + Skill v4",
        backend="deepseek",
        skill_version="v4",
        smoke_run_dir="experiments/final_smoke_deepseek_v4",
        test_run_dir="experiments/deepseek_skill_v4_test",
        delay=1.0,
    ),
)


def _read_image_ids(labels_path: Path) -> list[str]:
    if not labels_path.is_file():
        raise MatrixError(f"labels file not found: {labels_path}")

    image_ids: list[str] = []
    with labels_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if "image_id" not in (reader.fieldnames or []):
            raise MatrixError(f"labels file has no image_id column: {labels_path}")
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if image_id:
                image_ids.append(image_id)

    if len(image_ids) != len(set(image_ids)):
        raise MatrixError(f"duplicate image_id values in labels file: {labels_path}")
    return image_ids


def _read_results(results_path: Path) -> list[dict[str, str]]:
    if not results_path.is_file() or results_path.stat().st_size == 0:
        return []
    with results_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def _as_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes"}


def _validate_results(
    condition: Condition,
    rows: list[dict[str, str]],
    expected_ids: set[str],
    results_path: Path,
) -> None:
    image_ids = [(row.get("image_id") or "").strip() for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise MatrixError(f"{condition.name}: duplicate image_id in {results_path}")

    unexpected = sorted(set(image_ids) - expected_ids)
    if unexpected:
        raise MatrixError(
            f"{condition.name}: run directory contains unexpected image_id values: "
            f"{', '.join(unexpected)}"
        )

    for row in rows:
        skill_enabled = _as_bool(row.get("skill_enabled"))
        skill_version = (row.get("skill_version") or "").strip()
        skill_hash = (row.get("skill_hash") or "").strip()
        if condition.uses_skill:
            if not skill_enabled or skill_version != condition.skill_version:
                raise MatrixError(
                    f"{condition.name}: existing results have incompatible Skill metadata"
                )
            if not skill_hash:
                raise MatrixError(
                    f"{condition.name}: existing Skill result has an empty skill_hash"
                )
        elif skill_enabled or skill_version:
            raise MatrixError(
                f"{condition.name}: baseline run directory contains Skill-enabled results"
            )


def _redact_error_message(message: str) -> str:
    message = re.sub(
        r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    message = re.sub(
        r"(?i)((?:api[_-]?key|token)\s*[:=]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    return message


def _runner_command(
    condition: Condition,
    mode: str,
    labels_relative: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.runner",
        "--batch",
        "--backend",
        condition.backend,
        "--labels",
        labels_relative,
        "--images-dir",
        IMAGES_DIR,
        "--delay",
        str(condition.delay),
        "--run-dir",
        condition.run_dir(mode),
    ]
    if condition.uses_skill:
        command.extend(["--use-skill", "--skill-version", condition.skill_version or ""])
    if mode == "smoke":
        command.extend(["--limit", "1"])
    return command


def _run_attempt(
    condition: Condition,
    mode: str,
    labels_relative: str,
    completed_before: int,
    expected_n: int,
) -> AttemptOutcome:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        _runner_command(condition, mode, labels_relative),
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding=locale.getpreferredencoding(False),
        errors="replace",
        bufsize=1,
        env=environment,
    )
    if process.stdout is None:
        process.kill()
        raise MatrixError(f"{condition.name}: failed to capture runner output")

    completed_this_attempt = 0
    attempted_this_attempt = 0
    error_types: list[str] = []
    retryable_connection_errors: list[bool] = []

    for raw_line in process.stdout:
        line = raw_line.rstrip("\r\n")
        if DONE_RE.match(line):
            completed_this_attempt += 1
            attempted_this_attempt += 1
            completed = completed_before + completed_this_attempt
            print(
                f"[{condition.name}] progress: {completed}/{expected_n}; "
                "request succeeded",
                flush=True,
            )
            continue

        error_match = ERROR_RE.match(line)
        if error_match:
            attempted_this_attempt += 1
            error_type = error_match.group("error_type")
            error_types.append(error_type)
            raw_message = error_match.group("message")
            # Both current backends wrap SDK errors in RuntimeError. DeepSeek
            # preserves ``error_type=APIConnectionError`` in the message;
            # GLM preserves only the SDK text ``Connection error.``.
            retryable_connection_errors.append(
                error_type == "APIConnectionError"
                or "error_type=APIConnectionError" in raw_message
                or (
                    error_type == "RuntimeError"
                    and raw_message.rstrip().endswith("Connection error.")
                )
            )
            message = _redact_error_message(raw_message)
            print(
                f"[{condition.name}] request failed "
                f"(attempt {attempted_this_attempt}): {error_type}: {message}",
                flush=True,
            )

    returncode = process.wait()
    return AttemptOutcome(
        returncode=returncode,
        error_types=tuple(error_types),
        retryable_connection_errors=tuple(retryable_connection_errors),
    )


def _completed_ids(rows: list[dict[str, str]]) -> set[str]:
    return {(row.get("image_id") or "").strip() for row in rows}


def _run_condition(
    condition: Condition,
    mode: str,
    labels_relative: str,
    all_label_ids: list[str],
) -> ConditionSummary:
    if mode == "smoke":
        if not all_label_ids:
            raise MatrixError("smoke labels file is empty")
        expected_ids = {all_label_ids[0]}
        expected_n = 1
    else:
        expected_ids = set(all_label_ids)
        expected_n = TEST_EXPECTED_N

    run_dir = PROJECT_ROOT / condition.run_dir(mode)
    results_path = run_dir / "results.csv"
    rows = _read_results(results_path)
    _validate_results(condition, rows, expected_ids, results_path)
    completed = _completed_ids(rows)

    print(f"\nConfiguration: {condition.name}", flush=True)
    print(f"Run directory: {condition.run_dir(mode)}", flush=True)
    if completed:
        print(
            f"[{condition.name}] resume state: {len(completed)}/{expected_n} complete",
            flush=True,
        )

    if completed != expected_ids:
        first_outcome = _run_attempt(
            condition,
            mode,
            labels_relative,
            completed_before=len(completed),
            expected_n=expected_n,
        )
        rows = _read_results(results_path)
        _validate_results(condition, rows, expected_ids, results_path)
        completed = _completed_ids(rows)

        if mode == "smoke":
            if first_outcome.returncode != 0:
                raise MatrixError(
                    f"{condition.name}: runner exited with code "
                    f"{first_outcome.returncode}"
                )
            if first_outcome.error_types or completed != expected_ids:
                errors = ", ".join(first_outcome.error_types) or "unknown runner failure"
                raise MatrixError(f"{condition.name}: smoke failed: {errors}")
        elif first_outcome.returncode != 0 and completed == expected_ids:
            raise MatrixError(
                f"{condition.name}: runner exited with code "
                f"{first_outcome.returncode}"
            )
        elif completed != expected_ids:
            if not first_outcome.only_api_connection_errors:
                errors = ", ".join(first_outcome.error_types) or "unknown runner failure"
                raise MatrixError(
                    f"{condition.name}: incomplete after first attempt; errors: {errors}"
                )

            print(
                f"[{condition.name}] APIConnectionError detected; "
                "retrying incomplete samples once in the same run directory",
                flush=True,
            )
            retry_outcome = _run_attempt(
                condition,
                mode,
                labels_relative,
                completed_before=len(completed),
                expected_n=expected_n,
            )
            rows = _read_results(results_path)
            _validate_results(condition, rows, expected_ids, results_path)
            completed = _completed_ids(rows)
            if retry_outcome.returncode != 0 or completed != expected_ids:
                errors = ", ".join(retry_outcome.error_types) or "unknown runner failure"
                raise MatrixError(
                    f"{condition.name}: still incomplete after the single allowed retry; "
                    f"errors: {errors}"
                )

    if completed != expected_ids:
        raise MatrixError(
            f"{condition.name}: expected {expected_n} completed rows, found {len(completed)}"
        )

    parse_ok: bool | None = None
    if mode == "smoke":
        parse_ok = len(rows) == 1 and _as_bool(rows[0].get("parse_ok"))
        if not parse_ok:
            raise MatrixError(f"{condition.name}: smoke result has parse_ok=False")

    summary = ConditionSummary(
        name=condition.name,
        results_rows=len(rows),
        succeeded=len(completed),
        failed=expected_n - len(completed),
        parse_ok=parse_ok,
    )
    parse_text = f", parse_ok={summary.parse_ok}" if mode == "smoke" else ""
    print(
        f"[{condition.name}] complete: results_rows={summary.results_rows}, "
        f"succeeded={summary.succeeded}, failed={summary.failed}{parse_text}",
        flush=True,
    )
    return summary


def run_matrix(mode: str) -> list[ConditionSummary]:
    # Select and open only the split required by the requested mode. In
    # particular, smoke mode never opens data/splits/test.csv.
    labels_relative = SMOKE_LABELS if mode == "smoke" else TEST_LABELS
    labels_path = PROJECT_ROOT / labels_relative
    label_ids = _read_image_ids(labels_path)

    if mode == "test" and len(label_ids) != TEST_EXPECTED_N:
        raise MatrixError(
            f"test split must contain {TEST_EXPECTED_N} unique image_id values; "
            f"found {len(label_ids)}"
        )

    summaries: list[ConditionSummary] = []
    for condition in CONDITIONS:
        summaries.append(
            _run_condition(condition, mode, labels_relative, label_ids)
        )
    return summaries


def _print_summary(mode: str, summaries: list[ConditionSummary]) -> None:
    print(f"\nFinal matrix {mode} summary")
    for summary in summaries:
        if mode == "smoke":
            print(
                f"- {summary.name}: succeeded={summary.succeeded}, "
                f"failed={summary.failed}, parse_ok={summary.parse_ok}"
            )
        else:
            print(
                f"- {summary.name}: results_rows={summary.results_rows}, "
                f"succeeded={summary.succeeded}, failed={summary.failed}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed final smoke or held-out-test matrix sequentially."
    )
    parser.add_argument("--mode", choices=("smoke", "test"), required=True)
    args = parser.parse_args()

    try:
        summaries = run_matrix(args.mode)
    except MatrixError as exc:
        print(f"\nMATRIX STOPPED: {exc}", file=sys.stderr)
        return 1

    _print_summary(args.mode, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

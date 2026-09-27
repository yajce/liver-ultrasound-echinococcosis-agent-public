"""实验入口：单张 / 批量测试，结果写入 experiments/<run>/results.csv。

当前已实现：

    python -m src.runner                                    # mock 单条模拟样本
    python -m src.runner --batch                            # mock 批量（断点续跑）
    python -m src.runner --batch --backend glm              # GLM 真实批量（断点续跑）
    python -m src.runner --backend glm --image test.jpg     # GLM 真实单图链路

TODO（后续）:
    - config.yaml 驱动的模型选择与实验配置快照（load_config/build_agent/make_run_dir）
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from .agent import Agent, PredictionRecord
from .models.base import ModelBackend
from .models.mock_backend import MockBackend

BACKEND_CHOICES = ["mock", "glm", "deepseek"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_VERSIONS_ROOT = PROJECT_ROOT / "skills" / "echo-hydatid"

# 旧结果表结构仅用于无损升级已有 baseline 断点文件。
LEGACY_RESULTS_COLUMNS = [
    "image_id",
    "image_path",
    "true_label",
    "image_valid",
    "predicted_diagnosis",
    "predicted_subtype",
    "confidence",
    "parse_ok",
    "model",
    "skill_enabled",
    "prompt_version",
    "latency",
    "raw_response",
]

# results.csv 的列，与 agent.PredictionRecord 字段保持一致。
RESULTS_COLUMNS = [
    "image_id",
    "image_path",
    "true_label",
    "image_valid",
    "predicted_diagnosis",
    "predicted_subtype",
    "confidence",
    "parse_ok",
    "model",
    "skill_enabled",
    "skill_version",
    "skill_hash",
    "prompt_version",
    "latency",
    "raw_response",
]


def make_backend(name: str) -> ModelBackend:
    """按名称构造后端。glm 延迟导入，mock 链路不依赖 openai 包。"""
    if name == "mock":
        return MockBackend()
    if name == "glm":
        from .models.glm_backend import GLMBackend

        return GLMBackend()
    if name == "deepseek":
        from .models.deepseek_backend import DeepSeekBackend

        return DeepSeekBackend()
    raise ValueError(
        f"未知 backend: {name}（可选: {' | '.join(BACKEND_CHOICES)}）"
    )


def resolve_skill_path(skill_version: str) -> Path:
    """把 v1 / v2 等版本名解析为对应的 SKILL.md。"""
    version = skill_version.strip()
    if not version:
        raise ValueError("Skill 版本不能为空")
    if Path(version).name != version:
        raise ValueError(f"无效的 Skill 版本名: {skill_version!r}")

    skill_path = SKILL_VERSIONS_ROOT / version / "SKILL.md"
    if not skill_path.is_file():
        available = sorted(
            path.parent.name
            for path in SKILL_VERSIONS_ROOT.glob("*/SKILL.md")
            if path.is_file()
        )
        available_text = ", ".join(available) if available else "无"
        raise FileNotFoundError(
            f"Skill 版本 {version!r} 不存在: {skill_path}；"
            f"当前可用版本: {available_text}"
        )
    return skill_path


def make_experiment_agent(
    backend_name: str,
    skill_enabled: bool = False,
    skill_version: str = "v1",
) -> Agent:
    """构造 baseline 或指定 Skill 版本的同一 Agent 主链路。"""
    skill_path = resolve_skill_path(skill_version) if skill_enabled else None
    return Agent(
        backend=make_backend(backend_name),
        skill_enabled=skill_enabled,
        skill_path=skill_path,
        skill_version=skill_version if skill_enabled else None,
    )


def ensure_results_schema(results_path: Path) -> None:
    """确保已有 results.csv 使用当前列；旧 baseline 文件会原地无损升级。"""
    if not results_path.exists() or results_path.stat().st_size == 0:
        return

    with results_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        existing_columns = next(reader, [])
    if existing_columns == RESULTS_COLUMNS:
        return
    if existing_columns != LEGACY_RESULTS_COLUMNS:
        raise RuntimeError(
            f"已有 results.csv 列结构不兼容: {existing_columns}"
        )

    with results_path.open("r", newline="", encoding="utf-8") as csv_file:
        legacy_rows = list(csv.DictReader(csv_file))
    temporary_path = results_path.with_name(f".{results_path.name}.schema.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULTS_COLUMNS)
        writer.writeheader()
        for row in legacy_rows:
            row["skill_version"] = ""
            row["skill_hash"] = ""
            writer.writerow(row)
    temporary_path.replace(results_path)


def append_record(results_path: Path, record: PredictionRecord) -> None:
    """追加一行；文件不存在时先写表头。逐条落盘，中断后已完成样本不丢失。"""
    ensure_results_schema(results_path)
    write_header = not results_path.exists() or results_path.stat().st_size == 0
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(RESULTS_COLUMNS)
        writer.writerow([getattr(record, col) for col in RESULTS_COLUMNS])


def write_results_csv(run_dir: Path, records: list[PredictionRecord]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.csv"
    ensure_results_schema(results_path)
    for record in records:
        append_record(results_path, record)
    return results_path


def load_labels(labels_csv: str | Path) -> dict[str, str]:
    """读 labels.csv，返回 image_id -> true_label（保持文件行序）。"""
    labels: dict[str, str] = {}
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_id = (row.get("image_id") or "").strip()
            if image_id:
                labels[image_id] = (row.get("true_label") or "").strip()
    return labels


def read_done_image_ids(results_path: Path) -> set[str]:
    """读取已有 results.csv 中出现过的 image_id，用于断点续跑跳过。"""
    if not results_path.exists():
        return set()
    with open(results_path, newline="", encoding="utf-8") as f:
        return {
            (row.get("image_id") or "").strip() for row in csv.DictReader(f)
        }


def run_mock_sample(
    skill_enabled: bool = False, skill_version: str = "v1"
) -> tuple[PredictionRecord, Path]:
    """用 MockBackend 对一条模拟样本运行完整链路。

    图像路径仅为占位（MockBackend 不读取文件内容），true_label 为模拟标注。
    """
    agent = make_experiment_agent(
        "mock", skill_enabled=skill_enabled, skill_version=skill_version
    )
    image_path = Path("data/images/mock_sample_001.jpg")
    record = agent.analyze(image_path, true_label="echinococcosis")
    suffix = f"mock_skill_{skill_version}" if skill_enabled else "mock"
    run_dir = Path("experiments") / f"{datetime.now():%Y%m%d-%H%M%S}_{suffix}"
    results_path = write_results_csv(run_dir, [record])
    return record, results_path


def run_batch(
    labels_csv: str | Path,
    images_dir: str | Path,
    run_dir: str | Path,
    backend_name: str = "mock",
    limit: int | None = None,
    delay: float = 0.0,
    skill_enabled: bool = False,
    skill_version: str = "v1",
) -> tuple[Path, int, int, list[str]]:
    """按 labels.csv 批量运行指定后端，逐条追加到 results.csv。

    已存在于 results.csv 的 image_id 自动跳过（断点续跑）。
    单张失败只记录并继续，不写入伪造的 PredictionRecord，因此后续运行仍会重试。
    limit 只计算尚未完成且本次实际尝试的样本。
    返回 (results_path, 成功数, 跳过数, 失败 image_id 列表)。
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit 必须是正整数")
    if delay < 0:
        raise ValueError("delay 不能为负数")

    agent = make_experiment_agent(
        backend_name,
        skill_enabled=skill_enabled,
        skill_version=skill_version,
    )
    results_path = Path(run_dir) / "results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_results_schema(results_path)
    labels = load_labels(labels_csv)
    done = read_done_image_ids(results_path)

    pending: list[tuple[str, str]] = []
    n_skipped = 0
    for image_id, true_label in labels.items():
        if image_id in done:
            n_skipped += 1
            print(f"skip (done) : {image_id}")
            continue
        pending.append((image_id, true_label))

    if limit is not None:
        pending = pending[:limit]

    n_succeeded = 0
    failed_image_ids: list[str] = []
    for attempt_index, (image_id, true_label) in enumerate(pending):
        if attempt_index > 0 and delay > 0:
            time.sleep(delay)

        image_path = Path(images_dir) / image_id
        try:
            # 保持唯一主链路：Runner -> Agent -> Backend -> Parser -> Record。
            record = agent.analyze(image_path, true_label=true_label)
        except Exception as exc:
            failed_image_ids.append(image_id)
            print(
                f"error       : {image_id} -> "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        # 每张成功后立即打开、追加并关闭文件，确保结果逐条落盘。
        append_record(results_path, record)
        n_succeeded += 1
        print(f"done        : {image_id} -> {record.predicted_diagnosis} "
              f"(parse_ok={record.parse_ok})")
    return results_path, n_succeeded, n_skipped, failed_image_ids


def run_mock_batch(
    labels_csv: str | Path, images_dir: str | Path, run_dir: str | Path
) -> tuple[Path, int, int]:
    """兼容原有调用的 mock 批量包装。"""
    results_path, n_succeeded, n_skipped, _ = run_batch(
        labels_csv=labels_csv,
        images_dir=images_dir,
        run_dir=run_dir,
        backend_name="mock",
    )
    return results_path, n_succeeded, n_skipped


def run_single(
    image_path: str | Path,
    backend_name: str = "mock",
    true_label: str = "",
    run_dir: str | Path | None = None,
    skill_enabled: bool = False,
    skill_version: str = "v1",
) -> tuple[PredictionRecord, Path]:
    """单张图端到端链路：Agent -> Backend -> Parser -> PredictionRecord -> results.csv。

    真实后端（glm）会校验图像存在性与 API 配置，异常信息由后端抛出。
    """
    agent = make_experiment_agent(
        backend_name,
        skill_enabled=skill_enabled,
        skill_version=skill_version,
    )
    record = agent.analyze(Path(image_path), true_label=true_label)
    run_dir = run_dir or Path("experiments") / (
        f"{datetime.now():%Y%m%d-%H%M%S}_{backend_name}_single"
    )
    results_path = write_results_csv(Path(run_dir), [record])
    return record, results_path


def load_config(config_path: str = "config.yaml") -> dict:
    """TODO: 读取 config.yaml（密钥从 .env 加载，不经过本函数返回值）。"""
    raise NotImplementedError


def build_agent(config: dict) -> Agent:
    """TODO: 按 config['model']['backend'] 选择适配器，组装 Agent（含 Skill 开关）。"""
    raise NotImplementedError


def make_run_dir(experiments_dir: str, config: dict) -> str:
    """TODO: 创建 experiments/<日期>_<模型>_<skill开关>/ 并写入 config_used.yaml 快照。"""
    raise NotImplementedError


def _print_record(record: PredictionRecord, results_path: Path) -> None:
    print("run dir     :", results_path.parent)
    print("results.csv :", results_path)
    for col in RESULTS_COLUMNS:
        print(f"{col:<20}: {getattr(record, col)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="实验入口（mock / GLM 均支持单图与批量主链路）"
    )
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="mock",
                        help="模型后端（默认 mock，离线）")
    parser.add_argument("--batch", action="store_true",
                        help="按 labels.csv 批量运行，支持逐条落盘与断点续跑")
    parser.add_argument("--image", default=None,
                        help="单张图片路径；提供时走单图链路（glm 时文件必须存在）")
    parser.add_argument("--label", default="",
                        help="单图测试的真实标签（可选，仅用于记录）")
    parser.add_argument("--labels", default="data/labels.csv",
                        help="标签文件路径（默认 data/labels.csv）")
    parser.add_argument("--images-dir", default="data/images",
                        help="图像目录（默认 data/images）")
    parser.add_argument("--run-dir", default=None,
                        help="实验输出目录；批量模式下重复运行即断点续跑，"
                             "默认按时间戳新建")
    parser.add_argument("--limit", type=int, default=None,
                        help="批量模式下最多尝试前 N 个尚未完成的样本")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="批量模式下相邻 API 请求之间等待的秒数（默认 0）")
    parser.add_argument("--use-skill", action="store_true",
                        help="加载指定版本的 Skill 作为额外推理上下文")
    parser.add_argument("--skill-version", default=None,
                        help="Skill 版本目录名，如 v1；--use-skill 时默认 v1")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须是正整数")
    if args.delay < 0:
        parser.error("--delay 不能为负数")
    if not args.batch and (args.limit is not None or args.delay != 0):
        parser.error("--limit 和 --delay 仅用于 --batch 模式")
    if args.skill_version is not None and not args.use_skill:
        parser.error("--skill-version 必须与 --use-skill 一起使用")

    selected_skill_version = args.skill_version or "v1"
    if args.use_skill:
        try:
            resolve_skill_path(selected_skill_version)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))

    if args.batch:
        run_dir = args.run_dir or str(
            Path("experiments")
            / f"{datetime.now():%Y%m%d-%H%M%S}_{args.backend}"
        )
        results_path, n_succeeded, n_skipped, failed_image_ids = run_batch(
            labels_csv=args.labels,
            images_dir=args.images_dir,
            run_dir=run_dir,
            backend_name=args.backend,
            limit=args.limit,
            delay=args.delay,
            skill_enabled=args.use_skill,
            skill_version=selected_skill_version,
        )
        print(f"\nrun dir     : {results_path.parent}")
        print(f"results.csv : {results_path}")
        print(f"succeeded   : {n_succeeded}")
        print(f"failed      : {len(failed_image_ids)}")
        print(f"skipped     : {n_skipped}")
        if failed_image_ids:
            print(f"failed IDs  : {', '.join(failed_image_ids)}")
    elif args.image:
        record, results_path = run_single(
            args.image,
            args.backend,
            args.label,
            args.run_dir,
            skill_enabled=args.use_skill,
            skill_version=selected_skill_version,
        )
        _print_record(record, results_path)
    else:
        record, results_path = run_mock_sample(
            skill_enabled=args.use_skill,
            skill_version=selected_skill_version,
        )
        _print_record(record, results_path)


if __name__ == "__main__":
    main()

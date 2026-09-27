"""Agent：组装 prompt（可选加载 Skill）→ 调用模型后端 → 生成一条完整实验记录。"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from . import prompts
from .models.base import ModelBackend
from .parser import parse_response


@dataclass
class PredictionRecord:
    """单张图的完整实验记录，对应 results.csv 的一行。raw_response 必须始终保存。"""

    image_id: str               # 图像文件名（与 labels.csv 的 image_id 对应）
    image_path: str
    true_label: str             # 来自 labels.csv；无标注时为 ""
    image_valid: bool | None    # 模型判断的图像可用性；解析失败时为 None
    predicted_diagnosis: str    # echinococcosis | negative | not_applicable
    predicted_subtype: str | None   # 可选
    confidence: float | None
    parse_ok: bool
    model: str                  # backend.name + 模型名（如 "glm/glm-5.3-flash"）
    skill_enabled: bool
    skill_version: str
    skill_hash: str
    prompt_version: str
    latency: float              # 秒
    raw_response: str           # 模型原始回答，始终保存


class Agent:
    def __init__(
        self,
        backend: ModelBackend,
        skill_enabled: bool = False,
        skill_path: Path | None = None,
        skill_version: str | None = None,
    ):
        self.backend = backend
        self.skill_enabled = skill_enabled
        if skill_enabled:
            if skill_path is None:
                raise ValueError("skill_enabled=True 需要提供 skill_path")
            self.skill_path = Path(skill_path)
            if not self.skill_path.is_file():
                raise FileNotFoundError(f"Skill 文件不存在: {self.skill_path}")
            self.skill_version = (
                skill_version or self.skill_path.parent.name
            ).strip()
            if not self.skill_version:
                raise ValueError("启用 Skill 时 skill_version 不能为空")
            self.skill_hash = hashlib.sha256(
                self.skill_path.read_bytes()
            ).hexdigest()
        else:
            # baseline 模式不读取 Skill 文件，也不保留 Skill 元数据。
            self.skill_path = None
            self.skill_version = ""
            self.skill_hash = ""
        self.system_prompt, self.prompt_version = prompts.build_system_prompt(
            skill_enabled, self.skill_path
        )

    def analyze(self, image_path: Path, true_label: str = "") -> PredictionRecord:
        """单图主链路：generate -> parse -> 记录。

        - true_label 由 runner 查 labels.csv 传入（Agent 本身不读标签文件）
        - 模型调用异常不在此捕获，交给 runner 处理与记录
        """
        response = self.backend.generate(
            image_path=image_path,
            system_prompt=self.system_prompt,
            user_prompt=prompts.BASELINE_USER_PROMPT,
        )
        parsed = parse_response(response.text)
        return PredictionRecord(
            image_id=image_path.name,
            image_path=str(image_path),
            true_label=true_label,
            image_valid=parsed.image_valid,
            predicted_diagnosis=parsed.diagnosis,
            predicted_subtype=parsed.subtype,
            confidence=parsed.confidence,
            parse_ok=parsed.parse_ok,
            model=f"{self.backend.name}/{response.model_name}",
            skill_enabled=self.skill_enabled,
            skill_version=self.skill_version,
            skill_hash=self.skill_hash,
            prompt_version=self.prompt_version,
            latency=response.latency,
            raw_response=response.text,
        )

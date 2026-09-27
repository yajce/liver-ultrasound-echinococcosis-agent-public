"""大模型后端抽象接口。

所有实验代码只依赖本模块的 ModelBackend / ModelResponse，
不直接 import 任何具体厂商 SDK —— 这是模型可替换的关键。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelResponse:
    """后端统一返回结构。text 即 raw_response，实验中必须始终落盘。"""

    text: str                              # 模型原始回答
    latency: float                         # 单次调用耗时（秒）
    model_name: str                        # 实际调用的模型标识
    usage: dict[str, Any] | None = None    # token 用量等（厂商有提供则记录）


class ModelBackend(ABC):
    """大模型后端接口：输入图像与提示词，返回原始文本。"""

    name: str = "base"

    @abstractmethod
    def generate(
        self, image_path: Path, system_prompt: str, user_prompt: str
    ) -> ModelResponse:
        """单次多模态调用。

        TODO（由各适配器实现）:
        - 从 .env 读取本厂商 API key
        - 图像读取与编码（base64 / URL，按厂商要求）
        - 超时、重试与错误处理（错误应抛异常，由 runner 记录）
        - 记录 latency 与 usage
        """

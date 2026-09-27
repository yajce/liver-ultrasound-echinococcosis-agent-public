"""Claude 适配器（骨架，暂不实现 API 调用）。"""

from pathlib import Path

from .base import ModelBackend, ModelResponse


class ClaudeBackend(ModelBackend):
    name = "claude"

    def generate(
        self, image_path: Path, system_prompt: str, user_prompt: str
    ) -> ModelResponse:
        # TODO: 读取 ANTHROPIC_API_KEY，调用 Anthropic 多模态接口，返回 ModelResponse
        raise NotImplementedError

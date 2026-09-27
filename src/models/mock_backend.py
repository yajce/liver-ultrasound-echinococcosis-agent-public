"""Mock 后端：不访问网络，返回固定的、符合约定 JSON schema 的响应。

用途：离线验证 输入 → Agent → Backend → Parser → Result 主链路。
响应内容为流程验证占位文本，不包含任何医学判读知识。
"""

import json
import time
from pathlib import Path

from .base import ModelBackend, ModelResponse

_MOCK_RESPONSE = {
    "image_valid": True,
    "diagnosis": "echinococcosis",
    "confidence": 0.8,
    "subtype": None,
    "findings": "mock response for pipeline verification (no medical content)",
}


class MockBackend(ModelBackend):
    name = "mock"

    def generate(
        self, image_path: Path, system_prompt: str, user_prompt: str
    ) -> ModelResponse:
        start = time.perf_counter()
        time.sleep(0.05)  # 模拟调用耗时，便于确认 latency 字段被正确记录
        return ModelResponse(
            text=json.dumps(_MOCK_RESPONSE, ensure_ascii=False),
            latency=time.perf_counter() - start,
            model_name="mock-echo-hydatid-v0",
        )

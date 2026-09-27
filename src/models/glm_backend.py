"""GLM-5.3-Flash 适配器（并行智算云 MaaS 平台 · 清华算力券，OpenAI 兼容接口）。

全部连接信息来自 .env 环境变量，不在代码中硬编码：
- THU_MAAS_API_KEY   平台 API key（必填）
- THU_MAAS_BASE_URL  接口根地址（必填；OpenAI SDK 自动拼接 /chat/completions）
- THU_MAAS_MODEL     模型名（必填，如 GLM-5.3-Flash）

图像输入沿用 OpenAI 多模态格式：messages[].content[] 中 type=image_url 内容块，
本地图片以 Base64 Data URL 传入。

依赖: pip install openai python-dotenv
"""

import base64
import mimetypes
import os
import time
from pathlib import Path

from .base import ModelBackend, ModelResponse

try:
    from openai import OpenAI
except ImportError:  # 允许未装依赖时导入本模块，实例化时再报清晰错误
    OpenAI = None


def _image_to_data_url(image_path: Path) -> str:
    """本地图片 -> Base64 Data URL（OpenAI 兼容接口支持的图像传入方式）。"""
    mime, _ = mimetypes.guess_type(image_path.name)
    if mime is None or not mime.startswith("image/"):
        mime = "image/jpeg"  # 扩展名无法识别时的兜底
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


class GLMBackend(ModelBackend):
    name = "glm"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        if OpenAI is None:
            raise ImportError(
                "GLMBackend 需要 openai 包：pip install openai"
            )
        try:
            from dotenv import load_dotenv

            load_dotenv()  # 读取项目根目录 .env（不存在时无操作）
        except ImportError:
            pass  # 未装 python-dotenv 时退回系统环境变量
        self.model = model        # None 时 generate() 从 THU_MAAS_MODEL 读取
        self._api_key = api_key   # None 时从 THU_MAAS_API_KEY 读取
        self._base_url = base_url  # None 时从 THU_MAAS_BASE_URL 读取

    @staticmethod
    def _require(value: str | None, env_name: str, what: str) -> str:
        """取显式参数或环境变量，二者皆空则报清晰错误。"""
        resolved = (value or os.getenv(env_name, "")).strip()
        if not resolved:
            raise RuntimeError(f"未配置 {env_name}（{what}）：请在 .env 中填写")
        return resolved

    def generate(
        self, image_path: Path, system_prompt: str, user_prompt: str
    ) -> ModelResponse:
        api_key = self._require(self._api_key, "THU_MAAS_API_KEY", "平台 API key")
        base_url = self._require(self._base_url, "THU_MAAS_BASE_URL", "接口根地址")
        model = self._require(self.model, "THU_MAAS_MODEL", "模型名")
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        # TODO: 批量阶段复用 client（实例级缓存），单张调用暂不优化
        client = OpenAI(api_key=api_key, base_url=base_url)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_url(image_path)},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        start = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=model, messages=messages
            )
        except Exception as e:
            raise RuntimeError(
                f"GLM API 请求失败（model={model}, base_url={base_url}）: {e}"
            ) from e
        latency = time.perf_counter() - start

        usage = None
        if completion.usage is not None:
            usage = completion.usage.model_dump()
        return ModelResponse(
            text=completion.choices[0].message.content or "",
            latency=latency,
            model_name=model,
            usage=usage,
        )

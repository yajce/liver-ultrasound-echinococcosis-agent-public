"""DeepSeek-V4.1-Flash backend for the shared THU MaaS endpoint.

Connection settings are read from environment variables:

- ``THU_MAAS_API_KEY``: shared MaaS API key.
- ``THU_MAAS_BASE_URL``: shared OpenAI-compatible API base URL.
- ``THU_MAAS_DEEPSEEK_MODEL``: DeepSeek model name (optional).

The local image is sent as a Base64 data URL in an ``image_url`` content
block, matching the existing GLM backend's multimodal request format.
"""

import os
import time
from pathlib import Path

from .base import ModelBackend, ModelResponse
from .glm_backend import _image_to_data_url

try:
    from openai import OpenAI
except ImportError:  # Keep module importable and fail clearly on construction.
    OpenAI = None


DEFAULT_MODEL = "DeepSeek-V4.1-Flash"


class DeepSeekBackend(ModelBackend):
    """OpenAI-compatible multimodal backend for DeepSeek-V4.1-Flash."""

    name = "deepseek"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        if OpenAI is None:
            raise ImportError(
                "DeepSeekBackend requires the openai package: pip install openai"
            )
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        self.model = (
            model or os.getenv("THU_MAAS_DEEPSEEK_MODEL", DEFAULT_MODEL)
        ).strip()
        self._api_key = api_key
        self._base_url = base_url

    @staticmethod
    def _require(value: str | None, env_name: str, what: str) -> str:
        resolved = (value or os.getenv(env_name, "")).strip()
        if not resolved:
            raise RuntimeError(
                f"Missing {env_name} ({what}); configure it in the project .env file"
            )
        return resolved

    def generate(
        self, image_path: Path, system_prompt: str, user_prompt: str
    ) -> ModelResponse:
        api_key = self._require(
            self._api_key, "THU_MAAS_API_KEY", "shared MaaS API key"
        )
        base_url = self._require(
            self._base_url, "THU_MAAS_BASE_URL", "shared MaaS base URL"
        )
        model = self.model
        if not model:
            raise RuntimeError("DeepSeek model name cannot be empty")

        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

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
        except Exception as exc:
            raise RuntimeError(
                "DeepSeek API request failed "
                f"(model={model}, base_url={base_url}, "
                f"error_type={type(exc).__name__}): {exc}"
            ) from exc
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

"""Prompt 管理：baseline prompt 与 Skill prompt 分开维护。

版本号规则：每次修改 prompt 内容必须递增对应版本号。
实验结果中的 prompt_version 字段依赖它区分不同 prompt 下的实验。
"""

from pathlib import Path

BASELINE_PROMPT_VERSION = "baseline-v0.2"

# baseline（无 Skill）系统提示词：只规定任务与输出格式，不含医学判读知识。
# v0.2: 新增 image_valid 门控——非可用肝脏超声图像输出 not_applicable，不强制二分类。
BASELINE_SYSTEM_PROMPT = (
    "You are an assistant for a research experiment on liver ultrasound images.\n"
    "Task: first decide whether the given image is a usable liver ultrasound image "
    "for this task; if it is, decide whether it is suspected of echinococcosis.\n"
    "Reply with exactly one JSON object and no other text, using this schema:\n"
    "{\n"
    '  "image_valid": true or false, whether the image is a usable liver ultrasound image,\n'
    '  "diagnosis": "echinococcosis" | "negative" | "not_applicable",\n'
    '  "confidence": a number between 0 and 1, or null,\n'
    '  "subtype": a string or null, optional field, null if not applicable,\n'
    '  "findings": a short factual description of what you observe\n'
    "}\n"
    "Rules:\n"
    "- If the image is not a usable liver ultrasound image (for example a natural photo, "
    "a screenshot, a non-ultrasound image, or an unreadable image), you must output "
    '"image_valid": false, "diagnosis": "not_applicable", "confidence": null, '
    '"subtype": null.\n'
    "- Only when image_valid is true, choose diagnosis between "
    '"echinococcosis" and "negative", with a confidence between 0 and 1.\n'
    "- Never force a diagnosis on an image that is not a usable liver ultrasound image."
)

BASELINE_USER_PROMPT = (
    "Analyze the attached liver ultrasound image and reply with the JSON object "
    "as required."
)


def load_skill_text(skill_path: Path) -> str:
    """读取 SKILL.md 文本作为判读流程内容。"""
    resolved_path = Path(skill_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Skill 文件不存在: {resolved_path}")
    skill_text = resolved_path.read_text(encoding="utf-8")
    if not skill_text.strip():
        raise ValueError(f"Skill 文件为空: {resolved_path}")
    return skill_text


def build_system_prompt(
    skill_enabled: bool, skill_path: Path | None = None
) -> tuple[str, str]:
    """按实验设置组装 (system_prompt, prompt_version)。

    - skill_enabled=False -> baseline prompt
    - skill_enabled=True  -> baseline + Skill 判读流程文本
    """
    if not skill_enabled:
        return BASELINE_SYSTEM_PROMPT, BASELINE_PROMPT_VERSION
    if skill_path is None:
        raise ValueError("skill_enabled=True 需要提供 skill_path")
    # TODO: SKILL.md 目前是模板；正式 Skill 实验前需完成文献填充与校验
    skill_text = load_skill_text(skill_path)
    system_prompt = (
        BASELINE_SYSTEM_PROMPT
        + "\n\nFollow this review procedure step by step:\n\n"
        + skill_text
    )
    return system_prompt, BASELINE_PROMPT_VERSION

"""从模型原始回答中解析结构化结果。

原则：解析失败不丢弃样本 —— parse_ok=False，原始回答照常落盘，
便于离线修复解析逻辑后重新统计，无需重跑模型。
"""

import json
from dataclasses import dataclass

# diagnosis 允许的三个取值；image_valid=false 时只应为 not_applicable
VALID_DIAGNOSES = {"echinococcosis", "negative", "not_applicable"}


@dataclass
class ParsedResult:
    image_valid: bool | None    # 是否为可用于本任务的肝脏超声图像；缺失/类型错误为 None
    diagnosis: str              # "echinococcosis" | "negative" | "not_applicable"
    confidence: float | None    # 模型自评置信度 0-1，缺失或超范围时为 None
    subtype: str | None         # 可选字段：是否启用待数据标签确定后再决定
    findings: str               # 模型给出的判读依据/描述
    parse_ok: bool              # False 时以上字段不可信，只能依赖 raw_response


def parse_response(raw: str) -> ParsedResult:
    """从 raw_response 提取 JSON。

    实现：取首个 '{' 到最后一个 '}' 之间的文本做 json.loads
    （天然兼容 ```json 代码块包裹与前后多余文本），再做字段基本校验：
    - diagnosis 必须是三个合法取值之一，否则 parse_ok=False
    - image_valid 必须是 bool，否则为 None
    - 一致性归一化：image_valid=False 或 diagnosis=not_applicable 时，
      强制 diagnosis=not_applicable、confidence/subtype 置空
    """
    failed = ParsedResult(image_valid=None, diagnosis="", confidence=None,
                          subtype=None, findings="", parse_ok=False)
    try:
        start = raw.index("{")
        end = raw.rindex("}")
        data = json.loads(raw[start:end + 1])
    except ValueError:  # 无大括号或 JSON 非法
        return failed
    if not isinstance(data, dict):
        return failed

    diagnosis = data.get("diagnosis")
    if not isinstance(diagnosis, str):
        return failed
    diagnosis = diagnosis.strip().lower()
    if diagnosis not in VALID_DIAGNOSES:
        return failed

    image_valid = data.get("image_valid")
    if not isinstance(image_valid, bool):
        image_valid = None

    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        confidence = None
    elif not (0 <= confidence <= 1):
        confidence = None

    subtype = data.get("subtype")
    if not isinstance(subtype, str) or not subtype.strip():
        subtype = None

    findings = data.get("findings")
    if not isinstance(findings, str):
        findings = ""

    if image_valid is False or diagnosis == "not_applicable":
        image_valid = False
        diagnosis = "not_applicable"
        confidence = None
        subtype = None

    return ParsedResult(
        image_valid=image_valid,
        diagnosis=diagnosis,
        confidence=confidence,
        subtype=subtype,
        findings=findings,
        parse_ok=True,
    )

"""成员 C 负责：Day 2 验收脚本——一个端到端 mock 请求。

流程：自然语言需求 -> LLM（有 key 走百炼，无 key 走 Mock）-> JSON
      -> contracts.DesignSpec 校验 -> 打印结果并写日志。

运行（项目根目录）：
    .venv/Scripts/python.exe scripts/mock_request.py
    .venv/Scripts/python.exe scripts/mock_request.py "2.4GHz 带通滤波器，插损小于 2dB"
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rf_agent.adapters.llm import MissingAPIKeyError, chat_json, mock_chat_json
from rf_agent.contracts import DesignSpec
from rf_agent.logging_setup import get_logger

logger = get_logger("scripts.mock_request")

DEFAULT_REQUEST = "在 2.4 GHz 附近做一个带通滤波器，通带插损小于 2 dB，回波损耗优于 -10 dB"

SYSTEM_PROMPT = (
    "你是射频设计需求解析器。把用户需求抽取为 JSON："
    "topology、frequency_band_hz（Hz）、parameters（name/unit/lower/upper/nominal）、"
    "metrics（name/operator/target/band_hz）、needs_confirmation（缺失或含糊的字段列表）。"
    "operator 只能取 <=、>=、==、in_band 四个值之一，禁止使用 < 或 >。"
    "parameters 至少给一个可调几何参数：用户没说时按该拓扑的典型参数提出建议范围，"
    "并在 needs_confirmation 里写明这些参数是建议值、需工程师确认。"
    "其他不确定的字段也必须放进 needs_confirmation，不许编造。"
)

# 合同只允许 <=/>=/==/in_band；LLM 偶尔写成严格不等号，规范化为闭区间（工程上容差远大于等号差别）
OPERATOR_ALIASES = {"<": "<=", ">": ">=", "=": "==", "≤": "<=", "≥": ">="}


def normalize_operators(data: dict) -> None:
    for metric in data.get("metrics", []):
        op = metric.get("operator")
        if op in OPERATOR_ALIASES:
            metric["operator"] = OPERATOR_ALIASES[op]


def main() -> int:
    raw_request = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REQUEST
    logger.info("step1 收到原始需求: %s", raw_request)

    try:
        data = chat_json(SYSTEM_PROMPT, raw_request, schema=DesignSpec.model_json_schema())
        mode = "百炼 LLM"
    except MissingAPIKeyError:
        data = mock_chat_json(raw_request)
        mode = "Mock（未配置 DASHSCOPE_API_KEY）"
    logger.info("step2 LLM 返回 JSON，模式=%s", mode)

    meta = data.pop("_meta", None)
    data.pop("_mock", None)
    data.pop("note", None)
    # spec_id/raw_request/时间戳由脚本统一赋值，模型即使自作主张填了也忽略
    for field in ("spec_id", "raw_request", "created_at", "spec_version"):
        data.pop(field, None)
    normalize_operators(data)
    spec = DesignSpec(
        spec_id=f"spec-{uuid.uuid4().hex[:8]}",
        raw_request=raw_request,
        **data,
    )
    logger.info("step3 DesignSpec 校验通过: spec_id=%s", spec.spec_id)

    print(f"\n=== 端到端 mock 请求成功（{mode}）===")
    if meta:
        print(f"追溯信息: {json.dumps(meta, ensure_ascii=False)}")
    print(f"spec_id: {spec.spec_id}")
    print(f"拓扑: {spec.topology}  频段: {spec.frequency_band_hz} Hz")
    print(f"参数: {[p.name for p in spec.parameters]}")
    print(f"指标: {[m.name for m in spec.metrics]}")
    if spec.needs_confirmation:
        print(f"待工程师确认: {spec.needs_confirmation}")
    print("\n完整 JSON:")
    print(spec.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

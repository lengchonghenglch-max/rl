"""成员 C 负责：Day 2 验收脚本——一个端到端 mock 请求。

流程：自然语言需求 -> LLM（有 key 走百炼/Token Plan，无 key 走 Mock）-> JSON
      -> contracts.DesignSpec 校验 -> 打印结果并写日志。

解析逻辑统一放在 rf_agent.agent.parse_requirement_text（Day 3 起单一来源），
本脚本只负责演示和打印。

运行（项目根目录）：
    .venv/Scripts/python.exe scripts/mock_request.py
    .venv/Scripts/python.exe scripts/mock_request.py "2.4GHz 带通滤波器，插损小于 2dB"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rf_agent.agent import parse_requirement_text
from rf_agent.logging_setup import get_logger

logger = get_logger("scripts.mock_request")

DEFAULT_REQUEST = "在 2.4 GHz 附近做一个带通滤波器，通带插损小于 2 dB，回波损耗优于 -10 dB"


def main() -> int:
    raw_request = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REQUEST
    logger.info("step1 收到原始需求: %s", raw_request)

    spec, mode, meta = parse_requirement_text(raw_request)
    logger.info("step2 解析完成，模式=%s", mode)
    logger.info("step3 DesignSpec 校验通过: spec_id=%s", spec.spec_id)

    print(f"\n=== 端到端请求成功（{mode}）===")
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

"""成员 C 负责：阿里云百炼 LLM 客户端（手册 Day 2）。

规则：百炼调用只放在本文件；API key 从环境变量 DASHSCOPE_API_KEY 读取，
永不写进代码或提交进 git。无 key 时必须能落到 mock 模式。
"""

from __future__ import annotations

import os

def has_api_key() -> bool:
    return bool(os.environ.get("DASHSCOPE_API_KEY"))


def chat_json(system_prompt: str, user_message: str, schema: dict | None = None) -> dict:
    """请求 LLM 并返回 JSON。

    TODO(C) 实现要点：
    - dashscope SDK，OpenAI-compatible 结构化输出
    - 解析失败 -> 带错误信息重试（上限 2 次）
    - 无 API key -> 抛 MissingAPIKeyError，由上层切 mock
    - 记录 model 版本、prompt 版本，保证可追溯
    """
    raise NotImplementedError("成员 C 待实现")


def mock_chat_json(user_message: str) -> dict:
    """演示用固定回复（手册：演示中必须标记 Mock）。"""
    return {
        "_mock": True,
        "topology": "bandpass_filter",
        "frequency_band_hz": [2.3e9, 2.5e9],
        "note": "这是 Mock 数据，不是真实 LLM 解析结果",
    }

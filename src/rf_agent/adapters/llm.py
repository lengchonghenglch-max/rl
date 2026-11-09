"""成员 C 负责：阿里云百炼 LLM 客户端（手册 Day 2）。

规则：百炼调用只放在本文件；API key 从环境变量 DASHSCOPE_API_KEY 读取，
永不写进代码或提交进 git，也永不写进日志。无 key 时必须能落到 mock 模式。
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from rf_agent.logging_setup import get_logger

logger = get_logger("adapters.llm")

DEFAULT_MODEL = "qwen-plus"
PROMPT_VERSION = "v1"
MAX_PARSE_RETRIES = 2  # 解析失败后带错误信息重试的上限


class MissingAPIKeyError(RuntimeError):
    """环境变量里没有 DASHSCOPE_API_KEY。上层应捕获并切换到 mock 模式。"""


class LLMError(RuntimeError):
    """百炼调用失败或多次重试后仍拿不到合法 JSON。"""


def has_api_key() -> bool:
    load_dotenv()
    return bool(os.environ.get("DASHSCOPE_API_KEY"))


def _call_dashscope(model: str, messages: list[dict]) -> tuple[str, str]:
    """调用百炼，返回 (回复文本, request_id)。只有本函数 import dashscope。"""
    import dashscope
    from dashscope import Generation

    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    resp = Generation.call(
        model=model,
        messages=messages,
        result_format="message",
        response_format={"type": "json_object"},
    )
    if resp.status_code != 200:
        raise LLMError(f"百炼返回 {resp.status_code}: code={resp.code} message={resp.message}")
    content = resp.output.choices[0].message.content
    return content, resp.request_id


def _call_openai_compatible(base_url: str, model: str, messages: list[dict]) -> tuple[str, str]:
    """套餐专用 key（如 sk-sp- 前缀）走 OpenAI 兼容服务地址。

    base_url 形如 https://xxx.aliyuncs.com/v1，实际请求 {base_url}/chat/completions。
    """
    import requests

    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}"},
        json={
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise LLMError(f"OpenAI 兼容接口返回 {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content, data.get("id", "")


def _call_anthropic_compatible(base_url: str, model: str, messages: list[dict]) -> tuple[str, str]:
    """Token Plan 套餐（sk-sp- key）走 Anthropic 兼容接口，如 .../apps/anthropic。

    与 OpenAI 风格不同：system 单独成字段、max_tokens 必填、
    回复在 content 块数组里（可能夹带 thinking 块，只取 text 块）。
    """
    import requests

    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    conv = [m for m in messages if m["role"] != "system"]
    resp = requests.post(
        f"{base_url.rstrip('/')}/v1/messages",
        headers={
            "x-api-key": os.environ["DASHSCOPE_API_KEY"],
            "anthropic-version": "2023-06-01",
        },
        json={"model": model, "max_tokens": 8192, "system": system, "messages": conv},
        timeout=120,
    )
    if resp.status_code != 200:
        raise LLMError(f"Anthropic 兼容接口返回 {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    content = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return content, data.get("id", "")


def chat_json(system_prompt: str, user_message: str, schema: dict | None = None) -> dict:
    """请求 LLM 并返回 JSON dict。

    - 无 API key -> 抛 MissingAPIKeyError，由上层切 mock。
    - 回复不是合法 JSON -> 把错误信息喂回去重试，上限 MAX_PARSE_RETRIES 次。
    - 成功时在返回值里附 _meta（model/prompt_version/request_id），保证可追溯。
    """
    load_dotenv()
    if not has_api_key():
        raise MissingAPIKeyError(
            "缺少 DASHSCOPE_API_KEY：复制 .env.example 为 .env 并填入 key，或改用 mock 模式"
        )

    model = os.environ.get("DASHSCOPE_MODEL", DEFAULT_MODEL)
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "")
    if schema is not None:
        system_prompt = (
            f"{system_prompt}\n\n输出必须是符合以下 JSON Schema 的单个 JSON 对象：\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    last_error: Exception | None = None
    for attempt in range(1, MAX_PARSE_RETRIES + 2):
        if base_url.rstrip("/").endswith("/apps/anthropic"):
            content, request_id = _call_anthropic_compatible(base_url, model, messages)
        elif base_url:
            content, request_id = _call_openai_compatible(base_url, model, messages)
        else:
            content, request_id = _call_dashscope(model, messages)
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning(
                "llm_json_retry attempt=%s request_id=%s error=%s", attempt, request_id, exc
            )
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": f"上一条输出不是合法 JSON（{exc}）。请只输出修正后的 JSON，不要任何其他文字。",
                }
            )
            continue
        if not isinstance(data, dict):
            last_error = LLMError(f"期望 JSON 对象，实际是 {type(data).__name__}")
            logger.warning(
                "llm_json_retry attempt=%s request_id=%s error=%s", attempt, request_id, last_error
            )
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {"role": "user", "content": "请输出一个 JSON 对象，不要数组或其他类型。"}
            )
            continue
        logger.info(
            "llm_ok model=%s prompt_version=%s request_id=%s attempt=%s",
            model,
            PROMPT_VERSION,
            request_id,
            attempt,
        )
        data["_meta"] = {
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "request_id": request_id,
            "mock": False,
        }
        return data

    raise LLMError(f"重试 {MAX_PARSE_RETRIES} 次后仍未得到合法 JSON：{last_error}")


def mock_chat_json(user_message: str) -> dict:
    """演示用固定回复（手册：Mock 数据必须明确标记，不得冒充真实结果）。

    返回的字段与 DesignSpec 对齐，可直接通过 contracts.DesignSpec 校验，
    用于无 API key 时跑通端到端流程。
    """
    logger.info("llm_mock 已收到请求（%d 字符），返回 Mock 数据", len(user_message))
    return {
        "_mock": True,
        "topology": "bandpass_filter",
        "frequency_band_hz": [2.3e9, 2.5e9],
        "parameters": [
            {"name": "line_width_um", "unit": "um", "lower": 5.0, "upper": 50.0, "nominal": 20.0},
            {"name": "gap_um", "unit": "um", "lower": 5.0, "upper": 40.0},
        ],
        "metrics": [
            {
                "name": "insertion_loss_db",
                "operator": "<=",
                "target": 2.0,
                "band_hz": [2.35e9, 2.45e9],
            },
            {"name": "s11_db", "operator": "<=", "target": -10.0, "band_hz": [2.35e9, 2.45e9]},
        ],
        "needs_confirmation": ["基板材料与厚度未确认", "端口定义未确认"],
        "note": "这是 Mock 数据（simulated），不是真实 LLM 解析结果",
    }

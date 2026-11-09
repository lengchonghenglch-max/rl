"""成员 C 负责：Day 2 百炼客户端与日志的测试（不访问网络）。"""

import json

import pytest

from rf_agent.adapters import llm
from rf_agent.contracts import DesignSpec


def test_has_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    assert llm.has_api_key() is False
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    assert llm.has_api_key() is True


def test_chat_json_raises_without_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    with pytest.raises(llm.MissingAPIKeyError):
        llm.chat_json("system", "user")


def test_mock_chat_json_passes_design_spec():
    """Mock 回复必须能通过 B 维护的 DesignSpec 合同。"""
    data = llm.mock_chat_json("2.4GHz 带通滤波器")
    assert data["_mock"] is True
    data.pop("_mock")
    data.pop("note")
    spec = DesignSpec(spec_id="spec-test", raw_request="2.4GHz 带通滤波器", **data)
    assert spec.topology == "bandpass_filter"
    assert spec.needs_confirmation  # 含糊字段必须追问，不许为空


def test_chat_json_retries_on_bad_json(monkeypatch):
    """第一次回非 JSON，第二次回合法 JSON：应重试成功且带 _meta。"""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    calls = {"n": 0}

    def fake_call(model, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return "这不是JSON", "req-1"
        return json.dumps({"topology": "inductor"}), "req-2"

    monkeypatch.setattr(llm, "_call_dashscope", fake_call)
    data = llm.chat_json("system", "user")
    assert calls["n"] == 2
    assert data["topology"] == "inductor"
    assert data["_meta"]["mock"] is False
    assert data["_meta"]["request_id"] == "req-2"


def test_chat_json_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    calls = {"n": 0}

    def fake_call(model, messages):
        calls["n"] += 1
        return "永远不是JSON", f"req-{calls['n']}"

    monkeypatch.setattr(llm, "_call_dashscope", fake_call)
    with pytest.raises(llm.LLMError):
        llm.chat_json("system", "user")
    assert calls["n"] == llm.MAX_PARSE_RETRIES + 1


def test_chat_json_rejects_non_object(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    calls = {"n": 0}

    def fake_call(model, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return "[1, 2, 3]", "req-1"
        return json.dumps({"ok": True}), "req-2"

    monkeypatch.setattr(llm, "_call_dashscope", fake_call)
    data = llm.chat_json("system", "user")
    assert data["ok"] is True

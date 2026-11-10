"""成员 C 负责：Day 3 状态机测试（全离线，走 Mock，不访问网络）。"""

import pytest

from rf_agent import agent, storage, tasks
from rf_agent.adapters import llm
from rf_agent.contracts import DesignSpec, Phase


@pytest.fixture
def db(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture(autouse=True)
def export_dir(tmp_path, monkeypatch):
    """任务单导出目录指向临时目录，测试不污染仓库。"""
    monkeypatch.setattr(tasks, "EXPORT_DIR", tmp_path / "incoming")


@pytest.fixture
def offline(monkeypatch):
    """强制 Mock 模式：无 key 且不读 .env。"""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)


RAW = "在 2.4 GHz 附近做一个带通滤波器，通带插损小于 2 dB"


def _run_to_confirm(db):
    state = agent.create_project("p1", RAW, db)
    assert state.phase == Phase.RETRIEVE
    return agent.run_until_stopped(state, db)


def test_flow_stops_at_confirm_gate(db, offline):
    state = _run_to_confirm(db)
    assert state.phase == Phase.CONFIRM
    # mock spec 2 项 + mock 证据 2 项
    assert len(state.pending_confirmations) >= 2


def test_approve_then_sample_then_submit_stop_at_ingest(db, offline):
    _run_to_confirm(db)
    state = agent.approve("p1", approver="测试工程师", db_path=db)
    assert state.pending_confirmations == []
    state = agent.run_until_stopped(state, db)
    assert state.phase == Phase.INGEST  # Day 8 才接入，必须停住而不是跳过
    batch = state.latest_sample_batch
    assert batch is not None and len(batch.candidates) == 5
    assert all(c.feasible for c in batch.candidates)


def test_restart_recovery_from_sqlite(db, offline):
    _run_to_confirm(db)
    agent.approve("p1", db_path=db)
    agent.run_until_stopped(storage.load_state("p1", db), db)
    restored = storage.load_state("p1", db)  # 模拟进程重启
    assert restored.phase == Phase.INGEST
    assert restored.latest_sample_batch is not None


def test_audit_events_recorded(db, offline):
    _run_to_confirm(db)
    state = agent.approve("p1", approver="测试工程师", db_path=db)
    agent.run_until_stopped(state, db)
    messages = " ".join(ev["message"] for ev in storage.list_events("p1", db))
    assert "收到原始需求" in messages
    assert "parse 完成 mode=mock" in messages
    assert "停在人工确认" in messages
    assert "批准" in messages
    assert "sample 完成" in messages
    assert "submit 完成" in messages


def test_no_pending_skips_confirm(db, offline, monkeypatch):
    """既无待确认项也无待确认证据时，应直接走到采样。"""
    spec = DesignSpec(
        spec_id="spec-x",
        raw_request=RAW,
        topology="inductor",
        frequency_band_hz=(1e9, 2e9),
        parameters=[{"name": "w_um", "unit": "um", "lower": 1.0, "upper": 10.0}],
        metrics=[{"name": "q_factor", "operator": ">=", "target": 10.0}],
        needs_confirmation=[],
    )
    monkeypatch.setattr(agent, "parse_requirement_text", lambda raw: (spec, "mock", None))
    monkeypatch.setattr(agent, "mock_retrieve", lambda state: [])
    state = agent.create_project("p2", RAW, db)
    state = agent.run_until_stopped(state, db)
    assert state.phase == Phase.INGEST


def test_node_error_goes_to_failed_terminal(db, offline, monkeypatch):
    def boom(state):
        raise RuntimeError("检索服务炸了")

    monkeypatch.setattr(agent, "mock_retrieve", boom)
    state = agent.create_project("p3", RAW, db)
    state = agent.run_until_stopped(state, db)
    assert state.phase == Phase.FAILED
    assert any("检索服务炸了" in e for e in state.errors)
    # 终态存档后恢复也必须是 FAILED
    assert storage.load_state("p3", db).phase == Phase.FAILED


def test_approve_rejected_outside_confirm(db, offline):
    agent.create_project("p4", RAW, db)  # phase=RETRIEVE，未到闸门
    with pytest.raises(ValueError):
        agent.approve("p4", db_path=db)

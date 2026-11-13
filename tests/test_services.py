"""成员 C 负责：Day 7 服务层真实闭环测试（全离线，走 mock 解析）。

验证 WorkflowService 不再自建平行状态，而是驱动 agent.py 状态机 +
storage.py 的 WorkflowState（单一状态来源），审批后自动走 sample -> submit。
"""

import json
import sqlite3

import pytest

from rf_agent import rag, storage, tasks
from rf_agent.adapters import llm
from rf_agent.contracts import DataLabel
from rf_agent.services import workflow as wf_module
from rf_agent.services.workflow import WorkflowService

RAW = "在 2.4 GHz 附近做一个带通滤波器，通带插损小于 2 dB"


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(wf_module, "DB_PATH", tmp_path / "ui.db")
    monkeypatch.setattr(tasks, "EXPORT_DIR", tmp_path / "incoming")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)  # 离线：解析走 mock
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    return WorkflowService()


def _count_tasks(db_path) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]


def test_create_run_stops_at_confirm_gate(service):
    run = service.create_run(RAW)
    assert run["phase"] == "confirm"
    assert run["status"] == "waiting_approval"
    assert run["data_kind"] == "simulated"
    assert len(run["pending_confirmations"]) >= 2
    assert run["spec"] is not None and run["spec"]["raw_request"] == RAW
    expected = [e.model_dump(mode="json") for e in rag.retrieve(RAW, k=5)]
    assert run["evidence"] == expected


def test_state_single_source_in_workflow_state(service):
    """run 状态必须能从 storage.load_state 读回——证明没有第二套状态表。"""
    run = service.create_run(RAW)
    state = storage.load_state(run["run_id"], wf_module.DB_PATH)
    assert state.phase.value == run["phase"]
    assert state.design_spec is not None


def test_create_run_displays_cited_rag_evidence(service):
    query = "What does the LHS in LHS expansion algorithm enable when adding samples to an existing LHS design?"

    run = service.create_run(query)

    assert run["request"] == query
    assert run["evidence"]
    assert run["evidence"][0]["document_id"] == "a05"
    assert run["evidence"][0]["page"] == 1
    assert run["evidence"][0]["chunk_id"]
    assert run["evidence"][0]["source_type"] == "knowledge_suggestion"


def test_approve_runs_sample_submit_and_exports_tasks(service):
    run = service.create_run(RAW)
    approved = service.resolve_approval(run["run_id"], "approve", "参数范围合理", "key-0001")
    assert approved["phase"] == "ingest"
    assert approved["status"] == "tasks_exported"
    batch = approved["sample_batch"]
    assert batch is not None and len(batch["candidates"]) == 5
    assert batch["sampler"] == "lhs"
    saved = storage.load_state(run["run_id"], wf_module.DB_PATH)
    assert saved.search_space is not None and saved.search_space.approved_by == "前端UI"
    # 任务单真的入库 + 落盘（Day 6 成果被前端链路复用）
    assert _count_tasks(wf_module.DB_PATH) == 5
    csv_files = list(tasks.EXPORT_DIR.glob("tasks_*.csv"))
    assert len(csv_files) == 1


def test_reject_goes_failed_terminal(service):
    run = service.create_run(RAW)
    rejected = service.resolve_approval(run["run_id"], "reject", "基板未定，先退回", "key-0002")
    assert rejected["phase"] == "failed"
    assert rejected["status"] == "rejected"
    assert rejected["approval"]["action"] == "reject"
    assert rejected["sample_batch"] is None


def test_idempotency_key_replay_and_conflict(service):
    run = service.create_run(RAW)
    first = service.resolve_approval(run["run_id"], "approve", "OK", "key-0003")
    replay = service.resolve_approval(run["run_id"], "approve", "OK", "key-0003")
    assert replay["phase"] == first["phase"] == "ingest"
    with pytest.raises(ValueError):
        service.resolve_approval(run["run_id"], "reject", "换主意了", "key-0003")


def test_approval_only_at_confirm_gate(service):
    run = service.create_run(RAW)
    service.resolve_approval(run["run_id"], "approve", "OK", "key-0004")
    with pytest.raises(ValueError):
        service.resolve_approval(run["run_id"], "approve", "再批一次", "key-0005")


def test_list_runs_shape(service):
    run = service.create_run(RAW)
    rows = service.list_runs()
    assert any(r["run_id"] == run["run_id"] for r in rows)
    row = next(r for r in rows if r["run_id"] == run["run_id"])
    assert set(row) == {
        "run_id",
        "created_at",
        "updated_at",
        "phase",
        "status",
        "data_kind",
        "request",
    }


def test_report_contains_full_chain(service):
    run = service.create_run(RAW)
    service.resolve_approval(run["run_id"], "approve", "参数范围合理", "key-0006")
    md = service.report(run["run_id"], "md").decode()
    for section in ("设计规格", "审批", "采样与任务", "审计事件"):
        assert section in md
    assert "导出任务 5 个" in md
    assert "UI 批准" in md


def test_ingest_returns_declared_label(service):
    run = service.create_run(RAW)
    csv_bytes = b"frequency_hz,s11_db,s21_db\n2.4e9,-15.0,-1.5\n"
    info = service.ingest(run["run_id"], "measured.csv", csv_bytes, DataLabel.SIMULATION)
    assert info["data_kind"] == "simulation"  # 如实回显，不许把上传数据洗成 simulated
    arts = service.artifacts(run["run_id"])
    assert arts[0]["data_kind"] == "simulation"
    # 下载回读一致
    name, content, media = service.download(run["run_id"], arts[0]["artifact_id"])
    assert (name, content, media) == ("measured.csv", csv_bytes, "text/csv")


def test_snapshot_missing_run_raises_keyerror(service):
    with pytest.raises(KeyError):
        service.snapshot("run-nonexistent")


def _approved_run(service):
    run = service.create_run(RAW)
    return service.resolve_approval(run["run_id"], "approve", "范围已确认", "manual-loop-approval")


def test_manual_summary_results_fit_gp_and_approve_next_batch(service):
    run = _approved_run(service)
    sheet_name, task_sheet = service.task_sheet(run["run_id"])
    assert sheet_name.endswith(".csv")
    assert b"task_id" in task_sheet
    candidates = run["sample_batch"]["candidates"][:2]
    lines = ["task_id,status,simulator_version,insertion_loss_db,s11_db,error"]
    for index, candidate in enumerate(candidates):
        lines.append(
            f"task-{candidate['candidate_id']},success,ADS-2026,{1.0 + index * 0.2},-15.0,"
        )

    imported = service.ingest_task_results(run["run_id"], "summary.csv", "\n".join(lines).encode())

    assert imported["mode"] == "summary"
    assert imported["tasks"] == 2
    recommendation = service.recommend_next_batch(run["run_id"], batch_size=2)
    assert len(recommendation["candidates"]) == 2
    assert recommendation["metric"] == "insertion_loss_db"
    approved = service.approve_next_batch(run["run_id"], True, "继续优化")
    assert approved["approved"] is True
    assert len(approved["sample_batch"]["candidates"]) == 2
    assert service.snapshot(run["run_id"])["phase"] == "ingest"


def test_manual_curve_results_extract_metrics_in_requested_band(service):
    run = _approved_run(service)
    candidate = run["sample_batch"]["candidates"][0]
    content = (
        "task_id,frequency_hz,s11_db,s21_db,simulator_version\n"
        f"task-{candidate['candidate_id']},2400000000,-18,-1.2,HFSS-1\n"
        f"task-{candidate['candidate_id']},2450000000,-12,-1.8,HFSS-1\n"
    ).encode()

    imported = service.ingest_task_results(run["run_id"], "curves.csv", content)

    assert imported["mode"] == "curve"
    assert imported["tasks"] == 1
    with service._connect() as db:
        result = db.execute(
            "SELECT metrics_json FROM ui_task_results WHERE run_id=?", (run["run_id"],)
        ).fetchone()
    metrics = json.loads(result["metrics_json"])
    assert {item["name"]: item["value"] for item in metrics} == {
        "insertion_loss_db": 1.8,
        "s11_db": -12.0,
    }

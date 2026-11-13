"""应用服务层：UI / HTTP 的唯一入口，内部驱动 agent.py 真实状态机。

Day 7 真实闭环改造：
- 状态唯一来源是 storage.py 的 workflow_state 表（WorkflowState 快照），
  本服务不再维护平行的 ui_runs 状态表；
- create_run -> agent.create_project + run_until_stopped（parse/retrieve/confirm 闸门）；
- approve -> agent.approve + run_until_stopped（sample -> submit 导出任务单 -> 停在 INGEST）；
- reject -> 记审计事件并转 FAILED 终态；
- ui_artifacts（上传文件）与 ui_approval_log（审批理由 + 幂等键）是 UI 侧补充记录，
  不与领域状态重叠。
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

from rf_agent import agent, rag, storage
from rf_agent.contracts import DataLabel, Phase, WorkflowState
from rf_agent.services import mock_provider as mock

DB_PATH = Path("data/rf_agent.db")
MAX_UPLOAD = 5_000_000
ALLOWED_EXTENSIONS = {".csv", ".json"}


class WorkflowService:
    """Application boundary: all UI and HTTP operations go through this service."""

    def _connect(self) -> sqlite3.Connection:
        storage.init_db(DB_PATH)  # 幂等：workflow_state / events / tasks
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(DB_PATH, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute(
            """CREATE TABLE IF NOT EXISTS ui_artifacts (
            artifact_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, name TEXT NOT NULL,
            content BLOB NOT NULL, media_type TEXT NOT NULL, sha256 TEXT NOT NULL,
            data_kind TEXT NOT NULL, created_at TEXT NOT NULL)"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS ui_approval_log (
            idempotency_key TEXT PRIMARY KEY, run_id TEXT NOT NULL,
            action TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL)"""
        )
        return connection

    # ------------------------------------------------------------------
    # 领域状态：全部来自 storage.py 的 WorkflowState
    # ------------------------------------------------------------------

    def _latest_approval(self, run_id: str) -> dict | None:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT action, reason, created_at FROM ui_approval_log "
                "WHERE run_id=? ORDER BY rowid DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return dict(row) if row else None

    def _status(self, state: WorkflowState, approval: dict | None) -> str:
        if state.phase == Phase.CONFIRM and state.pending_confirmations:
            return "waiting_approval"
        if state.phase == Phase.FAILED:
            return "rejected" if approval and approval["action"] == "reject" else "failed"
        if state.phase == Phase.INGEST:
            return "tasks_exported"
        return state.phase.value

    def _snapshot(self, state: WorkflowState) -> dict:
        """WorkflowState -> UI 快照 dict（字段形状与旧版一致，页面不需要改）。"""
        approval = self._latest_approval(state.project_id)
        spec = state.design_spec
        batch = state.latest_sample_batch
        evidence = (
            [e.model_dump(mode="json") for e in rag.retrieve(spec.raw_request, k=5)] if spec else []
        )
        return {
            "run_id": state.project_id,
            "created_at": (spec.created_at if spec else state.updated_at).isoformat(),
            "updated_at": state.updated_at.isoformat(),
            "request": spec.raw_request if spec else "",
            "phase": state.phase.value,
            "status": self._status(state, approval),
            "data_kind": DataLabel.SIMULATED.value,
            "spec": spec.model_dump(mode="json") if spec else None,
            "evidence": evidence,
            "search_space": state.search_space.model_dump(mode="json") if state.search_space else None,
            "pending_confirmations": list(state.pending_confirmations),
            "approval": approval,
            "sample_batch": batch.model_dump(mode="json") if batch else None,
            "errors": list(state.errors),
        }

    def create_run(self, request: str) -> dict:
        """真实解析链路：有 LLM key 走 LLM，没有则自动落到 mock（agent.parse 内部决定）。"""
        run_id = f"run-{uuid.uuid4().hex[:10]}"
        state = agent.create_project(run_id, request, DB_PATH)
        state = agent.run_until_stopped(state, DB_PATH)
        return self._snapshot(state)

    def snapshot(self, run_id: str) -> dict:
        return self._snapshot(storage.load_state(run_id, DB_PATH))

    def list_runs(self) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT state_json FROM workflow_state ORDER BY updated_at DESC LIMIT 100"
            ).fetchall()
        keys = ("run_id", "created_at", "updated_at", "phase", "status", "data_kind", "request")
        return [
            {k: snap[k] for k in keys}
            for snap in (
                self._snapshot(WorkflowState.model_validate_json(row["state_json"])) for row in rows
            )
        ]

    def resolve_approval(self, run_id: str, action: str, reason: str, key: str) -> dict:
        state = storage.load_state(run_id, DB_PATH)  # 不存在 -> KeyError
        with closing(self._connect()) as db, db:
            previous = db.execute(
                "SELECT action FROM ui_approval_log WHERE idempotency_key=?", (key,)
            ).fetchone()
            if previous:
                if previous[0] != action:
                    raise ValueError("idempotency key already used for another action")
                return self._snapshot(state)
            if state.phase != Phase.CONFIRM or not state.pending_confirmations:
                raise ValueError("run is not awaiting approval")
            db.execute(
                "INSERT INTO ui_approval_log VALUES(?,?,?,?,?)",
                (key, run_id, action, reason, mock.iso_now()),
            )
        if action == "approve":
            storage.log_event(run_id, Phase.CONFIRM, f"UI 批准，理由：{reason}", DB_PATH)
            state = agent.approve(run_id, approver="前端UI", db_path=DB_PATH)
            state = agent.run_until_stopped(state, DB_PATH)  # sample -> submit -> 停在 INGEST
        else:
            storage.log_event(run_id, Phase.CONFIRM, f"UI 退回，理由：{reason}", DB_PATH)
            state.transition(Phase.FAILED)
            storage.save_state(state, DB_PATH)
        return self._snapshot(state)

    # ------------------------------------------------------------------
    # 文件上传 / artifact（UI 侧补充记录）
    # ------------------------------------------------------------------

    def ingest(
        self,
        run_id: str,
        filename: str,
        content: bytes,
        data_kind: DataLabel = DataLabel.SIMULATION,
    ) -> dict:
        storage.load_state(run_id, DB_PATH)  # run 必须存在
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError("only CSV and JSON files are accepted")
        if not content or len(content) > MAX_UPLOAD:
            raise ValueError("file must be between 1 byte and 5 MB")
        try:
            if suffix == ".csv":
                rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
                if not rows or not {"frequency_hz", "s11_db", "s21_db"}.issubset(rows[0]):
                    raise ValueError("CSV requires frequency_hz, s11_db, and s21_db columns")
            else:
                data = json.loads(content)
                if not isinstance(data, list) or not data:
                    raise ValueError("JSON must be a non-empty array of result rows")
                rows = data
                if not {"frequency_hz", "s11_db", "s21_db"}.issubset(rows[0]):
                    raise ValueError("JSON rows require frequency_hz, s11_db, and s21_db")
        except (UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
            raise ValueError(f"invalid {suffix[1:].upper()} file") from exc
        for row in rows[:100_000]:
            try:
                [float(row[k]) for k in ("frequency_hz", "s11_db", "s21_db")]
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError("result values must be numeric") from exc
        digest = mock.sha256(content)
        artifact_id = f"artifact-{uuid.uuid4().hex[:10]}"
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO ui_artifacts VALUES(?,?,?,?,?,?,?,?)",
                (
                    artifact_id,
                    run_id,
                    Path(filename).name,
                    content,
                    "text/csv" if suffix == ".csv" else "application/json",
                    digest,
                    data_kind.value,
                    mock.iso_now(),
                ),
            )
        storage.log_event(
            run_id,
            Phase.INGEST,
            f"上传结果文件 {filename}（{len(rows)} 行，标签 {data_kind.value}，SHA256 {digest[:12]}）",
            DB_PATH,
        )
        return {
            "artifact_id": artifact_id,
            "run_id": run_id,
            "name": Path(filename).name,
            "sha256": digest,
            "rows": len(rows),
            "data_kind": data_kind.value,  # 如实回显上传者声明的标签（修复旧版恒返 simulated）
        }

    def artifacts(self, run_id: str) -> list[dict]:
        storage.load_state(run_id, DB_PATH)
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT artifact_id, name, sha256, data_kind, created_at "
                "FROM ui_artifacts WHERE run_id=? ORDER BY created_at DESC",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def download(self, run_id: str, artifact_id: str) -> tuple[str, bytes, str]:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT name, content, media_type FROM ui_artifacts "
                "WHERE run_id=? AND artifact_id=?",
                (run_id, artifact_id),
            ).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        return row[0], row[1], row[2]

    def mock_artifact(self) -> tuple[str, bytes, str]:
        return mock.artifacts()

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------

    def report(self, run_id: str, format: str = "md") -> bytes:
        snap = self.snapshot(run_id)
        arts = self.artifacts(run_id)
        if format == "csv":
            out = io.StringIO()
            writer = csv.writer(out)
            writer.writerow(["run_id", "artifact", "sha256", "data_kind"])
            for item in arts:
                writer.writerow([run_id, item["name"], item["sha256"], item["data_kind"]])
            return out.getvalue().encode()

        n_tasks = 0
        batch = snap["sample_batch"]
        if batch:
            with closing(self._connect()) as db:
                n_tasks = db.execute(
                    "SELECT COUNT(*) FROM tasks WHERE batch_id=?", (batch["batch_id"],)
                ).fetchone()[0]

        lines = [
            f"# RF AI Agent Report: {run_id}",
            "",
            f"- Status: {snap['status']}",
            f"- Phase: {snap['phase']}",
            f"- data_kind: {snap['data_kind']}",
            f"- 原始需求: {snap['request']}",
            "",
        ]
        spec = snap["spec"]
        if spec:
            lines += [
                "## 设计规格",
                f"- 拓扑: {spec['topology']} · 频段: {spec['frequency_band_hz']}",
                *[
                    f"- 参数 {p['name']}: [{p['lower']}, {p['upper']}] {p['unit']}"
                    for p in spec["parameters"]
                ],
                *[f"- 指标 {m['name']} {m['operator']} {m['target']}" for m in spec["metrics"]],
                "",
            ]
        if snap["approval"]:
            a = snap["approval"]
            lines += ["## 审批", f"- {a['action']}（{a['created_at']}）：{a['reason']}", ""]
        if batch:
            batch_line = (
                f"- 批次 {batch['batch_id']}（{batch['sampler']}, seed={batch['seed']}）："
                f"{len(batch['candidates'])} 个候选点，导出任务 {n_tasks} 个"
            )
            lines += ["## 采样与任务", batch_line, ""]
        lines += ["## Artifacts"]
        lines += [
            f"- {item['name']} | SHA-256: `{item['sha256']}` | data_kind: `{item['data_kind']}`"
            for item in arts
        ] or ["- No uploaded artifacts"]
        events = storage.list_events(run_id, DB_PATH)
        if events:
            lines += ["", "## 审计事件"]
            lines += [f"- [{ev['phase']}] {ev['message']}" for ev in events]
        return "\n".join(lines).encode()


workflow = WorkflowService()

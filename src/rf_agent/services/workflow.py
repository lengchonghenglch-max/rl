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
import math
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

from rf_agent import agent, rag, storage
from rf_agent import tasks as task_module
from rf_agent.contracts import (
    DataLabel,
    MetricValue,
    Phase,
    SampleBatch,
    SimulationResult,
    TaskStatus,
    WorkflowState,
)
from rf_agent.services import mock_provider as mock

DB_PATH = Path("data/rf_agent.db")
MAX_UPLOAD = 5_000_000
ALLOWED_EXTENSIONS = {".csv", ".json"}


def _metric_unit(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(("_hz", "_khz", "_mhz", "_ghz")):
        return "Hz"
    if lowered.endswith("_db"):
        return "dB"
    return "unitless"


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
        connection.execute(
            """CREATE TABLE IF NOT EXISTS ui_task_results (
            task_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, metrics_json TEXT NOT NULL,
            simulator_version TEXT NOT NULL, source TEXT NOT NULL, status TEXT NOT NULL,
            error TEXT, artifact_id TEXT, created_at TEXT NOT NULL)"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS ui_run_tasks (
            run_id TEXT NOT NULL, task_id TEXT NOT NULL, PRIMARY KEY(run_id, task_id))"""
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
            "acquisition_batch": state.latest_acquisition.model_dump(mode="json") if state.latest_acquisition else None,
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
            self._register_batch_tasks(run_id, state)
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

    def task_sheet(self, run_id: str) -> tuple[str, bytes]:
        state = storage.load_state(run_id, DB_PATH)
        if state.latest_sample_batch is None or state.design_spec is None:
            raise ValueError("run has no task batch to export")
        _, _, csv_path, _ = task_module.export_tasks(state.latest_sample_batch, state.design_spec, DB_PATH)
        return csv_path.name, csv_path.read_bytes()

    def result_template(self, run_id: str, mode: str = "summary") -> tuple[str, bytes]:
        """Return a task-linked summary or S-parameter curve CSV template."""
        state = storage.load_state(run_id, DB_PATH)
        batch = state.latest_sample_batch
        if batch is None:
            raise ValueError("run has no exported tasks")
        out = io.StringIO(newline="")
        writer = csv.writer(out)
        if mode == "summary":
            metrics = [metric.name for metric in (state.design_spec.metrics if state.design_spec else [])]
            writer.writerow(["task_id", "status", "simulator_version", *metrics, "error"])
            for candidate in batch.candidates:
                task_id = f"task-{candidate.candidate_id}"
                writer.writerow([task_id, "success", "", *("" for _ in metrics), ""])
            return f"{run_id}_summary_results_template.csv", out.getvalue().encode("utf-8-sig")
        if mode == "curve":
            writer.writerow(["task_id", "frequency_hz", "s11_db", "s21_db", "simulator_version"])
            for candidate in batch.candidates:
                writer.writerow([f"task-{candidate.candidate_id}", "", "", "", ""])
            return f"{run_id}_curve_results_template.csv", out.getvalue().encode("utf-8-sig")
        raise ValueError("mode must be summary or curve")

    def ingest_task_results(self, run_id: str, filename: str, content: bytes) -> dict:
        """Import either wide summary rows or task_id-tagged S-parameter sweep rows."""
        state = storage.load_state(run_id, DB_PATH)
        if state.design_spec is None or state.latest_sample_batch is None:
            raise ValueError("run must have a design spec and an exported task batch")
        if not content or len(content) > MAX_UPLOAD:
            raise ValueError("file must be between 1 byte and 5 MB")
        if Path(filename).suffix.lower() != ".csv":
            raise ValueError("task result import currently accepts CSV files")
        try:
            rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
        except (UnicodeDecodeError, csv.Error) as exc:
            raise ValueError("invalid CSV file") from exc
        if not rows:
            raise ValueError("CSV has no result rows")
        with closing(self._connect()) as db:
            known_tasks = {
                row[0]
                for row in db.execute("SELECT task_id FROM ui_run_tasks WHERE run_id=?", (run_id,))
            }
        known_tasks.update(f"task-{candidate.candidate_id}" for candidate in state.latest_sample_batch.candidates)
        task_ids = {row.get("task_id", "").strip() for row in rows}
        if "" in task_ids or not task_ids.issubset(known_tasks):
            unknown = sorted(task_ids - known_tasks)
            raise ValueError(f"CSV task_id is missing or unknown: {unknown[:3]}")
        columns = set(rows[0])
        curve_mode = {"frequency_hz", "s11_db", "s21_db"}.issubset(columns)
        metric_specs = {metric.name: metric for metric in state.design_spec.metrics}
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["task_id"].strip(), []).append(row)
        parsed: dict[str, tuple[list[MetricValue], str, str, str | None]] = {}
        if curve_mode:
            for task_id, task_rows in grouped.items():
                points = []
                versions = {r.get("simulator_version", "").strip() for r in task_rows if r.get("simulator_version", "").strip()}
                if len(versions) > 1:
                    raise ValueError(f"task {task_id} has multiple simulator versions")
                for row in task_rows:
                    try:
                        points.append((float(row["frequency_hz"]), float(row["s11_db"]), float(row["s21_db"])))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"task {task_id} has non-numeric curve values") from exc
                if not all(all(math.isfinite(value) for value in point) and point[0] > 0 for point in points):
                    raise ValueError(f"task {task_id} curve values must be finite and frequency positive")
                metrics = []
                for name, spec in metric_specs.items():
                    low, high = spec.band_hz or state.design_spec.frequency_band_hz
                    in_band = [point for point in points if low <= point[0] <= high]
                    if not in_band:
                        continue
                    if name == "s11_db":
                        value = max(point[1] for point in in_band)
                    elif name == "insertion_loss_db":
                        value = max(-point[2] for point in in_band)
                    else:
                        continue
                    metrics.append(MetricValue(name=name, value=value, unit=_metric_unit(name), band_low_hz=low, band_high_hz=high))
                if not metrics:
                    raise ValueError(f"task {task_id} curve has no points in the requested metric bands")
                parsed[task_id] = (metrics, next(iter(versions), ""), "success", None)
        else:
            required = {"task_id", "status", "simulator_version"}
            if not required.issubset(columns):
                raise ValueError("summary CSV requires task_id, status, simulator_version, and metric columns")
            for task_id, task_rows in grouped.items():
                if len(task_rows) != 1:
                    raise ValueError(f"summary CSV must contain one row per task: {task_id}")
                row = task_rows[0]
                status = row.get("status", "").strip().lower()
                error = row.get("error", "").strip() or None
                if status not in {"success", "failed"}:
                    raise ValueError(f"task {task_id} status must be success or failed")
                metrics = []
                if status == "success":
                    for name, spec in metric_specs.items():
                        raw = row.get(name, "").strip()
                        if not raw:
                            raise ValueError(f"task {task_id} missing metric {name}")
                        try:
                            value = float(raw)
                        except ValueError as exc:
                            raise ValueError(f"task {task_id} metric {name} must be numeric") from exc
                        if not math.isfinite(value):
                            raise ValueError(f"task {task_id} metric {name} must be finite")
                        metrics.append(MetricValue(name=name, value=value, unit=_metric_unit(name), band_low_hz=spec.band_hz[0] if spec.band_hz else None, band_high_hz=spec.band_hz[1] if spec.band_hz else None))
                    if not metrics:
                        raise ValueError(f"task {task_id} has no supported metric columns")
                elif not error:
                    error = "Manual simulation marked failed"
                parsed[task_id] = (metrics, row.get("simulator_version", "").strip(), status, error)

        digest = mock.sha256(content)
        artifact_id = f"artifact-{uuid.uuid4().hex[:10]}"
        now = mock.iso_now()
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO ui_artifacts VALUES(?,?,?,?,?,?,?,?)",
                (artifact_id, run_id, Path(filename).name, content, "text/csv", digest, DataLabel.SIMULATION.value, now),
            )
            for task_id, (metrics, version, status, error) in parsed.items():
                db.execute(
                    """INSERT INTO ui_task_results VALUES(?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(task_id) DO UPDATE SET metrics_json=excluded.metrics_json,
                    simulator_version=excluded.simulator_version, source=excluded.source,
                    status=excluded.status, error=excluded.error, artifact_id=excluded.artifact_id,
                    created_at=excluded.created_at WHERE ui_task_results.run_id=excluded.run_id""",
                    (task_id, run_id, json.dumps([m.model_dump(mode="json") for m in metrics]), version, DataLabel.SIMULATION.value, status, error, artifact_id, now),
                )
        storage.log_event(run_id, Phase.INGEST, f"导入人工仿真结果 {filename}：{len(parsed)} 个任务，模式={'curve' if curve_mode else 'summary'}，SHA256 {digest[:12]}", DB_PATH)
        return {"artifact_id": artifact_id, "run_id": run_id, "tasks": len(parsed), "mode": "curve" if curve_mode else "summary", "sha256": digest}

    def recommend_next_batch(self, run_id: str, batch_size: int = 4, kind: str = "ei") -> dict:
        """Fit on imported real results and persist a human-reviewable acquisition batch."""
        from rf_agent.active_learning import fit_gp, select_next_batch

        state = storage.load_state(run_id, DB_PATH)
        if state.design_spec is None or state.search_space is None:
            raise ValueError("run has no approved design space")
        metric = state.design_spec.metrics[0].name
        with closing(self._connect()) as db:
            rows = db.execute(
                """SELECT r.*, t.payload_json FROM ui_task_results r
                JOIN tasks t ON t.task_id=r.task_id
                WHERE r.run_id=? AND r.source=? AND r.status='success'""",
                (run_id, DataLabel.SIMULATION.value),
            ).fetchall()
        results = []
        params = {}
        for row in rows:
            task = json.loads(row["payload_json"])
            task_metrics = [MetricValue.model_validate(item) for item in json.loads(row["metrics_json"])]
            if not any(item.name == metric for item in task_metrics):
                continue
            results.append(SimulationResult(task_id=row["task_id"], status=TaskStatus.SUCCESS, metrics=task_metrics, source=DataLabel.SIMULATION, simulator_version=row["simulator_version"]))
            values = task["parameter_values"]
            params[row["task_id"]] = {
                parameter.name: (values[parameter.name] - parameter.lower) / (parameter.upper - parameter.lower)
                for parameter in state.search_space.parameters
            }
        model = fit_gp(results, metric, params)
        batch = select_next_batch(model, state.search_space, budget_remaining=100, batch_size=batch_size, kind=kind)
        if not batch.candidates:
            raise ValueError("no feasible acquisition candidates found")
        state.latest_acquisition = batch
        state.transition(Phase.ACQUIRE)
        storage.save_state(state, DB_PATH)
        storage.log_event(run_id, Phase.ACQUIRE, f"GP {model.version} 基于 {len(results)} 个真实仿真点生成 {len(batch.candidates)} 个 {kind.upper()} 建议，等待工程师批准", DB_PATH)
        return {"model_version": model.version, "metric": metric, **batch.model_dump(mode="json")}

    def _register_batch_tasks(self, run_id: str, state: WorkflowState) -> None:
        if state.latest_sample_batch is None:
            return
        with closing(self._connect()) as db, db:
            db.executemany(
                "INSERT OR IGNORE INTO ui_run_tasks VALUES(?,?)",
                [(run_id, f"task-{candidate.candidate_id}") for candidate in state.latest_sample_batch.candidates],
            )

    def approve_next_batch(self, run_id: str, approve: bool, reason: str = "") -> dict:
        state = storage.load_state(run_id, DB_PATH)
        acquisition = state.latest_acquisition
        if state.phase != Phase.ACQUIRE or acquisition is None:
            raise ValueError("run has no acquisition batch awaiting review")
        if not approve:
            storage.log_event(run_id, Phase.ACQUIRE, f"工程师拒绝下一批建议：{reason}", DB_PATH)
            state.transition(Phase.DONE)
            storage.save_state(state, DB_PATH)
            return {"approved": False, "phase": state.phase.value}
        if state.design_spec is None:
            raise ValueError("run has no design spec")
        sample_batch = SampleBatch(batch_id=acquisition.batch_id, design_spec_id=acquisition.design_spec_id, seed=42, sampler=acquisition.candidates[0].candidate.sampler, candidates=[item.candidate for item in acquisition.candidates], created_at=acquisition.created_at)
        state.latest_sample_batch = sample_batch
        tasks, n_new, csv_path, json_path = task_module.export_tasks(sample_batch, state.design_spec, DB_PATH)
        self._register_batch_tasks(run_id, state)
        state.transition(Phase.INGEST)
        storage.save_state(state, DB_PATH)
        storage.log_event(run_id, Phase.ACQUIRE, f"工程师批准下一批：导出 {len(tasks)} 个任务（新增 {n_new}），{reason}", DB_PATH)
        return {"approved": True, "phase": state.phase.value, "sample_batch": sample_batch.model_dump(mode="json"), "csv_path": str(csv_path), "json_path": str(json_path)}

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
        with closing(self._connect()) as db:
            result_rows = db.execute(
                "SELECT task_id, status, metrics_json, simulator_version FROM ui_task_results WHERE run_id=? ORDER BY created_at",
                (run_id,),
            ).fetchall()
        lines += ["", "## 仿真结果"]
        lines += [
            f"- {row['task_id']} | {row['status']} | {row['simulator_version'] or 'version unspecified'} | "
            + ", ".join(f"{metric['name']}={metric['value']} {metric['unit']}" for metric in json.loads(row["metrics_json"]))
            for row in result_rows
        ] or ["- No task results imported"]
        if snap["acquisition_batch"]:
            acq = snap["acquisition_batch"]
            lines += ["", "## 下一轮建议", f"- 模型 {acq['model_version']} · {len(acq['candidates'])} 个候选点，待工程师批准"]
        events = storage.list_events(run_id, DB_PATH)
        if events:
            lines += ["", "## 审计事件"]
            lines += [f"- [{ev['phase']}] {ev['message']}" for ev in events]
        return "\n".join(lines).encode()


workflow = WorkflowService()

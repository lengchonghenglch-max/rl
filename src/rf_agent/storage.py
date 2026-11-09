"""成员 C 负责：SQLite 数据入库与追溯（手册模块 6 数据表）。

Day 3 先实现工作流存档两件套：
- workflow_state：每个项目一行，存 WorkflowState 的 JSON 快照（断点恢复用）
- events：流水账，每个节点发生了什么（审计用，只增不改）

runs / parameters / artifacts / metrics 表在 Day 8（仿真结果入库）时再建。
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from rf_agent.contracts import Phase, SimulationResult, SimulationTask, WorkflowState

DEFAULT_DB = Path("data/rf_agent.db")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DEFAULT_DB) -> None:
    """建表（幂等，可重复调用）。"""
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS workflow_state (
              project_id TEXT PRIMARY KEY,
              state_json TEXT NOT NULL,
              phase      TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
              event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
              project_id TEXT NOT NULL,
              phase      TEXT NOT NULL,
              message    TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            """
        )


def save_state(state: WorkflowState, db_path: Path = DEFAULT_DB) -> None:
    """把记事本快照写入/覆盖 projects 行，并记一条事件。"""
    now = datetime.now(UTC).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO workflow_state(project_id, state_json, phase, updated_at)
               VALUES(?,?,?,?)
               ON CONFLICT(project_id) DO UPDATE SET
                 state_json=excluded.state_json,
                 phase=excluded.phase,
                 updated_at=excluded.updated_at""",
            (state.project_id, state.model_dump_json(), state.phase.value, now),
        )


def load_state(project_id: str, db_path: Path = DEFAULT_DB) -> WorkflowState:
    """从存档恢复记事本；不存在时抛 KeyError。"""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT state_json FROM workflow_state WHERE project_id=?", (project_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"项目 {project_id} 不存在存档")
    return WorkflowState.model_validate_json(row["state_json"])


def log_event(project_id: str, phase: Phase, message: str, db_path: Path = DEFAULT_DB) -> None:
    """流水账：只追加，永不修改/删除（手册审计要求）。"""
    now = datetime.now(UTC).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO events(project_id, phase, message, created_at) VALUES(?,?,?,?)",
            (project_id, phase.value, message, now),
        )


def list_events(project_id: str, db_path: Path = DEFAULT_DB) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT phase, message, created_at FROM events WHERE project_id=? ORDER BY event_id",
            (project_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# 以下为 Day 8 仿真数据入库（runs/parameters/artifacts/metrics），暂未实现
# ---------------------------------------------------------------------------


def save_task(task: SimulationTask, db_path: Path = DEFAULT_DB) -> None:
    raise NotImplementedError("Day 6-8 实现：仿真任务入库")


def save_result(result: SimulationResult, db_path: Path = DEFAULT_DB) -> None:
    """入库并保留 source 标签；raw_file 记录路径与 SHA256。"""
    raise NotImplementedError("Day 8 实现：仿真结果入库")


def load_results(design_id: str, db_path: Path = DEFAULT_DB) -> list[SimulationResult]:
    """给 B 的 fit_gp 使用；必须原样带出 source 标签。"""
    raise NotImplementedError("Day 9 实现：结果查询")

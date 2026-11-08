"""成员 C 负责：SQLite 数据入库与追溯（手册模块 6 数据表）。

表结构按手册建议：runs / parameters / artifacts / metrics，
另加 workflow_state 用于任务恢复（Day 11）。
"""

from __future__ import annotations

from pathlib import Path

from rf_agent.contracts import SimulationResult, SimulationTask

DEFAULT_DB = Path("data/rf_agent.db")


def init_db(db_path: Path = DEFAULT_DB) -> None:
    """建表（手册 SQL：runs/parameters/artifacts/metrics + workflow_state）。"""
    raise NotImplementedError("成员 C 待实现")


def save_task(task: SimulationTask, db_path: Path = DEFAULT_DB) -> None:
    raise NotImplementedError("成员 C 待实现")


def save_result(result: SimulationResult, db_path: Path = DEFAULT_DB) -> None:
    """入库并保留 source 标签；raw_file 记录路径与 SHA256。"""
    raise NotImplementedError("成员 C 待实现")


def load_results(design_id: str, db_path: Path = DEFAULT_DB) -> list[SimulationResult]:
    """给 B 的 fit_gp 使用；必须原样带出 source 标签。"""
    raise NotImplementedError("成员 C 待实现")

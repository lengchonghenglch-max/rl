"""成员 C 负责：仿真 Adapter（手册模块 5 / Day 6 mock 版，Day 8 接结果导入）。

当前阶段：只做 Mock。真实 UltraEM 接入前必须完成手册
"厂商确认问题清单"，并用 ManualUltraEMAdapter 由人执行导入。
所有 mock 产物 source 必须是 DataLabel.SIMULATED，绝不允许标成 SIMULATION。
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from rf_agent.contracts import (
    DataLabel,
    MetricValue,
    SimulationResult,
    SimulationTask,
    TaskStatus,
    utc_now,
)
from rf_agent.logging_setup import get_logger

logger = get_logger("adapters.simulator")

SIMULATED_DIR = Path("data/simulated")


def submit(task: SimulationTask) -> str:
    """提交任务，返回 run_id。幂等：run_id 由 task_id 确定性派生，重复提交不变。"""
    run_id = f"run-{task.task_id}"
    logger.info("mock submit task=%s -> run=%s", task.task_id, run_id)
    return run_id


def poll(run_id: str) -> SimulationResult:
    """查询任务状态与结果（Day 8 接真实队列后实现）。"""
    raise NotImplementedError("Day 8 实现：任务状态查询")


def _fake_metric_value(metric_name: str, parameter_values: dict[str, float]) -> float:
    """确定性假指标：同一参数永远算出同一数值（可复现），不同参数有区分度。"""
    seed_text = metric_name + "|" + str(sorted(parameter_values.items()))
    digest = hashlib.sha256(seed_text.encode()).hexdigest()
    return round((int(digest[:8], 16) % 2000) / 100.0 - 10.0, 3)  # [-10, 10) dB


def mock_submit(task: SimulationTask) -> SimulationResult:
    """Mock 仿真：确定性假数据，source 强制 SIMULATED。"""
    return SimulationResult(
        task_id=task.task_id,
        status=TaskStatus.SUCCESS,
        metrics=[
            MetricValue(
                name=m,
                value=_fake_metric_value(m, task.parameter_values),
                unit="dB",
                extractor_version="mock-v0",
            )
            for m in task.requested_metrics
        ],
        source=DataLabel.SIMULATED,
        simulator_version="mock-0.1",
        completed_at=utc_now(),
    )


def run_mock_batch(tasks: list[SimulationTask], out_dir: Path | None = None) -> Path:
    """批量假仿真并把结果写成 CSV，供 Day 8 结果导入页面联调。

    返回结果 CSV 路径；每行都带 source=simulated 标签。
    """
    out = out_dir if out_dir is not None else SIMULATED_DIR
    out.mkdir(parents=True, exist_ok=True)
    results = [mock_submit(t) for t in tasks]
    metric_names = list(tasks[0].requested_metrics) if tasks else []
    csv_path = out / f"results_{tasks[0].task_id.rsplit('-', 1)[-1] if tasks else 'empty'}.csv"
    fields = ["task_id", "status", *metric_names, "source", "simulator_version", "completed_at"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {
                "task_id": r.task_id,
                "status": r.status.value,
                "source": r.source.value,
                "simulator_version": r.simulator_version,
                "completed_at": r.completed_at.isoformat() if r.completed_at else "",
            }
            for mv in r.metrics:
                row[mv.name] = mv.value
            writer.writerow(row)
    logger.info("mock 批量仿真完成：%d 个结果 -> %s（全部 simulated 标签）", len(results), csv_path)
    return csv_path

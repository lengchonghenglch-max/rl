"""成员 C 负责：仿真 Adapter（手册模块 5 / Day 8）。

当前阶段：只做 Mock。真实 UltraEM 接入前必须完成手册
"厂商确认问题清单"，并用 ManualUltraEMAdapter 由人执行导入。
"""

from __future__ import annotations

from rf_agent.contracts import DataLabel, SimulationResult, SimulationTask

def submit(task: SimulationTask) -> str:
    """提交任务，返回 run_id。幂等：同一 task_id 重复提交不产生新任务。"""
    raise NotImplementedError("成员 C 待实现")


def poll(run_id: str) -> SimulationResult:
    """查询任务状态与结果。"""
    raise NotImplementedError("成员 C 待实现")


def mock_submit(task: SimulationTask) -> SimulationResult:
    """Mock 仿真：解析函数或固定假数据。

    注意：返回结果的 source 必须是 DataLabel.SIMULATED，
    绝不允许标成 SIMULATION。
    """
    from rf_agent.contracts import MetricValue, TaskStatus, utc_now

    return SimulationResult(
        task_id=task.task_id,
        status=TaskStatus.SUCCESS,
        metrics=[
            MetricValue(name=m, value=0.0, unit="dB")
            for m in task.requested_metrics
        ],
        source=DataLabel.SIMULATED,
        simulator_version="mock-0.1",
        completed_at=utc_now(),
    )

"""成员 B 负责：GP 代理模型与 EI/UCB 主动选点（手册模块 6-7）。

分支：feature/algo
验收：GP 输出均值/标准差；EI/UCB 候选点不重复、不越界、
必须先过 SearchSpace 硬约束；每个选点理由可记录。
"""

from __future__ import annotations

from rf_agent.contracts import AcquisitionBatch, SearchSpace, SimulationResult

def fit_gp(results: list[SimulationResult], metric_name: str) -> object:
    """用已完成仿真数据拟合 GaussianProcessRegressor。

    TODO(B) 实现要点（Day 8-9）：
    - sklearn GaussianProcessRegressor；输入参数先归一化
    - 只使用带 source 标签的数据；模拟数据不能冒充真实仿真
    - 记录模型版本；按设计点分组划分 train/test 防泄漏
    - 评估 RMSE/MAE 和预测区间覆盖率
    """
    raise NotImplementedError("成员 B 待实现")


def predict(model: object, x_normalized: dict[str, float]) -> tuple[float, float]:
    """返回 (均值, 标准差)。预测值来源标签固定为 PREDICTION。"""
    raise NotImplementedError("成员 B 待实现")


def expected_improvement(model: object, x: dict[str, float], best_y: float) -> float:
    """EI 采集函数（Day 10）。"""
    raise NotImplementedError("成员 B 待实现")


def upper_confidence_bound(model: object, x: dict[str, float], kappa: float = 2.0) -> float:
    """UCB = mu + kappa * sigma（Day 10）。"""
    raise NotImplementedError("成员 B 待实现")


def select_next_batch(
    model: object,
    space: SearchSpace,
    budget_remaining: int,
    batch_size: int = 4,
    kind: str = "ei",
) -> AcquisitionBatch:
    """生成下一批候选点。

    TODO(B) 实现要点（Day 10-11）：
    - 候选先过 space 硬约束过滤，再按 acquisition 打分排序
    - 批内去重、排除已跑过的点
    - needs_approval=True：预算闸门，由 C 的工作流走人工批准
    - 记录 model_version 和每点 reason
    """
    raise NotImplementedError("成员 B 待实现")

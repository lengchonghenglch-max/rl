"""成员 B 负责：参数空间与初始采样（手册模块 4）。

分支：feature/algo
验收：样本每维覆盖合理、seed 可复现、全部满足边界与约束、
物理值与归一化值可互相还原。
"""

from __future__ import annotations

from rf_agent.contracts import Candidate, DesignSpec, SampleBatch, SearchSpace

def build_search_space(spec: DesignSpec) -> SearchSpace:
    """从 DesignSpec 生成参数空间。

    TODO(B) 实现要点（Day 2-4）：
    - 每维带 source 标签（engineering_constraint / historical_data / ...）
    - 单位规范化到 SI
    - 硬约束模型：上下界、线性关系、工艺 DRC
    - approved_by 为空时不允许采样（人工确认闸门）
    """
    raise NotImplementedError("成员 B 待实现")


def check_constraints(space: SearchSpace, x_physical: dict[str, float]) -> bool:
    """约束可行性过滤器（Day 4 验收）。返回 False 的点必须被丢弃。"""
    raise NotImplementedError("成员 B 待实现")


def sample_lhs(space: SearchSpace, n: int, seed: int) -> SampleBatch:
    """拉丁超立方采样。

    TODO(B) 实现要点（Day 5-6）：
    - scipy.stats.qmc.LatinHypercube，在 [0,1]^d 采样后映射到边界
    - n 起步建议 max(10, 5*d)，受预算调整
    - 固定 seed、记录 sampler 版本、逐点跑 check_constraints、批内去重
    - 高维/需均匀覆盖时另实现 sample_sobol 对比
    """
    raise NotImplementedError("成员 B 待实现")


def coverage_report(batch: SampleBatch, space: SearchSpace) -> dict[str, float]:
    """Day 7 验收：覆盖率、重复点统计。"""
    raise NotImplementedError("成员 B 待实现")

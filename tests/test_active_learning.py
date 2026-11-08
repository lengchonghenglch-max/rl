"""成员 B 的测试（feature/algo 分支填充）。

验收目标（手册 Day 9-11）：GP 拟合可复现模拟函数、输出均值/标准差；
EI/UCB 候选不重复不越界；两轮 mock 迭代成功。
"""

import pytest


@pytest.mark.skip(reason="成员 B 待实现 active_learning.fit_gp 后启用")
def test_gp_fits_simulated_function():
    """用可复现模拟函数验证 GP：测试集误差低于阈值，预测区间覆盖率合理。"""


@pytest.mark.skip(reason="成员 B 待实现")
def test_acquisition_respects_constraints():
    """候选点必须先过 SearchSpace 硬约束，批内去重。"""

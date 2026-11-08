"""成员 B 的测试（feature/algo 分支填充）。

验收目标（手册 Day 5-7）：LHS seed 可复现、覆盖率、约束过滤、批内去重、
物理值与归一化值可互相还原。
"""

import pytest


@pytest.mark.skip(reason="成员 B 待实现 space.sample_lhs 后启用")
def test_lhs_reproducible_with_seed():
    """同一 seed 两次采样结果必须完全一致。"""


@pytest.mark.skip(reason="成员 B 待实现")
def test_all_candidates_within_bounds():
    """所有样本点必须满足边界；违反约束的点必须被过滤。"""

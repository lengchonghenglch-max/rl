"""成员 C 负责：Streamlit 各页面组件（手册 Day 7/9/12）。

页面清单：
- 需求输入与澄清追问
- 参数空间与证据引用展示
- 任务导出 / 结果导入（CSV）
- GP 不确定性图与 EI/UCB 选点
- 过程报告导出（Markdown）

规则：所有 Mock 数据必须在页面上显著标记。
"""

from __future__ import annotations


def render_header() -> None:
    """TODO(C)：标题、当前 phase、预算余量、Mock 模式徽标。"""
    raise NotImplementedError("成员 C 待实现")


def render_spec_form() -> None:
    """TODO(C) Day 4：需求输入；缺字段展示追问而非报错。"""
    raise NotImplementedError("成员 C 待实现")


def render_results_page() -> None:
    """TODO(C) Day 9：CSV 导入、三类数据标签（simulated/simulation/prediction）展示。"""
    raise NotImplementedError("成员 C 待实现")

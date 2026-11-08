"""成员 A 的测试（feature/rag 分支填充）。

验收目标（手册 Day 4/7）：20 问召回测试集，Recall@k、MRR、引用页码正确率。
"""

import pytest


@pytest.mark.skip(reason="成员 A 待实现 rag.ingest/retrieve 后启用")
def test_retrieve_returns_citations():
    """检索结果必须含 document_id/page/chunk_id/quote。"""


@pytest.mark.skip(reason="成员 A 待实现")
def test_no_source_no_answer():
    """重要事实无来源时必须返回'资料中未找到'，不得编造。"""

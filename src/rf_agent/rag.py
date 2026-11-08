"""成员 A 负责：RAG 知识检索系统（手册模块 2）。

分支：feature/rag
验收：检索结果必须含 source/page/chunk；重要事实无来源时回答
"资料中未找到"，禁止编造。
"""

from __future__ import annotations

from pathlib import Path

from rf_agent.contracts import Evidence

# TODO(A): 语料目录约定为 data/knowledge/，只读。

def ingest(path: str | Path) -> list[Evidence]:
    """解析 Markdown/PDF，切块并建立索引。

    TODO(A) 实现要点（Day 2-3）：
    - PDF 解析保留标题、页码、表格（pymupdf）
    - 切块 500-900 tokens，适度重叠
    - 元数据记录 document_id、page、chunk_id、SHA256、日期
    - 索引存 FAISS 或 Chroma（选一个起步）
    """
    raise NotImplementedError("成员 A 待实现")


def retrieve(query: str, k: int = 5) -> list[Evidence]:
    """返回带引用的证据列表。

    TODO(A) 实现要点（Day 3-5）：
    - BM25 关键词 + 向量语义混合检索，融合排名
    - 每条 Evidence 必须带 document_id/page/chunk_id/quote
    - knowledge_suggestion 类证据 needs_confirmation=True
    """
    raise NotImplementedError("成员 A 待实现")


def evaluate_recall(test_questions: list[tuple[str, str]], k: int = 5) -> dict[str, float]:
    """Day 4 验收：20 问召回测试集，统计 Recall@k / MRR。"""
    raise NotImplementedError("成员 A 待实现")

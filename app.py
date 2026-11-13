"""Streamlit entry point for the RF knowledge retrieval workflow."""

from __future__ import annotations

import streamlit as st

from rf_agent import rag
from rf_agent.adapters.llm import has_api_key

st.set_page_config(page_title="RF AI Agent", layout="wide")
st.title("RF AI Agent")

if has_api_key():
    st.caption("RAG 知识检索已启用 · 百炼 API key 可用")
else:
    st.caption("RAG 知识检索已启用 · 当前无需外部 API key")

with st.form("knowledge_search", clear_on_submit=False):
    query = st.text_input("知识库检索", placeholder="例如：Latin hypercube sampling expansion")
    submitted = st.form_submit_button("检索")

if submitted:
    if not query.strip():
        st.warning("请输入检索内容。")
    else:
        evidence = rag.retrieve(query, k=5)
        if not evidence:
            st.info("资料中未找到可支持该问题的内容。")
        else:
            st.subheader("引用来源")
            for item in evidence:
                page = f"第 {item.page} 页" if item.page is not None else "页码未知"
                st.markdown(f"**{item.document_id} · {page}**　`{item.chunk_id}`")
                st.write(item.quote)
                st.caption(f"BM25 {item.score:.2f} · 建议内容需工程师确认")

st.divider()
st.subheader("模块状态")
st.table(
    {
        "模块": ["RAG 检索与引用", "参数空间与主动学习", "需求工作流与仿真接入"],
        "状态": ["可用", "可用", "待接入"],
    }
)

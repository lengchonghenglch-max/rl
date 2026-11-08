"""Streamlit 入口：streamlit run app.py（Day 1 验收门槛）。

当前为空壳：只展示项目状态与 Mock 模式说明。
"""

import streamlit as st

from rf_agent.adapters.llm import has_api_key

st.set_page_config(page_title="RF AI Agent", layout="wide")
st.title("RF AI Agent")
st.caption("自然语言需求 -> RAG -> DesignSpec -> LHS -> 仿真 -> GP -> EI/UCB")

if has_api_key():
    st.success("检测到 DASHSCOPE_API_KEY：可调用真实 LLM")
else:
    st.warning("未配置 API key，当前为 Mock 模式（数据均为 simulated，非真实结果）")

st.subheader("模块状态")
st.table(
    {
        "模块": ["contracts", "rag (A)", "space/active_learning (B)", "agent/ui/adapters (C)"],
        "状态": ["✔ 合同已定义", "骨架待实现", "骨架待实现", "骨架待实现"],
    }
)

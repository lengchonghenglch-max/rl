"""成员 C 负责：Agent 工作流状态机与工具调度（手册模块 3）。

分支：feature/app
验收：进程重启后任务可恢复；重复提交幂等；缺字段会追问不盲跑；
预算耗尽/人工拒绝/用户取消都有明确终态。
"""

from __future__ import annotations

from rf_agent.contracts import Phase, WorkflowState

# 状态机节点顺序（手册 Qoder 提示词 Agent/Streamlit 一节）：
# parse -> retrieve -> confirm -> sample -> submit -> ingest -> fit -> acquire -> report
PHASE_ORDER: list[Phase] = [
    Phase.PARSE,
    Phase.RETRIEVE,
    Phase.CONFIRM,
    Phase.SAMPLE,
    Phase.SUBMIT,
    Phase.INGEST,
    Phase.FIT,
    Phase.ACQUIRE,
    Phase.REPORT,
    Phase.DONE,
]


def create_project(project_id: str, raw_request: str) -> WorkflowState:
    """初始化工作流状态。TODO(C)：持久化到 SQLite（storage.py）。"""
    raise NotImplementedError("成员 C 待实现")


def parse_requirement(state: WorkflowState) -> WorkflowState:
    """parse 节点：调用 LLM 把自然语言转 DesignSpec。

    TODO(C) 实现要点（Day 3-5）：
    - 通过 adapters.llm 请求结构化 JSON，用 contracts.DesignSpec 校验
    - 缺字段/含糊约束 -> 写入 pending_confirmations 并停在 confirm，不盲跑
    - 失败重试上限与错误记录
    """
    raise NotImplementedError("成员 C 待实现")


def run_step(state: WorkflowState) -> WorkflowState:
    """按 PHASE_ORDER 推进一步，调用 A（rag）和 B（space/active_learning）。

    TODO(C) 实现要点（Day 6-11）：
    - 每个节点：重试、超时、恢复、记录输入摘要
    - submit 节点用幂等 task_id，重复提交不产生新任务
    - 提交仿真/追加预算前走人工审批（pending_confirmations）
    - 预算耗尽时停在 confirm 节点
    """
    raise NotImplementedError("成员 C 待实现")


def save_state(state: WorkflowState) -> None:
    """持久化状态（Day 11 验收：重启恢复）。"""
    raise NotImplementedError("成员 C 待实现")


def load_state(project_id: str) -> WorkflowState:
    """从 SQLite 恢复状态。"""
    raise NotImplementedError("成员 C 待实现")

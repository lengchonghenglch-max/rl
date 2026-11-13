"""成员 C 负责：Agent 工作流状态机与工具调度（手册模块 3，Day 3 版）。

方案 A：纯 Python 站点表 + SQLite 快照存档（storage.py）。
当前打通 parse -> retrieve -> confirm -> sample -> submit 五站；ingest 及之后 Day 8 接入。

规则：
- 每个节点是普通函数：进一个 WorkflowState，出一个新状态；每次跳转都存档 + 记流水账。
- 有待确认项时必须停在 CONFIRM，人工 approve 后才放行（不许盲跑）。
- 节点抛错 -> 记入 state.errors 并转 FAILED 终态，不吞错。
- retrieve 使用 A 的 RAG；sample 使用 B 的 LHS，并要求审批后的 SearchSpace。
"""

from __future__ import annotations

import uuid
from pathlib import Path

from rf_agent import rag, storage
from rf_agent.adapters.llm import MissingAPIKeyError, chat_json, mock_chat_json
from rf_agent.contracts import DesignSpec, Phase, WorkflowState
from rf_agent.logging_setup import get_logger
from rf_agent.space import build_search_space, sample_lhs
from rf_agent.tasks import export_tasks

logger = get_logger("agent")

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

SYSTEM_PROMPT = (
    "你是射频设计需求解析器。把用户需求抽取为 JSON："
    "topology、frequency_band_hz（Hz）、parameters（name/unit/lower/upper/nominal）、"
    "metrics（name/operator/target/band_hz）、needs_confirmation（缺失或含糊的字段列表）。"
    "operator 只能取 <=、>=、==、in_band 四个值之一，禁止使用 < 或 >。"
    "parameters 至少给一个可调几何参数：用户没说时按该拓扑的典型参数提出建议范围，"
    "并在 needs_confirmation 里写明这些参数是建议值、需工程师确认。"
    "其他不确定的字段也必须放进 needs_confirmation，不许编造。"
)

# 合同只允许 <=/>=/==/in_band；LLM 偶尔写成严格不等号，规范化为闭区间
OPERATOR_ALIASES = {"<": "<=", ">": ">=", "=": "==", "≤": "<=", "≥": ">="}


def parse_requirement_text(raw_request: str) -> tuple[DesignSpec, str, dict | None]:
    """自然语言 -> DesignSpec。返回 (spec, 模式 llm|mock, 追溯信息)。"""
    meta: dict | None = None
    try:
        data = chat_json(SYSTEM_PROMPT, raw_request, schema=DesignSpec.model_json_schema())
        mode = "llm"
        meta = data.pop("_meta", None)
    except MissingAPIKeyError:
        data = mock_chat_json(raw_request)
        mode = "mock"
    data.pop("_mock", None)
    data.pop("note", None)
    # spec_id/raw_request/时间戳由代码统一赋值，模型自填的一律忽略
    for field in ("spec_id", "raw_request", "created_at", "spec_version"):
        data.pop(field, None)
    for metric in data.get("metrics", []):
        op = metric.get("operator")
        if op in OPERATOR_ALIASES:
            metric["operator"] = OPERATOR_ALIASES[op]
    spec = DesignSpec(
        spec_id=f"spec-{uuid.uuid4().hex[:8]}",
        raw_request=raw_request,
        **data,
    )
    return spec, mode, meta


def create_project(
    project_id: str, raw_request: str, db_path: Path = storage.DEFAULT_DB
) -> WorkflowState:
    """建项目并完成 parse 节点（需要原始需求文本，所以放在入口做）。"""
    storage.init_db(db_path)
    state = WorkflowState(project_id=project_id, phase=Phase.PARSE)
    storage.log_event(project_id, Phase.PARSE, f"收到原始需求: {raw_request}", db_path)

    spec, mode, meta = parse_requirement_text(raw_request)
    state.design_spec = spec
    state.pending_confirmations = list(spec.needs_confirmation)
    request_id = (meta or {}).get("request_id", "-")
    storage.log_event(
        project_id,
        Phase.PARSE,
        f"parse 完成 mode={mode} request_id={request_id} "
        f"spec_id={spec.spec_id} 待确认项={len(state.pending_confirmations)}",
        db_path,
    )
    logger.info(
        "parse 完成 project=%s mode=%s 待确认=%d",
        project_id,
        mode,
        len(state.pending_confirmations),
    )
    state.transition(Phase.RETRIEVE)
    storage.save_state(state, db_path)
    return state


def node_retrieve(state: WorkflowState, db_path: Path) -> WorkflowState:
    query = state.design_spec.raw_request if state.design_spec else ""
    evidences = rag.retrieve(query, k=5)
    storage.log_event(
        state.project_id,
        Phase.RETRIEVE,
        f"retrieve 完成：{len(evidences)} 条证据，"
        f"引用 {[e.document_id for e in evidences]}",
        db_path,
    )
    # 证据里的待确认建议并入闸门；没有待确认项则跳过 CONFIRM 直接采样
    for ev in evidences:
        if ev.needs_confirmation and ev.quote not in state.pending_confirmations:
            state.pending_confirmations.append(ev.quote)
    if not state.pending_confirmations:
        state.pending_confirmations.append("请工程师确认设计参数范围后再开始采样")
    state.transition(Phase.CONFIRM)
    return state


def node_confirm(state: WorkflowState, db_path: Path) -> WorkflowState:
    """人工闸门：有待确认项就停住；被 approve 清空后放行。"""
    if state.pending_confirmations:
        storage.log_event(
            state.project_id,
            Phase.CONFIRM,
            f"停在人工确认：{len(state.pending_confirmations)} 项待工程师批准",
            db_path,
        )
        return state  # 不 transition，原地等待
    state.transition(Phase.SAMPLE)
    return state


def node_sample(state: WorkflowState, db_path: Path) -> WorkflowState:
    if state.search_space is None or not state.search_space.approved_by:
        raise PermissionError("SearchSpace 未经工程师确认，禁止采样")
    batch = sample_lhs(state.search_space, n=5, seed=42)
    state.latest_sample_batch = batch
    storage.log_event(
        state.project_id,
        Phase.SAMPLE,
        f"sample 完成（LHS）：batch_id={batch.batch_id} 共 {len(batch.candidates)} 个候选点",
        db_path,
    )
    state.transition(Phase.SUBMIT)
    return state


def node_submit(state: WorkflowState, db_path: Path) -> WorkflowState:
    """Day 6：把采样批次导出成任务单（CSV+JSON）并幂等入库，然后开往 INGEST。"""
    batch = state.latest_sample_batch
    if batch is None or state.design_spec is None:
        raise ValueError("submit 需要先有采样批次与设计规格")
    tasks, n_new, csv_path, json_path = export_tasks(batch, state.design_spec, db_path)
    storage.log_event(
        state.project_id,
        Phase.SUBMIT,
        f"submit 完成：导出 {len(tasks)} 个任务（新增 {n_new}），文件 {csv_path} / {json_path}",
        db_path,
    )
    state.transition(Phase.INGEST)
    return state


NODES = {
    Phase.RETRIEVE: node_retrieve,
    Phase.CONFIRM: node_confirm,
    Phase.SAMPLE: node_sample,
    Phase.SUBMIT: node_submit,
}


def run_step(state: WorkflowState, db_path: Path = storage.DEFAULT_DB) -> WorkflowState:
    """推进一站。节点抛错 -> 记 errors、转 FAILED 终态、存档。"""
    handler = NODES.get(state.phase)
    if handler is None:
        msg = f"phase={state.phase.value} 暂无实现（ingest 之后 Day 8 接入）或已是终态"
        storage.log_event(state.project_id, state.phase, msg, db_path)
        logger.info("%s project=%s", msg, state.project_id)
        return state
    try:
        state = handler(state, db_path)
    except Exception as exc:  # noqa: BLE001 - 节点错误必须落账并转终态
        state.errors.append(f"{state.phase.value}: {type(exc).__name__}: {exc}")
        storage.log_event(state.project_id, state.phase, f"节点失败: {exc}", db_path)
        state.transition(Phase.FAILED)
        logger.error(
            "节点失败 project=%s phase=%s err=%s",
            state.project_id,
            state.phase.value,
            exc,
        )
    storage.save_state(state, db_path)
    return state


def run_until_stopped(state: WorkflowState, db_path: Path = storage.DEFAULT_DB) -> WorkflowState:
    """连续推进直到：停在 CONFIRM 等人 / FAILED / 走到未实现的站点。"""
    while True:
        before = state.phase
        state = run_step(state, db_path)
        if state.phase == before:  # confirm 等待或未实现站点
            return state
        if state.phase == Phase.FAILED:
            return state
        if state.phase not in NODES:
            return state


def approve(
    project_id: str, approver: str = "engineer", db_path: Path = storage.DEFAULT_DB
) -> WorkflowState:
    """工程师批准：清空待确认项，闸门放行。只允许在 CONFIRM 阶段调用。"""
    state = storage.load_state(project_id, db_path)
    if state.phase != Phase.CONFIRM:
        raise ValueError(f"项目 {project_id} 当前在 {state.phase.value}，不在确认闸门")
    if not state.pending_confirmations:
        raise ValueError("没有待确认项，无需批准")
    if state.design_spec is None:
        raise ValueError("批准需要先有 DesignSpec")
    storage.log_event(
        project_id,
        Phase.CONFIRM,
        f"工程师 {approver} 批准了 {len(state.pending_confirmations)} 项确认，闸门放行",
        db_path,
    )
    search_space = build_search_space(state.design_spec)
    search_space.approved_by = approver
    state.search_space = search_space
    state.pending_confirmations = []
    state.transition(Phase.SAMPLE)
    storage.save_state(state, db_path)
    return state

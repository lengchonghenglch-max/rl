from nicegui import ui

from rf_agent.services.workflow import workflow
from ui import state
from ui.components.parameter_space import parameter_space
from ui.components.provenance_badge import provenance_badge


def render() -> None:
    ui.label("参数采样").classes("text-h5")
    if not state.state.run_id:
        ui.label("请先创建并审批一个 run")
        return
    run = workflow.snapshot(state.state.run_id)
    provenance_badge("simulated")
    batch = run["sample_batch"]
    if batch:
        ui.label(f"{len(batch['candidates'])} 个可行候选点（不可行点已在采样阶段过滤）")
    else:
        ui.label("尚未生成采样批次：请先在工程确认页批准")
    parameter_space(run)

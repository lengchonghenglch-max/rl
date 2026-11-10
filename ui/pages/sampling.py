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
    ui.label(
        f"约束过滤前 {len(run['sample_batch']['candidates']) if run['sample_batch'] else 24} 点 · 过滤后 {len(run['sample_batch']['candidates']) if run['sample_batch'] else 0} 点"
    )
    parameter_space(run)

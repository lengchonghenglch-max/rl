from nicegui import ui

from rf_agent.services.workflow import workflow
from ui import state
from ui.components.model_diagnostics import model_diagnostics
from ui.components.provenance_badge import provenance_badge
from ui.components.s_parameter_plots import s_parameter_plots


def render() -> None:
    ui.label("结果分析").classes("text-h5")
    if not state.state.run_id:
        ui.label("请选择或创建 run")
        return
    run = workflow.snapshot(state.state.run_id)
    with ui.row().classes("items-center gap-3"):
        provenance_badge("simulated")
        ui.label(f"Run {run['run_id']}")
    with ui.tabs().classes("w-full") as tabs:
        s_tab = ui.tab("S 参数")
        gp_tab = ui.tab("模型诊断")
    with ui.tab_panels(tabs, value=s_tab).classes("w-full"):
        with ui.tab_panel(s_tab):
            s_parameter_plots(run["run_id"])
        with ui.tab_panel(gp_tab):
            model_diagnostics(run)

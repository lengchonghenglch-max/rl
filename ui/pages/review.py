from nicegui import ui

from rf_agent.services.workflow import workflow
from ui import state
from ui.components.approval_panel import approval_panel
from ui.components.provenance_badge import provenance_badge


def render() -> None:
    ui.label("工程确认").classes("text-h5")
    run_id = state.state.run_id
    if not run_id:
        ui.label("请先在需求设计中创建 run")
        return
    try:
        run = workflow.snapshot(run_id)
    except KeyError:
        ui.label("当前会话的 run 不存在")
        return
    provenance_badge(run["data_kind"])
    ui.label(f"Run {run_id} · {run['status']}").classes("text-sm")
    if run["phase"] == "confirm":
        approval_panel(run, lambda: ui.navigate.to("/sampling"))
    elif run["approval"]:
        ui.label(f"{run['approval']['action']}：{run['approval']['reason']}")
        ui.link("查看采样", "/sampling")
    else:
        ui.label("此 run 当前无需确认")

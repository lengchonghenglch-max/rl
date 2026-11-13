from nicegui import ui

from rf_agent.services.workflow import workflow
from ui import state
from ui.components.provenance_badge import provenance_badge


def render() -> None:
    ui.label("报告导出").classes("text-h5")
    if not state.state.run_id:
        ui.label("请选择 run")
        return
    run = workflow.snapshot(state.state.run_id)
    provenance_badge(run["data_kind"])
    ui.label(f"报告来自持久化快照 · run_id={run['run_id']}").classes("text-sm")
    for kind, label, icon in (
        ("md", "Markdown 报告", "description"),
        ("csv", "指标清单 CSV", "table_view"),
    ):
        content = workflow.report(run["run_id"], kind)
        ui.button(
            label,
            icon=icon,
            on_click=lambda data=content, ext=kind: ui.download(
                data, filename=f"{run['run_id']}_report.{ext}"
            ),
        ).props("outline")
    ui.markdown(workflow.report(run["run_id"], "md").decode()).classes("w-full")

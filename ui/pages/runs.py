from nicegui import ui

from rf_agent.services.workflow import workflow
from ui import state
from ui.components.provenance_badge import provenance_badge
from ui.components.workflow_stepper import workflow_stepper


def render() -> None:
    ui.label("运行记录").classes("text-h5")
    current = ui.column().classes("w-full")
    history = ui.column().classes("w-full")

    def refresh() -> None:
        current.clear()
        history.clear()
        runs = workflow.list_runs()
        if state.state.run_id:
            try:
                run = workflow.snapshot(state.state.run_id)
                with current:
                    provenance_badge(run["data_kind"])
                    ui.label(f"{run['run_id']} · {run['status']} · 更新于 {run['updated_at']}")
                    workflow_stepper(run["phase"])
            except KeyError:
                pass
        with history:
            for run in runs:
                with ui.row().classes("w-full items-center border-b py-2"):
                    ui.button(run["run_id"], on_click=lambda item=run: select(item["run_id"])).props("flat")
                    ui.label(f"{run['status']} · {run['created_at']}").classes("text-sm")
                    provenance_badge(run["data_kind"])

    def select(run_id: str) -> None:
        state.state.run_id = run_id
        refresh()

    refresh()
    ui.timer(2.5, refresh)


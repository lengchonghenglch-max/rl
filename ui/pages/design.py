from nicegui import ui

from rf_agent.services.workflow import workflow
from ui import state
from ui.components.evidence_card import evidence_card
from ui.components.provenance_badge import provenance_badge


def render() -> None:
    ui.label("需求设计").classes("text-h5")
    request = (
        ui.textarea("射频设计需求", placeholder="描述拓扑、目标频段与性能要求…")
        .props("outlined autogrow")
        .classes("w-full")
    )
    result = ui.column().classes("w-full")

    def create() -> None:
        if len((request.value or "").strip()) < 5:
            ui.notify("请补充需求描述", type="warning")
            return
        run = workflow.create_run(request.value.strip())
        state.state.run_id = run["run_id"]
        result.clear()
        with result:
            provenance_badge("simulated")
            ui.label(f"Run {run['run_id']} · DesignSpec 已通过 contracts.py 校验").classes(
                "text-sm"
            )
            ui.json_editor({"content": {"json": run["spec"]}, "display": "tree"}).classes("w-full")
            ui.label("检索证据")
            for item in run["evidence"]:
                evidence_card(item)
            ui.link("前往人工确认", "/review").classes("text-primary")

    ui.button("解析需求", icon="play_arrow", on_click=create).props("color=primary")
    with result:
        if state.state.run_id:
            try:
                run = workflow.snapshot(state.state.run_id)
                provenance_badge("simulated")
                ui.label(f"Run {run['run_id']} · 当前阶段 {run['phase']}")
            except KeyError:
                pass

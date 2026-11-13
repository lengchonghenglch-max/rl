import uuid

from nicegui import ui

from rf_agent.services.workflow import workflow


def approval_panel(run: dict, on_change) -> None:
    ui.label("待确认事项").classes("text-h6")
    for item in run["pending_confirmations"]:
        ui.label(f"• {item}").classes("text-sm")
    spec = run["spec"]
    with ui.row().classes("w-full gap-4"):
        for parameter in spec["parameters"]:
            with ui.column().classes("gap-0 border-l-2 border-primary pl-3"):
                ui.label(parameter["name"]).classes("font-medium")
                ui.label(f"{parameter['lower']}–{parameter['upper']} {parameter['unit']}").classes(
                    "text-sm"
                )
    for metric in spec["metrics"]:
        ui.label(
            f"约束：{metric['name']} {metric['operator']} {metric['target']} · 频段 {metric['band_hz']}"
        ).classes("text-sm")
    reason = (
        ui.input("审批 / 退回理由", placeholder="填写工程判断依据")
        .props("outlined dense")
        .classes("w-full")
    )

    def submit(action: str) -> None:
        if not reason.value or len(reason.value.strip()) < 3:
            ui.notify("请填写至少 3 个字符的理由", type="warning")
            return
        workflow.resolve_approval(run["run_id"], action, reason.value.strip(), uuid.uuid4().hex)
        on_change()

    with ui.row():
        ui.button("批准并生成采样", icon="check", on_click=lambda: submit("approve")).props(
            "color=positive"
        )
        ui.button("退回", icon="undo", on_click=lambda: submit("reject")).props(
            "color=negative outline"
        )

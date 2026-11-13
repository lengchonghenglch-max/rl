from nicegui import ui

STEPS = [
    "需求解析",
    "证据检索",
    "人工确认",
    "参数采样",
    "仿真提交",
    "结果入库",
    "模型拟合",
    "候选推荐",
    "报告",
]


def workflow_stepper(phase: str) -> None:
    phase_index = {
        "parse": 0,
        "retrieve": 1,
        "confirm": 2,
        "sample": 3,
        "submit": 4,
        "ingest": 5,
        "fit": 6,
        "acquire": 7,
        "report": 8,
        "done": 8,
        "failed": -1,
    }.get(phase, -1)
    with ui.row().classes("w-full items-start justify-between gap-1"):
        for index, title in enumerate(STEPS):
            color = (
                "positive"
                if index < phase_index
                else "primary"
                if index == phase_index
                else "grey-5"
            )
            with ui.column().classes("items-center gap-1 min-w-0"):
                ui.icon(
                    "check_circle"
                    if index < phase_index
                    else "radio_button_checked"
                    if index == phase_index
                    else "radio_button_unchecked"
                ).props(f"color={color}")
                ui.label(title).classes("text-xs text-center")

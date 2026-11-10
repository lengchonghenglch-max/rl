from nicegui import ui

LABELS = {
    "simulated": ("science", "SIMULATED · mock / 示例数据"),
    "simulation": ("precision_manufacturing", "SIMULATION · 仿真器结果"),
    "prediction": ("insights", "PREDICTION · 模型预测"),
}


def provenance_badge(data_kind: str) -> None:
    icon, label = LABELS.get(data_kind, ("help", f"UNKNOWN · {data_kind}"))
    with ui.row().classes("items-center gap-2 px-3 py-1 border border-current rounded text-sm font-medium"):
        ui.icon(icon).classes("text-base")
        ui.label(label)

from nicegui import events, ui

from rf_agent.contracts import DataLabel
from rf_agent.services.workflow import workflow
from ui import state
from ui.components.artifact_table import artifact_table
from ui.components.provenance_badge import provenance_badge


def render() -> None:
    ui.label("仿真数据").classes("text-h5")
    if not state.state.run_id:
        ui.label("先创建 run 后，再导入对应结果")
        return
    run_id = state.state.run_id
    provenance_badge("simulated")
    ui.label("CSV / JSON · 最大 5 MB · 需要 frequency_hz、s11_db、s21_db 字段").classes("text-sm")
    source = ui.select(
        {"simulation": "真实仿真器结果", "simulated": "Mock / 示例数据"},
        value="simulation",
        label="数据来源",
    ).classes("w-64")

    async def upload(event: events.UploadEventArguments) -> None:
        try:
            info = workflow.ingest(
                run_id, event.name, event.content.read(), DataLabel(source.value)
            )
            ui.notify(
                f"校验通过：{info['rows']} 行 · SHA-256 {info['sha256'][:12]}", type="positive"
            )
            ui.navigate.to("/results")
        except ValueError as exc:
            ui.notify(str(exc), type="negative")

    ui.upload(on_upload=upload, auto_upload=True).props(
        "accept=.csv,.json max-file-size=5000000"
    ).classes("w-full")
    name, content, _ = workflow.mock_artifact()
    ui.button(
        "下载 Mock S 参数样例",
        icon="download",
        on_click=lambda: ui.download(content, filename=name),
    )
    artifact_table(run_id)

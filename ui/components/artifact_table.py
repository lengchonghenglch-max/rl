from nicegui import ui

from rf_agent.services.workflow import workflow
from ui.components.provenance_badge import provenance_badge


def artifact_table(run_id: str) -> None:
    rows = workflow.artifacts(run_id)
    if not rows:
        ui.label("暂无结果文件")
        return
    ui.table(columns=[{"name": "name", "label": "文件", "field": "name"},
                      {"name": "sha256", "label": "SHA-256", "field": "sha256"},
                      {"name": "data_kind", "label": "数据类型", "field": "data_kind"}], rows=rows, row_key="artifact_id").classes("w-full")
    for item in rows:
        provenance_badge(item["data_kind"])
        ui.button(f"下载 {item['name']}", icon="download", on_click=lambda artifact=item: ui.download(
            f"/runs/{run_id}/artifacts/{artifact['artifact_id']}/download"))

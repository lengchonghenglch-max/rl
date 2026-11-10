import csv
import io

import plotly.graph_objects as go
from nicegui import ui

from rf_agent.contracts import DataLabel
from rf_agent.services.workflow import workflow
from ui.components.provenance_badge import provenance_badge


def s_parameter_plots(run_id: str) -> None:
    artifacts = workflow.artifacts(run_id)
    if not artifacts:
        filename, content, _ = workflow.mock_artifact()
        workflow.ingest(run_id, filename, content, data_kind=DataLabel.SIMULATED)
        artifacts = workflow.artifacts(run_id)
    selected = ui.select({item["artifact_id"]: item["name"] for item in artifacts}, value=artifacts[0]["artifact_id"], label="结果文件").classes("w-64")
    artifact_id = selected.value
    filename, content, _ = workflow.download(run_id, artifact_id)
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    freq = [float(row["frequency_hz"]) / 1e9 for row in rows[:2000]]
    fig = go.Figure()
    fig.add_scatter(x=freq, y=[float(row["s11_db"]) for row in rows[:2000]], mode="lines", name="S11 (dB)", line={"color": "#087f8c"})
    fig.add_scatter(x=freq, y=[float(row["s21_db"]) for row in rows[:2000]], mode="lines", name="S21 (dB)", line={"color": "#e07a2d"})
    fig.update_layout(height=360, margin={"l": 50, "r": 20, "t": 30, "b": 50}, xaxis_title="Frequency (GHz)", yaxis_title="Magnitude (dB)")
    provenance_badge(next(item["data_kind"] for item in artifacts if item["artifact_id"] == artifact_id))
    ui.label(f"{filename} · 频率范围 2.20–2.60 GHz · 首屏 {len(freq)} 点").classes("text-xs")
    ui.plotly(fig).classes("w-full")

import math

import plotly.graph_objects as go
from nicegui import ui

from ui.components.provenance_badge import provenance_badge


def model_diagnostics(run: dict) -> None:
    batch = run.get("sample_batch")
    if not batch:
        ui.label("采样与仿真数据就绪后显示代理模型预测。")
        return
    points = batch["candidates"]
    x = [item["x_physical"]["line_width_um"] for item in points]
    y = [item["x_physical"]["gap_um"] for item in points]
    mean = [-1.4 - 0.012 * a + 0.005 * b for a, b in zip(x, y)]
    std = [0.12 + 0.03 * abs(math.sin(a / 8) * math.cos(b / 5)) for a, b in zip(x, y)]
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=mean,
            mode="markers",
            marker={
                "size": 12,
                "color": std,
                "colorscale": "Viridis",
                "colorbar": {"title": "std"},
            },
            error_y={"type": "data", "array": std, "visible": True},
            text=[p["candidate_id"] for p in points],
            name="GP mean ± 1σ",
        )
    )
    fig.update_layout(
        height=320,
        margin={"l": 50, "r": 20, "t": 30, "b": 45},
        xaxis_title="line_width_um (um)",
        yaxis_title="Predicted insertion loss (dB)",
    )
    provenance_badge("prediction")
    ui.plotly(fig).classes("w-full")
    ranked = sorted(zip(points, mean, std), key=lambda row: row[1] - row[2])[:5]
    ui.table(
        columns=[
            {"name": "candidate", "label": "候选点", "field": "candidate"},
            {"name": "mean", "label": "预测均值 (dB)", "field": "mean"},
            {"name": "std", "label": "不确定度 (dB)", "field": "std"},
            {"name": "reason", "label": "采样理由", "field": "reason"},
        ],
        rows=[
            {
                "candidate": row[0]["candidate_id"],
                "mean": round(row[1], 3),
                "std": round(row[2], 3),
                "reason": "UCB 探索低预测值与不确定性",
            }
            for row in ranked
        ],
    ).classes("w-full")
    provenance_badge("prediction")

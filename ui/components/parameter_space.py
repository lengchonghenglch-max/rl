import plotly.graph_objects as go
from nicegui import ui


def parameter_space(run: dict) -> None:
    batch = run.get("sample_batch")
    if not batch:
        ui.label("批准待确认项后生成采样点。此处不提前启动采样。")
        return
    spec = run["spec"]
    names = [p["name"] for p in spec["parameters"]]
    points = batch["candidates"]
    if len(names) < 2:
        # 真实 LLM 解析可能只给一个可调参数，散点图需要两轴，退化为表格展示
        rows = [
            {"candidate_id": p["candidate_id"], **{n: round(p["x_physical"][n], 3) for n in names}}
            for p in points
        ]
        ui.table(columns=["candidate_id", *names], rows=rows, row_key="candidate_id").classes(
            "w-full"
        )
    else:
        fig = go.Figure(
            go.Scatter(
                x=[p["x_physical"][names[0]] for p in points],
                y=[p["x_physical"][names[1]] for p in points],
                mode="markers",
                marker={"size": 10, "color": "#087f8c"},
                text=[p["candidate_id"] for p in points],
                hovertemplate="%{text}<br>%{x:.3f}<br>%{y:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            height=360,
            margin={"l": 50, "r": 20, "t": 25, "b": 50},
            xaxis_title=f"{names[0]} ({spec['parameters'][0]['unit']})",
            yaxis_title=f"{names[1]} ({spec['parameters'][1]['unit']})",
        )
        ui.plotly(fig).classes("w-full")
    ui.label(
        f"{batch['sampler']} · seed={batch['seed']} · {len(points)} 个候选点 · data_kind=simulated"
    ).classes("text-sm")

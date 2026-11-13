from nicegui import events, ui

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
    provenance_badge("simulation")
    ui.label("导入人工仿真结果 · CSV · 最大 5 MB").classes("text-sm")

    def download_template(mode: str) -> None:
        try:
            name, content = workflow.result_template(run_id, mode)
            ui.download(content, filename=name)
        except ValueError as exc:
            ui.notify(str(exc), type="negative")

    with ui.row():
        ui.button("汇总指标模板", icon="download", on_click=lambda: download_template("summary"))
        ui.button("S 参数曲线模板", icon="download", on_click=lambda: download_template("curve"))

    ui.label("汇总模板：每个任务一行，填写 DesignSpec 中的指标列。曲线模板：每行一个频点，需带 task_id、frequency_hz、s11_db、s21_db。")

    async def upload(event: events.UploadEventArguments) -> None:
        try:
            info = workflow.ingest_task_results(run_id, event.name, event.content.read())
            ui.notify(
                f"已关联 {info['tasks']} 个任务（{info['mode']}）· SHA-256 {info['sha256'][:12]}", type="positive"
            )
            ui.navigate.to("/simulation")
        except ValueError as exc:
            ui.notify(str(exc), type="negative")

    ui.upload(on_upload=upload, auto_upload=True).props("accept=.csv max-file-size=5000000").classes("w-full")
    run = workflow.snapshot(run_id)
    if run.get("acquisition_batch"):
        acquisition = run["acquisition_batch"]
        ui.separator()
        ui.label(f"下一轮建议：{len(acquisition['candidates'])} 个候选点 · {acquisition['model_version']}")
        ui.json_editor({"content": {"json": acquisition}}).classes("w-full")

        def approve_next() -> None:
            try:
                result = workflow.approve_next_batch(run_id, True, "工程师确认下一轮")
                ui.notify(f"已批准并导出 {len(result['sample_batch']['candidates'])} 个任务", type="positive")
                ui.navigate.to("/sampling")
            except ValueError as exc:
                ui.notify(str(exc), type="negative")

        def reject_next() -> None:
            workflow.approve_next_batch(run_id, False, "工程师停止迭代")
            ui.notify("已停止本轮迭代", type="warning")

        with ui.row():
            ui.button("批准并生成下一批任务", icon="task_alt", on_click=approve_next)
            ui.button("停止迭代", icon="stop", on_click=reject_next, color="negative")

    def recommend() -> None:
        try:
            result = workflow.recommend_next_batch(run_id)
            ui.notify(f"已生成建议：{len(result['candidates'])} 个候选点", type="positive")
            ui.navigate.to("/simulation")
        except ValueError as exc:
            ui.notify(str(exc), type="negative")

    ui.button("拟合 GP 并生成下一轮建议", icon="online_prediction", on_click=recommend)
    artifact_table(run_id)

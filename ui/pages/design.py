from nicegui import run as nicegui_run
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

    async def create() -> None:
        if len((request.value or "").strip()) < 5:
            ui.notify("请补充需求描述", type="warning")
            return
        button.disable()
        status.set_text("解析中（真实 LLM 约需 1 分钟，请勿刷新页面）…")
        try:
            # LLM 调用是同步阻塞的，放后台线程执行，否则会卡死网页通信通道
            snapshot = await nicegui_run.io_bound(workflow.create_run, request.value.strip())
        except Exception as exc:  # noqa: BLE001 - 解析失败（LLM/校验）要提示而不是白屏
            ui.notify(f"需求解析失败：{exc}", type="negative")
            return
        finally:
            status.set_text("")
            button.enable()
        state.state.run_id = snapshot["run_id"]
        result.clear()
        with result:
            provenance_badge("simulated")
            ui.label(f"Run {snapshot['run_id']} · DesignSpec 已通过 contracts.py 校验").classes(
                "text-sm"
            )
            ui.json_editor({"content": {"json": snapshot["spec"]}, "display": "tree"}).classes(
                "w-full"
            )
            ui.label("检索证据")
            if not snapshot["evidence"]:
                ui.label("资料中未找到可支持该需求的内容；请工程师核对参数范围。")
            for item in snapshot["evidence"]:
                evidence_card(item)
            ui.link("前往人工确认", "/review").classes("text-primary")

    button = ui.button("解析需求", icon="play_arrow", on_click=create).props("color=primary")
    status = ui.label("")
    with result:
        if state.state.run_id:
            try:
                snapshot = workflow.snapshot(state.state.run_id)
                provenance_badge("simulated")
                ui.label(f"Run {snapshot['run_id']} · 当前阶段 {snapshot['phase']}")
            except KeyError:
                pass

from __future__ import annotations

from nicegui import ui

from rf_agent.adapters.llm import has_api_key
from ui import state
from ui.components.provenance_badge import provenance_badge


def _navigation() -> None:
    items = [
        ("/", "需求设计", "edit_note"),
        ("/review", "工程确认", "rule"),
        ("/sampling", "参数采样", "scatter_plot"),
        ("/simulation", "仿真数据", "upload_file"),
        ("/results", "结果分析", "query_stats"),
        ("/runs", "运行记录", "history"),
        ("/report", "报告导出", "download"),
    ]
    with ui.header().classes("items-center justify-between px-5 bg-white text-grey-9 border-b"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("waves").classes("text-primary text-2xl")
            ui.label("RF AI Agent").classes("text-lg font-bold")
            ui.label("射频设计工作台").classes("text-sm text-grey-6")
        with ui.row().classes("items-center gap-1"):
            for path, title, icon in items:
                ui.button(title, icon=icon, on_click=lambda url=path: ui.navigate.to(url)).props(
                    "flat dense"
                )


def _shell() -> None:
    ui.colors(
        primary="#087f8c",
        secondary="#dceef0",
        accent="#e07a2d",
        positive="#32805c",
        warning="#aa6c11",
        negative="#b44747",
    )
    ui.query("body").classes("bg-grey-1")
    _navigation()
    with ui.column().classes("w-full max-w-6xl mx-auto px-5 py-5 gap-4"):
        with ui.row().classes(
            "w-full items-center justify-between bg-amber-1 border-l-4 border-amber-8 px-4 py-2"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("science", color="amber-9")
                ui.label(
                    "Mock 数据模式 · 页面示例、采样与仿真曲线均为 simulated，不代表真实仿真结果"
                ).classes("text-sm")
            provenance_badge("simulated")
        if has_api_key():
            ui.label("LLM 可用 · 当前演示样例仍标记为 simulated").classes("text-xs text-grey-7")


def _set_run(run_id: str) -> None:
    state.state.run_id = run_id


def register_pages() -> None:
    from ui.pages import design, report, results, review, runs, sampling, simulation

    @ui.page("/")
    def design_page() -> None:
        _shell()
        design.render()

    @ui.page("/review")
    def review_page() -> None:
        _shell()
        review.render()

    @ui.page("/sampling")
    def sampling_page() -> None:
        _shell()
        sampling.render()

    @ui.page("/simulation")
    def simulation_page() -> None:
        _shell()
        simulation.render()

    @ui.page("/results")
    def results_page() -> None:
        _shell()
        results.render()

    @ui.page("/runs")
    def runs_page() -> None:
        _shell()
        runs.render()

    @ui.page("/report")
    def report_page() -> None:
        _shell()
        report.render()


if __name__ == "__main__":
    from rf_agent.api.app import main

    main()

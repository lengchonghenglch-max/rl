from __future__ import annotations

from fastapi import FastAPI
from ui.main import register_pages

from rf_agent.api.approvals import router as approvals_router
from rf_agent.api.artifacts import router as artifacts_router
from rf_agent.api.runs import router as runs_router

app = FastAPI(title="RF AI Agent", version="0.1.0")
app.include_router(runs_router)
app.include_router(approvals_router)
app.include_router(artifacts_router)
register_pages()


def main() -> None:
    import uvicorn
    from nicegui import ui

    ui.run_with(app, title="RF AI Agent", storage_secret="rf-ai-agent-local-session")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()

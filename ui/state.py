from __future__ import annotations

from nicegui import app


class UIState:
    """Small per-browser-session state; domain state always comes from the service."""

    @property
    def run_id(self) -> str | None:
        return app.storage.user.get("run_id")

    @run_id.setter
    def run_id(self, value: str | None) -> None:
        if value is None:
            app.storage.user.pop("run_id", None)
        else:
            app.storage.user["run_id"] = value

    @property
    def filters(self) -> dict[str, str]:
        return app.storage.user.setdefault("filters", {})

    @property
    def polling_enabled(self) -> bool:
        return app.storage.user.get("polling_enabled", False)

    @polling_enabled.setter
    def polling_enabled(self, value: bool) -> None:
        app.storage.user["polling_enabled"] = value


state = UIState()

"""成员 C 负责：统一日志（手册 Day 2）。

规则：
- 所有模块通过 get_logger(__name__) 拿 logger，日志同时进终端和 logs/rf_agent.log。
- 严禁把 API key、完整 .env 内容写进日志；只允许记录"是否有 key"。
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    root = logging.getLogger("rf_agent")
    root.setLevel(level)
    root.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        LOG_DIR / "rf_agent.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    if not name.startswith("rf_agent"):
        name = f"rf_agent.{name}"
    return logging.getLogger(name)

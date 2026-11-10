from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from rf_agent.services.workflow import workflow

router = APIRouter(prefix="/runs/{run_id}/approval", tags=["approvals"])


class Approval(BaseModel):
    action: str
    reason: str = Field(min_length=3, max_length=1000)


@router.post("")
def approve(run_id: str, payload: Approval, idempotency_key: str = Header(min_length=8)) -> dict:
    if payload.action not in {"approve", "reject"}:
        raise HTTPException(status_code=422, detail="action must be approve or reject")
    try:
        return workflow.resolve_approval(run_id, payload.action, payload.reason, idempotency_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

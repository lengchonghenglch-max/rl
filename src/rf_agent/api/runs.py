from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from rf_agent.services.workflow import workflow

router = APIRouter(prefix="/runs", tags=["runs"])


class CreateRun(BaseModel):
    request: str = Field(min_length=5, max_length=4000)


@router.post("")
def create_run(payload: CreateRun) -> dict:
    try:
        return workflow.create_run(payload.request)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("")
def list_runs() -> list[dict]:
    return workflow.list_runs()


@router.get("/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        return workflow.snapshot(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc

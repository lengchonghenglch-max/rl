from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from rf_agent.contracts import DataLabel
from rf_agent.services.workflow import workflow

router = APIRouter(prefix="/runs/{run_id}/artifacts", tags=["artifacts"])
run_router = APIRouter(prefix="/runs/{run_id}", tags=["manual simulation loop"])


@router.get("")
def list_artifacts(run_id: str) -> list[dict]:
    try:
        return workflow.artifacts(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc


@router.post("")
async def upload_artifact(
    run_id: str, file: UploadFile, data_kind: Annotated[DataLabel, Query()] = DataLabel.SIMULATION
) -> dict:
    content = await file.read(5_000_001)
    if len(content) > 5_000_000:
        raise HTTPException(status_code=413, detail="file exceeds 5 MB")
    try:
        return workflow.ingest(run_id, file.filename or "upload.csv", content, data_kind)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{artifact_id}/download")
def download_artifact(run_id: str, artifact_id: str) -> Response:
    try:
        name, content, media_type = workflow.download(run_id, artifact_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="artifact not found for this run") from exc
    return Response(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@run_router.get("/result-template/{mode}")
def result_template(run_id: str, mode: str) -> Response:
    try:
        name, content = workflow.result_template(run_id, mode)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(content, media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@run_router.get("/task-sheet")
def task_sheet(run_id: str) -> Response:
    try:
        name, content = workflow.task_sheet(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(content, media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@run_router.post("/simulation-results")
async def upload_simulation_results(run_id: str, file: UploadFile) -> dict:
    content = await file.read(5_000_001)
    if len(content) > 5_000_000:
        raise HTTPException(status_code=413, detail="file exceeds 5 MB")
    try:
        return workflow.ingest_task_results(run_id, file.filename or "results.csv", content)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class RecommendBatch(BaseModel):
    batch_size: int = Field(default=4, ge=1, le=20)
    kind: str = "ei"


class ApproveBatch(BaseModel):
    approve: bool
    reason: str = ""


@run_router.post("/recommend-next")
def recommend_next(run_id: str, payload: RecommendBatch) -> dict:
    try:
        return workflow.recommend_next_batch(run_id, payload.batch_size, payload.kind)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@run_router.post("/approve-next")
def approve_next(run_id: str, payload: ApproveBatch) -> dict:
    try:
        return workflow.approve_next_batch(run_id, payload.approve, payload.reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

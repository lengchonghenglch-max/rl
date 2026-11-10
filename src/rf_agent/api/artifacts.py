from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, UploadFile
from fastapi.responses import Response

from rf_agent.contracts import DataLabel
from rf_agent.services.workflow import workflow

router = APIRouter(prefix="/runs/{run_id}/artifacts", tags=["artifacts"])


@router.get("")
def list_artifacts(run_id: str) -> list[dict]:
    try:
        return workflow.artifacts(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc


@router.post("")
async def upload_artifact(run_id: str, file: UploadFile,
                          data_kind: Annotated[DataLabel, Query()] = DataLabel.SIMULATION) -> dict:
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
    return Response(content, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="{name}"'})

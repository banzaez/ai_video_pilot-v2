"""FastAPI routes for pipeline jobs."""

from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app.jobs.runner import (
    active_snapshot,
    enqueue_and_maybe_start,
    restart_job,
    stop_job,
)
from app.jobs.store import list_jobs, load_job, read_log_tail


def _check_auth(authorization: str | None = Header(default=None)) -> None:
    token = os.environ.get("JOBS_API_TOKEN", "").strip()
    if not token:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    if authorization[7:].strip() != token:
        raise HTTPException(status_code=403, detail="invalid token")


class CreateJobBody(BaseModel):
    input: str = Field(..., description="session:01_20260601 or data/video/…")
    stage: str | None = None
    stage_from: str | None = Field(default=None, alias="from")
    stage_to: str | None = Field(default=None, alias="to")

    model_config = {"populate_by_name": True}


def create_app() -> FastAPI:
    app = FastAPI(title="AI Video Pilot Jobs", version="1.0")

    @app.get("/api/jobs/health")
    def health() -> dict[str, str]:
        return {"ok": "1"}

    @app.get("/api/jobs/active")
    def active(_: None = Depends(_check_auth)) -> dict[str, Any]:
        snap = active_snapshot()
        return {"job": snap}

    @app.get("/api/jobs")
    def jobs_list(
        limit: int = Query(50, ge=1, le=200),
        _: None = Depends(_check_auth),
    ) -> dict[str, Any]:
        items = [j.to_dict() for j in list_jobs(limit=limit)]
        return {"jobs": items}

    @app.get("/api/jobs/{job_id}")
    def job_get(job_id: str, _: None = Depends(_check_auth)) -> dict[str, Any]:
        job = load_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="not found")
        return job.to_dict()

    @app.get("/api/jobs/{job_id}/log")
    def job_log(
        job_id: str,
        offset: int = Query(0, ge=0),
        _: None = Depends(_check_auth),
    ) -> dict[str, Any]:
        if not load_job(job_id):
            raise HTTPException(status_code=404, detail="not found")
        return read_log_tail(job_id, offset=offset)

    @app.post("/api/jobs")
    def job_create(body: CreateJobBody, _: None = Depends(_check_auth)) -> dict[str, Any]:
        try:
            job = enqueue_and_maybe_start(
                input_path=body.input,
                stage=body.stage,
                stage_from=body.stage_from,
                stage_to=body.stage_to,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return job.to_dict()

    @app.post("/api/jobs/{job_id}/stop")
    def job_stop(job_id: str, _: None = Depends(_check_auth)) -> dict[str, Any]:
        try:
            job = stop_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return job.to_dict()

    @app.post("/api/jobs/{job_id}/restart")
    def job_restart(job_id: str, _: None = Depends(_check_auth)) -> dict[str, Any]:
        try:
            job = restart_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return job.to_dict()

    return app


app = create_app()

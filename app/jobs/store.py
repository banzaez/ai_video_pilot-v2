"""Disk store for pipeline jobs: data/jobs/{id}/meta.json + log.txt."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from app.jobs.models import Job, JobStatus

_SESSION_RE = re.compile(r"^session:\d{2}_\d{8}$")
_SESSION_KEY_RE = re.compile(r"^\d{2}_\d{8}$")


def project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", ".."))


def jobs_root(root: str | None = None) -> str:
    base = root or project_root()
    path = os.path.join(base, "data", "jobs")
    os.makedirs(path, exist_ok=True)
    return path


def job_dir(job_id: str, root: str | None = None) -> str:
    return os.path.join(jobs_root(root), job_id)


def meta_path(job_id: str, root: str | None = None) -> str:
    return os.path.join(job_dir(job_id, root), "meta.json")


def log_path(job_id: str, root: str | None = None) -> str:
    return os.path.join(job_dir(job_id, root), "log.txt")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def new_job_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def save_job(job: Job, root: str | None = None) -> None:
    d = job_dir(job.id, root)
    os.makedirs(d, exist_ok=True)
    if not job.log_path:
        job.log_path = log_path(job.id, root)
    path = meta_path(job.id, root)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(job.to_dict(), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_job(job_id: str, root: str | None = None) -> Job | None:
    path = meta_path(job_id, root)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    job = Job.from_dict(data)
    if not job.log_path:
        job.log_path = log_path(job_id, root)
    return job


def list_jobs(limit: int = 50, root: str | None = None) -> list[Job]:
    base = jobs_root(root)
    ids: list[str] = []
    for name in os.listdir(base):
        if os.path.isfile(meta_path(name, root)):
            ids.append(name)
    ids.sort(reverse=True)
    out: list[Job] = []
    for jid in ids[: max(1, int(limit))]:
        job = load_job(jid, root)
        if job:
            out.append(job)
    return out


def find_active(root: str | None = None) -> Job | None:
    for job in list_jobs(limit=200, root=root):
        if job.status in ("pending", "running"):
            return job
    return None


def find_running(root: str | None = None) -> Job | None:
    for job in list_jobs(limit=200, root=root):
        if job.status == "running":
            return job
    return None


def read_log_tail(job_id: str, offset: int = 0, root: str | None = None) -> dict[str, Any]:
    path = log_path(job_id, root)
    if not os.path.isfile(path):
        return {"offset": 0, "size": 0, "text": ""}
    size = os.path.getsize(path)
    off = max(0, min(int(offset), size))
    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    return {"offset": off + len(raw), "size": size, "text": text}


def append_log(job_id: str, text: str, root: str | None = None) -> None:
    if not text:
        return
    path = log_path(job_id, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def create_job_record(
    *,
    input_path: str,
    cmd: list[str],
    stage: str | None = None,
    stage_from: str | None = None,
    stage_to: str | None = None,
    root: str | None = None,
) -> Job:
    jid = new_job_id()
    job = Job(
        id=jid,
        status="pending",
        input=input_path,
        cmd=cmd,
        created_at=_now_iso(),
        stage=stage,
        stage_from=stage_from,
        stage_to=stage_to,
        log_path=log_path(jid, root),
    )
    save_job(job, root)
    open(job.log_path, "a", encoding="utf-8").close()
    return job


def update_status(
    job: Job,
    status: JobStatus,
    *,
    pid: int | None = None,
    exit_code: int | None = None,
    error: str | None = None,
    root: str | None = None,
) -> Job:
    job.status = status
    if pid is not None:
        job.pid = pid
    if exit_code is not None:
        job.exit_code = exit_code
    if error is not None:
        job.error = error
    if status == "running" and not job.started_at:
        job.started_at = _now_iso()
    if status in ("done", "failed", "stopped"):
        job.ended_at = _now_iso()
    save_job(job, root)
    return job


def validate_input_path(raw: str, root: str | None = None) -> str:
    """Return normalized input for CLI or raise ValueError."""
    s = str(raw or "").strip()
    if not s:
        raise ValueError("input пустой")
    if _SESSION_RE.match(s):
        return s
    if _SESSION_KEY_RE.match(s):
        return f"session:{s}"
    base = root or project_root()
    if os.path.isabs(s):
        path = os.path.normpath(s)
    else:
        path = os.path.normpath(os.path.join(base, s))
    video_root = os.path.normpath(os.path.join(base, "data", "video"))
    if not (path == video_root or path.startswith(video_root + os.sep)):
        raise ValueError("input должен быть session:… или путём под data/video")
    if not os.path.exists(path):
        raise ValueError(f"путь не найден: {s}")
    # Prefer relative for CLI
    try:
        rel = os.path.relpath(path, base)
        if not rel.startswith(".."):
            return rel.replace("\\", "/")
    except ValueError:
        pass
    return path

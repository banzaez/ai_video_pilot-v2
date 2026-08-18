"""Build CLI and spawn pipeline processes."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from app.jobs.models import STAGE_WHITELIST, Job
from app.jobs.store import (
    append_log,
    find_running,
    list_jobs,
    load_job,
    project_root,
    update_status,
    validate_input_path,
)

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_active_proc: subprocess.Popen | None = None
_active_job_id: str | None = None
_pump_thread: threading.Thread | None = None


def resolve_python(root: str | None = None) -> str:
    base = root or project_root()
    for cand in (
        os.path.join(base, "venv", "bin", "python"),
        os.path.join(base, ".venv", "bin", "python"),
    ):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return sys.executable


def _check_stage(name: str | None, field: str) -> str | None:
    if name is None or name == "":
        return None
    s = str(name).strip()
    if s not in STAGE_WHITELIST:
        raise ValueError(f"недопустимый {field}: {s}")
    return s


def build_cmdline(
    *,
    input_path: str,
    stage: str | None = None,
    stage_from: str | None = None,
    stage_to: str | None = None,
    root: str | None = None,
) -> list[str]:
    """Whitelist-safe argv for python -m app.main."""
    inp = validate_input_path(input_path, root)
    st = _check_stage(stage, "stage")
    fr = _check_stage(stage_from, "from")
    to = _check_stage(stage_to, "to")

    if st and (fr or to):
        raise ValueError("укажите либо stage, либо from/to, не оба")
    if to and not fr:
        raise ValueError("--to требует --from")
    if not st and not fr:
        raise ValueError("нужен stage или from")

    py = resolve_python(root)
    cmd = [py, "-m", "app.main", "--input", inp]
    if st:
        cmd.extend(["--stage", st])
    else:
        assert fr is not None
        cmd.extend(["--from", fr])
        if to:
            cmd.extend(["--to", to])
    return cmd


def _normalize_chunk(raw: bytes) -> str:
    """tqdm \\r → newlines for log file."""
    text = raw.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def _pump_output(job_id: str, proc: subprocess.Popen, root: str | None) -> None:
    assert proc.stdout is not None
    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            append_log(job_id, _normalize_chunk(chunk), root)
    finally:
        code = proc.wait()
        with _lock:
            global _active_proc, _active_job_id
            if _active_job_id == job_id:
                _active_proc = None
                _active_job_id = None
        job = load_job(job_id, root)
        if not job:
            return
        if job.status == "stopped":
            update_status(job, "stopped", exit_code=code, root=root)
        elif code == 0:
            update_status(job, "done", exit_code=0, root=root)
            append_log(job_id, f"\n[jobs] exit 0\n", root)
        else:
            update_status(job, "failed", exit_code=code, root=root)
            append_log(job_id, f"\n[jobs] exit {code}\n", root)
        # Start next pending if any
        _try_start_next(root)


def _try_start_next(root: str | None) -> None:
    with _lock:
        if _active_proc is not None:
            return
    pendings = [j for j in list_jobs(limit=200, root=root) if j.status == "pending"]
    pendings.sort(key=lambda j: j.created_at)
    if not pendings:
        return
    start_job(pendings[0].id, root=root)


def start_job(job_id: str, root: str | None = None) -> Job:
    root = root or project_root()
    job = load_job(job_id, root)
    if not job:
        raise ValueError(f"job не найден: {job_id}")
    if job.status not in ("pending", "stopped", "failed"):
        if job.status == "running":
            return job
        raise ValueError(f"job в статусе {job.status}, нельзя стартовать")

    with _lock:
        global _active_proc, _active_job_id, _pump_thread
        running = find_running(root)
        if running and running.id != job_id:
            # Keep pending; queue will pick it
            if job.status != "pending":
                update_status(job, "pending", root=root)
            raise ValueError(f"уже running: {running.id}")
        if _active_proc is not None:
            raise ValueError("уже есть активный процесс")

        cwd = root
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        append_log(job_id, f"[jobs] start: {' '.join(job.cmd)}\n", root)
        proc = subprocess.Popen(
            job.cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            bufsize=0,
        )
        _active_proc = proc
        _active_job_id = job_id
        update_status(job, "running", pid=proc.pid, root=root)
        t = threading.Thread(target=_pump_output, args=(job_id, proc, root), daemon=True)
        _pump_thread = t
        t.start()
        return load_job(job_id, root) or job


def stop_job(job_id: str, root: str | None = None) -> Job:
    root = root or project_root()
    job = load_job(job_id, root)
    if not job:
        raise ValueError(f"job не найден: {job_id}")
    if job.status == "pending":
        update_status(job, "stopped", exit_code=None, root=root)
        append_log(job_id, "[jobs] cancelled (was pending)\n", root)
        return load_job(job_id, root) or job
    if job.status != "running":
        return job

    with _lock:
        proc = _active_proc if _active_job_id == job_id else None
        pid = job.pid
    update_status(job, "stopped", root=root)
    append_log(job_id, "\n[jobs] stopping…\n", root)
    try:
        if proc and proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        elif pid:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    except ProcessLookupError:
        pass
    # Wait briefly then SIGKILL
    deadline = time.time() + 5
    while time.time() < deadline:
        if proc and proc.poll() is not None:
            break
        time.sleep(0.2)
    else:
        try:
            if proc and proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            elif pid:
                os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return load_job(job_id, root) or job


def enqueue_and_maybe_start(
    *,
    input_path: str,
    stage: str | None = None,
    stage_from: str | None = None,
    stage_to: str | None = None,
    root: str | None = None,
) -> Job:
    from app.jobs.store import create_job_record

    root = root or project_root()
    cmd = build_cmdline(
        input_path=input_path,
        stage=stage,
        stage_from=stage_from,
        stage_to=stage_to,
        root=root,
    )
    job = create_job_record(
        input_path=validate_input_path(input_path, root),
        cmd=cmd,
        stage=stage,
        stage_from=stage_from,
        stage_to=stage_to,
        root=root,
    )
    if find_running(root):
        append_log(job.id, "[jobs] queued (another job is running)\n", root)
        return job
    try:
        return start_job(job.id, root=root)
    except ValueError as exc:
        append_log(job.id, f"[jobs] queued: {exc}\n", root)
        return load_job(job.id, root) or job


def restart_job(job_id: str, root: str | None = None) -> Job:
    root = root or project_root()
    src = load_job(job_id, root)
    if not src:
        raise ValueError(f"job не найден: {job_id}")
    if src.status == "running":
        stop_job(job_id, root=root)
        time.sleep(0.3)
    return enqueue_and_maybe_start(
        input_path=src.input,
        stage=src.stage,
        stage_from=src.stage_from,
        stage_to=src.stage_to,
        root=root,
    )


def active_snapshot(root: str | None = None) -> dict[str, Any] | None:
    from app.jobs.store import find_active

    job = find_running(root) or find_active(root)
    if not job:
        return None
    return job.to_dict()

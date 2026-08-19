"""Job records for pipeline runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

JobStatus = Literal["pending", "running", "done", "failed", "stopped"]

STAGE_WHITELIST = frozenset(
    {
        "info",
        "detect",
        "tracklets",
        "pose",
        "feet",
        "tracklet_reid",
        "tracklet_link",
        "track",
        "camera_link",
        "day_link",
        "all",
        "no_merge",
    }
)


@dataclass
class Job:
    id: str
    status: JobStatus
    input: str
    cmd: list[str]
    created_at: str
    stage: str | None = None
    stage_from: str | None = None
    stage_to: str | None = None
    pid: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    exit_code: int | None = None
    error: str | None = None
    log_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        return cls(
            id=str(data["id"]),
            status=data.get("status") or "pending",  # type: ignore[arg-type]
            input=str(data.get("input") or ""),
            cmd=list(data.get("cmd") or []),
            created_at=str(data.get("created_at") or ""),
            stage=data.get("stage"),
            stage_from=data.get("stage_from") or data.get("from"),
            stage_to=data.get("stage_to") or data.get("to"),
            pid=data.get("pid"),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            exit_code=data.get("exit_code"),
            error=data.get("error"),
            log_path=str(data.get("log_path") or ""),
        )

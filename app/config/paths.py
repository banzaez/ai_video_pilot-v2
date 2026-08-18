"""Пути к артефактам пайплайна."""

from __future__ import annotations

import os

from app.config.settings import Settings
from app.session.discover import parse_session_input


def video_stem(settings: Settings) -> str:
    sk = session_key(settings)
    if sk:
        return sk
    return os.path.splitext(os.path.basename(str(settings.input_path)))[0]


def session_key(settings: Settings) -> str | None:
    """Ключ session из input_path (`session:01_20260601`)."""
    return parse_session_input(str(settings.input_path))


def video_work_dir(settings: Settings) -> str:
    """Папка артефактов: session → {key}/; legacy → {stem}/."""
    root = settings.json_output_dir or "data/results"
    sk = session_key(settings)
    if sk:
        return os.path.join(str(root), sk)
    return os.path.join(str(root), os.path.splitext(os.path.basename(str(settings.input_path)))[0])


def info_json_path(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), "info.json")


def detections_json_path(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), "detections.json")


def tracking_json_path(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), "tracking.json")


def poses_json_path(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), "poses.json")


def feet_json_path(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), "feet.json")


def tracklet_frames_json_path(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), "tracklet_frames.json")


def tracklets_json_path(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), "tracklets.json")


def tracklet_reid_json_path(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), "tracklet_reid.json")


def tracklet_pose_cache_path(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), "tracklet_pose_cache.json")


def tracklet_reid_npz_path(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), "tracklet_reid.npz")


def tracklet_links_json_path(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), "tracklet_links.json")


def tracklet_crops_dir(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), settings.tracklet_crops_dir)


def tracks_json_path(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), "tracks.json")


def camera_face_json_path(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), "camera_face.json")


def camera_face_npz_path(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), "camera_face.npz")


def camera_links_json_path(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), "camera_links.json")


def face_crops_dir(settings: Settings) -> str:
    return os.path.join(video_work_dir(settings), getattr(settings, "camera_link_face_crops_dir", "face_crops"))


def json_output_path(settings: Settings) -> str:
    path = tracking_json_path(settings)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def results_root(settings: Settings) -> str:
    return str(settings.json_output_dir or "data/results")


def cameras_dir(settings: Settings) -> str:
    cfg = os.path.abspath(settings.config_path or "config.yaml")
    cfg_dir = os.path.dirname(cfg)
    if os.path.basename(cfg_dir) == "config" and os.path.basename(os.path.dirname(cfg_dir)) == "data":
        project_root = os.path.dirname(os.path.dirname(cfg_dir))
    else:
        project_root = cfg_dir
    return os.path.join(project_root, "data", "maps", "cameras")

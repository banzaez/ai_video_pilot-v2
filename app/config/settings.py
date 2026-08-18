"""Dataclass настроек пайплайна."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DETECTION_BACKEND_CHOICES = (
    "yolo",
    "rtdetr",
    "rtdetr_v2",
)
TRACKER_CHOICES = (
    "bytetrack",
    "botsort",
    "ocsort",
    "deepocsort",
    "fasttrack",
    "tracktrack",
)
TRACKLET_MODE_CHOICES = (
    "direct",
    "tracklet_global",
)
STAGE_CHOICES = (
    "info",
    "detect",
    "tracklets",
    "tracklet_reid",
    "tracklet_link",
    "track",
    "pose",
    "feet",
    "camera_link",
    "all",
    "no_merge",  # алиас all (раньше: без финального link)
)


@dataclass
class NvrSettings:
    enabled: bool = True
    host: str = "183.88.220.86"
    port: int = 8003
    username: str = "admin"
    password: str = ""
    track_ids: list[str] = field(default_factory=lambda: ["1401", "1501"])
    track_camera_map: dict[str, str] = field(
        default_factory=lambda: {"1401": "Camera_01", "1501": "Camera_02"}
    )
    search_lookback_hours: int = 24
    connect_timeout_sec: float = 30.0
    read_timeout_search_sec: float = 120.0
    read_timeout_download_sec: float = 900.0


@dataclass
class Settings:
    nvr: NvrSettings = field(default_factory=NvrSettings)
    detection_backend: str = "yolo"
    models_dir: str = "data/models/detect"
    model_path: str = "data/models/detect/yolo26n.pt"
    conf: float = 0.35
    classes: list[int] = field(default_factory=lambda: [0])
    device: str = "auto"
    quantize: int | None = 16
    batch_size: int = 16
    imgsz: int = 640
    detect_every_n: int = 1
    nms_iou: float = 0.5

    tracker_type: str = "bytetrack"
    tracker_params: dict[str, Any] = field(default_factory=dict)
    min_bbox_area: float = 0.0
    min_bbox_side: float = 0.0
    min_track_sec: float = 0.0

    tracklet_mode: str = "tracklet_global"
    tracklet_local_tracker: str = "bytetrack"
    tracklet_local_config: str | None = None
    tracklet_local_params: dict[str, Any] = field(default_factory=dict)
    tracklet_min_obs: int = 2
    tracklet_min_sec: float = 0.0
    tracklet_reid_top_k: int = 3
    tracklet_reid_pick: str = "spread"
    tracklet_reid_backend: str = "solider"
    tracklet_reid_model: str = "osnet_x1_0"
    tracklet_reid_weights: str = "data/models/reid/osnet_x1_0_msmt17.pth"
    tracklet_reid_device: str = "cpu"
    tracklet_reid_batch_size: int = 32
    tracklet_reid_save_crops: bool = False
    tracklet_reid_solider_weights: str = "data/models/reid/solider_swin_base_msmt17.pth"
    tracklet_reid_solider_semantic_weight: float = 0.2
    tracklet_reid_solider_image_size: tuple[int, int] = (384, 128)
    tracklet_reid_solider_transformer: str = "swin_base_patch4_window7_224"
    tracklet_reid_pad: float = 0.04
    tracklet_crops_dir: str = "tracklet_crops"
    tracklet_link_max_gap_sec: float = 20.0
    tracklet_link_min_reid_score: float = 0.55
    tracklet_link_pass1_min_score: float = 0.70
    tracklet_link_max_spatial_px: float = 700.0
    tracklet_link_max_spatial_m: float = 4.0
    tracklet_link_motion_sigma_px: float = 180.0
    tracklet_link_motion_sigma_m: float = 1.5
    tracklet_link_size_log_scale: float = 0.45
    tracklet_link_w_reid: float = 0.55
    tracklet_link_w_motion: float = 0.25
    tracklet_link_w_size: float = 0.10
    tracklet_link_w_gap: float = 0.10
    tracklet_link_pass2_min_score: float = 0.0
    tracklet_link_pass4_max_overlap_sec: float = 2.0
    tracklet_link_pass4_min_reid: float = 0.95
    tracklet_link_pass4_min_score: float = 0.85
    tracklet_link_window_sec: float = 120.0
    tracklet_link_window_overlap_sec: float = 15.0
    tracklet_link_solver: str = "hungarian"
    tracklet_link_torso_height_m: float = 0.0
    feet_torso_height_m: float = 0.0
    feet_person_height_m: float = 1.70
    feet_smooth_window: int = 5
    feet_max_speed_mps: float = 2.0
    pose_model: str = "yolo26s-pose.pt"
    pose_conf: float = 0.25
    pose_kpt_min: float = 0.25
    pose_every_n: int = 4

    camera_link_enabled: bool = True
    camera_link_model: str = "buffalo_l"
    camera_link_face_models: tuple[str, ...] = ("buffalo_l", "antelopev2")
    camera_link_face_top_k: int = 5
    camera_link_face_max_attempts: int = 15
    camera_link_min_face_score: float = 0.60
    camera_link_save_face_crops: bool = True
    camera_link_face_crops_dir: str = "face_crops"
    camera_link_max_gap_sec: float = 120.0
    camera_link_max_speed_mps: float = 3.5
    camera_link_motion_sigma_m: float = 2.0
    camera_link_w_face: float = 0.50
    camera_link_w_reid: float = 0.30
    camera_link_w_motion: float = 0.20
    camera_link_min_combo_score: float = 0.55
    camera_link_solver: str = "hungarian"

    input_path: str = "data/video"
    json_output_dir: str | None = "data/results"
    workers: int | None = None

    stage: str = "all"
    stage_onward: bool = False
    stage_until: str | None = None

    config_path: str = "config.yaml"

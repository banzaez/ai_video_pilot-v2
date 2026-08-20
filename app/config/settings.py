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
    "day_link",
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
    track_camera_map: dict[str, str] = field(default_factory=dict)
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
    tracklet_reid_pose_weight: float = 0.40
    tracklet_reid_crowd_penalty: float = 0.50
    tracklet_reid_min_completeness: float = 0.20
    tracklet_reid_feathering_enabled: bool = False
    tracklet_reid_feathering_mode: str = "pose"
    tracklet_reid_feathering_sigma: float = 15.0
    tracklet_reid_feathering_bone_thickness: float = 0.18
    tracklet_reid_feathering_bg_color: tuple[int, int, int] = (128, 128, 128)
    tracklet_reid_trim_enabled: bool = True
    tracklet_reid_trim_start: int = 2
    tracklet_reid_trim_end: int = 2
    tracklet_reid_trim_min_len: int = 8
    tracklet_reid_kinematic_enabled: bool = True
    tracklet_reid_kinematic_max_speed_ratio: float = 3.0
    tracklet_reid_kinematic_max_area_ratio: float = 2.2
    tracklet_reid_color_enabled: bool = True
    tracklet_reid_color_min_similarity: float = 0.50
    tracklet_reid_color_min_candidates: int = 3
    tracklet_crops_dir: str = "tracklet_crops"
    tracklet_link_max_gap_sec: float = 20.0
    tracklet_link_max_overlap_sec: float = 2.0
    tracklet_link_min_reid_score: float = 0.55
    tracklet_link_pass0_min_reid: float = 0.0
    tracklet_link_pass0_min_score: float = 0.0
    tracklet_link_pass_alone_enabled: bool = True
    tracklet_link_pass_alone_radius_m: float = 2.0
    tracklet_link_pass_alone_max_gap_sec: float = 15.0
    tracklet_link_pass_alone_max_dist_m: float = 3.0
    tracklet_link_pass_alone_max_speed_mps: float = 2.0
    tracklet_link_pass_alone_min_reid: float = 0.0
    tracklet_link_w_reid: float = 0.85
    tracklet_link_w_gap: float = 0.15
    feet_torso_height_m: float = 0.0
    feet_person_height_m: float = 1.70
    feet_smooth_window: int = 5
    feet_max_speed_mps: float = 2.0
    pose_model: str = "yolo26s-pose.pt"
    pose_conf: float = 0.25
    pose_kpt_min: float = 0.25

    camera_link_enabled: bool = False
    camera_link_model: str = "buffalo_l"
    camera_link_face_models: tuple[str, ...] = ("buffalo_l", "antelopev2")
    camera_link_face_top_k: int = 5
    camera_link_face_max_attempts: int = 7
    camera_link_face_min_gap_sec: float = 0.5
    camera_link_face_dup_cos: float = 0.97
    camera_link_min_face_score: float = 0.60
    camera_link_min_pose_face_score: float = 0.35
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

    # --- Global Day Link Settings (day_link) ---
    day_link_enabled: bool = True
    day_link_top_k: int = 0
    day_link_save_crops: bool = True
    day_link_max_gap_sec: float = 300.0

    # Pass 0: Alone Geo (совпадение на 2D-карте + изоляция на всех камерах)
    day_link_pass0_enabled: bool = True
    day_link_pass0_radius_m: float = 2.0
    day_link_pass0_min_overlap_sec: float = 5.0
    day_link_pass0_max_gap_sec: float = 10.0
    day_link_pass0_max_dist_m: float = 3.0
    day_link_pass0_max_speed_mps: float = 2.5
    day_link_pass0_min_reid: float = 0.90

    # Pass 1: Strict ReID (ReID >= 0.96)
    day_link_pass1_enabled: bool = True
    day_link_pass1_min_reid: float = 0.96
    day_link_pass1_max_gap_sec: float = 300.0

    input_path: str = "data/video"
    json_output_dir: str | None = "data/results"
    workers: int | None = None

    stage: str = "all"
    stage_onward: bool = False
    stage_until: str | None = None

    config_path: str = "config.yaml"

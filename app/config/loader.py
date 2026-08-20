"""Загрузка YAML + CLI → Settings."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import yaml

from app.config.settings import (
    DETECTION_BACKEND_CHOICES,
    STAGE_CHOICES,
    TRACKER_CHOICES,
    TRACKLET_MODE_CHOICES,
    Settings,
)
from app.model_cache import DETECT_MODELS_DIR, REID_MODELS_DIR, join_models_path

logger = logging.getLogger(__name__)


def _solider_image_size(raw: Any) -> tuple[int, int]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return (int(raw[0]), int(raw[1]))
    return (384, 128)


def _link_section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    raw = cfg.get(name)
    return raw if isinstance(raw, dict) else {}


def _link_value(
    cfg: dict[str, Any],
    section: dict[str, Any],
    key: str,
    *flat_keys: str,
    default: Any,
) -> Any:
    if key in section:
        return section[key]
    for flat_key in flat_keys:
        if flat_key in cfg:
            return cfg[flat_key]
    return default


def _parse_day_link_cfg(dl: dict[str, Any]) -> dict[str, Any]:
    """Nested day_link.candidates / pass0 / pass1 + fallback на плоские ключи."""
    cand = _link_section(dl, "candidates")
    p0 = _link_section(dl, "pass0")
    p1 = _link_section(dl, "pass1")

    def g(section: dict[str, Any], key: str, *flat: str, default: Any) -> Any:
        return _link_value(dl, section, key, *flat, default=default)

    return {
        "day_link_enabled": bool(dl.get("enabled", True)),
        "day_link_top_k": int(dl.get("top_k", 0) or 0),
        "day_link_save_crops": bool(dl.get("save_crops", True)),
        "day_link_max_gap_sec": float(g(cand, "max_gap_sec", "max_gap_sec", default=300.0)),
        # Pass 0: Alone Geo
        "day_link_pass0_enabled": bool(g(p0, "enabled", "pass0_enabled", default=True)),
        "day_link_pass0_radius_m": float(g(p0, "radius_m", "pass0_radius_m", default=2.0)),
        "day_link_pass0_min_overlap_sec": float(
            g(p0, "min_overlap_sec", "pass0_min_overlap_sec", default=5.0)
        ),
        "day_link_pass0_max_gap_sec": float(g(p0, "max_gap_sec", "pass0_max_gap_sec", default=10.0)),
        "day_link_pass0_max_dist_m": float(g(p0, "max_dist_m", "pass0_max_dist_m", default=3.0)),
        "day_link_pass0_max_speed_mps": float(
            g(p0, "max_speed_mps", "pass0_max_speed_mps", default=2.5)
        ),
        "day_link_pass0_min_reid": float(g(p0, "min_reid", "pass0_min_reid", default=0.90)),
        # Pass 1: Strict ReID
        "day_link_pass1_enabled": bool(g(p1, "enabled", "pass1_enabled", default=True)),
        "day_link_pass1_min_reid": float(g(p1, "min_reid", "pass1_min_reid", default=0.96)),
        "day_link_pass1_max_gap_sec": float(
            g(p1, "max_gap_sec", "pass1_max_gap_sec", default=300.0)
        ),
    }


def _parse_tracklet_link_cfg(link_cfg: dict[str, Any]) -> dict[str, Any]:
    """Nested link.candidates / combo / pass0 / pass_alone_geo + fallback на плоские ключи."""
    cand = _link_section(link_cfg, "candidates")
    combo = _link_section(link_cfg, "combo")
    p0 = _link_section(link_cfg, "pass0")
    p_alone = _link_section(link_cfg, "pass_alone_geo")

    def g(section: dict[str, Any], key: str, *flat: str, default: Any) -> Any:
        return _link_value(link_cfg, section, key, *flat, default=default)

    return {
        "max_gap_sec": float(g(cand, "max_gap_sec", "max_gap_sec", default=20.0)),
        "max_overlap_sec": float(g(cand, "max_overlap_sec", "max_overlap_sec", default=2.0)),
        "min_reid_score": float(g(cand, "min_reid_score", "min_reid_score", default=0.55)),
        "w_reid": float(g(combo, "w_reid", "w_reid", default=0.85)),
        "w_gap": float(g(combo, "w_gap", "w_gap", default=0.15)),
        "pass0_min_reid": float(g(p0, "min_reid", "pass0_min_reid", default=0.0)),
        "pass0_min_score": float(
            g(p0, "min_score", "pass0_min_score", "min_combo", default=0.0)
        ),
        "pass_alone_enabled": bool(p_alone.get("enabled", True)),
        "pass_alone_radius_m": float(g(p_alone, "radius_m", "pass_alone_radius_m", default=2.0)),
        "pass_alone_max_gap_sec": float(g(p_alone, "max_gap_sec", "pass_alone_max_gap_sec", default=15.0)),
        "pass_alone_max_dist_m": float(g(p_alone, "max_dist_m", "pass_alone_max_dist_m", default=3.0)),
        "pass_alone_max_speed_mps": float(g(p_alone, "max_speed_mps", "pass_alone_max_speed_mps", default=2.0)),
        "pass_alone_min_reid": float(g(p_alone, "min_reid", "pass_alone_min_reid", default=0.0)),
    }


def _tracker_inline_params(raw: dict[str, Any]) -> dict[str, Any]:
    inline: dict[str, Any] = {}
    for key, value in raw.items():
        if key in ("type", "tracker", "config") or value is None:
            continue
        inline[str(key)] = value
    return inline


def _reid_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "device",
        "batch_size",
        "solider_weights",
        "solider_semantic_weight",
        "solider_image_size",
        "solider_transformer",
        "reid_device",
    )
    out: dict[str, Any] = {}
    for key in keys:
        if key in raw:
            out[key] = raw[key]
    if "reid_device" in out and "device" not in out:
        out["device"] = out.pop("reid_device")
    return out


def _resolve_reid_settings(
    reid_cfg: dict[str, Any],
    *,
    reid_models_dir: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = {**reid_cfg, **(overrides or {})}
    models_dir = str(reid_models_dir or REID_MODELS_DIR).strip() or REID_MODELS_DIR
    return {
        "backend": "solider",
        "device": str(merged.get("device") or "cpu"),
        "batch_size": max(1, int(merged.get("batch_size", 32))),
        "solider_weights": join_models_path(
            models_dir,
            str(merged.get("solider_weights") or "solider_swin_base_msmt17.pth"),
        ),
        "solider_semantic_weight": float(merged.get("solider_semantic_weight", 0.2)),
        "solider_image_size": _solider_image_size(merged.get("solider_image_size")),
        "solider_transformer": str(
            merged.get("solider_transformer") or "swin_base_patch4_window7_224"
        ),
    }


def load_yaml(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        logger.warning("Конфиг не найден: %s — используем значения по умолчанию", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_quantize(quantize: int | str | None, device: str) -> int | None:
    """Нормализовать quantize; на CPU всегда FP32 (None)."""
    if device == "cpu":
        return None
    if quantize is None:
        return None
    if isinstance(quantize, str):
        q = quantize.strip().lower()
        if q in ("", "none", "fp32", "32"):
            return None
        if q in ("fp16", "16", "half"):
            return 16
        if q in ("int8", "8"):
            return 8
        raise ValueError(f"Неизвестный quantize: {quantize!r}")
    q_int = int(quantize)
    if q_int in (32, 0):
        return None
    return q_int


def resolve_half(half: bool, device: str) -> bool:
    """Совместимость: True → FP16, если не CPU."""
    return resolve_quantize(16 if half else None, device) == 16


def _resolve_input_path(cli: str | None, yaml_path: str | None) -> str:
    yaml_path = (yaml_path or "data/video").strip() or "data/video"
    if not cli:
        return yaml_path
    cli = cli.strip()
    if cli == "0" or os.path.isdir(cli) or os.path.isfile(cli):
        return cli
    search = yaml_path if os.path.isdir(yaml_path) else os.path.dirname(yaml_path)
    if search:
        cand = os.path.join(search, os.path.basename(cli))
        if os.path.isfile(cand):
            return cand
    return cli


def _parse_detection_model_cfg(
    det_cfg: dict[str, Any],
    backend: str,
    *,
    models_dir: str,
) -> dict[str, Any]:
    """Общие параметры Ultralytics + path для выбранного backend."""
    backend = backend.lower().strip()
    cfg: dict[str, Any] = {}
    shared_keys = ("classes", "quantize", "batch_size", "imgsz", "half", "path")

    for key in shared_keys:
        if key in det_cfg:
            cfg[key] = det_cfg[key]

    # legacy: подгруппы yolo/rtdetr (path и переопределения)
    legacy = dict(det_cfg.get(backend) or {})
    for key in shared_keys:
        if key in legacy:
            cfg[key] = legacy[key]

    models = det_cfg.get("models")
    if isinstance(models, dict) and backend in models:
        cfg["path"] = models[backend]

    if "path" not in cfg:
        cfg["path"] = {
            "yolo": "yolo26n.pt",
            "rtdetr": "rtdetr-l.pt",
            "rtdetr_v2": "PekingU/rtdetr_v2_r50vd",
        }[backend]

    if backend == "rtdetr_v2":
        # Hugging Face id или локальная папка — не склеивать с models_dir
        raw_path = str(cfg.get("path") or "").strip()
        if raw_path and ("/" in raw_path.replace("\\", "/") or os.path.isdir(raw_path)):
            cfg["path"] = raw_path
        else:
            cfg["path"] = join_models_path(models_dir, raw_path)
    else:
        cfg["path"] = join_models_path(models_dir, str(cfg["path"]))
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Person detection and tracking (YOLO / RT-DETR / RT-DETRv2 + ByteTrack family)"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Папка с видео или один файл (путь или имя). Без файла — все ролики в папке",
    )
    parser.add_argument(
        "--day",
        type=str,
        default=None,
        help="День для обработки (формат YYYYMMDD или YYYY-MM-DD)",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--detection-backend",
        type=str,
        choices=list(DETECTION_BACKEND_CHOICES),
        default=None,
        help="yolo | rtdetr | rtdetr_v2 (иначе detection.backend из YAML)",
    )
    parser.add_argument("--tracker", type=str, choices=list(TRACKER_CHOICES), default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--detect-every-n", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    _RANGE_STAGES = [s for s in STAGE_CHOICES if s not in ("all", "no_merge")]
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--stage",
        type=str,
        choices=list(STAGE_CHOICES),
        default=None,
        help="Только эта стадия: info … feet; all=всё; no_merge=алиас all",
    )
    stage_group.add_argument(
        "--from",
        dest="from_stage",
        type=str,
        choices=_RANGE_STAGES,
        default=None,
        help="С этой стадии (можно с --to). Без --to — до конца",
    )
    parser.add_argument(
        "--to",
        dest="to_stage",
        type=str,
        choices=_RANGE_STAGES,
        default=None,
        help="До этой стадии включительно (только вместе с --from)",
    )
    return parser


def settings_from_sources(args: argparse.Namespace | None = None) -> Settings:
    """Приоритет: CLI > YAML > defaults."""
    if args is None:
        args = build_arg_parser().parse_args([])

    cfg = load_yaml(args.config)
    det_cfg = cfg.get("detection") or cfg.get("model") or {}
    models_dir = str(
        cfg.get("models_dir")
        or det_cfg.get("models_dir")
        or DETECT_MODELS_DIR
    ).strip() or DETECT_MODELS_DIR
    backend = str(getattr(args, "detection_backend", None) or det_cfg.get("backend") or "yolo").lower()
    if backend not in DETECTION_BACKEND_CHOICES:
        known = ", ".join(DETECTION_BACKEND_CHOICES)
        raise ValueError(f"Неизвестный detection.backend '{backend}'. Доступны: {known}")

    active_cfg = _parse_detection_model_cfg(
        det_cfg,
        backend,
        models_dir=models_dir,
    )
    tracker_cfg = cfg.get("tracker") or {}
    tracklet_cfg = cfg.get("tracklet_pipeline") or {}
    tracklet_tracker_cfg = tracklet_cfg.get("tracker") or tracklet_cfg.get("local") or {}
    tracklet_reid_cfg = tracklet_cfg.get("reid") or {}
    tracklet_link_cfg = tracklet_cfg.get("link") or {}
    best_frames_cfg = cfg.get("best_frames") or tracklet_cfg.get("best_frames") or {}
    pose_cfg = tracklet_cfg.get("pose") if isinstance(tracklet_cfg.get("pose"), dict) else (cfg.get("pose") or {})
    feet_cfg = tracklet_cfg.get("feet") if isinstance(tracklet_cfg.get("feet"), dict) else (cfg.get("feet") or {})
    link = _parse_tracklet_link_cfg(tracklet_link_cfg)
    reid_models_dir = str(cfg.get("reid_models_dir") or REID_MODELS_DIR).strip() or REID_MODELS_DIR
    tracklet_reid = _resolve_reid_settings(
        cfg.get("reid") or {},
        reid_models_dir=reid_models_dir,
        overrides=_reid_overrides(tracklet_reid_cfg),
    )
    filt_cfg = cfg.get("tracker_filters") or {}
    pipe_cfg = cfg.get("pipeline") or {}
    video_cfg = cfg.get("video") or {}

    device = resolve_device(args.device or det_cfg.get("device", "auto"))
    if "quantize" in active_cfg:
        quantize_raw = active_cfg.get("quantize")
    elif "quantize" in det_cfg:
        quantize_raw = det_cfg.get("quantize")
    elif "half" in active_cfg:
        quantize_raw = 16 if bool(active_cfg.get("half")) else None
    elif "half" in det_cfg:
        quantize_raw = 16 if bool(det_cfg.get("half")) else None
    else:
        quantize_raw = 16
    quantize = resolve_quantize(quantize_raw, device)

    workers = args.workers if args.workers is not None else video_cfg.get("workers")
    if workers is not None:
        workers = int(workers)

    tracker_type = str(args.tracker or tracker_cfg.get("type", "bytetrack")).lower()
    if tracker_type not in TRACKER_CHOICES:
        raise ValueError(
            f"Неизвестный tracker '{tracker_type}'. Доступны: {', '.join(TRACKER_CHOICES)}"
        )
    from app.tracker_config import load_tracker_params

    tracker_inline = _tracker_inline_params(tracker_cfg)
    tracker_params = load_tracker_params(
        tracker_type,
        inline_params=tracker_inline or None,
    )

    tracklet_mode = str(tracklet_cfg.get("mode") or "tracklet_global").lower().strip()
    if tracklet_mode not in TRACKLET_MODE_CHOICES:
        known = ", ".join(TRACKLET_MODE_CHOICES)
        raise ValueError(f"Неизвестный tracklet_pipeline.mode '{tracklet_mode}'. Доступны: {known}")

    tracklet_local_tracker = str(
        tracklet_tracker_cfg.get("type") or tracklet_tracker_cfg.get("tracker") or "bytetrack"
    ).lower()
    if tracklet_local_tracker not in TRACKER_CHOICES:
        raise ValueError(
            f"Неизвестный tracklet tracker '{tracklet_local_tracker}'. "
            f"Доступны: {', '.join(TRACKER_CHOICES)}"
        )
    tracklet_inline = _tracker_inline_params(tracklet_tracker_cfg)
    tracklet_local_params = load_tracker_params(
        tracklet_local_tracker,
        inline_params=tracklet_inline or None,
    )

    settings = Settings(
        models_dir=models_dir,
        detection_backend=backend,
        model_path=join_models_path(models_dir, args.model) if args.model else active_cfg.get("path"),
        conf=args.conf if args.conf is not None else float(det_cfg.get("conf", 0.35)),
        classes=list(active_cfg.get("classes", det_cfg.get("classes", [0]))),
        device=device,
        quantize=quantize,
        batch_size=args.batch_size or int(active_cfg.get("batch_size", det_cfg.get("batch_size", 16))),
        imgsz=args.imgsz if args.imgsz is not None else int(active_cfg.get("imgsz", det_cfg.get("imgsz", 640))),
        detect_every_n=max(
            1,
            int(
                args.detect_every_n
                if args.detect_every_n is not None
                else det_cfg.get("detect_every_n", det_cfg.get("frame_stride", 1))
            ),
        ),
        nms_iou=float(det_cfg.get("iou", 0.5)),
        tracker_type=tracker_type,
        tracker_params=tracker_params,
        min_bbox_area=float(
            filt_cfg.get("min_bbox_area", tracker_cfg.get("min_bbox_area", 0))
        ),
        min_bbox_side=float(
            filt_cfg.get("min_bbox_side", tracker_cfg.get("min_bbox_side", 0))
        ),
        min_track_sec=float(
            filt_cfg.get("min_track_sec", tracker_cfg.get("min_track_sec", 0))
        ),
        tracklet_mode=tracklet_mode,
        tracklet_local_tracker=tracklet_local_tracker,
        tracklet_local_config=None,
        tracklet_local_params=tracklet_local_params,
        tracklet_min_obs=max(1, int(tracklet_cfg.get("min_obs", 2))),
        tracklet_min_sec=float(tracklet_cfg.get("min_sec", 0)),
        tracklet_reid_top_k=max(1, int(tracklet_reid_cfg.get("top_k", 3))),
        tracklet_reid_pick=str(tracklet_reid_cfg.get("pick") or "spread").lower(),
        tracklet_reid_backend=tracklet_reid["backend"],
        tracklet_reid_model="osnet_x1_0",
        tracklet_reid_weights=join_models_path(reid_models_dir, "osnet_x1_0_msmt17.pth"),
        tracklet_reid_device=tracklet_reid["device"],
        tracklet_reid_batch_size=tracklet_reid["batch_size"],
        tracklet_reid_save_crops=bool(tracklet_reid_cfg.get("save_crops", False)),
        tracklet_reid_solider_weights=tracklet_reid["solider_weights"],
        tracklet_reid_solider_semantic_weight=tracklet_reid["solider_semantic_weight"],
        tracklet_reid_solider_image_size=tracklet_reid["solider_image_size"],
        tracklet_reid_solider_transformer=tracklet_reid["solider_transformer"],
        tracklet_reid_pad=float(best_frames_cfg.get("pad", tracklet_reid_cfg.get("pad", 0.04))),
        tracklet_reid_pose_weight=float(best_frames_cfg.get("pose_weight", tracklet_reid_cfg.get("pose_weight", 0.40))),
        tracklet_reid_crowd_penalty=float(best_frames_cfg.get("crowd_penalty", tracklet_reid_cfg.get("crowd_penalty", 0.50))),
        tracklet_reid_min_completeness=float(best_frames_cfg.get("min_completeness", tracklet_reid_cfg.get("min_completeness", 0.20))),
        tracklet_reid_feathering_enabled=bool(
            best_frames_cfg.get("feathering", {}).get("enabled", False)
            if isinstance(best_frames_cfg.get("feathering"), dict)
            else best_frames_cfg.get("feathering_enabled", False)
        ),
        tracklet_reid_feathering_mode=str(
            best_frames_cfg.get("feathering", {}).get("mode", "pose")
            if isinstance(best_frames_cfg.get("feathering"), dict)
            else best_frames_cfg.get("feathering_mode", "pose")
        ),
        tracklet_reid_feathering_sigma=float(
            best_frames_cfg.get("feathering", {}).get("sigma", 15.0)
            if isinstance(best_frames_cfg.get("feathering"), dict)
            else best_frames_cfg.get("feathering_sigma", 15.0)
        ),
        tracklet_reid_feathering_bone_thickness=float(
            best_frames_cfg.get("feathering", {}).get("bone_thickness", 0.18)
            if isinstance(best_frames_cfg.get("feathering"), dict)
            else best_frames_cfg.get("feathering_bone_thickness", 0.18)
        ),
        tracklet_reid_feathering_bg_color=tuple(
            best_frames_cfg.get("feathering", {}).get("bg_color", [128, 128, 128])
            if isinstance(best_frames_cfg.get("feathering"), dict)
            else [128, 128, 128]
        ),
        tracklet_reid_trim_enabled=bool(
            best_frames_cfg.get("trim", {}).get("enabled", True)
            if isinstance(best_frames_cfg.get("trim"), dict)
            else best_frames_cfg.get("trim_enabled", True)
        ),
        tracklet_reid_trim_start=int(
            best_frames_cfg.get("trim", {}).get("start_frames", best_frames_cfg.get("trim", {}).get("start", 2))
            if isinstance(best_frames_cfg.get("trim"), dict)
            else best_frames_cfg.get("trim_start", 2)
        ),
        tracklet_reid_trim_end=int(
            best_frames_cfg.get("trim", {}).get("end_frames", best_frames_cfg.get("trim", {}).get("end", 2))
            if isinstance(best_frames_cfg.get("trim"), dict)
            else best_frames_cfg.get("trim_end", 2)
        ),
        tracklet_reid_trim_min_len=int(
            best_frames_cfg.get("trim", {}).get("min_track_len", best_frames_cfg.get("trim", {}).get("min_len", 8))
            if isinstance(best_frames_cfg.get("trim"), dict)
            else best_frames_cfg.get("trim_min_len", 8)
        ),
        tracklet_reid_kinematic_enabled=bool(
            best_frames_cfg.get("kinematic", {}).get("enabled", True)
            if isinstance(best_frames_cfg.get("kinematic"), dict)
            else best_frames_cfg.get("kinematic_enabled", True)
        ),
        tracklet_reid_kinematic_max_speed_ratio=float(
            best_frames_cfg.get("kinematic", {}).get("max_speed_ratio", 3.0)
            if isinstance(best_frames_cfg.get("kinematic"), dict)
            else best_frames_cfg.get("kinematic_max_speed_ratio", 3.0)
        ),
        tracklet_reid_kinematic_max_area_ratio=float(
            best_frames_cfg.get("kinematic", {}).get("max_area_ratio", 2.2)
            if isinstance(best_frames_cfg.get("kinematic"), dict)
            else best_frames_cfg.get("kinematic_max_area_ratio", 2.2)
        ),
        tracklet_reid_color_enabled=bool(
            best_frames_cfg.get("color_consistency", {}).get("enabled", True)
            if isinstance(best_frames_cfg.get("color_consistency"), dict)
            else best_frames_cfg.get("color_consistency_enabled", True)
        ),
        tracklet_reid_color_min_similarity=float(
            best_frames_cfg.get("color_consistency", {}).get("min_similarity", 0.50)
            if isinstance(best_frames_cfg.get("color_consistency"), dict)
            else best_frames_cfg.get("color_min_similarity", 0.50)
        ),
        tracklet_reid_color_min_candidates=int(
            best_frames_cfg.get("color_consistency", {}).get("min_candidates", 3)
            if isinstance(best_frames_cfg.get("color_consistency"), dict)
            else best_frames_cfg.get("color_min_candidates", 3)
        ),
        tracklet_crops_dir=str(tracklet_reid_cfg.get("crops_dir") or "tracklet_crops"),
        tracklet_link_max_gap_sec=float(link["max_gap_sec"]),
        tracklet_link_max_overlap_sec=float(link.get("max_overlap_sec", 2.0)),
        tracklet_link_min_reid_score=float(link["min_reid_score"]),
        tracklet_link_pass0_min_reid=float(link["pass0_min_reid"]),
        tracklet_link_pass0_min_score=float(link.get("pass0_min_score", 0.0)),
        tracklet_link_pass_alone_enabled=bool(link.get("pass_alone_enabled", True)),
        tracklet_link_pass_alone_radius_m=float(link.get("pass_alone_radius_m", 2.0)),
        tracklet_link_pass_alone_max_gap_sec=float(link.get("pass_alone_max_gap_sec", 15.0)),
        tracklet_link_pass_alone_max_dist_m=float(link.get("pass_alone_max_dist_m", 3.0)),
        tracklet_link_pass_alone_max_speed_mps=float(link.get("pass_alone_max_speed_mps", 2.0)),
        tracklet_link_pass_alone_min_reid=float(link.get("pass_alone_min_reid", 0.0)),
        tracklet_link_w_reid=float(link["w_reid"]),
        tracklet_link_w_gap=float(link["w_gap"]),
        feet_torso_height_m=float(feet_cfg.get("torso_height_m", 0.0)),
        feet_person_height_m=float(feet_cfg.get("person_height_m", 1.70)),
        feet_smooth_window=max(1, int(feet_cfg.get("smooth_window", 5))),
        feet_max_speed_mps=float(feet_cfg.get("max_speed_mps", 2.0)),
        pose_model=str(pose_cfg.get("model") or "yolo26s-pose.pt"),
        pose_conf=float(pose_cfg.get("conf", 0.25)),
        pose_kpt_min=float(pose_cfg.get("kpt_min", 0.25)),
        camera_link_enabled=bool((cfg.get("camera_link") or {}).get("enabled", False)),
        camera_link_model=str(
            ((cfg.get("camera_link") or {}).get("face_models") or [None])[0]
            or (cfg.get("camera_link") or {}).get("face_model", "buffalo_l")
        ),
        camera_link_face_models=tuple(
            str(m)
            for m in (
                (cfg.get("camera_link") or {}).get("face_models")
                or [(cfg.get("camera_link") or {}).get("face_model", "buffalo_l")]
            )
            if str(m).strip()
        ),
        camera_link_face_top_k=max(1, int((cfg.get("camera_link") or {}).get("face_top_k", 5))),
        camera_link_face_max_attempts=max(1, int((cfg.get("camera_link") or {}).get("face_max_attempts", 7))),
        camera_link_face_min_gap_sec=max(0.0, float((cfg.get("camera_link") or {}).get("face_min_gap_sec", 0.5))),
        camera_link_face_dup_cos=float((cfg.get("camera_link") or {}).get("face_dup_cos", 0.97)),
        camera_link_min_face_score=float((cfg.get("camera_link") or {}).get("min_face_score", 0.60)),
        camera_link_min_pose_face_score=float((cfg.get("camera_link") or {}).get("min_pose_face_score", 0.35)),
        camera_link_save_face_crops=bool((cfg.get("camera_link") or {}).get("save_face_crops", True)),
        camera_link_face_crops_dir=str((cfg.get("camera_link") or {}).get("face_crops_dir", "face_crops")),
        camera_link_max_gap_sec=float((cfg.get("camera_link") or {}).get("max_gap_sec", 120.0)),
        camera_link_max_speed_mps=float((cfg.get("camera_link") or {}).get("max_speed_mps", 3.5)),
        camera_link_motion_sigma_m=float((cfg.get("camera_link") or {}).get("motion_sigma_m", 2.0)),
        camera_link_w_face=float((cfg.get("camera_link") or {}).get("w_face", 0.50)),
        camera_link_w_reid=float((cfg.get("camera_link") or {}).get("w_reid", 0.30)),
        camera_link_w_motion=float((cfg.get("camera_link") or {}).get("w_motion", 0.20)),
        camera_link_min_combo_score=float((cfg.get("camera_link") or {}).get("min_combo_score", 0.55)),
        camera_link_solver=str((cfg.get("camera_link") or {}).get("solver", "hungarian")).lower(),
        # Global Day Link (day_link)
        **_parse_day_link_cfg(cfg.get("day_link") or {}),
        input_path=_resolve_input_path(
            (
                f"day:{args.day}"
                if getattr(args, "day", None) and not getattr(args, "input", None)
                else getattr(args, "input", None)
            ),
            video_cfg.get("input_path", "data/video"),
        ),
        json_output_dir=video_cfg.get("json_output_dir", "data/results"),
        workers=workers,
        stage=(
            str(args.from_stage).lower()
            if getattr(args, "from_stage", None)
            else str(args.stage or pipe_cfg.get("stage", "all")).lower()
        ),
        stage_onward=bool(getattr(args, "from_stage", None)),
        stage_until=(
            str(args.to_stage).lower()
            if getattr(args, "to_stage", None)
            else None
        ),
        config_path=args.config,
    )
    if settings.stage not in STAGE_CHOICES:
        raise ValueError(
            f"Неизвестный stage '{settings.stage}'. Доступны: {', '.join(STAGE_CHOICES)}"
        )
    if settings.stage_until and not settings.stage_onward:
        raise ValueError("--to можно только вместе с --from")
    if settings.stage_until and settings.stage_until not in (
        s for s in STAGE_CHOICES if s not in ("all", "no_merge")
    ):
        raise ValueError(f"Неизвестный --to '{settings.stage_until}'")
    return settings

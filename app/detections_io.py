"""Парсинг detections.json → per-frame dict."""

from __future__ import annotations


def detections_from_json(data: dict) -> dict[int, list]:
    """JSON frame_index (1-based) → 0-based кадры без пустых."""
    out: dict[int, list] = {}
    for i, frame in enumerate(data.get("frames") or []):
        if not isinstance(frame, dict):
            raise ValueError(f"detections.frames[{i}]: ожидался объект")
        try:
            fi = int(frame["frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"detections.frames[{i}]: нет frame_index") from exc
        dets = frame.get("detections") or []
        if not dets:
            continue
        cleaned: list[dict] = []
        for j, det in enumerate(dets):
            if not isinstance(det, dict):
                raise ValueError(f"detections.frames[{i}].detections[{j}]: ожидался объект")
            bbox = det.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                raise ValueError(f"detections.frames[{i}].detections[{j}]: нет bbox xyxy")
            try:
                conf = float(det.get("confidence") or 0.0)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"detections.frames[{i}].detections[{j}]: bad confidence") from exc
            item = dict(det)
            item["bbox"] = [float(x) for x in bbox[:4]]
            item["confidence"] = conf
            if "track_id" in item:
                item["track_id"] = int(item["track_id"])
            if "tracklet_id" in item:
                item["tracklet_id"] = int(item["tracklet_id"])
            cleaned.append(item)
        out[fi - 1] = cleaned
    return out

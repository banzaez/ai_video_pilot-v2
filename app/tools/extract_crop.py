"""
Быстрое извлечение кропов детекций напрямую из видеофайла.
Используется сервером админки для On-Demand генерации превью без сохранения тысяч файлов на диск.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import cv2


def extract_crops(
    video_path: str,
    items: list[dict],
    pad_ratio: float = 0.05,
    quality: int = 85,
) -> dict:
    """
    items: list of dicts with:
      - frame: int
      - bbox: [x1, y1, x2, y2]
      - output: str (file path to save)
    """
    if not os.path.exists(video_path):
        return {"ok": False, "error": f"Video not found: {video_path}"}

    if not items:
        return {"ok": True, "extracted": 0}

    # Сортируем по номеру кадра для последовательного чтения видео
    sorted_items = sorted(items, key=lambda it: it.get("frame", 0))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "error": f"Failed to open video: {video_path}"}

    extracted_count = 0
    current_frame_idx = -1

    try:
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for item in sorted_items:
            target_f = int(item.get("frame", 0))
            out_path = item.get("output")
            bbox = item.get("bbox")
            if not out_path or not bbox or len(bbox) != 4:
                continue

            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                extracted_count += 1
                continue

            # Если целевой кадр дальше текущего на 1-5 кадров — быстрее прочитать read(),
            # иначе — сделать seek через CAP_PROP_POS_FRAMES.
            if current_frame_idx >= 0 and 0 < (target_f - current_frame_idx) <= 5:
                frame = None
                while current_frame_idx < target_f:
                    ret, frame = cap.read()
                    current_frame_idx += 1
                    if not ret:
                        break
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
                ret, frame = cap.read()
                current_frame_idx = target_f
                if not ret:
                    continue

            if frame is None or frame.size == 0:
                continue

            # Координаты bbox [x1, y1, x2, y2] с небольшим паддингом
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bw = x2 - x1
            bh = y2 - y1
            px = bw * pad_ratio
            py = bh * pad_ratio

            ix1 = max(0, int(round(x1 - px)))
            iy1 = max(0, int(round(y1 - py)))
            ix2 = min(video_w, int(round(x2 + px)))
            iy2 = min(video_h, int(round(y2 + py)))

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            crop = frame[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                continue

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
            extracted_count += 1

    finally:
        cap.release()

    return {"ok": True, "extracted": extracted_count}


def main():
    parser = argparse.ArgumentParser(description="Extract crops from video on demand")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--items", help="JSON string or path to JSON file with list of items")
    parser.add_argument("--frame", type=int, help="Single frame number (0-based, OpenCV POS_FRAMES)")
    parser.add_argument("--bbox", nargs=4, type=float, help="Single bbox: x1 y1 x2 y2")
    parser.add_argument("--output", help="Single output path")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (1-100)")

    args = parser.parse_args()

    if args.items:
        if os.path.exists(args.items):
            with open(args.items, "r", encoding="utf-8") as f:
                items = json.load(f)
        else:
            items = json.loads(args.items)
    elif args.frame is not None and args.bbox and args.output:
        items = [{"frame": args.frame, "bbox": args.bbox, "output": args.output}]
    else:
        print(json.dumps({"ok": False, "error": "Invalid arguments. Provide --items or --frame+--bbox+--output"}))
        sys.exit(1)

    result = extract_crops(args.video, items, quality=args.quality)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

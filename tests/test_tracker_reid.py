"""Тесты поддержки with_reid в трекере (parallel_tracker / tracklet_pipeline)."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from app.parallel_tracker import (
    TickFrameReader,
    associate_tracks,
    create_tracker,
)


class TestTrackerReid(unittest.TestCase):
    def test_create_tracker_without_reid(self) -> None:
        tracker = create_tracker("tracktrack", overrides={"with_reid": False})
        self.assertFalse(getattr(tracker.args, "with_reid", False))
        self.assertIsNone(getattr(tracker, "encoder", None))

    def test_create_tracker_with_reid_config(self) -> None:
        # При with_reid: True дефолтная модель yolo26n-reid.onnx
        tracker = create_tracker(
            "tracktrack",
            overrides={
                "with_reid": True,
                "model": "auto",
                "reid_weight": 0.30,
            },
        )
        self.assertTrue(getattr(tracker.args, "with_reid", False))
        self.assertTrue(str(getattr(tracker.args, "model", "")).endswith("yolo26n-reid.onnx"))
        self.assertIsNotNone(getattr(tracker.args, "device", None))

    def test_associate_tracks_without_video(self) -> None:
        all_dets = {
            0: [{"bbox": [100, 100, 150, 200], "confidence": 0.9, "cls": 0}],
            1: [{"bbox": [102, 101, 152, 201], "confidence": 0.9, "cls": 0}],
            2: [{"bbox": [104, 102, 154, 202], "confidence": 0.9, "cls": 0}],
        }
        tracked = associate_tracks(
            all_dets,
            tracker_type="tracktrack",
            total_frames=3,
            tracker_overrides={"with_reid": False},
            nms_iou=0.5,
            detect_every_n=1,
            video_source=None,
        )
        self.assertEqual(len(tracked), 3)
        self.assertEqual(len(tracked[0]), 1)
        self.assertIn("track_id", tracked[0][0])

    def test_tick_frame_reader(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path, fourcc, 25.0, (64, 64))
            for i in range(10):
                frame = np.full((64, 64, 3), fill_value=i * 20, dtype=np.uint8)
                out.write(frame)
            out.release()

            reader = TickFrameReader(video_path)
            self.assertTrue(reader.is_opened())

            f0 = reader.get_frame(0)
            self.assertIsNotNone(f0)
            self.assertEqual(f0.shape, (64, 64, 3))
            self.assertEqual(f0[0, 0, 0], 0)

            # Пропуск кадров до индекса 3
            f3 = reader.get_frame(3)
            self.assertIsNotNone(f3)
            self.assertAlmostEqual(int(f3[0, 0, 0]), 60, delta=10)

            # Скачок назад на индекс 1
            f1 = reader.get_frame(1)
            self.assertIsNotNone(f1)
            self.assertAlmostEqual(int(f1[0, 0, 0]), 20, delta=10)

            reader.close()
            self.assertFalse(reader.is_opened())

    def test_associate_tracks_with_reid_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path, fourcc, 25.0, (64, 64))
            for i in range(5):
                frame = np.full((64, 64, 3), fill_value=i * 30, dtype=np.uint8)
                out.write(frame)
            out.release()

            all_dets = {
                0: [{"bbox": [10, 10, 40, 50], "confidence": 0.9, "cls": 0}],
                2: [{"bbox": [12, 11, 42, 51], "confidence": 0.9, "cls": 0}],
                4: [{"bbox": [14, 12, 44, 52], "confidence": 0.9, "cls": 0}],
            }

            # Мокаем encoder трекера, чтобы не загружать реальную ONNX-модель из интернета в unit-тесте
            dummy_encoder = MagicMock(side_effect=lambda img, dets: [np.ones((128,), dtype=np.float32) for _ in range(len(dets))])

            with patch("ultralytics.trackers.utils.reid.build_encoder", return_value=dummy_encoder):
                tracked = associate_tracks(
                    all_dets,
                    tracker_type="tracktrack",
                    total_frames=5,
                    tracker_overrides={
                        "with_reid": True,
                        "reid_weight": 0.25,
                    },
                    nms_iou=0.5,
                    detect_every_n=2,
                    video_source=video_path,
                )

            self.assertEqual(len(tracked), 3)
            self.assertIn(0, tracked)
            self.assertIn(2, tracked)
            self.assertIn(4, tracked)
            self.assertEqual(dummy_encoder.call_count, 3)

    def test_associate_tracks_with_session_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test_seg.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path, fourcc, 25.0, (64, 64))
            for i in range(4):
                frame = np.full((64, 64, 3), fill_value=i * 40, dtype=np.uint8)
                out.write(frame)
            out.release()

            manifest = {
                "stage": "info",
                "kind": "camera_day",
                "session_key": "test_sess",
                "frame_count": 4,
                "parts": [
                    {
                        "path": video_path,
                        "frame_offset": 0,
                        "frame_count": 4,
                    }
                ],
            }

            all_dets = {
                0: [{"bbox": [10, 10, 40, 50], "confidence": 0.9, "cls": 0}],
                2: [{"bbox": [12, 11, 42, 51], "confidence": 0.9, "cls": 0}],
            }

            dummy_encoder = MagicMock(side_effect=lambda img, dets: [np.ones((128,), dtype=np.float32) for _ in range(len(dets))])

            with patch("ultralytics.trackers.utils.reid.build_encoder", return_value=dummy_encoder):
                tracked = associate_tracks(
                    all_dets,
                    tracker_type="tracktrack",
                    total_frames=4,
                    tracker_overrides={
                        "with_reid": True,
                        "reid_weight": 0.25,
                    },
                    nms_iou=0.5,
                    detect_every_n=2,
                    video_source="session:test_sess",
                    manifest=manifest,
                )

            self.assertEqual(len(tracked), 2)
            self.assertIn(0, tracked)
            self.assertIn(2, tracked)
            self.assertEqual(dummy_encoder.call_count, 2)


if __name__ == "__main__":
    unittest.main()

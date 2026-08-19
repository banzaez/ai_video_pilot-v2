"""Тесты для TrackBestFramesPicker."""

import os
import tempfile
import unittest
import numpy as np

from app.crops.track_best_frames import (
    ScoredTrackFrame,
    TrackBestFramesPicker,
    TrackFrameCandidate,
    compute_frame_crowd_penalty,
    compute_geometry_score,
    extract_face_box_from_pose,
    extract_face_crop_from_person,
    load_pose_cache,
)
from app.pose.types import PoseResult


class TestTrackBestFrames(unittest.TestCase):
    def test_compute_geometry_score(self):
        det_good = {"bbox": [100, 100, 200, 400], "confidence": 0.9}
        s_good = compute_geometry_score(det_good, frame_w=1920, frame_h=1080)
        self.assertGreater(s_good, 0.7)

        # Обрезан по границе кадра
        det_edge = {"bbox": [0, 100, 100, 400], "confidence": 0.9}
        s_edge = compute_geometry_score(det_edge, frame_w=1920, frame_h=1080)
        self.assertLess(s_edge, s_good)

    def test_compute_frame_crowd_penalty(self):
        target = [100, 100, 200, 400]
        self.assertEqual(compute_frame_crowd_penalty(target, []), 1.0)

        # Другой человек далеко
        far_person = [{"bbox": [500, 500, 600, 800]}]
        self.assertEqual(compute_frame_crowd_penalty(target, far_person), 1.0)

        # Другой человек сильно перекрывает
        overlapping_person = [{"bbox": [120, 120, 220, 420]}]
        penalty = compute_frame_crowd_penalty(target, overlapping_person)
        self.assertLess(penalty, 0.8)

    def test_picker_without_pose(self):
        picker = TrackBestFramesPicker(pose_service=None)
        fake_img = np.zeros((480, 640, 3), dtype=np.uint8)

        candidates = [
            TrackFrameCandidate(
                frame_index=0,
                image=fake_img,
                target_det={"bbox": [50, 50, 150, 350], "confidence": 0.95},
                all_dets=[],
                tracklet_id=1,
            ),
            TrackFrameCandidate(
                frame_index=10,
                image=fake_img,
                target_det={"bbox": [50, 50, 150, 350], "confidence": 0.40},
                all_dets=[],
                tracklet_id=1,
            ),
            TrackFrameCandidate(
                frame_index=20,
                image=fake_img,
                target_det={"bbox": [0, 50, 100, 350], "confidence": 0.90},
                all_dets=[{"bbox": [20, 60, 120, 360]}],
                tracklet_id=1,
            ),
        ]

        scored = picker.score_candidates_batch(candidates)
        self.assertEqual(len(scored), 3)
        self.assertGreater(scored[0].score, scored[1].score)
        self.assertGreater(scored[0].score, scored[2].score)

        picked = picker.pick_best_for_tracklet(candidates, top_k=2)
        self.assertEqual(len(picked), 2)

    def test_picker_for_group(self):
        picker = TrackBestFramesPicker(pose_service=None)
        fake_img = np.zeros((480, 640, 3), dtype=np.uint8)

        cands_t1 = [
            TrackFrameCandidate(
                frame_index=i,
                image=fake_img,
                target_det={"bbox": [100, 100, 200, 400], "confidence": 0.9},
                all_dets=[],
                tracklet_id=1,
            )
            for i in range(10)
        ]
        cands_t2 = [
            TrackFrameCandidate(
                frame_index=20 + i,
                image=fake_img,
                target_det={"bbox": [150, 100, 250, 400], "confidence": 0.85},
                all_dets=[],
                tracklet_id=2,
            )
            for i in range(10)
        ]

        by_tid = {1: cands_t1, 2: cands_t2}
        picked = picker.pick_best_for_group(by_tid, top_k=4)
        self.assertEqual(len(picked), 4)
        picked_tids = {p.tracklet_id for p in picked}
        self.assertIn(1, picked_tids)
        self.assertIn(2, picked_tids)

    def test_extract_face_box_and_crop(self):
        kxy = [
            [150.0, 120.0],
            [140.0, 110.0],
            [160.0, 110.0],
            [130.0, 115.0],
            [170.0, 115.0],
            [120.0, 180.0],
            [180.0, 180.0],
        ] + [[0.0, 0.0]] * 10
        kcf = [0.9, 0.9, 0.9, 0.8, 0.8, 0.9, 0.9] + [0.0] * 10
        pose = PoseResult(bbox=[100, 100, 200, 400], confidence=0.9, kxy=kxy, kcf=kcf)

        fbox = extract_face_box_from_pose(pose, [100, 100, 200, 400], frame_w=1920, frame_h=1080)
        self.assertLess(fbox[0], 150)
        self.assertGreater(fbox[2], 150)
        self.assertLess(fbox[1], 120)
        self.assertGreater(fbox[3], 120)

        crop_img = np.ones((300, 100, 3), dtype=np.uint8) * 200
        crop_roi = (100, 100, 200, 400)
        fcrop, fbox2 = extract_face_crop_from_person(
            crop_img, crop_roi, pose, {"bbox": [100, 100, 200, 400]}
        )
        self.assertIsNotNone(fcrop)
        self.assertGreater(fcrop.shape[0], 10)
        self.assertGreater(fcrop.shape[1], 10)

        fbox_fallback = extract_face_box_from_pose(None, [100, 100, 200, 400])
        self.assertEqual(fbox_fallback[1], 100.0)
        self.assertLessEqual(fbox_fallback[3], 220.0)

    def test_pick_best_faces(self):
        picker = TrackBestFramesPicker(pose_service=None)
        fake_img = np.zeros((480, 640, 3), dtype=np.uint8)

        cands = [
            TrackFrameCandidate(
                frame_index=0,
                image=fake_img,
                target_det={"bbox": [100, 100, 200, 400], "confidence": 0.9},
                all_dets=[],
                tracklet_id=1,
            ),
            TrackFrameCandidate(
                frame_index=5,
                image=fake_img,
                target_det={"bbox": [100, 100, 200, 400], "confidence": 0.8},
                all_dets=[],
                tracklet_id=1,
            ),
        ]

        scored = picker.score_candidates_batch(cands, extract_faces=True)
        self.assertEqual(len(scored), 2)
        self.assertIsNotNone(scored[0].face_crop)
        self.assertIsNotNone(scored[0].face_bbox)

        faces_picked = picker.pick_best_faces_for_group({1: cands}, top_k=1)
        self.assertEqual(len(faces_picked), 1)
        self.assertIsNotNone(faces_picked[0].face_crop)

    def test_pose_caching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.json")
            picker = TrackBestFramesPicker(pose_service=None)
            fake_img = np.zeros((480, 640, 3), dtype=np.uint8)

            cands = [
                TrackFrameCandidate(
                    frame_index=1,
                    image=fake_img,
                    target_det={"bbox": [50, 50, 150, 350], "confidence": 0.9},
                    all_dets=[],
                    tracklet_id=10,
                )
            ]

            scored1 = picker.score_candidates_batch(cands, cache_path=cache_file)
            self.assertEqual(len(scored1), 1)
            self.assertTrue(os.path.isfile(cache_file))

            loaded = load_pose_cache(cache_file)
            self.assertEqual(len(loaded), 1)

            scored2 = picker.score_candidates_batch(cands, cache_path=cache_file)
            self.assertEqual(len(scored2), 1)
            self.assertEqual(scored2[0].score, scored1[0].score)

            miss = load_pose_cache(cache_file, pose_model="other-pose.pt", kpt_min=0.25)
            self.assertEqual(miss, {})


if __name__ == "__main__":
    unittest.main()

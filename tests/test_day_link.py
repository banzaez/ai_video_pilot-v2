"""Тесты для глобальной стадии межкамерной склейки дня (stage_day_link, без лиц)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from app.config.settings import Settings
from app.global_id.group_reid import _tracklet_groups
from app.global_id.stage_day_link import link_day_tracks


def _emb(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, 1e-9)


def _node(
    uid: str,
    *,
    camera: str,
    camera_index: int,
    track_id: int,
    t0: float,
    t1: float,
    p0: list[float] | None,
    p1: list[float] | None,
    reid: np.ndarray | None = None,
) -> dict:
    bbox = [100.0, 80.0, 180.0, 260.0]
    return {
        "uid": uid,
        "session_key": uid.split("#")[0],
        "camera": camera,
        "camera_index": camera_index,
        "track_id": track_id,
        "f0": 0,
        "f1": 1,
        "t0": t0,
        "t1": t1,
        "duration_sec": t1 - t0,
        "p0": p0,
        "p1": p1,
        "map_p0": list(p0) if p0 else None,
        "map_p1": list(p1) if p1 else None,
        "map_src0": "kpt_map" if p0 else "",
        "map_src1": "kpt_map" if p1 else "",
        "bbox0": list(bbox),
        "bbox1": list(bbox),
        "h": 180.0,
        "w": 80.0,
        "avg_speed_mps": 0.2,
        "n_frames": 10,
        "reid_embs": reid,
        "has_reid": reid is not None,
        "crops": ["g_0001_k0_f1.jpg"] if reid is not None else [],
        "best_crop": "g_0001_k0_f1.jpg" if reid is not None else None,
    }


def _settings(**overrides) -> Settings:
    s = Settings()
    s.day_link_pass1_min_score = 0.70
    s.day_link_pass1_min_reid = 0.85
    s.day_link_pass2_min_score = 0.95
    s.day_link_pass4_min_score = 0.70
    s.day_link_pass4_min_reid = 0.0
    s.day_link_max_overlap_sec = 20.0
    s.day_link_pass4_max_overlap_sec = 20.0
    s.day_link_min_reid_score = 0.85
    s.day_link_max_spatial_m = 4.0
    s.day_link_max_spatial_px = 0.0
    s.day_link_motion_sigma_m = 3.0
    s.day_link_w_reid = 0.65
    s.day_link_w_motion = 0.20
    s.day_link_w_size = 0.0
    s.day_link_w_gap = 0.15
    s.day_link_max_gap_sec = 300.0
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


class TestDayLink(unittest.TestCase):
    def test_link_day_tracks_cross_camera(self):
        reid = _emb([1.0, 0.0, 0.0, 0.0])
        nodes = [
            _node(
                "01_20260401#1",
                camera="Camera_01",
                camera_index=1,
                track_id=1,
                t0=10.0,
                t1=20.0,
                p0=[1800.0, 2000.0],
                p1=[2000.0, 2000.0],
                reid=reid,
            ),
            _node(
                "02_20260401#1",
                camera="Camera_02",
                camera_index=2,
                track_id=1,
                t0=25.0,
                t1=40.0,
                p0=[2160.0, 2000.0],
                p1=[2400.0, 2000.0],
                reid=reid,
            ),
        ]
        result = link_day_tracks(nodes, _settings())
        self.assertEqual(result["stats"]["n_tracks_total"], 2)
        self.assertEqual(result["n_persons"], 1)
        self.assertEqual(result["stats"]["n_multi_cam_persons"], 1)
        self.assertEqual(len(result["edges"]), 1)
        self.assertIn(result["edges"][0]["pass"], (1, 2))
        self.assertIsNotNone(result["edges"][0].get("reid"))
        self.assertNotIn("face", result["edges"][0])

    def test_cross_camera_simultaneous_overlap(self):
        reid = _emb([1.0, 0.0, 0.0, 0.0])
        nodes = [
            _node(
                "01_20260401#1",
                camera="Camera_01",
                camera_index=1,
                track_id=1,
                t0=10.0,
                t1=50.0,
                p0=[2000.0, 2000.0],
                p1=[2100.0, 2000.0],
                reid=reid,
            ),
            _node(
                "02_20260401#1",
                camera="Camera_02",
                camera_index=2,
                track_id=1,
                t0=12.0,
                t1=48.0,
                p0=[2160.0, 2000.0],
                p1=[2200.0, 2000.0],
                reid=reid,
            ),
            _node(
                "03_20260401#1",
                camera="Camera_03",
                camera_index=3,
                track_id=1,
                t0=10.0,
                t1=50.0,
                p0=[1900.0, 2000.0],
                p1=[2050.0, 2000.0],
                reid=reid,
            ),
        ]
        result = link_day_tracks(nodes, _settings())
        self.assertEqual(result["n_persons"], 1)
        self.assertEqual(result["stats"]["n_multi_cam_persons"], 1)
        self.assertEqual(result["persons"][0]["n_cameras"], 3)

    def test_same_camera_overlap_not_merged(self):
        reid = _emb([1.0, 0.0, 0.0, 0.0])
        nodes = [
            _node(
                "01_20260401#1",
                camera="Camera_01",
                camera_index=1,
                track_id=1,
                t0=10.0,
                t1=30.0,
                p0=[1800.0, 2000.0],
                p1=[2000.0, 2000.0],
                reid=reid,
            ),
            _node(
                "02_20260401#1",
                camera="Camera_02",
                camera_index=2,
                track_id=1,
                t0=35.0,
                t1=50.0,
                p0=[2160.0, 2000.0],
                p1=[2400.0, 2000.0],
                reid=reid,
            ),
            _node(
                "01_20260401#2",
                camera="Camera_01",
                camera_index=1,
                track_id=2,
                t0=12.0,
                t1=28.0,
                p0=[1900.0, 2100.0],
                p1=[2100.0, 2100.0],
                reid=reid,
            ),
        ]
        result = link_day_tracks(nodes, _settings())
        self.assertEqual(result["n_persons"], 2)
        cams_of_multi = [p for p in result["persons"] if p["n_tracks"] > 1]
        self.assertTrue(cams_of_multi)
        for person in result["persons"]:
            by_cam: dict[str, list[tuple[float, float]]] = {}
            for tr in person["tracks"]:
                by_cam.setdefault(tr["camera"], []).append((tr["t0"], tr["t1"]))
            for spans in by_cam.values():
                spans.sort()
                for i in range(len(spans) - 1):
                    self.assertLessEqual(spans[i][1], spans[i + 1][0] + 1e-6)

    def test_pass4_handover_overlap(self):
        reid = _emb([0.0, 1.0, 0.0, 0.0])
        nodes = [
            _node(
                "01_20260401#1",
                camera="Camera_01",
                camera_index=1,
                track_id=1,
                t0=10.0,
                t1=30.0,
                p0=[2000.0, 2000.0],
                p1=[2100.0, 2000.0],
                reid=reid,
            ),
            _node(
                "02_20260401#1",
                camera="Camera_02",
                camera_index=2,
                track_id=1,
                t0=25.0,
                t1=45.0,
                p0=[2200.0, 2000.0],
                p1=[2300.0, 2000.0],
                reid=reid,
            ),
        ]
        result = link_day_tracks(nodes, _settings(day_link_pass1_min_score=0.99, day_link_pass1_min_reid=0.99, day_link_pass2_min_score=0.99))
        self.assertEqual(result["n_persons"], 1)
        self.assertTrue(result["edges"])
        self.assertEqual(result["edges"][0]["pass"], 4)
        self.assertTrue(result["edges"][0]["is_overlap"])

    def test_long_gap_still_linked_in_pass1(self):
        reid = _emb([0.0, 0.0, 1.0, 0.0])
        nodes = [
            _node(
                "01_20260401#1",
                camera="Camera_01",
                camera_index=1,
                track_id=1,
                t0=0.0,
                t1=400.0,
                p0=[2000.0, 2000.0],
                p1=[2100.0, 2000.0],
                reid=reid,
            ),
            _node(
                "02_20260401#1",
                camera="Camera_02",
                camera_index=2,
                track_id=1,
                t0=410.0,
                t1=450.0,
                p0=[2200.0, 2000.0],
                p1=[2300.0, 2000.0],
                reid=reid,
            ),
        ]
        result = link_day_tracks(nodes, _settings())
        self.assertEqual(result["n_persons"], 1)
        self.assertEqual(result["stats"]["n_multi_cam_persons"], 1)

    def test_spatial_cutoff_rejects_pair(self):
        reid = _emb([1.0, 1.0, 0.0, 0.0])
        nodes = [
            _node(
                "01_20260401#1",
                camera="Camera_01",
                camera_index=1,
                track_id=1,
                t0=10.0,
                t1=20.0,
                p0=[0.0, 0.0],
                p1=[0.0, 0.0],
                reid=reid,
            ),
            _node(
                "02_20260401#1",
                camera="Camera_02",
                camera_index=2,
                track_id=1,
                t0=21.0,
                t1=30.0,
                p0=[1600.0, 0.0],
                p1=[1700.0, 0.0],
                reid=reid,
            ),
        ]
        result = link_day_tracks(nodes, _settings(day_link_max_spatial_m=4.0))
        self.assertEqual(result["n_persons"], 2)
        self.assertEqual(result["edges"], [])

    def test_min_reid_score_rejects_pair(self):
        nodes = [
            _node(
                "01_20260401#1",
                camera="Camera_01",
                camera_index=1,
                track_id=1,
                t0=10.0,
                t1=20.0,
                p0=[2000.0, 2000.0],
                p1=[2100.0, 2000.0],
                reid=_emb([1.0, 0.0, 0.0, 0.0]),
            ),
            _node(
                "02_20260401#1",
                camera="Camera_02",
                camera_index=2,
                track_id=1,
                t0=25.0,
                t1=40.0,
                p0=[2160.0, 2000.0],
                p1=[2300.0, 2000.0],
                reid=_emb([0.0, 1.0, 0.0, 0.0]),
            ),
        ]
        result = link_day_tracks(nodes, _settings(day_link_min_reid_score=0.85))
        self.assertEqual(result["n_persons"], 2)
        self.assertEqual(result["edges"], [])

    def test_tracklet_groups_multi_and_solo(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "tracklet_links.json"), "w", encoding="utf-8") as f:
                json.dump({"tracklet_to_global": {"3": 1, "7": 1, "2": 2}}, f)
            groups = _tracklet_groups(tmp, {1, 2, 9})
            self.assertEqual(groups[1], [3, 7])
            self.assertEqual(groups[2], [2])
            self.assertEqual(groups[9], [9])


if __name__ == "__main__":
    unittest.main()

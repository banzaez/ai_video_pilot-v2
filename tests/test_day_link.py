"""Тесты для глобальной стадии межкамерной склейки дня (stage_day_link: Pass 0 Alone Geo, Pass 1 ReID)."""

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
        "f1": int(round((t1 - t0) * 25)),
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
        "n_frames": max(1, int(round((t1 - t0) * 25))),
        "reid_embs": reid,
        "has_reid": reid is not None,
        "crops": ["g_0001_k0_f1.jpg"] if reid is not None else [],
        "best_crop": "g_0001_k0_f1.jpg" if reid is not None else None,
    }


def _settings(**overrides) -> Settings:
    s = Settings()
    s.day_link_enabled = True
    s.day_link_max_gap_sec = 300.0
    s.day_link_pass0_enabled = True
    s.day_link_pass0_radius_m = 2.0
    s.day_link_pass0_min_overlap_sec = 5.0
    s.day_link_pass0_max_gap_sec = 10.0
    s.day_link_pass0_max_dist_m = 3.0
    s.day_link_pass0_max_speed_mps = 2.5
    s.day_link_pass0_min_reid = 0.90
    s.day_link_pass1_enabled = True
    s.day_link_pass1_min_reid = 0.96
    s.day_link_pass1_max_gap_sec = 300.0
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


class TestDayLink(unittest.TestCase):
    def test_pass0_simultaneous_overlap_linked(self):
        """Две камеры видят человека одновременно в одном месте карты >= 5с при ReID >= 0.90 -> Pass 0."""
        reid = _emb([1.0, 0.0, 0.0, 0.0])
        nodes = [
            _node(
                "01_20260401#1",
                camera="Camera_01",
                camera_index=1,
                track_id=1,
                t0=10.0,
                t1=20.0,
                p0=[2000.0, 2000.0],
                p1=[2050.0, 2000.0],
                reid=reid,
            ),
            _node(
                "02_20260401#1",
                camera="Camera_02",
                camera_index=2,
                track_id=1,
                t0=12.0,
                t1=19.0,
                p0=[2020.0, 2000.0],
                p1=[2060.0, 2000.0],
                reid=reid,
            ),
        ]
        result = link_day_tracks(nodes, _settings())
        self.assertEqual(result["n_persons"], 1)
        self.assertEqual(result["stats"]["pass0_merges"], 1)
        self.assertEqual(result["stats"]["pass1_merges"], 0)
        self.assertEqual(result["edges"][0]["pass"], 0)
        self.assertIn("Pass 0 (Alone Geo)", result["edges"][0]["reason"])

    def test_pass0_sequential_gap_linked(self):
        """Последовательный переход между камерами с зазором <= 10с и ReID >= 0.90 -> Pass 0."""
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
                t0=24.0,
                t1=35.0,
                p0=[2160.0, 2000.0],  # 160 px = 1 метр
                p1=[2300.0, 2000.0],
                reid=reid,
            ),
        ]
        result = link_day_tracks(nodes, _settings())
        self.assertEqual(result["n_persons"], 1)
        self.assertEqual(result["stats"]["pass0_merges"], 1)
        self.assertEqual(result["edges"][0]["pass"], 0)

    def test_pass0_rejected_when_other_person_nearby(self):
        """Pass 0 отклоняется, если рядом (в радиусе 2м) на карте находится посторонний человек."""
        reid_a = _emb([1.0, 0.0, 0.0, 0.0])
        reid_other = _emb([0.0, 0.0, 1.0, 0.0])
        nodes = [
            _node(
                "01_20260401#1",
                camera="Camera_01",
                camera_index=1,
                track_id=1,
                t0=10.0,
                t1=20.0,
                p0=[2000.0, 2000.0],
                p1=[2000.0, 2000.0],
                reid=reid_a,
            ),
            _node(
                "02_20260401#1",
                camera="Camera_02",
                camera_index=2,
                track_id=1,
                t0=12.0,
                t1=18.0,
                p0=[2000.0, 2000.0],
                p1=[2000.0, 2000.0],
                reid=reid_a,
            ),
            # Посторонний человек на 3-й камере в той же точке карты
            _node(
                "03_20260401#9",
                camera="Camera_03",
                camera_index=3,
                track_id=9,
                t0=11.0,
                t1=19.0,
                p0=[2050.0, 2000.0],  # 50px = 0.3м < 2.0м
                p1=[2050.0, 2000.0],
                reid=reid_other,
            ),
        ]
        # Так как ReID_A == 1.0 >= 0.96, Pass 0 отклонится из-за соседа, но Pass 1 склеит по высокому ReID
        result = link_day_tracks(nodes, _settings(day_link_pass1_min_reid=0.99))  # чтобы Pass 1 не склеил
        # Без Pass 1 склейки не будет
        self.assertEqual(result["stats"]["pass0_merges"], 0)

    def test_pass0_rejected_low_reid(self):
        """Pass 0 отклоняется, если ReID < 0.90."""
        nodes = [
            _node(
                "01_20260401#1",
                camera="Camera_01",
                camera_index=1,
                track_id=1,
                t0=10.0,
                t1=20.0,
                p0=[2000.0, 2000.0],
                p1=[2000.0, 2000.0],
                reid=_emb([1.0, 0.0, 0.0, 0.0]),
            ),
            _node(
                "02_20260401#1",
                camera="Camera_02",
                camera_index=2,
                track_id=1,
                t0=12.0,
                t1=18.0,
                p0=[2000.0, 2000.0],
                p1=[2000.0, 2000.0],
                reid=_emb([0.0, 1.0, 0.0, 0.0]),  # ReID = 0.0
            ),
        ]
        result = link_day_tracks(nodes, _settings())
        self.assertEqual(result["n_persons"], 2)
        self.assertEqual(result["edges"], [])

    def test_pass1_strict_reid_links_distant_tracks(self):
        """Pass 1 склеивает треки по строгому ReID >= 0.96 даже при большом зазоре или без карты."""
        reid = _emb([1.0, 0.0, 0.0, 0.0])
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
                t0=60.0,
                t1=80.0,
                p0=[5000.0, 5000.0],  # далеко на карте
                p1=[5000.0, 5000.0],
                reid=reid,
            ),
        ]
        result = link_day_tracks(nodes, _settings())
        self.assertEqual(result["n_persons"], 1)
        self.assertEqual(result["stats"]["pass1_merges"], 1)
        self.assertEqual(result["edges"][0]["pass"], 1)
        self.assertIn("Pass 1 (ReID)", result["edges"][0]["reason"])

    def test_same_camera_overlap_never_merged(self):
        """Треки на одной камере, перекрывающиеся по времени, никогда не объединяются."""
        reid = _emb([1.0, 0.0, 0.0, 0.0])
        nodes = [
            _node(
                "01_20260401#1",
                camera="Camera_01",
                camera_index=1,
                track_id=1,
                t0=10.0,
                t1=30.0,
                p0=[2000.0, 2000.0],
                p1=[2000.0, 2000.0],
                reid=reid,
            ),
            _node(
                "01_20260401#2",
                camera="Camera_01",
                camera_index=1,
                track_id=2,
                t0=12.0,
                t1=28.0,
                p0=[2000.0, 2000.0],
                p1=[2000.0, 2000.0],
                reid=reid,
            ),
        ]
        result = link_day_tracks(nodes, _settings())
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

    def test_day_reid_caching_and_invalidation(self):
        from app.global_id.group_reid import (
            DAY_REID_CACHE_NAME,
            TrackGroupReid,
            is_day_reid_cache_valid,
            load_day_reid_cache,
            save_day_reid_cache,
        )
        import time

        with tempfile.TemporaryDirectory() as tmp:
            tracking_fp = os.path.join(tmp, "tracking.json")
            with open(tracking_fp, "w", encoding="utf-8") as f:
                json.dump({"frames": [{"frame_index": 1, "detections": [{"track_id": 1}, {"track_id": 2}]}]}, f)

            settings = _settings()
            track_ids = {1, 2}
            top_k = 3
            cache_fp = os.path.join(tmp, DAY_REID_CACHE_NAME)

            # 1. Сначала кэша нет
            is_valid, _ = is_day_reid_cache_valid(tmp, settings, track_ids, top_k)
            self.assertFalse(is_valid)

            # 2. Сохраняем кэш
            embs1 = np.ones((3, 512), dtype=np.float32)
            embs2 = np.zeros((3, 512), dtype=np.float32)
            sample_out = {
                1: TrackGroupReid(track_id=1, embs=embs1, crop_files=["c1.jpg"], n_tracklets=1),
                2: TrackGroupReid(track_id=2, embs=embs2, crop_files=["c2.jpg"], n_tracklets=2),
            }
            save_day_reid_cache(cache_fp, sample_out, track_ids, top_k, settings)

            # 3. Кэш валиден и загружается мгновенно
            is_valid, meta = is_day_reid_cache_valid(tmp, settings, track_ids, top_k)
            self.assertTrue(is_valid)
            self.assertIsNotNone(meta)

            loaded = load_day_reid_cache(cache_fp, meta, track_ids)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[1].crop_files, ["c1.jpg"])
            self.assertEqual(loaded[2].n_tracklets, 2)
            np.testing.assert_array_equal(loaded[1].embs, embs1)

            # 4. Обновляем tracking.json (симуляция перезапуска ранних стадий)
            time.sleep(0.01)
            with open(tracking_fp, "w", encoding="utf-8") as f:
                json.dump({"frames": [{"frame_index": 1, "detections": [{"track_id": 1}, {"track_id": 2}, {"track_id": 3}]}]}, f)

            # Кэш автоматически инвалидируется!
            is_valid_after_touch, _ = is_day_reid_cache_valid(tmp, settings, {1, 2, 3}, top_k)
            self.assertFalse(is_valid_after_touch)


if __name__ == "__main__":
    unittest.main()

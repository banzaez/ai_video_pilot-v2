"""Тесты для стадии склейки треклетов (tracklet link, Pass 0)."""

from __future__ import annotations

import unittest
import numpy as np

from app.tracklet.link_mcf import link_tracklets


def _emb(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, 1e-9)


class TestTrackletLinkPass0(unittest.TestCase):
    def test_pass0_links_sequential_tracklets(self) -> None:
        e1 = _emb([1.0, 0.0, 0.0])
        e2 = _emb([0.98, 0.05, 0.0])

        t1 = {
            "tracklet_id": 1,
            "t0": 0.0,
            "t1": 2.0,
            "p0": [100.0, 100.0],
            "p1": [120.0, 100.0],
            "h": 150.0,
            "w": 60.0,
        }
        t2 = {
            "tracklet_id": 2,
            "t0": 3.0,
            "t1": 5.0,
            "p0": [130.0, 100.0],
            "p1": [150.0, 100.0],
            "h": 150.0,
            "w": 60.0,
        }
        embeddings = {1: e1, 2: e2}

        res = link_tracklets(
            [t1, t2],
            embeddings,
            max_gap_sec=5.0,
            min_reid_score=0.90,
            pass0_min_reid=0.95,
            pass0_min_score=0.70,
        )

        self.assertEqual(res["pass0_merged"], 1)
        self.assertEqual(len(res["groups"]), 1)
        self.assertEqual(res["groups"][0], [1, 2])
        self.assertEqual(res["tracklet_to_global"]["1"], res["tracklet_to_global"]["2"])
        self.assertEqual(len(res["edges"]), 1)
        self.assertEqual(res["edges"][0]["pass"], 0)

    def test_pass0_rejects_low_reid(self) -> None:
        e1 = _emb([1.0, 0.0, 0.0])
        e2 = _emb([0.0, 1.0, 0.0])  # Orthogonal

        t1 = {
            "tracklet_id": 1,
            "t0": 0.0,
            "t1": 2.0,
            "p0": [100.0, 100.0],
            "p1": [120.0, 100.0],
        }
        t2 = {
            "tracklet_id": 2,
            "t0": 3.0,
            "t1": 5.0,
            "p0": [130.0, 100.0],
            "p1": [150.0, 100.0],
        }
        embeddings = {1: e1, 2: e2}

        res = link_tracklets(
            [t1, t2],
            embeddings,
            max_gap_sec=5.0,
            min_reid_score=0.90,
            pass0_min_reid=0.95,
            pass0_min_score=0.70,
        )

        self.assertEqual(res["pass0_merged"], 0)
        self.assertEqual(len(res["groups"]), 2)
        self.assertNotEqual(res["tracklet_to_global"]["1"], res["tracklet_to_global"]["2"])


class TestTrackletLinkPassAloneGeo(unittest.TestCase):
    def test_pass_alone_geo_links_isolated_tracklets(self) -> None:
        """Одиночный человек пропал и появился в радиусе 1.5м, вокруг никого не было -> склейка."""
        e1 = _emb([1.0, 0.0, 0.0])
        e2 = _emb([0.0, 1.0, 0.0])  # ReID ортогонален, pass0 не возьмет

        t1 = {
            "tracklet_id": 1,
            "f0": 1,
            "f1": 50,
            "t0": 0.0,
            "t1": 2.0,
            "map_p0": [100.0, 100.0],
            "map_p1": [120.0, 100.0],
            "h": 150.0,
        }
        t2 = {
            "tracklet_id": 2,
            "f0": 75,
            "f1": 125,
            "t0": 3.0,
            "t1": 5.0,
            "map_p0": [130.0, 100.0],  # 10 px на карте = 10/160 = 0.06 м
            "map_p1": [150.0, 100.0],
            "h": 150.0,
        }
        embeddings = {1: e1, 2: e2}

        res = link_tracklets(
            [t1, t2],
            embeddings,
            max_gap_sec=5.0,
            min_reid_score=0.90,
            pass0_min_reid=0.95,
            pass0_min_score=0.70,
            pass_alone_enabled=True,
            pass_alone_radius_m=2.0,
            pass_alone_max_gap_sec=5.0,
            pass_alone_max_dist_m=3.0,
            pass_alone_max_speed_mps=2.0,
        )

        self.assertEqual(res["pass0_merged"], 0)
        self.assertEqual(res["pass_alone_merged"], 1)
        self.assertEqual(len(res["groups"]), 1)
        self.assertEqual(res["groups"][0], [1, 2])
        self.assertEqual(res["tracklet_to_global"]["1"], res["tracklet_to_global"]["2"])
        # Проверяем, что ребро помечено pass=1
        alone_edges = [e for e in res["edges"] if e.get("pass") == 1]
        self.assertEqual(len(alone_edges), 1)
        self.assertIn("Pass 1 (Alone Geo)", alone_edges[0]["reason"])

    def test_pass_alone_geo_rejects_when_other_person_nearby(self) -> None:
        """Если рядом (< 2м) шел посторонний человек 3, склейка не должна произойти."""
        e1 = _emb([1.0, 0.0, 0.0])
        e2 = _emb([0.0, 1.0, 0.0])
        e3 = _emb([0.0, 0.0, 1.0])

        t1 = {
            "tracklet_id": 1,
            "f0": 1,
            "f1": 50,
            "t0": 0.0,
            "t1": 2.0,
            "map_p0": [100.0, 100.0],
            "map_p1": [120.0, 100.0],
            "h": 150.0,
        }
        t2 = {
            "tracklet_id": 2,
            "f0": 75,
            "f1": 125,
            "t0": 3.0,
            "t1": 5.0,
            "map_p0": [130.0, 100.0],
            "map_p1": [150.0, 100.0],
            "h": 150.0,
        }
        # Третий человек шел прямо рядом с t1 (расстояние 40 px = 0.25 м < 2.0 м)
        t3 = {
            "tracklet_id": 3,
            "f0": 10,
            "f1": 40,
            "t0": 0.4,
            "t1": 1.6,
            "map_p0": [110.0, 140.0],
            "map_p1": [115.0, 140.0],
            "h": 150.0,
        }
        embeddings = {1: e1, 2: e2, 3: e3}

        res = link_tracklets(
            [t1, t2, t3],
            embeddings,
            max_gap_sec=5.0,
            min_reid_score=0.90,
            pass0_min_reid=0.95,
            pass0_min_score=0.70,
            pass_alone_enabled=True,
            pass_alone_radius_m=2.0,
            pass_alone_max_gap_sec=5.0,
            pass_alone_max_dist_m=3.0,
            pass_alone_max_speed_mps=2.0,
        )

        self.assertEqual(res["pass0_merged"], 0)
        self.assertEqual(res["pass_alone_merged"], 0)
        self.assertEqual(len(res["groups"]), 3)

    def test_pass_alone_geo_rejects_large_spatial_dist(self) -> None:
        """Если трек появился слишком далеко (> max_dist_m), склейка отклоняется."""
        e1 = _emb([1.0, 0.0, 0.0])
        e2 = _emb([0.0, 1.0, 0.0])

        t1 = {
            "tracklet_id": 1,
            "f0": 1,
            "f1": 50,
            "t0": 0.0,
            "t1": 2.0,
            "map_p0": [100.0, 100.0],
            "map_p1": [120.0, 100.0],
            "h": 150.0,
        }
        # Расстояние 1000 px = 6.25 м > 3.0 м
        t2 = {
            "tracklet_id": 2,
            "f0": 75,
            "f1": 125,
            "t0": 3.0,
            "t1": 5.0,
            "map_p0": [1120.0, 100.0],
            "map_p1": [1150.0, 100.0],
            "h": 150.0,
        }
        embeddings = {1: e1, 2: e2}

        res = link_tracklets(
            [t1, t2],
            embeddings,
            max_gap_sec=5.0,
            min_reid_score=0.90,
            pass0_min_reid=0.95,
            pass0_min_score=0.70,
            pass_alone_enabled=True,
            pass_alone_radius_m=2.0,
            pass_alone_max_gap_sec=5.0,
            pass_alone_max_dist_m=3.0,
            pass_alone_max_speed_mps=2.0,
        )

        self.assertEqual(res["pass_alone_merged"], 0)
        self.assertEqual(len(res["groups"]), 2)


class TestTrackletLinkOverlap(unittest.TestCase):
    def test_link_allows_overlap_up_to_max_overlap_sec(self) -> None:
        """Перекрытие на 1.0с (<= max_overlap_sec=2.0) разрешено и склеивается."""
        e1 = _emb([1.0, 0.0, 0.0])
        e2 = _emb([1.0, 0.0, 0.0])
        t1 = {
            "tracklet_id": 1,
            "f0": 1,
            "f1": 50,
            "t0": 0.0,
            "t1": 3.0,  # Заканчивается на 3.0с
            "bbox0": [10, 10, 50, 150],
            "bbox1": [20, 10, 60, 150],
        }
        t2 = {
            "tracklet_id": 2,
            "f0": 35,
            "f1": 85,
            "t0": 2.0,  # Начинается на 2.0с (наложение 1.0с)
            "t1": 5.0,
            "bbox0": [22, 10, 62, 150],
            "bbox1": [30, 10, 70, 150],
        }
        embeddings = {1: e1, 2: e2}

        res = link_tracklets(
            [t1, t2],
            embeddings,
            max_gap_sec=10.0,
            max_overlap_sec=2.0,
            min_reid_score=0.90,
            pass0_min_reid=0.95,
            pass0_min_score=0.70,
        )
        self.assertEqual(res["pass0_merged"], 1)
        self.assertEqual(len(res["groups"]), 1)
        self.assertEqual(res["tracklet_to_global"]["1"], res["tracklet_to_global"]["2"])

    def test_link_rejects_overlap_exceeding_max_overlap_sec(self) -> None:
        """Перекрытие на 3.5с (> max_overlap_sec=2.0) отбрасывается."""
        e1 = _emb([1.0, 0.0, 0.0])
        e2 = _emb([1.0, 0.0, 0.0])
        t1 = {
            "tracklet_id": 1,
            "f0": 1,
            "f1": 80,
            "t0": 0.0,
            "t1": 5.0,
            "bbox0": [10, 10, 50, 150],
            "bbox1": [20, 10, 60, 150],
        }
        t2 = {
            "tracklet_id": 2,
            "f0": 20,
            "f1": 100,
            "t0": 1.5,  # Наложение 3.5с > 2.0с
            "t1": 6.0,
            "bbox0": [22, 10, 62, 150],
            "bbox1": [30, 10, 70, 150],
        }
        embeddings = {1: e1, 2: e2}

        res = link_tracklets(
            [t1, t2],
            embeddings,
            max_gap_sec=10.0,
            max_overlap_sec=2.0,
            min_reid_score=0.90,
            pass0_min_reid=0.95,
            pass0_min_score=0.70,
        )
        self.assertEqual(res["pass0_merged"], 0)
        self.assertEqual(len(res["groups"]), 2)


if __name__ == "__main__":
    unittest.main()

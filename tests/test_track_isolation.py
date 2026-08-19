"""Модульные тесты для сервиса TrackNeighborhoodIndex (пространственно-временная изоляция и соседи)."""

from __future__ import annotations

import unittest
from app.global_id.isolation import TrackNeighborhoodIndex, calc_points_dist_m, point_to_segment_dist


class TestTrackNeighborhoodIndex(unittest.TestCase):
    def test_dist_helpers(self) -> None:
        # Точка на карте: 160 px = 1 м
        p1 = (100.0, 100.0, "map")
        p2 = (260.0, 100.0, "map")
        self.assertAlmostEqual(calc_points_dist_m(p1, p2, scale_px_per_m=160.0), 1.0, places=2)

        # Расстояние от точки до отрезка
        p = (10.0, 5.0)
        a = (0.0, 0.0)
        b = (20.0, 0.0)
        self.assertAlmostEqual(point_to_segment_dist(p, a, b), 5.0, places=2)

    def test_is_track_alone_true_when_no_others(self) -> None:
        t1 = {
            "tracklet_id": 1,
            "f0": 1,
            "f1": 30,
            "map_p0": [100.0, 100.0],
            "map_p1": [150.0, 100.0],
        }
        idx = TrackNeighborhoodIndex.from_tracklets([t1])
        self.assertTrue(idx.is_track_alone(1, radius_m=2.0))
        self.assertEqual(idx.get_neighbors(1, radius_m=2.0), [])

    def test_is_track_alone_false_when_neighbor_approaches(self) -> None:
        # t1 и t2 идут параллельно на расстоянии 80 px (0.5м < 2.0м)
        t1 = {
            "tracklet_id": 1,
            "f0": 1,
            "f1": 50,
            "map_p0": [100.0, 100.0],
            "map_p1": [200.0, 100.0],
        }
        t2 = {
            "tracklet_id": 2,
            "f0": 10,
            "f1": 40,
            "map_p0": [120.0, 180.0],  # dy = 80 px = 0.5 м
            "map_p1": [180.0, 180.0],
        }
        idx = TrackNeighborhoodIndex.from_tracklets([t1, t2])
        self.assertFalse(idx.is_track_alone(1, radius_m=2.0))
        self.assertFalse(idx.is_track_alone(2, radius_m=2.0))

        # Но в радиусе 0.3м они уже не соседи
        self.assertTrue(idx.is_track_alone(1, radius_m=0.3))

        # Детализация соседей
        neighbors = idx.get_neighbors(1, radius_m=2.0)
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0].other_track_id, 2)
        self.assertAlmostEqual(neighbors[0].min_dist_m, 0.5, places=2)
        self.assertEqual(neighbors[0].first_frame, 10)
        self.assertEqual(neighbors[0].last_frame, 40)
        self.assertEqual(neighbors[0].contact_frames_count, 31)

    def test_pair_transition_clear_with_own_exclusion(self) -> None:
        """При проверке пары A -> B треклет A не должен считаться помехой для B."""
        t1 = {
            "tracklet_id": 1,
            "f0": 1,
            "f1": 50,
            "map_p0": [100.0, 100.0],
            "map_p1": [150.0, 100.0],
        }
        t2 = {
            "tracklet_id": 2,
            "f0": 51,
            "f1": 100,
            "map_p0": [152.0, 100.0],
            "map_p1": [200.0, 100.0],
        }
        idx = TrackNeighborhoodIndex.from_tracklets([t1, t2])

        # Пара должна быть чиста от посторонних
        self.assertTrue(idx.is_pair_transition_clear(1, 2, radius_m=2.0))

        # Если появляется третий треклет рядом с t1
        t3 = {
            "tracklet_id": 3,
            "f0": 10,
            "f1": 30,
            "map_p0": [110.0, 120.0],  # dy = 20 px = 0.125м
            "map_p1": [130.0, 120.0],
        }
        idx_with_t3 = TrackNeighborhoodIndex.from_tracklets([t1, t2, t3])
        self.assertFalse(idx_with_t3.is_pair_transition_clear(1, 2, radius_m=2.0))

    def test_area_clear_during_gap_detection(self) -> None:
        """Если в момент разрыва между A и B через зону прошел C, зона не чиста."""
        t1 = {
            "tracklet_id": 1,
            "f0": 1,
            "f1": 20,
            "map_p0": [100.0, 100.0],
            "map_p1": [120.0, 100.0],
        }
        t2 = {
            "tracklet_id": 2,
            "f0": 40,
            "f1": 60,
            "map_p0": [140.0, 100.0],
            "map_p1": [160.0, 100.0],
        }
        # t3 активен в кадры 25-35 и проходит прямо через отрезок [120, 100] -> [140, 100]
        t3 = {
            "tracklet_id": 3,
            "f0": 25,
            "f1": 35,
            "map_p0": [130.0, 105.0],
            "map_p1": [130.0, 95.0],
        }
        idx = TrackNeighborhoodIndex.from_tracklets([t1, t2, t3])
        self.assertFalse(idx.is_pair_transition_clear(1, 2, radius_m=2.0))

    def test_from_feet_doc(self) -> None:
        feet_doc = {
            "frames": [
                {
                    "frame_index": 1,
                    "points": [{"track_id": 10, "map": [100.0, 100.0]}],
                },
                {
                    "frame_index": 2,
                    "points": [
                        {"track_id": 10, "map": [105.0, 100.0]},
                        {"track_id": 20, "map": [500.0, 500.0]},
                    ],
                },
            ]
        }
        idx = TrackNeighborhoodIndex.from_feet_doc(feet_doc)
        self.assertTrue(idx.is_track_alone(10, radius_m=2.0))
        self.assertTrue(idx.is_track_alone(20, radius_m=2.0))


if __name__ == "__main__":
    unittest.main()

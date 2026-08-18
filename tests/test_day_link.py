"""Тесты для глобальной стадии межкамерной склейки дня (stage_day_link)."""

import unittest
from app.config.settings import Settings
from app.global_id.stage_day_link import link_day_tracks


class TestDayLink(unittest.TestCase):
    def test_link_day_tracks_cross_camera(self):
        settings = Settings()
        settings.day_link_pass0_min_score = 0.80
        settings.day_link_pass1_min_score = 0.70

        # Моделируем 2 трека на разных камерах одного человека
        # Камера 1: t0=10.0, t1=20.0, выход в (2000, 2000)
        # Камера 2: t0=25.0, t1=40.0, вход в (2160, 2000) (смещение 1 метр = 160 px за 5 сек => 0.2 м/с)
        nodes = [
            {
                "uid": "01_20260401#1",
                "session_key": "01_20260401",
                "camera": "Camera_01",
                "camera_index": 1,
                "track_id": 1,
                "f0": 250,
                "f1": 500,
                "t0": 10.0,
                "t1": 20.0,
                "duration_sec": 10.0,
                "p0": [1800.0, 2000.0],
                "p1": [2000.0, 2000.0],
                "v_out": (0.2, 0.0),
                "avg_speed_mps": 0.2,
                "n_frames": 250,
                "reid_embs": None,
                "has_reid": False,
                "has_face": True,
                "face_embs_by_model": {"buffalo_l": None},
                "face_pose_weights_by_model": {"buffalo_l": None},
                "face_crops": ["face1.jpg"],
                "best_face_score": 0.85,
            },
            {
                "uid": "02_20260401#1",
                "session_key": "02_20260401",
                "camera": "Camera_02",
                "camera_index": 2,
                "track_id": 1,
                "f0": 625,
                "f1": 1000,
                "t0": 25.0,
                "t1": 40.0,
                "duration_sec": 15.0,
                "p0": [2160.0, 2000.0],
                "p1": [2400.0, 2000.0],
                "v_out": (0.2, 0.0),
                "avg_speed_mps": 0.2,
                "n_frames": 375,
                "reid_embs": None,
                "has_reid": False,
                "has_face": True,
                "face_embs_by_model": {"buffalo_l": None},
                "face_pose_weights_by_model": {"buffalo_l": None},
                "face_crops": ["face2.jpg"],
                "best_face_score": 0.82,
            },
        ]

        result = link_day_tracks(nodes, settings)
        self.assertIn("persons", result)
        self.assertIn("stats", result)
        self.assertEqual(result["stats"]["n_tracks_total"], 2)


if __name__ == "__main__":
    unittest.main()

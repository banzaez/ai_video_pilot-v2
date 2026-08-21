"""Тесты для всех режимов отбора кадров (pick modes) в TrackBestFramesPicker."""

import unittest
import numpy as np

from app.crops.track_best_frames import (
    ScoredTrackFrame,
    TrackBestFramesPicker,
)


class TestPickModes(unittest.TestCase):
    def setUp(self):
        self.picker = TrackBestFramesPicker(pose_service=None)

    def _create_scored_frames(self, count: int) -> list[ScoredTrackFrame]:
        """Создает тестовый список ScoredTrackFrame с возрастающими индексами кадров."""
        frames = []
        for i in range(count):
            score = 0.5 + 0.4 * np.sin(i / 3.0)
            frames.append(
                ScoredTrackFrame(
                    frame_index=i * 10,
                    target_det={"bbox": [100, 100, 200, 400], "confidence": 0.9},
                    score=float(score),
                    geom_score=0.8,
                    pose_score=float(score),
                    completeness=0.8,
                    face_conf=0.7,
                    crowd_penalty=1.0,
                    n_poses_in_crop=1,
                    pose_result=None,
                    crop_image=None,
                    tracklet_id=1,
                )
            )
        return frames

    def test_empty_and_short_tracklets(self):
        """Проверка пустых и коротких треков (меньше top_k)."""
        # Пустой список
        self.assertEqual(self.picker.pick_best_from_scored([], top_k=5), [])

        # 1 кадр
        f1 = self._create_scored_frames(1)
        picked = self.picker.pick_best_from_scored(f1, top_k=5, mode="first")
        self.assertEqual(len(picked), 1)
        self.assertEqual(picked[0].frame_index, 0)

        # 3 кадра при top_k=5
        f3 = self._create_scored_frames(3)
        picked = self.picker.pick_best_from_scored(f3, top_k=5, mode="middle")
        self.assertEqual(len(picked), 3)

    def test_mode_first(self):
        """Режим 'first': отбор из начального сегмента (первые fraction %)."""
        frames = self._create_scored_frames(50)  # frame_index: 0, 10, ..., 490
        # 20% от 50 = первые 10 кадров (индексы 0..90)
        picked = self.picker.pick_best_from_scored(frames, top_k=3, mode="first", fraction=0.20)
        self.assertEqual(len(picked), 3)
        for p in picked:
            self.assertLessEqual(p.frame_index, 90)

    def test_mode_last(self):
        """Режим 'last': отбор из конечного сегмента (последние fraction %)."""
        frames = self._create_scored_frames(50)  # frame_index: 0..490
        # 20% от 50 = последние 10 кадров (индексы 400..490)
        picked = self.picker.pick_best_from_scored(frames, top_k=3, mode="last", fraction=0.20)
        self.assertEqual(len(picked), 3)
        for p in picked:
            self.assertGreaterEqual(p.frame_index, 400)

    def test_mode_middle(self):
        """Режим 'middle': отбор из центрального сегмента вокруг середины трека."""
        frames = self._create_scored_frames(50)  # frame_index: 0..490, середина ~ 240-250
        # 20% от 50 = 10 кадров вокруг индекса 25 (индексы 20..29 -> frame_index 200..290)
        picked = self.picker.pick_best_from_scored(frames, top_k=3, mode="middle", fraction=0.20)
        self.assertEqual(len(picked), 3)
        for p in picked:
            self.assertGreaterEqual(p.frame_index, 190)
            self.assertLessEqual(p.frame_index, 310)

    def test_mode_score_pure_best(self):
        """Режим 'score': абсолютный топ-K по скору качества."""
        frames = self._create_scored_frames(30)
        frames[5].score = 0.99
        frames[15].score = 0.98
        frames[25].score = 0.97

        picked = self.picker.pick_best_from_scored(frames, top_k=3, mode="score")
        self.assertEqual(len(picked), 3)
        picked_indices = [p.frame_index for p in picked]
        self.assertEqual(picked_indices, [50, 150, 250])

    def test_mode_spread_uniform(self):
        """Режим 'spread': строго равномерное распределение по длине трека."""
        frames = self._create_scored_frames(31)  # 0, 10, ..., 300
        picked = self.picker.pick_best_from_scored(frames, top_k=4, mode="spread")
        self.assertEqual(len(picked), 4)
        self.assertEqual(picked[0].frame_index, 0)
        self.assertEqual(picked[-1].frame_index, 300)

    def test_mode_best_windowed(self):
        """Режим 'best': деление на top_k окон во времени с лучшим по скору в каждом окне."""
        frames = self._create_scored_frames(30)
        picked = self.picker.pick_best_from_scored(frames, top_k=3, mode="best")
        self.assertEqual(len(picked), 3)
        indices = [p.frame_index for p in picked]
        self.assertEqual(indices, sorted(indices))

    def test_guarantee_at_least_one_frame(self):
        """Гарантия возврата >= 1 кадра даже при экстремально малом fraction."""
        frames = self._create_scored_frames(10)
        for mode in ["first", "last", "middle", "best", "spread", "score"]:
            picked = self.picker.pick_best_from_scored(frames, top_k=1, mode=mode, fraction=0.01)
            self.assertGreaterEqual(len(picked), 1)


if __name__ == "__main__":
    unittest.main()

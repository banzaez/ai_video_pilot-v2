"""Тесты для фильтров аномальных и краевых кадров трека."""

import unittest
import cv2
import numpy as np

from app.crops.filters import (
    OutlierFilterConfig,
    compute_hist_similarity,
    extract_clothing_hsv_hist,
    filter_color_outliers,
    filter_kinematic_outliers,
    filter_track_outlier_candidates,
    trim_edge_items,
)
from app.crops.track_best_frames import TrackBestFramesPicker, TrackFrameCandidate


class TestTrackBestFramesFilters(unittest.TestCase):
    def test_trim_edge_items(self):
        items = list(range(10))

        # Длинный трек: обрезаем 2 с начала и 2 с конца
        trimmed = trim_edge_items(items, trim_start=2, trim_end=2, min_len=8)
        self.assertEqual(trimmed, [2, 3, 4, 5, 6, 7])

        # Короткий трек (< min_len=8): не обрезаем
        short_items = list(range(5))
        self.assertEqual(trim_edge_items(short_items, trim_start=2, trim_end=2, min_len=8), [0, 1, 2, 3, 4])

        # Трим только с начала
        trimmed_start = trim_edge_items(items, trim_start=3, trim_end=0, min_len=8)
        self.assertEqual(trimmed_start, [3, 4, 5, 6, 7, 8, 9])

    def test_kinematic_outlier_filter_speed_jump(self):
        # Создаем 8 кандидатов с плавным движением (x смещается на 5 px каждый кадр),
        # кроме кадра 0, который совершает скачок на 200 px (ошибка инициализации трекера)
        cands = []
        # Кадр 0: резкий скачок (x=400)
        cands.append(
            TrackFrameCandidate(
                frame_index=0,
                target_det={"bbox": [400, 100, 450, 250], "confidence": 0.9},
                all_dets=[],
                tracklet_id=1,
            )
        )
        # Кадры 1..7: плавное движение от x=100
        for fi in range(1, 8):
            x = 100 + fi * 5
            cands.append(
                TrackFrameCandidate(
                    frame_index=fi,
                    target_det={"bbox": [x, 100, x + 50, 250], "confidence": 0.9},
                    all_dets=[],
                    tracklet_id=1,
                )
            )

        filtered = filter_kinematic_outliers(cands, max_speed_ratio=3.0)
        # Кадр 0 должен быть отфильтрован
        filtered_frames = [c.frame_index for c in filtered]
        self.assertNotIn(0, filtered_frames)
        self.assertIn(1, filtered_frames)
        self.assertIn(7, filtered_frames)

    def test_kinematic_outlier_filter_area_jump(self):
        # Кадр 7 имеет аномально огромный BBox (скачок площади в 4 раза)
        cands = []
        for fi in range(7):
            cands.append(
                TrackFrameCandidate(
                    frame_index=fi,
                    target_det={"bbox": [100 + fi * 5, 100, 150 + fi * 5, 250], "confidence": 0.9},
                    all_dets=[],
                    tracklet_id=1,
                )
            )
        # Кадр 7: огромный бокс
        cands.append(
            TrackFrameCandidate(
                frame_index=7,
                target_det={"bbox": [50, 50, 250, 450], "confidence": 0.9},
                all_dets=[],
                tracklet_id=1,
            )
        )

        filtered = filter_kinematic_outliers(cands, max_area_ratio=2.0)
        filtered_frames = [c.frame_index for c in filtered]
        self.assertNotIn(7, filtered_frames)
        self.assertIn(0, filtered_frames)

    def test_color_consistency_filter(self):
        # Создаем 8 кандидатов: 7 кадров в синей одежде и 1 краевой кадр (кадр 0) в ярко-красной
        cands = []

        # Кадр 0: красный кроп
        red_crop = np.zeros((100, 50, 3), dtype=np.uint8)
        red_crop[:, :] = [0, 0, 255]  # BGR: Red
        cands.append(
            TrackFrameCandidate(
                frame_index=0,
                target_det={"bbox": [100, 100, 150, 200], "confidence": 0.9},
                all_dets=[],
                crop_image=red_crop,
                tracklet_id=1,
            )
        )

        # Кадры 1..7: синие кропы
        blue_crop = np.zeros((100, 50, 3), dtype=np.uint8)
        blue_crop[:, :] = [255, 0, 0]  # BGR: Blue
        for fi in range(1, 8):
            cands.append(
                TrackFrameCandidate(
                    frame_index=fi,
                    target_det={"bbox": [100, 100, 150, 200], "confidence": 0.9},
                    all_dets=[],
                    crop_image=blue_crop.copy(),
                    tracklet_id=1,
                )
            )

        filtered = filter_color_outliers(cands, min_similarity=0.50, edge_only=True)
        filtered_frames = [c.frame_index for c in filtered]
        self.assertNotIn(0, filtered_frames)
        self.assertIn(1, filtered_frames)
        self.assertIn(7, filtered_frames)

    def test_color_consistency_filter_3_frames(self):
        # 3 кадра: кадр 0 (красный), кадр 1 (синий), кадр 2 (синий)
        red_crop = np.zeros((100, 50, 3), dtype=np.uint8)
        red_crop[:, :] = [0, 0, 255]
        blue_crop = np.zeros((100, 50, 3), dtype=np.uint8)
        blue_crop[:, :] = [255, 0, 0]

        cands = [
            TrackFrameCandidate(
                frame_index=0,
                target_det={"bbox": [100, 100, 150, 200], "confidence": 0.9},
                all_dets=[],
                crop_image=red_crop,
                tracklet_id=1,
            ),
            TrackFrameCandidate(
                frame_index=1,
                target_det={"bbox": [100, 100, 150, 200], "confidence": 0.9},
                all_dets=[],
                crop_image=blue_crop.copy(),
                tracklet_id=1,
            ),
            TrackFrameCandidate(
                frame_index=2,
                target_det={"bbox": [100, 100, 150, 200], "confidence": 0.9},
                all_dets=[],
                crop_image=blue_crop.copy(),
                tracklet_id=1,
            ),
        ]

        filtered = filter_color_outliers(cands, min_similarity=0.50, min_candidates=3, edge_only=True)
        filtered_frames = [c.frame_index for c in filtered]
        self.assertEqual(filtered_frames, [1, 2])

    def test_picker_filter_candidates_integration(self):
        picker = TrackBestFramesPicker(
            pose_service=None,
            trim_enabled=True,
            trim_start=1,
            trim_end=1,
            trim_min_len=6,
            kinematic_enabled=False,
            color_consistency_enabled=False,
        )

        fake_img = np.zeros((100, 50, 3), dtype=np.uint8)
        cands = [
            TrackFrameCandidate(
                frame_index=i,
                target_det={"bbox": [10, 10, 40, 90], "confidence": 0.9},
                all_dets=[],
                crop_image=fake_img,
                tracklet_id=1,
            )
            for i in range(8)
        ]

        cleaned = picker.filter_candidates(cands)
        cleaned_frames = [c.frame_index for c in cleaned]
        # Кадры 0 и 7 должны быть отрезаны триммингом
        self.assertEqual(cleaned_frames, [1, 2, 3, 4, 5, 6])

        # pick_best_for_tracklet должен успешно отработать
        best = picker.pick_best_for_tracklet(cands, top_k=3, filter_outliers=True)
        self.assertEqual(len(best), 3)
        best_frames = [b.frame_index for b in best]
        self.assertNotIn(0, best_frames)
        self.assertNotIn(7, best_frames)


if __name__ == "__main__":
    unittest.main()

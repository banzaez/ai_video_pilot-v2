"""Тесты для мягкого гауссова затухания краев кропа (Gaussian / Alpha Feathering)."""

from __future__ import annotations

import unittest
import numpy as np

from app.crops.geometry import (
    apply_gaussian_feathering,
    create_pose_alpha_mask,
)
from app.pose.types import PoseResult


class TestPoseFeathering(unittest.TestCase):
    def test_create_pose_alpha_mask_with_pose(self):
        # 17 COCO-точек для фигуры человека в кропе 200x100
        # Head (0..4), Torso (5,6,11,12), Limbs (7..10, 13..16)
        kxy = [
            [50.0, 30.0],  # 0: nose
            [45.0, 25.0],  # 1: left eye
            [55.0, 25.0],  # 2: right eye
            [40.0, 28.0],  # 3: left ear
            [60.0, 28.0],  # 4: right ear
            [35.0, 60.0],  # 5: left shoulder
            [65.0, 60.0],  # 6: right shoulder
            [25.0, 90.0],  # 7: left elbow
            [75.0, 90.0],  # 8: right elbow
            [20.0, 120.0], # 9: left wrist
            [80.0, 120.0], # 10: right wrist
            [40.0, 110.0], # 11: left hip
            [60.0, 110.0], # 12: right hip
            [38.0, 150.0], # 13: left knee
            [62.0, 150.0], # 14: right knee
            [36.0, 190.0], # 15: left ankle
            [64.0, 190.0], # 16: right ankle
        ]
        kcf = [0.9] * 17
        pose = PoseResult(bbox=[10.0, 10.0, 90.0, 190.0], confidence=0.95, kxy=kxy, kcf=kcf)

        crop_shape = (200, 100)
        mask = create_pose_alpha_mask(
            crop_shape,
            pose,
            crop_roi=(0, 0, 100, 200),
            kpt_min=0.2,
            sigma=10.0,
            mode="pose",
        )

        self.assertEqual(mask.shape, (200, 100))
        self.assertEqual(mask.dtype, np.float32)
        self.assertTrue(0.0 <= mask.min() <= 1.0)
        self.assertTrue(0.0 <= mask.max() <= 1.0)
        # В центре торса (x=50, y=85) маска должна быть близка к 1.0
        self.assertGreater(mask[85, 50], 0.8)
        # В дальнем углу кропа (x=5, y=5) маска должна затухать к 0.0
        self.assertLess(mask[5, 5], 0.3)

    def test_create_pose_alpha_mask_fallback_ellipse(self):
        # Без позы — проверка fallback
        crop_shape = (150, 80)
        mask = create_pose_alpha_mask(
            crop_shape,
            None,
            crop_roi=(0, 0, 80, 150),
            sigma=8.0,
            mode="pose",
        )

        self.assertEqual(mask.shape, (150, 80))
        self.assertEqual(mask.dtype, np.float32)
        # Центр должен быть ярким, углы темными
        self.assertGreater(mask[75, 40], 0.7)
        self.assertLess(mask[0, 0], 0.3)

    def test_apply_gaussian_feathering(self):
        # Белое изображение 100x100
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # Маска с 1.0 в центре и 0.0 по краям
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[40:60, 40:60] = 1.0

        blended = apply_gaussian_feathering(img, mask, bg_color=(128, 128, 128))

        self.assertEqual(blended.shape, (100, 100, 3))
        self.assertEqual(blended.dtype, np.uint8)
        # В центре (маска=1.0) пиксели белые (255)
        self.assertTrue(np.all(blended[50, 50] == [255, 255, 255]))
        # По краям (маска=0.0) пиксели серые (128)
        self.assertTrue(np.all(blended[10, 10] == [128, 128, 128]))


if __name__ == "__main__":
    unittest.main()

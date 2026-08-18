"""Тесты для TrackBestFramesPicker."""

import numpy as np

from app.crops.track_best_frames import (
    ScoredTrackFrame,
    TrackBestFramesPicker,
    TrackFrameCandidate,
    compute_frame_crowd_penalty,
    compute_geometry_score,
)
from app.pose.types import PoseResult


def test_compute_geometry_score():
    det_good = {"bbox": [100, 100, 200, 400], "confidence": 0.9}  # aspect = 300 / 100 = 3.0
    s_good = compute_geometry_score(det_good, frame_w=1920, frame_h=1080)
    assert s_good > 0.7

    # Обрезан по границе кадра
    det_edge = {"bbox": [0, 100, 100, 400], "confidence": 0.9}
    s_edge = compute_geometry_score(det_edge, frame_w=1920, frame_h=1080)
    assert s_edge < s_good


def test_compute_frame_crowd_penalty():
    target = [100, 100, 200, 400]
    # Нет других людей
    assert compute_frame_crowd_penalty(target, []) == 1.0

    # Другой человек далеко (нет пересечения)
    far_person = [{"bbox": [500, 500, 600, 800]}]
    assert compute_frame_crowd_penalty(target, far_person) == 1.0

    # Другой человек сильно перекрывает целевого
    overlapping_person = [{"bbox": [120, 120, 220, 420]}]
    penalty = compute_frame_crowd_penalty(target, overlapping_person)
    assert penalty < 0.8


def test_picker_without_pose():
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
            target_det={"bbox": [0, 50, 100, 350], "confidence": 0.90},  # край кадра
            all_dets=[{"bbox": [20, 60, 120, 360]}],  # окклюзия
            tracklet_id=1,
        ),
    ]

    scored = picker.score_candidates_batch(candidates)
    assert len(scored) == 3
    # Первый кадр должен быть наилучшим по качеству
    assert scored[0].score > scored[1].score
    assert scored[0].score > scored[2].score

    picked = picker.pick_best_for_tracklet(candidates, top_k=2)
    assert len(picked) == 2


def test_picker_for_group():
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
    assert len(picked) == 4
    # Проверяем, что представлены оба треклета
    picked_tids = {p.tracklet_id for p in picked}
    assert 1 in picked_tids and 2 in picked_tids


def test_extract_face_box_and_crop():
    from app.crops.track_best_frames import (
        extract_face_box_from_pose,
        extract_face_crop_from_person,
    )

    # 1. С позой лица (нос 0, глаза 1,2, уши 3,4)
    kxy = [
        [150.0, 120.0],  # nose
        [140.0, 110.0],  # left eye
        [160.0, 110.0],  # right eye
        [130.0, 115.0],  # left ear
        [170.0, 115.0],  # right ear
        [120.0, 180.0],  # left shoulder
        [180.0, 180.0],  # right shoulder
    ] + [[0.0, 0.0]] * 10
    kcf = [0.9, 0.9, 0.9, 0.8, 0.8, 0.9, 0.9] + [0.0] * 10
    pose = PoseResult(bbox=[100, 100, 200, 400], confidence=0.9, kxy=kxy, kcf=kcf)

    fbox = extract_face_box_from_pose(pose, [100, 100, 200, 400], frame_w=1920, frame_h=1080)
    assert fbox[0] < 150 < fbox[2]
    assert fbox[1] < 120 < fbox[3]

    # Проверка вырезки кропа
    crop_img = np.ones((300, 100, 3), dtype=np.uint8) * 200
    crop_roi = (100, 100, 200, 400)
    fcrop, fbox2 = extract_face_crop_from_person(
        crop_img, crop_roi, pose, {"bbox": [100, 100, 200, 400]}
    )
    assert fcrop is not None
    assert fcrop.shape[0] > 10 and fcrop.shape[1] > 10

    # 2. Fallback без позы
    fbox_fallback = extract_face_box_from_pose(None, [100, 100, 200, 400])
    assert fbox_fallback[1] == 100.0
    assert fbox_fallback[3] <= 220.0


def test_pick_best_faces():
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
    assert len(scored) == 2
    assert scored[0].face_crop is not None
    assert scored[0].face_bbox is not None

    faces_picked = picker.pick_best_faces_for_group({1: cands}, top_k=1)
    assert len(faces_picked) == 1
    assert faces_picked[0].face_crop is not None


def test_pose_caching(tmp_path=None):
    import tempfile
    import os
    from app.crops.track_best_frames import load_pose_cache

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

        # 1-й прогон: запись в кэш
        scored1 = picker.score_candidates_batch(cands, cache_path=cache_file)
        assert len(scored1) == 1
        assert os.path.isfile(cache_file)

        loaded = load_pose_cache(cache_file)
        assert len(loaded) == 1

        # 2-й прогон: чтение из кэша
        scored2 = picker.score_candidates_batch(cands, cache_path=cache_file)
        assert len(scored2) == 1
        assert scored2[0].score == scored1[0].score

        # Другая модель — miss, файл не подходит
        miss = load_pose_cache(cache_file, pose_model="other-pose.pt", kpt_min=0.25)
        assert miss == {}

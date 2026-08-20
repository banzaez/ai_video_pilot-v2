# Зеркалируемые модули Python ↔ TypeScript

Админка и offline-пайплайн не делят runtime — логика **дублируется намеренно**. При изменении одной стороны сверяйте вторую и прогоняйте fixture-тесты.

| Домен | Python | TypeScript | Fixture-тесты |
|-------|--------|------------|---------------|
| Ноги (bbox + прилавки) | `app/global_id/feet.py` | `admin/src/feet.ts` | `admin/src/feet.fixtures.test.ts` |
| 3D-луч / fitRayPose | `app/global_id/camera_pose.py` | `admin/src/cameraPose.ts` | `tests/test_camera_pose.py`, `admin/src/cameraPose.test.ts` |
| Фингерпринт калибровки | `app/global_id/calib_fingerprint.py` | `admin/src/calibFingerprint.ts` | `admin/src/calibFingerprint.test.ts` |
| Stale per-video | `app/artifact_meta.py` `stale_stages_report` | `admin/server/media-handlers.ts` `staleStagesReport` | `tests/test_artifact_meta.py` |


# Зеркалируемые модули Python ↔ TypeScript

Админка и offline-пайплайн не делят runtime — логика **дублируется намеренно**. При изменении одной стороны сверяйте вторую и прогоняйте fixture-тесты.

| Домен | Python | TypeScript | Fixture-тесты |
|-------|--------|------------|---------------|
| Ноги (bbox + прилавки) | `app/global_id/feet.py` | `admin/src/feet.ts` | `tests/test_global_id.py` |
| 3D-луч / fitRayPose | `app/global_id/camera_pose.py` | `admin/src/cameraPose.ts` | `tests/test_camera_pose.py`, `admin/src/cameraPose.test.ts` |
| Фингерпринт калибровки | `app/global_id/calib_fingerprint.py` | `admin/src/calibFingerprint.ts` | `tests/test_stage_feet.py`, `admin/src/calibFingerprint.test.ts` |
| body_calib k (чтение старого JSON) | `app/maps/body_calib_data.py` | `pickKForShoulder` в `feet.ts` | `tests/test_feet_fixtures.py`, `admin/src/feet.fixtures.test.ts` |
| Stale per-video | `app/artifact_meta.py` `stale_stages_report` | `admin/server/media-handlers.ts` `staleStagesReport` | `tests/test_artifact_meta.py` |

Константы `FALLBACK_K = 2.0`, `MIN_SAMPLES = 30` — держать синхронными в `body_calib_data.py` и `feet.ts`.

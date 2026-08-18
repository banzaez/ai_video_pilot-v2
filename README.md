# AI Video Person Tracking (YOLO26 / YOLOv8 + трекеры Ultralytics)

Офлайн-детекция и трекинг людей на видеофайле.

## Возможности
- YOLO26 / YOLOv8 / YOLOv11 + трекер (по умолчанию ByteTrack, `data/models/yolo26n.pt`)
- Фильтрация классов COCO (`person`)
- Двухстадийный пайплайн: батч-детекция → трекинг по боксам
- Tracklets + ReID + link: склейка коротких треков в группы внутри ролика
- Отрисовка bbox / ID / траекторий
- Артефакты: `data/results/{session_key}/` (camera-day) или `data/results/{имя видео}/` (legacy lite)
- Конфиг YAML + CLI (CLI имеет приоритет)

## Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp config.yaml.example config.yaml
```

## Использование

### Видеофайл (настройки из `config.yaml`)
```bash
python -m app.main
```

### Camera-day sessions (prod)

Несколько частей одной камеры за день (`Camera_01_nvr_local_…`) лежат **плоско** в `data/video/`.
Папка `data/video/lite/` — dev-формат, в discovery **не участвует**.

Ключ session: `01_20260601` = `{камера:02d}_{YYYYMMDD}` по времени начала первой части.
Manifest: `data/results/01_20260601/info.json` (`kind: camera_day`, массив `parts` с offsets).

```bash
# одна session
python -m app.main --input session:01_20260601 --stage all

# все sessions в data/video/
python -m app.main --input data/video --from detect --to link
```

Legacy (один файл из lite):

```bash
python -m app.main --input data/video/lite/foo.mp4
```

### Один файл, а не вся папка
```bash
python -m app.main --input Camera_01_….mp4
```

### Stage 2 — трекинг (по detections JSON)
```bash
python -m app.main --stage track
```
Пишет `data/results/{видео}/tracking.json` (боксы по кадрам) и `tracks.json` — сводку по
каждому треку: `id` вида `01#1` (камера#локальный id), `t0`/`t1`, точки и края кадра
при появлении и уходе (`enter`/`exit`), медианный размер. По краю выхода и времени
ищется продолжение трека на соседней камере.

### Stage 2a–2c — tracklets → ReID → link
```bash
python -m app.main --stage tracklets
python -m app.main --stage tracklet_reid
python -m app.main --stage tracklet_link
```
Короткие треклеты ByteTrack склеиваются в длинные треки (`tracklet_links.json`). ReID — SOLIDER (веса в `reid:`). Кропы для ReID пишутся в `tracklet_crops/` при `save_crops`.

Инференс-код SOLIDER: `app/third_party/solider_reid/` (без train). Веса в git не коммитятся.

### Stage pose / feet
```bash
python -m app.main --stage pose
python -m app.main --stage feet
```

### Stage link — группы треков внутри ролика
Пары из tracklet ReID со score ≥ `link.pass1_min_score` → `auto_groups` / `groups` в `link.json` (вкладка «Склейки»).

```yaml
link:
  enabled: true
  group_mode: complete_link
  min_score: 0.95
  pass1_min_score: 0.98
  max_map_m: 3.0
```

Полный прогон: `--stage all`. Без склеек: `--stage no_merge` (detect→feet).

### Ускорение detect (CoreML)

На macOS пайплайн подхватывает `.mlpackage` рядом с `.pt` (если есть). Экспорт один раз:

```bash
pip install 'coremltools>=9.0'   # только для экспорта
python scripts/export_yolo_runtime.py \
  --detect data/models/yolo26s.pt \
  --format coreml \
  --imgsz 1280
```

**Ограничение:** статический CoreML в Ultralytics не умеет `batch>1` (падает с `IndexError`), поэтому на Mac
инференс идёт с **`batch_size=1`**. Это **медленнее**, чем батч на `.pt` (например 32) — так и задумано,
переделывать runtime не нужно. В `config.yaml` пути остаются на `.pt`.

Для Linux/CUDA: `--format onnx`. Модели между роликами одного `run()` кэшируются в памяти.

С указанной стадии и дальше:
```bash
python -m app.main --from track
```
Диапазон стадий (включительно):
```bash
python -m app.main --from track --to link
```
`--from pose` = pose + feet + link.  
`--from track --to feet` = track + pose + feet. Нельзя вместе с `--stage`.

### Версии артефактов

В каждый JSON стадии (`info`…`link`) пишется блок `artifact`:
`file_version`, `written_at`, `inputs` (отпечаток родителя).
Админка сравнивает цепочку и показывает устаревшие стадии + команду вида
`python -m app.main --from track --to link`.

### Job API + вкладка «Пайплайн»

Отдельный процесс запускает стадии и пишет логи в `data/jobs/{id}/`. Админка (вкладка **Пайплайн**) ходит на него через Vite proxy `/api/jobs` → `:8765`.

```bash
# зависимости (если ещё не стоят)
pip install 'fastapi>=0.115,<1' 'uvicorn>=0.32,<1'

# терминал 1 — Job API
./venv/bin/python -m app.jobs

# терминал 2 — админка
cd admin && npm run dev
```

Опционально: `JOBS_API_TOKEN=secret` на API и `VITE_JOBS_API_TOKEN=secret` в admin — Bearer auth (для VPS).

На VPS: systemd unit на `python -m app.jobs` (host `0.0.0.0` или только localhost за nginx), статику `admin/dist` отдаёт nginx; `/api/jobs` проксируется на Job API. Медиа/карты пока остаются на Vite middleware или отдельном static+API хосте.

### Трекер (Stage 2a)

Параметры ByteTrack — в `tracklet_pipeline.tracker`:

```yaml
tracklet_pipeline:
  tracker:
    type: bytetrack
    track_high_thresh: 0.25
    track_buffer: 11
    match_thresh: 0.78
```

CLI: `--tracker botsort` (дефолты Ultralytics). GMC в пайплайне выключен.

**Общие** (есть почти у всех):

| Параметр | Для чего |
|---|---|
| `track_high_thresh` | С какого conf детекция идёт в «уверенный» матч. Выше → меньше ложных треков, можно терять слабые боксы |
| `track_low_thresh` | Ниже high: «слабые» боксы только продлевают уже живой трек, новый ID не заводят |
| `new_track_thresh` | Мин. conf, чтобы **создать** новый ID. Выше → меньше мусорных треков |
| `track_buffer` | Сколько шагов трекера помнить пропавший объект (не видеокадры). Больше → лучше переживает дыры, риск склеить чужой ID |
| `match_thresh` | Насколько похожи боксы по IoU, чтобы считать одним человеком. Ниже → легче склеить, выше → чаще рвётся ID |
| `fuse_score` | Мешать conf и IoU при матче. Часто рвёт ID на средних conf — у нас обычно `false` |

**GMC / ReID** (`botsort`, `deepocsort`, `tracktrack`):

| Параметр | Для чего |
|---|---|
| `gmc_method` | Компенсация движения камеры. Нужен на PTZ; на фикс. камере `none` |
| `with_reid` | Матч по внешности. У нас Stage 2 без кадров — флаг пока бесполезен |
| `proximity_thresh` | Мин. IoU, чтобы вообще смотреть внешность |
| `appearance_thresh` | Насколько похожа внешность для склейки ID (выше = строже) |
| `model` | `auto` или путь к ReID-модели |

**OC-SORT / Deep OC-SORT:**

| Параметр | Для чего |
|---|---|
| `delta_t` | За сколько шагов считать направление движения |
| `inertia` | Насколько штрафовать «ломаную» траекторию |
| `use_byte` | Второй проход как у ByteTrack по слабым детекциям |
| `alpha_fixed_emb` | (deep) Как быстро обновлять эмбеддинг внешности |

**FastTracker** (окклюзии):

| Параметр | Для чего |
|---|---|
| `enlarge_bbox_occ` | Расширить bbox, пока человека закрыли |
| `dampen_motion_occ` | Притормозить предсказание движения под окклюзией |
| `occ_cover_thresh` | Какая доля перекрытия = «закрыт» |
| `occ_reappear_window` | Как долго после пропажи ещё искать того же человека |
| `init_iou_suppress` | Не плодить новый ID поверх уже существующего трека |

**TrackTrack** (мульти-сигналы):

| Параметр | Для чего |
|---|---|
| `iou_weight` / `reid_weight` / `conf_weight` / `angle_weight` | Веса в cost: геометрия / внешность / conf / углы бокса |
| `tai_thr` / `min_track_len` | Осторожный старт новых треков (NMS + мин. история) |
| `lost_match_thr` | Ослабленный матч для давно lost (длинные окклюзии) |

## Параметры CLI
| Аргумент | Описание |
|---|---|
| `--config` | Путь к YAML |
| `--input` | Папка или один видеофайл |
| `--stage` | `info` \| `detect` \| `tracklets` \| `tracklet_reid` \| `tracklet_link` \| `track` \| `pose` \| `feet` \| `link` \| `all` \| `no_merge` |
| `--from` | С этой стадии (до конца или до `--to`) |
| `--to` | До этой стадии включительно (только с `--from`) |

## Структура `app/`
- `main.py` — CLI
- `config/` — YAML + CLI → Settings
- `pipeline.py` — офлайн: detect → tracklets → ReID → track → pose → feet → link
- `tracklet/` — короткие треки, ReID и склейка в длинные
- `crops/` — геометрия кропа для tracklet ReID
- `link/` — группы треков внутри ролика (`link.json`)
- `global_id/` — pose / feet / проекции на карту
- `reid.py` — эмбеддинги кропов (SOLIDER / OSNet)
- `parallel_tracker.py` — Stage 1 detect + Stage 2 трекинг
- `viz.py` — отрисовка
- `io/` — видео и JSON

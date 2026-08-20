# Руководство по CLI (Command Line Interface)

В проекте управление запуском пайплайна видеоаналитики, обработкой сессий (Camera-Day Sessions), вспомогательными утилитами и выгрузкой видео осуществляется через интерфейс командной строки (CLI).

---

## 1. Основной пайплайн: `python -m app.main`

Точка входа: [`app/main.py`](file:///Users/mac/Documents/Projects/Pycharm/ai_video_pilot-v2/app/main.py)  
Модуль конфигурации: [`app/config/loader.py`](file:///Users/mac/Documents/Projects/Pycharm/ai_video_pilot-v2/app/config/loader.py)  
Запуск подпроцессов / фоновых задач: [`app/jobs/runner.py`](file:///Users/mac/Documents/Projects/Pycharm/ai_video_pilot-v2/app/jobs/runner.py)

### Приоритет применения настроек
1. **CLI-аргументы** (наивысший приоритет)
2. **YAML-конфиг** (`config.yaml` или переданный через `--config`)
3. **Значения по умолчанию в коде** (`app/config/settings.py`)

---

### Доступные аргументы

| Аргумент | Тип / Варианты | По умолчанию | Описание |
|---|---|---|---|
| `--config` | `str` | `config.yaml` | Путь к файлу конфигурации YAML |
| `--input` | `str` | `data/video` (из YAML) | Путь к файлу/папке, либо ключ сессии: `session:<KEY>` или `<KEY>` (например, `session:01_20260601`) |
| `--stage` | `str` (см. ниже) | `all` | Запустить ровно одну указанную стадию |
| `--from` | `str` (см. ниже) | `None` | Запустить диапазон стадий, начиная с указанной |
| `--to` | `str` (см. ниже) | `None` | Завершить выполнение на указанной стадии (только вместе с `--from`) |
| `--detection-backend` | `yolo` \| `rtdetr` \| `rtdetr_v2` | из YAML (`yolo`) | Архитектура/движок детектора объектов |
| `--model` | `str` | из YAML | Путь к файлу весов модели детекции |
| `--tracker` | `bytetrack` \| `botsort` \| `ocsort` \| `deepocsort` \| `fasttrack` \| `tracktrack` | из YAML (`bytetrack`) | Алгоритм трекинга |
| `--tracker-with-reid` / `--no-tracker-with-reid` | `flag` | из YAML (`false`) | Включить/выключить ReID-ассоциацию внутри трекера |
| `--tracker-reid-model` | `str` | из YAML (`yolo26n-reid.onnx`) | Модель ReID для онлайн-трекера |
| `--conf` | `float` (0.0 .. 1.0) | из YAML (`0.35`) | Порог уверенности детектора (confidence) |
| `--device` | `auto` \| `cuda` \| `mps` \| `cpu` | `auto` | Вычислительное устройство для инференса |
| `--batch-size` | `int` | из YAML (`16`) | Размер батча кадров при детекции |
| `--imgsz` | `int` | из YAML (`640`) | Разрешение входного кадра для нейросети |
| `--detect-every-n` | `int` | из YAML (`1`) | Запускать детекцию каждый N-й кадр (пропуск кадров) |
| `--workers` | `int` | `None` (авто) | Количество параллельных процессов обработки видео |

---

### Стадии пайплайна (`--stage`, `--from`, `--to`)

Пайплайн выполняет обработку видео строго по цепочке:

1. **`info`** — Извлечение метаданных видео или формирование манифеста сессии (`info.json`).
2. **`detect`** — Пакетная детекция людей на кадрах (`detections.json`).
3. **`tracklets`** — Формирование надежных коротких треклетов локальным трекером (`tracklets.json`).
4. **`tracklet_reid`** — Извлечение визуальных ReID-эмбеддингов внешности для треклетов (SOLIDER / OSNet) (`tracklet_reid.json`).
5. **`tracklet_link`** — Склейка треклетов в длинные треки на основе пространственно-временной близости и сходства ReID (`tracklet_link.json`).
6. **`track`** — Формирование финальных локальных треков с интерполяцией пропусков (`track.json`).
7. **`pose`** — Детекция скелетных ключевых точек людей (YOLOv8/11/26 Pose) (`pose.json`).
8. **`feet`** — Расчет положения стоп людей и их проекция на 2D-план/карту помещения с учетом калибровки камер (`feet.json`).
9. **`camera_link`** — Межкамерный глобальный трекинг: объединение треков с разных камер в глобальные ID (лицо Buffalo-L / Antelopev2 + ReID + топология) (`camera_link.json`).

*Специальные значения для `--stage`:*
* `all` — Выполнить все стадии от `info` до `camera_link`.
* `no_merge` — Алиас для `all`.

---

## 2. Работа с сессиями (Camera-Day Sessions)

В проекте реализован механизм сессий (группировка частей записей за день по одной камере, склеиваемых в общий манифест `info.json`).

### Формат именования prod-файлов
```
Camera_<camera_idx>_<source>_<YYYYMMDDHHMMSS>_<YYYYMMDDHHMMSS>_<segment>.mp4
# Пример:
Camera_01_nvr_local_20260601100000_20260601101500_tid1401s001.mp4
```

### Ключ сессии (`session_key`)
Формируется как `<CAMERA_INDEX>_<YYYYMMDD>` (например, `01_20260601` — Камера 1 за 1 июня 2026).

### Способы передачи сессии в CLI через `--input`:
1. `session:<SESSION_KEY>` (например, `session:01_20260601`)
2. `<SESSION_KEY>` (например, `01_20260601`)
3. Путь к папке `data/video/` — если в папке лежат prod-имена, пайплайн автоматически обнаружит все доступные сессии (`discover_sessions`).
4. Путь к одному из файлов сессии — пайплайн автоматически найдет сессию, к которой относится этот фрагмент.

---

## 3. Примеры использования CLI

### Примеры запуска для сессий

```bash
# 1. Запустить полный пайплайн для конкретной суточной сессии камеры
python -m app.main --input session:01_20260601

# Короткий синтаксис (без префикса session:)
python -m app.main --input 01_20260601

# 2. Выполнить только стадии ReID и склейки треклетов для сессии
python -m app.main --input session:01_20260601 --from tracklet_reid --to tracklet_link

# 3. Пересчитать только положение ног и проекцию на карту (feet) для сессии
python -m app.main --input session:01_20260601 --stage feet

# 4. Обработать сессию на GPU CUDA с кастомным конфигом
python -m app.main --input session:01_20260601 --config configs/prod.yaml --device cuda --batch-size 32
```

### Примеры для обычных файлов и директорий

```bash
# Обработать все видео из папки по умолчанию (data/video)
python -m app.main

# Обработать один конкретный ролик со своим конфигом
python -m app.main --config my_config.yaml --input data/video/Camera_01_20260601_100000.mp4

# Выполнить шаги от детекции до стоп (включительно) для файла
python -m app.main --input data/video/cam1.mp4 --from detect --to feet

# Использовать RT-DETRv2 на Apple Silicon GPU (mps) с батчем 32
python -m app.main --detection-backend rtdetr_v2 --device mps --batch-size 64

# Детекция каждый 2-й кадр с трекером BotSORT и порогом conf=0.4
python -m app.main --detect-every-n 2 --tracker botsort --conf 0.4

# Параллельная обработка видео в 4 потока
python -m app.main --workers 4
```

---

## 4. Скачивание записей с NVR: `scripts/fetch_nvr_clips.py`

Скрипт для прямого взаимодействия с видеорегистраторами Hikvision (через ISAPI) без использования БД и пайплайна.

### Параметры

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--at` | *Обязательный* | Целевое время на NVR (формат `"YYYY-MM-DD HH:MM"`) |
| `--list` | `False` | Только показать найденные фрагменты (без скачивания) |
| `--download` | `True` | Скачать найденные фрагменты |
| `--out` | `.` (текущая) | Каталог для сохранения `.mp4` файлов |
| `--tracks` | из `.env` / `1401,1501` | Список ID каналов NVR через запятую (например, `1401,1501`) |
| `--host` | из `.env` / `183.88.220.86` | IP-адрес / хост NVR |
| `--port` | из `.env` / `8003` | Порт ISAPI |
| `--user` | из `.env` / `admin` | Логин NVR |
| `--password` | из `.env` / `""` | Пароль NVR |
| `--lookback-hours` | `24` | Глубина поиска относительно `--at` (в часах) |

### Примеры
```bash
# 1. Посмотреть доступные фрагменты на 1 июня 10:20 (камеры 14 и 15)
python scripts/fetch_nvr_clips.py --at "2026-06-01 10:20" --list

# 2. Скачать ролики в папку data/video/
python scripts/fetch_nvr_clips.py --at "2026-06-01 10:20" --out data/video

# 3. Скачать с указанием другого NVR и каналов
python scripts/fetch_nvr_clips.py --at "2026-06-01 10:20" \
    --host 192.168.1.100 --port 8000 --user admin --password secret \
    --tracks 1601,1701 --out data/video
```

---

## 5. Извлечение кропов: `app/tools/extract_crop.py`

Утилита для быстрого On-Demand вырезания фрагментов кадров (bbox людей / лиц) из видео без пересохранения всех кадров на диск.

### Примеры

#### Одиночный кроп:
```bash
python -m app.tools.extract_crop \
    --video data/video/cam1.mp4 \
    --frame 120 \
    --bbox 100.0 150.0 300.0 500.0 \
    --output data/results/cam1/crops/crop_120.jpg \
    --quality 90
```

#### Пакетное извлечение по JSON-файлу:
```bash
python -m app.tools.extract_crop \
    --video data/video/cam1.mp4 \
    --items items.json \
    --quality 85
```
*Формат `items.json`:*
```json
[
  {
    "frame": 120,
    "bbox": [100.0, 150.0, 300.0, 500.0],
    "output": "data/results/cam1/crops/120.jpg"
  }
]
```

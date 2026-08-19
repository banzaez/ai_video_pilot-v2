"""
python3 scripts/convert/transcode_videos.py \
  --path data/video/20260817 \
  --out data/video/convert \
  --codec libx264 \
  --every 5
"""

import argparse
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_stop = threading.Event()
_procs_lock = threading.Lock()
_procs: set[subprocess.Popen] = set()
_progress_lock = threading.Lock()
_progress_lines = 0
_progress_state: dict[str, str] = {}
_progress_order: list[str] = []


class Stopped(Exception):
    """Остановка по Ctrl+C / SIGTERM."""


def request_stop(_signum=None, _frame=None) -> None:
    if _stop.is_set():
        with _procs_lock:
            for proc in list(_procs):
                _kill_proc(proc, force=True)
        return
    _stop.set()
    sys.stderr.write("\nОстановка (Ctrl+C). Жду завершения ffmpeg…\n")
    sys.stderr.flush()
    with _procs_lock:
        for proc in list(_procs):
            _kill_proc(proc, force=False)


def _kill_proc(proc: subprocess.Popen, *, force: bool) -> None:
    if proc.poll() is not None:
        return
    try:
        if force:
            proc.kill()
        else:
            proc.terminate()
    except OSError:
        return
    try:
        proc.wait(timeout=2 if force else 4)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass


def is_apple_silicon():
    """Проверяет, является ли система Mac на Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _videotoolbox_args(codec: str) -> list:
    """Аппаратный энкодер Media Engine (M1/M2/M3): скорость важнее энергосбережения."""
    enc = "h264_videotoolbox" if codec == "libx264" else "hevc_videotoolbox"
    return [
        "-c:v",
        enc,
        "-q:v",
        "55",
        "-prio_speed",
        "1",
        "-power_efficient",
        "0",
    ]


def _nvenc_video_args(codec: str, cq: int) -> list:
    """H.264/H.265 на GPU NVIDIA: VBR + CQ (ниже cq = лучше картинка, тяжелее файл)."""
    enc = "h264_nvenc" if codec == "libx264" else "hevc_nvenc"
    return [
        "-c:v",
        enc,
        "-preset",
        "p5",
        "-rc",
        "vbr",
        "-cq",
        str(cq),
        "-b:v",
        "0",
    ]


def _cpu_video_args(codec: str, *, fast: bool = False) -> list:
    preset = "veryfast" if fast else "medium"
    return [
        "-c:v",
        codec,
        "-crf",
        "23" if codec == "libx264" else "28",
        "-preset",
        preset,
        "-threads",
        "0",
    ]


def _frame_step_args(every: int) -> list[str]:
    """Оставить каждый N-й кадр, таймкоды с нуля (иначе плеер не стартует)."""
    if every <= 1:
        return []
    return ["-vf", f"framestep={every},setpts=PTS-STARTPTS"]


def _mux_args() -> list[str]:
    # pcm_mulaw и прочее с NVR нельзя copy в MP4 — берём только видео.
    return [
        "-map",
        "0:v:0",
        "-an",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        "-max_muxing_queue_size",
        "2048",
    ]


def _probe_duration(path: str) -> float | None:
    try:
        raw = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=20,
        )
        val = float(raw.strip())
        if val > 0 and val != float("inf"):
            return val
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, OSError):
        return None
    return None


def _hms(sec: float | None) -> str:
    if sec is None or sec < 0 or sec != sec:
        return "--:--"
    total = int(sec)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _parse_out_time(raw: str) -> float | None:
    raw = raw.strip()
    if not raw or raw.startswith("N"):
        return None
    parts = raw.split(":")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return None


def _parse_speed(raw: str) -> float | None:
    raw = raw.strip().rstrip("x")
    if not raw or raw.startswith("N"):
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if val > 0 else None


def _short_name(name: str, width: int) -> str:
    if width <= 3:
        return "…"
    if len(name) <= width:
        return name
    return name[: max(1, width - 1)] + "…"


def _progress_erase() -> None:
    global _progress_lines
    if _progress_lines and sys.stderr.isatty():
        sys.stderr.write(f"\033[{_progress_lines}A\033[J")
        _progress_lines = 0


def _progress_paint() -> None:
    global _progress_lines
    if not sys.stderr.isatty() or not _progress_order:
        return
    block = "\n".join(_progress_state[k] for k in _progress_order if k in _progress_state)
    if not block:
        return
    sys.stderr.write(block + "\n")
    sys.stderr.flush()
    _progress_lines = block.count("\n") + 1


def _progress_log(msg: str) -> None:
    with _progress_lock:
        _progress_erase()
        print(msg, flush=True)
        _progress_paint()


def _progress_finish(label: str, msg: str) -> None:
    with _progress_lock:
        _progress_state.pop(label, None)
        if label in _progress_order:
            _progress_order.remove(label)
        _progress_erase()
        print(msg, flush=True)
        _progress_paint()


def _progress_reset() -> None:
    with _progress_lock:
        _progress_erase()
        _progress_state.clear()
        _progress_order.clear()
        sys.stderr.flush()


def _draw_progress(
    *,
    label: str,
    index: int,
    total: int,
    out_time: float | None,
    duration: float | None,
    speed: float | None,
) -> None:
    cols = max(60, shutil.get_terminal_size((100, 24)).columns)
    pct = None
    if duration and duration > 0 and out_time is not None:
        pct = min(1.0, max(0.0, out_time / duration))
    bar_w = 22
    if pct is None:
        bar = "░" * bar_w
        pct_s = "  --%"
    else:
        fill = int(round(bar_w * pct))
        bar = "█" * fill + "░" * (bar_w - fill)
        pct_s = f"{pct * 100:5.1f}%"
    eta = ""
    if duration and out_time is not None and speed and out_time < duration:
        eta = f"  ETA {_hms((duration - out_time) / speed)}"
    speed_s = f"{speed:.1f}x" if speed else "--x"
    prefix = f"[{index}/{total}] "
    times = f"{_hms(out_time)}/{_hms(duration)}"
    tail = f"  {bar} {pct_s}  {times}  {speed_s}{eta}"
    name_w = max(8, cols - len(prefix) - len(tail) - 1)
    line = f"{prefix}{_short_name(label, name_w)}{tail}"
    if len(line) > cols:
        line = line[: cols - 1]
    if not sys.stderr.isatty():
        print(line, flush=True)
        return
    with _progress_lock:
        if label not in _progress_state:
            _progress_order.append(label)
        _progress_state[label] = line
        _progress_erase()
        _progress_paint()


def _run_ffmpeg(
    command: list[str],
    *,
    label: str = "",
    duration: float | None = None,
    index: int = 1,
    total: int = 1,
) -> None:
    if _stop.is_set():
        raise Stopped()
    cmd = [command[0], "-nostats", "-loglevel", "error", "-progress", "pipe:1", *command[1:]]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    err_chunks: list[str] = []

    def _drain_err() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            err_chunks.append(line)

    err_thread = threading.Thread(target=_drain_err, daemon=True)
    err_thread.start()
    with _procs_lock:
        _procs.add(proc)
    out_time = 0.0
    speed: float | None = None
    last_draw = 0.0
    interval = 0.12 if sys.stderr.isatty() else 1.0
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.strip()
            if not line or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key == "out_time":
                parsed = _parse_out_time(val)
                if parsed is not None:
                    out_time = parsed
            elif key in ("out_time_us", "out_time_ms"):
                try:
                    us = int(val)
                except ValueError:
                    us = -1
                if us >= 0:
                    out_time = us / 1_000_000
            elif key == "speed":
                speed = _parse_speed(val)
            elif key == "progress":
                now = time.monotonic()
                if val == "end" or now - last_draw >= interval:
                    last_draw = now
                    _draw_progress(
                        label=label or os.path.basename(command[-1]),
                        index=index,
                        total=total,
                        out_time=out_time,
                        duration=duration,
                        speed=speed,
                    )
        code = proc.wait()
    finally:
        with _procs_lock:
            _procs.discard(proc)
        err_thread.join(timeout=1)
    if _stop.is_set():
        raise Stopped()
    if code != 0:
        err = "".join(err_chunks).strip()
        raise subprocess.CalledProcessError(code, command, stderr=err)


def _restore_inplace(
    inplace: bool, source_path: str, input_file: str, output_path: str, part_path: str | None = None
) -> None:
    for path in (part_path, output_path):
        if path and path != input_file and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    if inplace and input_file != source_path and os.path.exists(input_file) and not os.path.exists(source_path):
        try:
            os.rename(input_file, source_path)
        except OSError:
            pass


def _finish_output(part_path: str, output_path: str) -> None:
    if part_path == output_path:
        return
    os.replace(part_path, output_path)


def _ffmpeg_prefix(input_file: str, *, hw: str | None = None) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-fflags", "+genpts"]
    if hw == "vld":
        cmd += ["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld"]
    elif hw == "vt":
        cmd += ["-hwaccel", "videotoolbox"]
    cmd += ["-i", input_file]
    return cmd


def _vt_commands(input_file: str, output_path: str, codec: str, *, every: int) -> list[list[str]]:
    """Сначала zero-copy GPU, затем decode CPU + encode VT (если формат не в VideoToolbox)."""
    video_args = _videotoolbox_args(codec)
    step = _frame_step_args(every)
    tail = [*step, *video_args, *_mux_args(), "-y", output_path]
    commands: list[list[str]] = []
    # Фильтр кадров нельзя вешать на videotoolbox_vld — нужен CPU-кадр.
    if every <= 1:
        commands.append([*_ffmpeg_prefix(input_file, hw="vld"), *tail])
    commands.append([*_ffmpeg_prefix(input_file, hw="vt"), *tail])
    commands.append([*_ffmpeg_prefix(input_file), *tail])
    return commands


def transcode_one(
    filename: str,
    source_dir: str,
    output_dir: str,
    codec: str,
    *,
    use_nvenc: bool,
    nvenc_cq: int,
    auto_videotoolbox: bool,
    every: int,
    index: int = 1,
    total: int = 1,
) -> tuple[str, bool, str]:
    source_path = os.path.join(source_dir, filename)
    if _stop.is_set():
        return filename, False, "остановлено"
    inplace = os.path.abspath(source_dir) == os.path.abspath(output_dir)
    input_file = source_path
    output_path = source_path

    if inplace:
        name, ext = os.path.splitext(filename)
        temp_orig = os.path.join(source_dir, f"{name}_orig{ext}")
        try:
            os.rename(source_path, temp_orig)
            input_file = temp_orig
            output_path = source_path
        except Exception as e:
            return filename, False, f"ошибка подготовки: {e}"
    else:
        output_path = os.path.join(output_dir, filename)
        input_file = source_path

    part_path = f"{os.path.splitext(output_path)[0]}.part{os.path.splitext(output_path)[1]}"
    step = _frame_step_args(every)
    if use_nvenc:
        commands = [
            [
                *_ffmpeg_prefix(input_file),
                *step,
                *_nvenc_video_args(codec, nvenc_cq),
                *_mux_args(),
                "-y",
                part_path,
            ]
        ]
    elif auto_videotoolbox:
        commands = _vt_commands(input_file, part_path, codec, every=every)
    else:
        commands = [
            [
                *_ffmpeg_prefix(input_file),
                *step,
                *_cpu_video_args(codec),
                *_mux_args(),
                "-y",
                part_path,
            ]
        ]

    duration = _probe_duration(input_file)
    run_kw = dict(label=filename, duration=duration, index=index, total=total)

    last_err = ""
    try:
        if _stop.is_set():
            raise Stopped()
        for i, command in enumerate(commands):
            try:
                _run_ffmpeg(command, **run_kw)
                _finish_output(part_path, output_path)
                return filename, True, output_path
            except subprocess.CalledProcessError as e:
                last_err = (e.stderr or str(e)).strip() or str(e)
                if os.path.exists(part_path):
                    try:
                        os.remove(part_path)
                    except OSError:
                        pass
                if i + 1 < len(commands) and not _stop.is_set():
                    _progress_log(f"  {filename}: VideoToolbox zero-copy не подошёл, повтор без hwaccel decode")

        if auto_videotoolbox and not _stop.is_set():
            try:
                _run_ffmpeg(
                    [
                        *_ffmpeg_prefix(input_file),
                        *_frame_step_args(every),
                        *_cpu_video_args(codec, fast=True),
                        *_mux_args(),
                        "-y",
                        part_path,
                    ],
                    **run_kw,
                )
                _finish_output(part_path, output_path)
                _progress_log(f"  {filename}: fallback CPU veryfast")
                return filename, True, output_path
            except subprocess.CalledProcessError as e:
                last_err = (e.stderr or str(e)).strip() or str(e)
    except Stopped:
        _restore_inplace(inplace, source_path, input_file, output_path, part_path)
        return filename, False, "остановлено"

    _restore_inplace(inplace, source_path, input_file, output_path, part_path)
    return filename, False, last_err or "ffmpeg error"


def transcode_videos(
    source_dir,
    output_dir,
    codec="libx265",
    *,
    use_nvenc=False,
    nvenc_cq=26,
    jobs=1,
    every=1,
):
    """
    Перекодирует видео из source_dir в output_dir.
    После успеха удаляет исходник из source_dir.
    """
    extensions = (".mp4", ".mkv", ".mov", ".avi", ".ts")

    if os.path.isfile(source_dir):
        files = [os.path.basename(source_dir)]
        source_dir = os.path.dirname(source_dir) or "."
    elif not os.path.isdir(source_dir):
        print(f"Ошибка: Входной путь {source_dir} не найден.")
        return
    else:
        files = [f for f in os.listdir(source_dir) if f.lower().endswith(extensions)]
        files = [f for f in files if not f.endswith(("_orig.mp4", "_orig.mkv", "_orig.mov", "_orig.avi", "_orig.ts"))]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана выходная директория: {output_dir}")

    if not files:
        print(f"Видео файлы в {source_dir} не найдены.")
        return

    every = max(1, int(every))
    auto_videotoolbox = is_apple_silicon() and not use_nvenc
    jobs = max(1, jobs)
    jobs = min(jobs, max(1, len(files)))

    print(f"Входная папка: {os.path.abspath(source_dir)}")
    print(f"Выходная папка: {os.path.abspath(output_dir)}")
    print(f"Найдено файлов: {len(files)}")
    print(f"Параллельность: {jobs}")
    if every > 1:
        print(f"Кадры: каждый {every}-й (длительность исходника)")
    print("Остановка: Ctrl+C (повторно — сразу убить ffmpeg)")

    if use_nvenc:
        vcodec = "h264_nvenc" if codec == "libx264" else "hevc_nvenc"
        print(f"Кодирование: GPU NVENC ({vcodec}), CQ={nvenc_cq}")
    elif auto_videotoolbox:
        vcodec = "h264_videotoolbox" if codec == "libx264" else "hevc_videotoolbox"
        print(f"Кодирование: Apple VideoToolbox ({vcodec}), prio_speed, jobs={jobs}")
    else:
        print(f"Кодирование: CPU ({codec})")

    prev_int = signal.signal(signal.SIGINT, request_stop)
    prev_term = signal.signal(signal.SIGTERM, request_stop)

    ok = 0
    stopped = 0
    failed = 0
    try:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = [
                pool.submit(
                    transcode_one,
                    filename,
                    source_dir,
                    output_dir,
                    codec,
                    use_nvenc=use_nvenc,
                    nvenc_cq=nvenc_cq,
                    auto_videotoolbox=auto_videotoolbox,
                    every=every,
                    index=i + 1,
                    total=len(files),
                )
                for i, filename in enumerate(files)
            ]
            for fut in as_completed(futs):
                filename, success, detail = fut.result()
                if success:
                    ok += 1
                    _progress_finish(filename, f"Успешно: {filename} -> {detail}")
                elif detail == "остановлено":
                    stopped += 1
                    _progress_finish(filename, f"Остановлено: {filename}")
                else:
                    failed += 1
                    _progress_finish(filename, f"Ошибка {filename}: {detail}")
    finally:
        _progress_reset()
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)

    if _stop.is_set():
        print(f"\nПрервано: готово {ok}/{len(files)}, остановлено {stopped}")
    else:
        print(f"\nГотово: {ok}/{len(files)}" + (f", ошибок {failed}" if failed else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт для перекодировки видео с перемещением результата.")
    parser.add_argument(
        "--path",
        type=str,
        default=".",
        help='Папка с исходными видео (по умолчанию текущая ".").',
    )
    parser.add_argument(
        "--out",
        type=str,
        default="..",
        help='Папка для сохранения результатов (по умолчанию родительская "..").',
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="libx264",
        choices=["libx264", "libx265"],
        help="Кодек: libx265 или libx264 (для веб-плеера обычно libx264).",
    )
    parser.add_argument(
        "--nvenc",
        action="store_true",
        help="Кодировать на NVIDIA (h264_nvenc / hevc_nvenc). Нужны драйвер и ffmpeg с NVENC.",
    )
    parser.add_argument(
        "--nvenc-cq",
        type=int,
        default=26,
        metavar="N",
        help="NVENC качество (аналог «силы» CRF): 18–28, выше — меньше файл, ниже — лучше картинка. По умолчанию 26.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        metavar="N",
        help="Брать каждый N-й кадр (1 = все). Длительность ролика та же, fps ниже. Ускоряет кодирование.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Сколько файлов кодировать сразу. По умолчанию 1.",
    )

    args = parser.parse_args()
    transcode_videos(
        source_dir=args.path,
        output_dir=args.out,
        codec=args.codec,
        use_nvenc=args.nvenc,
        nvenc_cq=args.nvenc_cq,
        jobs=args.jobs,
        every=args.every,
    )

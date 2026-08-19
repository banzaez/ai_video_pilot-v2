import argparse
import os
import platform
import signal
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_stop = threading.Event()
_procs_lock = threading.Lock()
_procs: set[subprocess.Popen] = set()


class Stopped(Exception):
    """Остановка по Ctrl+C / SIGTERM."""


def request_stop(_signum=None, _frame=None) -> None:
    if _stop.is_set():
        with _procs_lock:
            for proc in list(_procs):
                _kill_proc(proc, force=True)
        return
    _stop.set()
    print("\nОстановка (Ctrl+C). Жду завершения ffmpeg…")
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


def _run_ffmpeg(command: list[str]) -> None:
    if _stop.is_set():
        raise Stopped()
    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    with _procs_lock:
        _procs.add(proc)
    try:
        code = proc.wait()
    finally:
        with _procs_lock:
            _procs.discard(proc)
    if _stop.is_set():
        raise Stopped()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def _restore_inplace(inplace: bool, source_path: str, input_file: str, output_path: str) -> None:
    if output_path != input_file and os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError:
            pass
    if inplace and input_file != source_path and os.path.exists(input_file) and not os.path.exists(source_path):
        try:
            os.rename(input_file, source_path)
        except OSError:
            pass


def _vt_commands(input_file: str, output_path: str, codec: str) -> list[list[str]]:
    """Сначала zero-copy GPU, затем decode CPU + encode VT (если формат не в VideoToolbox)."""
    video_args = _videotoolbox_args(codec)
    tail = [*video_args, "-c:a", "copy", "-max_muxing_queue_size", "2048", "-y", output_path]
    return [
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-hwaccel",
            "videotoolbox",
            "-hwaccel_output_format",
            "videotoolbox_vld",
            "-i",
            input_file,
            *tail,
        ],
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            input_file,
            *tail,
        ],
    ]


def transcode_one(
    filename: str,
    source_dir: str,
    output_dir: str,
    codec: str,
    *,
    use_nvenc: bool,
    nvenc_cq: int,
    auto_videotoolbox: bool,
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

    if use_nvenc:
        commands = [
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-i",
                input_file,
                *_nvenc_video_args(codec, nvenc_cq),
                "-c:a",
                "copy",
                "-y",
                output_path,
            ]
        ]
    elif auto_videotoolbox:
        commands = _vt_commands(input_file, output_path, codec)
    else:
        commands = [
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-i",
                input_file,
                *_cpu_video_args(codec),
                "-c:a",
                "copy",
                "-y",
                output_path,
            ]
        ]

    last_err = ""
    try:
        if _stop.is_set():
            raise Stopped()
        for i, command in enumerate(commands):
            try:
                _run_ffmpeg(command)
                return filename, True, output_path
            except subprocess.CalledProcessError as e:
                last_err = str(e)
                if os.path.exists(output_path) and output_path != input_file:
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
                if i + 1 < len(commands) and not _stop.is_set():
                    print(f"  {filename}: VideoToolbox zero-copy не подошёл, повтор без hwaccel decode")

        if auto_videotoolbox and not _stop.is_set():
            try:
                _run_ffmpeg(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-nostdin",
                        "-i",
                        input_file,
                        *_cpu_video_args(codec, fast=True),
                        "-c:a",
                        "copy",
                        "-y",
                        output_path,
                    ]
                )
                print(f"  {filename}: fallback CPU veryfast")
                return filename, True, output_path
            except subprocess.CalledProcessError as e:
                last_err = str(e)
    except Stopped:
        _restore_inplace(inplace, source_path, input_file, output_path)
        return filename, False, "остановлено"

    _restore_inplace(inplace, source_path, input_file, output_path)
    return filename, False, last_err or "ffmpeg error"


def transcode_videos(
    source_dir,
    output_dir,
    codec="libx265",
    *,
    use_nvenc=False,
    nvenc_cq=26,
    jobs=0,
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

    auto_videotoolbox = is_apple_silicon() and not use_nvenc
    if jobs <= 0:
        jobs = 2 if auto_videotoolbox else 1
    jobs = max(1, min(jobs, len(files)))

    print(f"Входная папка: {os.path.abspath(source_dir)}")
    print(f"Выходная папка: {os.path.abspath(output_dir)}")
    print(f"Найдено файлов: {len(files)}")
    print(f"Параллельность: {jobs}")
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
                )
                for filename in files
            ]
            for fut in as_completed(futs):
                filename, success, detail = fut.result()
                if success:
                    ok += 1
                    print(f"Успешно: {filename} -> {detail}")
                elif detail == "остановлено":
                    stopped += 1
                    print(f"Остановлено: {filename}")
                else:
                    failed += 1
                    print(f"Ошибка {filename}: {detail}")
    finally:
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
        "--jobs",
        type=int,
        default=0,
        metavar="N",
        help="Сколько файлов кодировать сразу. 0 = авто (2 на Apple Silicon, 1 иначе).",
    )

    args = parser.parse_args()
    transcode_videos(
        source_dir=args.path,
        output_dir=args.out,
        codec=args.codec,
        use_nvenc=args.nvenc,
        nvenc_cq=args.nvenc_cq,
        jobs=args.jobs,
    )

import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import type { TrackingData } from "../types";
import {
  partAtTime,
  partDurationSec,
  partIndexOf,
  sessionDurationSec,
  type SessionPart,
} from "../session";
import {
  buildActivityCurve,
  buildTrackKeyframes,
  colorForTrackId,
  detectionsAtFrame,
  formatDuration,
  frameAtTime,
  frameIndexAtTime,
  parseTimecode,
  resolveDetectEveryN,
  type MergeTimeline,
} from "../utils";

type Props = {
  videoUrl: string | null;
  tracking: TrackingData | null;
  sessionParts?: SessionPart[] | null;
  showLabels: boolean;
  showTrails: boolean;
  trailLength: number;
  onFrameChange?: (frame: number) => void;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  highlightIds?: number[];
  /** track_id → group_id (merge) */
  groupByTrack?: Record<number, number>;
  mergeTimeline?: MergeTimeline | null;
  /**
   * На вкладке Склейки: треки из фильтра — обычные цвета;
   * остальные рисуются серым. null = без приглушения.
   */
  focusTrackIds?: number[] | null;
  /** Если true и задан focusTrackIds, не рисовать невыбранные треки вовсе */
  hideUnfocused?: boolean;
  compact?: boolean;
};

/** Seek по глобальному времени сессии (с учётом кусков). */
export type TrackingPlayerHandle = {
  seekToGlobal: (tSec: number, playAfter?: boolean) => void;
};

type JogStripProps = {
  globalSec: number;
  durationSec: number;
  fps: number;
  onSeek: (sec: number) => void;
};

function JogStripRuler({ globalSec, durationSec, fps, onSeek }: JogStripProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDragging = useRef(false);
  const startX = useRef(0);
  const startSec = useRef(0);
  const lastMoveTime = useRef(0);
  const lastMoveX = useRef(0);
  const velocity = useRef(0);
  const animRaf = useRef<number | null>(null);

  const [jogScale, setJogScale] = useState<"1x" | "5x" | "30x">("5x");

  const onSeekRef = useRef(onSeek);
  onSeekRef.current = onSeek;

  // Базовый масштаб: 1x = 60px/s (кадры), 5x = 14px/s (секунды), 30x = 2.5px/s (минуты)
  const pxPerSec = jogScale === "1x" ? 60 : jogScale === "5x" ? 14 : 2.5;
  const frameStep = 1 / Math.max(fps, 1);

  const [containerWidth, setContainerWidth] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const updateSize = () => {
      const w = Math.round(canvas.getBoundingClientRect().width || canvas.parentElement?.clientWidth || 300);
      if (w > 0) setContainerWidth(w);
    };
    updateSize();
    const ro = new ResizeObserver(updateSize);
    ro.observe(canvas);
    window.addEventListener("resize", updateSize);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", updateSize);
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(rect.width) || containerWidth || Math.round(canvas.parentElement?.clientWidth || 300);
    const h = 24;

    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // Фон шкалы
    ctx.fillStyle = "#0c1410";
    ctx.fillRect(0, 0, w, h);

    const cx = w / 2;
    const minSec = Math.max(0, globalSec - cx / pxPerSec);
    const maxSec = Math.min(durationSec || globalSec + 60, globalSec + cx / pxPerSec);

    // Интервал подписей (чтобы между текстами всегда было >= 60px)
    const stepInterval = jogScale === "30x" ? 30 : jogScale === "5x" ? 5 : 1;
    const startIntSec = Math.floor(minSec / stepInterval) * stepInterval;
    const endIntSec = Math.ceil(maxSec / stepInterval) * stepInterval;

    // Рисуем деления и таймкоды
    ctx.font = '8.5px "IBM Plex Mono", monospace';
    ctx.textAlign = "center";
    ctx.textBaseline = "top";

    for (let s = startIntSec; s <= endIntSec; s += stepInterval) {
      const x = cx + (s - globalSec) * pxPerSec;

      // Главная секундная засечка
      ctx.strokeStyle = "#4da65e";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, 8);
      ctx.stroke();

      // Подпись времени
      ctx.fillStyle = "#8a9a90";
      ctx.fillText(formatDuration(s), x, 10);

      // Промежуточные деления
      const subSteps = jogScale === "30x" ? 6 : jogScale === "5x" ? 5 : 5;
      for (let sub = 1; sub < subSteps; sub++) {
        const subX = x + (sub * (stepInterval / subSteps)) * pxPerSec;
        if (subX >= 0 && subX <= w) {
          ctx.strokeStyle = "rgba(255,255,255,0.15)";
          ctx.beginPath();
          ctx.moveTo(subX, 0);
          ctx.lineTo(subX, sub === 3 && jogScale === "30x" ? 5 : 3);
          ctx.stroke();
        }
      }
    }

    // Центральный курсор (красный треугольник + вертикальная линия)
    ctx.strokeStyle = "#ff4d4d";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, h);
    ctx.stroke();

    ctx.fillStyle = "#ff4d4d";
    ctx.beginPath();
    ctx.moveTo(cx - 3, 0);
    ctx.lineTo(cx + 3, 0);
    ctx.lineTo(cx, 4);
    ctx.closePath();
    ctx.fill();
  }, [globalSec, durationSec, fps, pxPerSec, jogScale, containerWidth]);

  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    if (animRaf.current != null) cancelAnimationFrame(animRaf.current);
    e.currentTarget.setPointerCapture(e.pointerId);
    isDragging.current = true;
    startX.current = e.clientX;
    startSec.current = globalSec;
    lastMoveX.current = e.clientX;
    lastMoveTime.current = performance.now();
    velocity.current = 0;
  }

  const pendingSeek = useRef<number | null>(null);
  const rafSeek = useRef<number | null>(null);

  function triggerSeek(sec: number) {
    pendingSeek.current = sec;
    if (rafSeek.current == null) {
      rafSeek.current = requestAnimationFrame(() => {
        rafSeek.current = null;
        if (pendingSeek.current != null) {
          onSeekRef.current(pendingSeek.current);
        }
      });
    }
  }

  function onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!isDragging.current) return;
    const now = performance.now();
    const dt = Math.max(1, now - lastMoveTime.current);
    const dxSinceLast = e.clientX - lastMoveX.current;
    velocity.current = dxSinceLast / dt; // px/ms
    lastMoveX.current = e.clientX;
    lastMoveTime.current = now;

    // Ускорители Shift / Alt
    const mult = e.shiftKey ? 5 : e.altKey ? 25 : 1;
    const totalDx = (e.clientX - startX.current) * mult;
    const dSec = -totalDx / pxPerSec;
    const next = Math.max(0, Math.min(durationSec || (startSec.current + dSec), startSec.current + dSec));
    triggerSeek(next);
  }

  function onPointerUp(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!isDragging.current) return;
    isDragging.current = false;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {}

    if (pendingSeek.current != null) {
      onSeekRef.current(pendingSeek.current);
    }

    // Инерционное затухание при резком смахивании
    if (Math.abs(velocity.current) > 0.25) {
      let currentVel = velocity.current * 18; // px per frame
      let curSec = globalSec;

      const coast = () => {
        if (Math.abs(currentVel) < 0.15 || isDragging.current) return;
        curSec = Math.max(0, Math.min(durationSec || (curSec - currentVel / pxPerSec), curSec - currentVel / pxPerSec));
        triggerSeek(curSec);
        currentVel *= 0.88; // трение/затухание
        animRaf.current = requestAnimationFrame(coast);
      };
      animRaf.current = requestAnimationFrame(coast);
    }
  }

  function onWheel(e: React.WheelEvent<HTMLCanvasElement>) {
    e.preventDefault();
    if (animRaf.current != null) cancelAnimationFrame(animRaf.current);
    const delta = e.deltaY || e.deltaX;
    if (Math.abs(delta) > 0) {
      const mult = e.shiftKey ? 10 : e.altKey ? 60 : jogScale === "30x" ? 10 : jogScale === "5x" ? 2 : 1;
      const step = frameStep * mult;
      const next = Math.max(0, Math.min(durationSec || globalSec, globalSec + (delta > 0 ? step : -step)));
      triggerSeek(next);
    }
  }

  return (
    <div className="session-jog-wrap">
      <canvas
        ref={canvasRef}
        className="session-jog-canvas"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onWheel={onWheel}
        title="Лента перемотки: тяните мышью или крутите колесико (Shift = 5x, Alt = 25x)"
      />
      <div className="session-jog-modes">
        <button
          type="button"
          className={`session-jog-mode-btn${jogScale === "1x" ? " is-active" : ""}`}
          onClick={() => setJogScale("1x")}
          title="Покадровая точность (1x)"
        >
          Кадры
        </button>
        <button
          type="button"
          className={`session-jog-mode-btn${jogScale === "5x" ? " is-active" : ""}`}
          onClick={() => setJogScale("5x")}
          title="Быстрая перемотка секундами (5x)"
        >
          Сек (5x)
        </button>
        <button
          type="button"
          className={`session-jog-mode-btn${jogScale === "30x" ? " is-active" : ""}`}
          onClick={() => setJogScale("30x")}
          title="Сверхбыстрая перемотка минутами (30x)"
        >
          Мин (30x)
        </button>
      </div>
    </div>
  );
}

export const TrackingPlayer = forwardRef<TrackingPlayerHandle, Props>(function TrackingPlayer(
  {
    videoUrl,
    tracking,
    sessionParts = null,
    showLabels,
    showTrails,
    trailLength,
    onFrameChange,
    videoRef: externalVideoRef,
    highlightIds = [],
    groupByTrack = {},
    mergeTimeline = null,
    focusTrackIds = null,
    hideUnfocused = false,
    compact = false,
  },
  ref,
) {
  const internalVideoRef = useRef<HTMLVideoElement>(null);
  const videoRef = externalVideoRef || internalVideoRef;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [persons, setPersons] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [globalSec, setGlobalSec] = useState(0);
  const [activePartIdx, setActivePartIdx] = useState(0);
  const scrubbingRef = useRef(false);
  const hideUnfocusedRef = useRef(hideUnfocused);
  hideUnfocusedRef.current = hideUnfocused;

  const isSession = !!(sessionParts && sessionParts.length > 0);
  const fps = tracking?.fps ?? 25;

  const keyframes = useMemo(
    () => (tracking ? buildTrackKeyframes(tracking) : null),
    [tracking],
  );
  const detectEveryN = useMemo(
    () => (tracking ? resolveDetectEveryN(tracking) : 1),
    [tracking],
  );

  const durationSec = useMemo(() => {
    if (isSession && sessionParts) {
      const fromParts = sessionDurationSec(sessionParts, fps);
      if (fromParts > 0) return fromParts;
    }
    if (tracking?.frame_count && tracking.fps) {
      return tracking.frame_count / tracking.fps;
    }
    return 0;
  }, [isSession, sessionParts, fps, tracking]);

  const partMarks = useMemo(() => {
    if (!isSession || !sessionParts || durationSec <= 0) return [];
    return sessionParts.map((p, i) => ({
      index: i,
      pct: (p.time_offset_sec / durationSec) * 100,
      label: String(i + 1),
    }));
  }, [isSession, sessionParts, durationSec]);

  const trailHistory = useRef<Map<number, Array<[number, number]>>>(new Map());
  const highlightRef = useRef(new Set(highlightIds));
  highlightRef.current = new Set(highlightIds);
  const groupByTrackRef = useRef(groupByTrack);
  groupByTrackRef.current = groupByTrack;
  const mergeTimelineRef = useRef(mergeTimeline);
  mergeTimelineRef.current = mergeTimeline;
  const focusTrackIdsRef = useRef<Set<number> | null>(null);
  focusTrackIdsRef.current = focusTrackIds != null ? new Set(focusTrackIds) : null;
  const onFrameChangeRef = useRef(onFrameChange);
  onFrameChangeRef.current = onFrameChange;
  const lastNotifiedFrame = useRef<number | null>(null);

  useEffect(() => {
    if (lastNotifiedFrame.current === currentFrame) return;
    lastNotifiedFrame.current = currentFrame;
    onFrameChangeRef.current?.(currentFrame);
  }, [currentFrame]);

  const activePartRef = useRef<SessionPart | null>(null);

  const seekToGlobal = useCallback(
    (tSec: number, playAfter = false) => {
      const video = videoRef.current;
      if (!video) return;
      const t = Math.max(0, tSec);
      if (!isSession || !sessionParts?.length) {
        setGlobalSec(t);
        video.currentTime = t;
        if (playAfter) video.play().catch(() => {});
        return;
      }
      const hit = partAtTime(sessionParts, t, fps);
      if (!hit) return;
      const wasPlaying = playAfter || !video.paused;
      const same = activePartRef.current?.name === hit.part.name;
      activePartRef.current = hit.part;
      setActivePartIdx(partIndexOf(sessionParts, hit.part));
      setGlobalSec(Math.max(0, Math.min(t, durationSec || t)));
      if (!same) {
        video.src = hit.part.videoUrl;
      }
      const apply = () => {
        if (typeof (video as HTMLVideoElement & { fastSeek?: (t: number) => void }).fastSeek === "function") {
          (video as HTMLVideoElement & { fastSeek: (t: number) => void }).fastSeek(hit.localTime);
        } else {
          video.currentTime = hit.localTime;
        }
        if (wasPlaying) video.play().catch(() => {});
      };
      if (!same) {
        const onReady = () => {
          video.removeEventListener("loadeddata", onReady);
          apply();
        };
        video.addEventListener("loadeddata", onReady);
        video.load();
      } else {
        apply();
      }
    },
    [videoRef, isSession, sessionParts, fps, durationSec],
  );

  useImperativeHandle(ref, () => ({ seekToGlobal }), [seekToGlobal]);

  const scrubRafRef = useRef<number | null>(null);
  const frameStep = 1 / Math.max(fps, 1);

  function nudge(deltaSec: number) {
    const next = Math.max(0, Math.min(durationSec || globalSec, globalSec + deltaSec));
    scrubbingRef.current = false;
    seekToGlobal(next, false);
  }

  function onScrubInput(value: number) {
    scrubbingRef.current = true;
    setGlobalSec(value);
    if (scrubRafRef.current != null) cancelAnimationFrame(scrubRafRef.current);
    scrubRafRef.current = requestAnimationFrame(() => {
      scrubRafRef.current = null;
      seekToGlobal(value, false);
    });
  }

  function onScrubCommit(value: number) {
    scrubbingRef.current = false;
    if (scrubRafRef.current != null) {
      cancelAnimationFrame(scrubRafRef.current);
      scrubRafRef.current = null;
    }
    seekToGlobal(value, playing);
  }

  const lastSourceKey = useRef<string | null>(null);
  const currentSourceKey = videoUrl || sessionParts?.map((p) => p.name).join(",") || null;

  useEffect(() => {
    trailHistory.current.clear();
    if (lastSourceKey.current !== currentSourceKey) {
      lastSourceKey.current = currentSourceKey;
      if (sessionParts?.length) {
        activePartRef.current = sessionParts[0] ?? null;
        setActivePartIdx(0);
        setGlobalSec(0);
      } else {
        activePartRef.current = null;
        setActivePartIdx(0);
        setGlobalSec(0);
      }
    }
  }, [currentSourceKey, sessionParts]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !sessionParts?.length) return;

    const syncUi = () => {
      if (scrubbingRef.current) return;
      const part = activePartRef.current ?? sessionParts[0]!;
      const g = part.time_offset_sec + video.currentTime;
      setGlobalSec(g);
      setActivePartIdx(partIndexOf(sessionParts, part));
    };

    const onTimeUpdate = () => {
      const part = activePartRef.current;
      if (!part) return;
      const g = part.time_offset_sec + video.currentTime;
      const hit = partAtTime(sessionParts, g, fps);
      if (!hit || hit.part.name === part.name) {
        syncUi();
        return;
      }
      const wasPlaying = !video.paused;
      activePartRef.current = hit.part;
      setActivePartIdx(partIndexOf(sessionParts, hit.part));
      video.src = hit.part.videoUrl;
      const onReady = () => {
        video.removeEventListener("loadeddata", onReady);
        video.currentTime = hit.localTime;
        if (wasPlaying) video.play().catch(() => {});
      };
      video.addEventListener("loadeddata", onReady);
      video.load();
    };

    const onEnded = () => {
      const part = activePartRef.current;
      if (!part) return;
      const idx = sessionParts.findIndex((p) => p.name === part.name);
      const next = idx >= 0 ? sessionParts[idx + 1] : null;
      if (!next) {
        setPlaying(false);
        syncUi();
        return;
      }
      activePartRef.current = next;
      setActivePartIdx(idx + 1);
      video.src = next.videoUrl;
      const onReady = () => {
        video.removeEventListener("loadeddata", onReady);
        video.currentTime = 0;
        video.play().catch(() => {});
      };
      video.addEventListener("loadeddata", onReady);
      video.load();
    };

    video.addEventListener("timeupdate", onTimeUpdate);
    video.addEventListener("ended", onEnded);
    video.addEventListener("seeked", syncUi);
    return () => {
      video.removeEventListener("timeupdate", onTimeUpdate);
      video.removeEventListener("ended", onEnded);
      video.removeEventListener("seeked", syncUi);
    };
  }, [sessionParts, fps, videoRef]);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    let raf = 0;
    let lastUi = 0;
    const MAX_SIDE = 1280;

    const draw = () => {
      const ctx = canvas.getContext("2d");
      const stage = stageRef.current;
      if (!ctx) return;

      const vw = video.videoWidth || tracking?.width || 0;
      const vh = video.videoHeight || tracking?.height || 0;
      if (!vw || !vh) return;

      const fit = Math.min(1, MAX_SIDE / Math.max(vw, vh));
      const cw = Math.max(1, Math.round(vw * fit));
      const ch = Math.max(1, Math.round(vh * fit));
      if (canvas.width !== cw || canvas.height !== ch) {
        canvas.width = cw;
        canvas.height = ch;
      }

      // object-fit: contain → чёрные поля; canvas должен совпасть с картинкой, не со stage
      if (stage) {
        const videoRect = video.getBoundingClientRect();
        const stageRect = stage.getBoundingClientRect();
        if (videoRect.width > 0 && videoRect.height > 0) {
          const scale = Math.min(videoRect.width / vw, videoRect.height / vh);
          const dw = vw * scale;
          const dh = vh * scale;
          canvas.style.left = `${videoRect.left - stageRect.left + (videoRect.width - dw) / 2}px`;
          canvas.style.top = `${videoRect.top - stageRect.top + (videoRect.height - dh) / 2}px`;
          canvas.style.width = `${dw}px`;
          canvas.style.height = `${dh}px`;
        }
      }

      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, cw, ch);
      ctx.setTransform(cw / vw, 0, 0, ch / vh, 0, 0);

      if (!tracking || !keyframes) return;

      const gSec = sessionParts?.length
        ? (activePartRef.current?.time_offset_sec ?? 0) + video.currentTime
        : video.currentTime;
      const frameFloat = frameAtTime(gSec, tracking.fps, tracking.frame_count);
      const idx = frameIndexAtTime(gSec, tracking.fps, tracking.frame_count);
      const focus = focusTrackIdsRef.current;
      const detections = detectionsAtFrame(keyframes, frameFloat, detectEveryN);
      // Сначала серые (вне фильтра), поверх — треки фильтра
      const ordered =
        focus == null
          ? detections
          : [
              ...detections.filter((d) => !focus.has(d.track_id)),
              ...detections.filter((d) => focus.has(d.track_id)),
            ];
      const count = focus == null ? detections.length : detections.filter((d) => focus.has(d.track_id)).length;

      const now = performance.now();
      const pushUi = video.paused || video.ended || now - lastUi >= 125;
      if (pushUi) {
        lastUi = now;
        setCurrentFrame((prev) => (prev === idx ? prev : idx));
        setPersons((prev) => (prev === count ? prev : count));
      }

      const activeIds = new Set<number>();
      const DIM = "#6b736c";
      const curSec = video?.currentTime ?? (fps > 0 ? currentFrame / fps : 0);

      for (const det of ordered) {
        activeIds.add(det.track_id);
        const [x1, y1, x2, y2] = det.bbox.map((v) => Math.round(v)) as [
          number,
          number,
          number,
          number,
        ];

        // Resolve exact fragment track_id and group_id at current time
        let fragTrackId = det.track_id;
        let gid = groupByTrackRef.current[det.track_id];
        const mt = mergeTimelineRef.current;
        if (mt && mt.tracks.length) {
          const match = mt.tracks.find(
            (tr: { global_id?: number | null; track_id: number; t0: number; t1: number; group_id?: number | null }) =>
              (tr.global_id === det.track_id || tr.track_id === det.track_id) &&
              curSec >= tr.t0 - 0.5 &&
              curSec <= tr.t1 + 0.5,
          );
          if (match) {
            fragTrackId = match.track_id;
            if (match.group_id != null) gid = match.group_id;
          }
        }

        const dimmed = focus != null && !focus.has(det.track_id) && !focus.has(fragTrackId);
        if (dimmed && hideUnfocusedRef.current) continue;
        const highlighted = !dimmed && (highlightRef.current.has(det.track_id) || highlightRef.current.has(fragTrackId));
        const color = dimmed ? DIM : colorForTrackId(fragTrackId);
        const bc: [number, number] = [Math.round((x1 + x2) / 2), y2];

        let hist = trailHistory.current.get(det.track_id);
        if (!hist) {
          hist = [];
          trailHistory.current.set(det.track_id, hist);
        }
        const last = hist[hist.length - 1];
        if (!last || last[0] !== bc[0] || last[1] !== bc[1]) {
          hist.push([bc[0], bc[1]]);
          if (hist.length > trailLength) hist.shift();
        }

        ctx.globalAlpha = dimmed ? 0.78 : 1;
        if (highlighted) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = Math.max(5, Math.round(vw / 240));
          ctx.strokeRect(x1 - 1, y1 - 1, x2 - x1 + 2, y2 - y1 + 2);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(
          dimmed ? 2 : highlighted ? 3.5 : 2,
          Math.round(vw / (highlighted ? 300 : dimmed ? 520 : 480)),
        );
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        if (showLabels) {
          const label =
            typeof gid === "number"
              ? `t${fragTrackId} g${gid} ${det.confidence.toFixed(2)}`
              : `t${fragTrackId} ${det.confidence.toFixed(2)}`;
          ctx.font = `600 ${Math.max(12, Math.round(vw / 55))}px "IBM Plex Mono", monospace`;
          const pad = 4;
          const metrics = ctx.measureText(label);
          const th = Math.max(14, Math.round(vw / 50));
          const ty = Math.max(y1 - 6, th + 2);
          if (highlighted) {
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(x1 - 1, ty - th - 1, metrics.width + pad * 2 + 2, th + 4);
          }
          ctx.fillStyle = color;
          ctx.fillRect(x1, ty - th, metrics.width + pad * 2, th + 2);
          ctx.fillStyle = "#fff";
          ctx.fillText(label, x1 + pad, ty - 4);
        }

        if (showTrails && hist.length > 1) {
          ctx.beginPath();
          ctx.strokeStyle = color;
          ctx.lineWidth = Math.max(dimmed ? 1.5 : 2, Math.round(vw / 640));
          ctx.moveTo(hist[0][0], hist[0][1]);
          for (let i = 1; i < hist.length; i++) {
            ctx.lineTo(hist[i][0], hist[i][1]);
          }
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }

      for (const id of [...trailHistory.current.keys()]) {
        if (!activeIds.has(id)) trailHistory.current.delete(id);
      }
    };

    const loop = () => {
      draw();
      if (!video.paused && !video.ended) {
        raf = requestAnimationFrame(loop);
      }
    };
    const kick = () => {
      cancelAnimationFrame(raf);
      draw();
      if (!video.paused && !video.ended) {
        raf = requestAnimationFrame(loop);
      }
    };

    video.addEventListener("play", kick);
    video.addEventListener("pause", kick);
    video.addEventListener("seeked", kick);
    video.addEventListener("loadeddata", kick);
    window.addEventListener("resize", kick);
    kick();
    return () => {
      cancelAnimationFrame(raf);
      video.removeEventListener("play", kick);
      video.removeEventListener("pause", kick);
      video.removeEventListener("seeked", kick);
      video.removeEventListener("loadeddata", kick);
      window.removeEventListener("resize", kick);
    };
  }, [tracking, keyframes, detectEveryN, showLabels, showTrails, trailLength, videoRef, focusTrackIds, sessionParts]);

  type ZoomMode = "all" | "1h" | "10m" | "1m";
  const [zoomMode, setZoomMode] = useState<ZoomMode>("all");
  const [playbackSpeed, setPlaybackSpeed] = useState(2);
  const [jumpText, setJumpText] = useState("");
  const [isEditingTime, setIsEditingTime] = useState(false);

  const SPEEDS = [0.5, 1, 1.5, 2, 4, 8];

  function cycleSpeed() {
    const next = SPEEDS[(SPEEDS.indexOf(playbackSpeed) + 1) % SPEEDS.length]!;
    setPlaybackSpeed(next);
    if (videoRef.current) videoRef.current.playbackRate = next;
  }

  function togglePlay() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.playbackRate = playbackSpeed;
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  }

  function handleTimeSubmit(e: React.FormEvent) {
    e.preventDefault();
    const sec = parseTimecode(jumpText);
    if (sec != null && Number.isFinite(sec)) {
      seekToGlobal(sec, playing);
    }
    setIsEditingTime(false);
  }

  function handleWheel(e: React.WheelEvent) {
    e.preventDefault();
    const delta = e.deltaY || e.deltaX;
    if (Math.abs(delta) > 0) {
      const step = e.shiftKey ? 1 : frameStep;
      nudge(delta > 0 ? step : -step);
    }
  }

  // График плотности треков (активности людей)
  const activityCurve = useMemo(() => {
    return buildActivityCurve(tracking, durationSec, 200);
  }, [tracking, durationSec]);

  if (!videoUrl) {
    return (
      <div className="player empty">
        <p>Выберите видео, чтобы начать просмотр</p>
      </div>
    );
  }

  const partCount = sessionParts?.length ?? 0;
  const scrubMax = Math.max(durationSec, 0.001);

  // Вычисление границ окна таймлайна при зуме
  const zoomSpan = zoomMode === "1h" ? 3600 : zoomMode === "10m" ? 600 : zoomMode === "1m" ? 60 : scrubMax;
  let zoomLo = 0;
  let zoomHi = scrubMax;
  if (zoomMode !== "all" && scrubMax > zoomSpan) {
    const half = zoomSpan / 2;
    zoomLo = Math.max(0, globalSec - half);
    zoomHi = Math.min(scrubMax, zoomLo + zoomSpan);
    zoomLo = Math.max(0, zoomHi - zoomSpan);
  }
  const zoomStep = zoomMode === "1m" ? frameStep : zoomMode === "10m" ? 0.04 : zoomMode === "1h" ? 0.2 : 0.5;
  const curPct = Math.min(
    100,
    Math.max(0, ((globalSec - zoomLo) / Math.max(zoomHi - zoomLo, 0.001)) * 100),
  );

  return (
    <div className={`player${isSession ? " is-session" : ""}`}>
      <div className="stage" ref={stageRef} onWheel={handleWheel} title="Колесико мыши: покадровая перемотка">
        <video
          ref={videoRef}
          src={videoUrl}
          playsInline
          preload="metadata"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onSeeking={() => {
            trailHistory.current.clear();
          }}
          onSeeked={() => {
            trailHistory.current.clear();
          }}
        />
        <canvas ref={canvasRef} className="overlay" />
      </div>

      {!compact && (
        <div className="session-scrub-stack">
          {/* 1. Верхняя панель: Выбор частей сессии и Зум таймлайна */}
          <div className="session-top-bar">
            <div className="session-top-left">
              {sessionParts && sessionParts.length > 1 ? (
                <div className="session-parts-pills">
                  <span className="session-bar-label">Части:</span>
                  {sessionParts.map((p, i) => {
                    const dur = partDurationSec(p, fps);
                    return (
                      <button
                        key={p.name}
                        type="button"
                        className={`session-part-pill${i === activePartIdx ? " is-active" : ""}`}
                        onClick={() => seekToGlobal(p.time_offset_sec, playing)}
                        title={`Часть ${i + 1}: ${formatDuration(p.time_offset_sec)}–${formatDuration(p.time_offset_sec + dur)} (${p.name})`}
                      >
                        <span className="session-part-pill-num">Ч.{i + 1}</span>
                        <span className="session-part-pill-sep">|</span>
                        <span className="session-part-pill-time">{formatDuration(p.time_offset_sec)}</span>
                      </button>
                    );
                  })}
                </div>
              ) : (
                <span className="session-bar-label mono">
                  {tracking ? `FPS: ${fps} | Кадров: ${tracking.frame_count ?? "—"}` : ""}
                </span>
              )}
            </div>

            <div className="session-top-right">
              {activityCurve.maxCount > 0 && (
                <span className="session-activity-badge" title="Максимальное скопление людей в кадре">
                  👥 пик {activityCurve.maxCount}
                </span>
              )}

              <div className="session-zoom-pills">
                <span className="session-bar-label">Зум:</span>
                <button
                  type="button"
                  className={`session-zoom-pill${zoomMode === "all" ? " is-active" : ""}`}
                  onClick={() => setZoomMode("all")}
                  title="Вся сессия"
                >
                  Всё
                </button>
                {scrubMax > 3600 && (
                  <button
                    type="button"
                    className={`session-zoom-pill${zoomMode === "1h" ? " is-active" : ""}`}
                    onClick={() => setZoomMode("1h")}
                    title="Окно 1 час"
                  >
                    1ч
                  </button>
                )}
                {scrubMax > 600 && (
                  <button
                    type="button"
                    className={`session-zoom-pill${zoomMode === "10m" ? " is-active" : ""}`}
                    onClick={() => setZoomMode("10m")}
                    title="Окно 10 минут"
                  >
                    10м
                  </button>
                )}
                <button
                  type="button"
                  className={`session-zoom-pill${zoomMode === "1m" ? " is-active" : ""}`}
                  onClick={() => setZoomMode("1m")}
                  title="Окно 1 минута"
                >
                  1м
                </button>
              </div>
            </div>
          </div>

          {/* 2. Главный таймлайн сессии со встроенной волной активности */}
          <div className="session-scrub-main" onWheel={handleWheel}>
            <div className="session-scrub-track">
              {/* Волна плотности треков (Heatmap Curve) прямо внутри таймлайна */}
              {activityCurve.svgPath && (
                <svg
                  className="session-activity-svg"
                  viewBox="0 0 100 24"
                  preserveAspectRatio="none"
                  aria-hidden
                >
                  <defs>
                    <linearGradient id="activityGradUnified" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#8ce698" stopOpacity="0.8" />
                      <stop offset="50%" stopColor="#4da65e" stopOpacity="0.35" />
                      <stop offset="100%" stopColor="#1e382b" stopOpacity="0.05" />
                    </linearGradient>
                  </defs>
                  <path d={activityCurve.svgPath} fill="url(#activityGradUnified)" />
                  <path d={activityCurve.strokePath} fill="none" stroke="#8ce698" strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
                </svg>
              )}

              {/* Красный визир в стиле ленты точной перемотки (треугольник + линия) */}
              <div className="session-scrub-cursor" style={{ left: `${curPct}%` }} aria-hidden>
                <div className="session-scrub-cursor-head" />
                <div className="session-scrub-cursor-line" />
              </div>

              <input
                type="range"
                min={zoomLo}
                max={Math.max(zoomHi, zoomLo + zoomStep)}
                step={zoomStep}
                value={Math.min(Math.max(globalSec, zoomLo), zoomHi)}
                onChange={(e) => onScrubInput(Number(e.target.value))}
                onPointerUp={(e) => onScrubCommit(Number((e.target as HTMLInputElement).value))}
                onKeyUp={(e) => onScrubCommit(Number((e.target as HTMLInputElement).value))}
                aria-label="Полоса времени"
              />
              {zoomMode === "all" && (
                <div className="session-scrub-marks" aria-hidden>
                  {partMarks.slice(1).map((m) => (
                    <span key={m.index} className="session-scrub-mark" style={{ left: `${m.pct}%` }} />
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* 3. Кинетическая покадровая шкала-лента (Jog-Strip Ruler) */}
          <JogStripRuler
            globalSec={globalSec}
            durationSec={durationSec}
            fps={fps}
            onSeek={(s) => seekToGlobal(s, playing)}
          />

          {/* 4. Нижняя строка контролов: Play, Speed, Nudge, Timecode */}
          <div className="session-scrub-controls">
            <div className="session-ctrl-left">
              <button type="button" className="session-play" onClick={togglePlay} title={playing ? "Пауза (Пробел)" : "Пуск (Пробел)"}>
                {playing ? "❚❚" : "▶"}
              </button>

              <button
                type="button"
                className="session-speed-btn"
                onClick={cycleSpeed}
                title="Скорость воспроизведения"
              >
                {playbackSpeed}x
              </button>

              <div className="session-nudge">
                {scrubMax > 600 && (
                  <button type="button" title="−5 минут" onClick={() => nudge(-300)}>
                    −5m
                  </button>
                )}
                <button type="button" title="−1 минута" onClick={() => nudge(-60)}>
                  −1m
                </button>
                <button type="button" title="−10 секунд" onClick={() => nudge(-10)}>
                  −10s
                </button>
                <button type="button" title="−1 кадр (Колесико вниз)" onClick={() => nudge(-frameStep)}>
                  −1f
                </button>
                <button type="button" title="+1 кадр (Колесико вверх)" onClick={() => nudge(frameStep)}>
                  +1f
                </button>
                <button type="button" title="+10 секунд" onClick={() => nudge(10)}>
                  +10s
                </button>
                <button type="button" title="+1 минута" onClick={() => nudge(60)}>
                  +1m
                </button>
                {scrubMax > 600 && (
                  <button type="button" title="+5 минут" onClick={() => nudge(300)}>
                    +5m
                  </button>
                )}
              </div>
            </div>

            <div className="session-ctrl-right">
              {zoomMode !== "all" && (
                <span className="session-zoom-range mono">
                  [{formatDuration(zoomLo)}–{formatDuration(zoomHi)}]
                </span>
              )}

              {isEditingTime ? (
                <form onSubmit={handleTimeSubmit} className="session-time-edit">
                  <input
                    type="text"
                    autoFocus
                    className="session-time-input mono"
                    value={jumpText}
                    placeholder="ЧЧ:ММ:СС"
                    onChange={(e) => setJumpText(e.target.value)}
                    onBlur={() => setIsEditingTime(false)}
                  />
                </form>
              ) : (
                <span
                  className="session-scrub-time mono session-time-clickable"
                  title="Нажмите, чтобы ввести время"
                  onClick={() => {
                    setJumpText(formatDuration(globalSec));
                    setIsEditingTime(true);
                  }}
                >
                  {formatDuration(globalSec)} / {formatDuration(durationSec)}
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {!compact && (
        <div className="hud">
          <span className="hud-state">{playing ? `PLAY (${playbackSpeed}x)` : "PAUSE"}</span>
          {isSession && partCount > 0 && (
            <span className="hud-part">
              часть {activePartIdx + 1}/{partCount}
            </span>
          )}
          <span>
            кадр {currentFrame}
            {tracking ? ` / ${tracking.frame_count}` : ""}
            <em className="hud-sub"> (0-based {Math.max(0, currentFrame - 1)})</em>
          </span>
          <span>человек: {persons}</span>
          {tracking && <span>{tracking.fps} fps</span>}
          {isSession && sessionParts && sessionParts[activePartIdx] && (
            <span className="hud-sub" title={sessionParts[activePartIdx]!.name}>
              {sessionParts[activePartIdx]!.name} (
              {formatDuration(sessionParts[activePartIdx]!.time_offset_sec)}–
              {formatDuration(
                sessionParts[activePartIdx]!.time_offset_sec +
                  partDurationSec(sessionParts[activePartIdx]!, fps),
              )}
              )
            </span>
          )}
        </div>
      )}
    </div>
  );
});

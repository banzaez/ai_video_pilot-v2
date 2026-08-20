import { useEffect, useMemo, useRef, useState } from "react";
import type { FeetDoc, TrackingData } from "../types";
import {
  drawCameraPlacement,
  normalizePlacement,
  type CameraPlacement,
  type HomographyDoc,
  type Mat3,
  type Pt,
} from "../homography";
import { MAP_H, MAP_W, drawFloorGrid, isGridFloorplan } from "../mapGrid";
import type { CountersDoc } from "../counters";
import { buildFeetIndex, resolveFeetOnMap } from "../feetIndex";
import {
  buildTrackKeyframes,
  colorForTrackId,
  detectionsAtFrame,
  frameAtTime,
  resolveDetectEveryN,
  type MergeTimeline,
} from "../utils";

export type MapCameraMark = {
  key: string;
  placement: CameraPlacement;
};

export type MapCalibPoint = {
  key: string;
  index: number;
  map: Pt;
};

import type { FeetSource } from "../feet";

/** Точка человека на плане (мультикам, вкладка «Люди»). */
export type MapLiveMarker = {
  camera: string;
  trackId: number;
  map: Pt;
  live?: boolean;
  dimmed?: boolean;
  feetSource?: FeetSource;
  confidence?: number;
};

type Props = {
  floorplanUrl: string;
  homography: HomographyDoc | null;
  tracking: TrackingData | null;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  /** Плеер смонтирован — переподписать play/pause на новый <video>. */
  videoActive?: boolean;
  currentFrame: number;
  showTrails: boolean;
  focusTrackIds?: number[] | null;
  groupByTrack?: Record<number, number>;
  mergeTimeline?: MergeTimeline | null;
  calibPairs?: { image: Pt; map: Pt }[];
  allCalibPoints?: MapCalibPoint[];
  camerasOnMap?: MapCameraMark[];
  activeCameraKey?: string | null;
  counters?: CountersDoc | null;
  compact?: boolean;
  /** Готовые точки на плане — без привязки к одному tracking/H */
  markers?: MapLiveMarker[];
  feetDoc?: FeetDoc | null;
};

export function MapFloorView({
  floorplanUrl,
  homography,
  tracking,
  videoRef,
  videoActive = false,
  currentFrame,
  showTrails,
  focusTrackIds = null,
  groupByTrack = {},
  mergeTimeline = null,
  calibPairs: _calibPairs,
  allCalibPoints: _allCalibPoints = [],
  camerasOnMap = [],
  activeCameraKey = null,
  counters = null,
  compact = false,
  markers = [],
  feetDoc = null,
}: Props) {
  const stageRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const gridRef = useRef<HTMLCanvasElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const trailRef = useRef<Map<number, Pt[]>>(new Map());
  const [rotDeg, setRotDeg] = useState(0);
  const useGrid = isGridFloorplan(floorplanUrl) || floorplanUrl === "grid" || !floorplanUrl;
  const feetIndex = useMemo(() => buildFeetIndex(feetDoc), [feetDoc]);
  const currentFrameRef = useRef(currentFrame);
  currentFrameRef.current = currentFrame;
  const drawRef = useRef<() => void>(() => {});
  const videoRefProp = videoRef;

  function normRot(deg: number): number {
    return ((Math.round(deg / 45) * 45) % 360 + 360) % 360;
  }

  useEffect(() => {
    trailRef.current.clear();
  }, [tracking, homography?.H, floorplanUrl, markers]);

  useEffect(() => {
    if (!useGrid) return;
    const grid = gridRef.current;
    if (!grid) return;
    if (grid.width !== MAP_W || grid.height !== MAP_H) {
      grid.width = MAP_W;
      grid.height = MAP_H;
    }
    const ctx = grid.getContext("2d");
    if (ctx) drawFloorGrid(ctx, MAP_W, MAP_H);
  }, [useGrid, floorplanUrl]);

  useEffect(() => {
    const img = imgRef.current;
    const grid = gridRef.current;
    const canvas = canvasRef.current;
    const stage = stageRef.current;
    if (!canvas || !stage) return;
    const bg = useGrid ? grid : img;
    if (!bg) return;

    let raf = 0;

    const draw = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const mw = useGrid ? MAP_W : img?.naturalWidth || MAP_W;
      const mh = useGrid ? MAP_H : img?.naturalHeight || MAP_H;
      if (!mw || !mh) return;

      const rect = stage.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) return;
      const fit = Math.min(rect.width / mw, rect.height / mh);
      const cw = Math.max(1, Math.round(mw * fit));
      const ch = Math.max(1, Math.round(mh * fit));
      if (canvas.width !== mw || canvas.height !== mh) {
        canvas.width = mw;
        canvas.height = mh;
      }
      canvas.style.width = `${cw}px`;
      canvas.style.height = `${ch}px`;
      canvas.style.left = `${(rect.width - cw) / 2}px`;
      canvas.style.top = `${(rect.height - ch) / 2}px`;
      canvas.style.transform = rotDeg ? `rotate(${rotDeg}deg)` : "";
      canvas.style.transformOrigin = "50% 50%";

      if (useGrid && grid) {
        grid.style.width = `${cw}px`;
        grid.style.height = `${ch}px`;
        grid.style.left = `${(rect.width - cw) / 2}px`;
        grid.style.top = `${(rect.height - ch) / 2}px`;
        grid.style.transform = rotDeg ? `rotate(${rotDeg}deg)` : "";
        grid.style.transformOrigin = "50% 50%";
      } else if (img) {
        img.style.transform = rotDeg ? `rotate(${rotDeg}deg)` : "";
        img.style.transformOrigin = "50% 50%";
      }

      ctx.clearRect(0, 0, mw, mh);

      for (const c of counters?.counters ?? []) {
        if (!c.map || c.map.length < 3) continue;
        ctx.beginPath();
        ctx.moveTo(c.map[0]![0], c.map[0]![1]);
        for (let i = 1; i < c.map.length; i++) ctx.lineTo(c.map[i]![0], c.map[i]![1]);
        ctx.closePath();
        ctx.fillStyle = "rgba(90, 90, 90, 0.18)";
        ctx.fill();
        ctx.strokeStyle = "#5a5a5a";
        ctx.lineWidth = Math.max(2, mw / 900);
        ctx.stroke();
      }

      const livePlacement = normalizePlacement(homography?.placement);
      const byKey = new Map<string, CameraPlacement>();
      for (const c of camerasOnMap) {
        const pl = normalizePlacement(c.placement);
        if (pl) byKey.set(c.key, pl);
      }
      const activeKey = activeCameraKey ?? homography?.camera_key ?? null;
      if (livePlacement && activeKey) byKey.set(activeKey, livePlacement);

      for (const [key, pl] of byKey) {
        drawCameraPlacement(ctx, pl, `cam ${key}`, {
          active: key === activeKey,
          dimmed: activeKey != null && key !== activeKey,
          mapW: mw,
          cameraKey: key,
        });
      }

      const H = (homography?.H?.length === 9 ? (homography.H as Mat3) : null) ?? null;
      const useMarkers = markers.length > 0;

      if (useMarkers) {
        const markerR = Math.max(12, mw / 60);
        for (const mk of markers) {
          const [u, v] = mk.map;
          if (u < -mw * 0.2 || v < -mh * 0.2 || u > mw * 1.2 || v > mh * 1.2) continue;
          const dimmed = mk.dimmed ?? !mk.live;
          const lowConf = (mk.confidence ?? 1) < 0.65;
          const color = dimmed ? "#8a9188" : colorForTrackId(mk.trackId);
          const r = dimmed ? markerR * 0.75 : lowConf ? markerR * 0.85 : markerR;

          ctx.globalAlpha = dimmed ? 0.5 : lowConf ? 0.82 : 1;
          ctx.beginPath();
          ctx.arc(u, v, r + 3, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(0,0,0,0.35)";
          ctx.fill();
          ctx.beginPath();
          ctx.arc(u, v, r, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = Math.max(2, mw / 450);
          ctx.stroke();
          if (lowConf && !dimmed) {
            ctx.setLineDash([Math.max(3, mw / 200), Math.max(3, mw / 200)]);
            ctx.beginPath();
            ctx.arc(u, v, r + 5, 0, Math.PI * 2);
            ctx.strokeStyle = "rgba(200, 120, 40, 0.85)";
            ctx.lineWidth = Math.max(2, mw / 500);
            ctx.stroke();
            ctx.setLineDash([]);
          }

          const camId = mk.camera
            .replace(/^Camera[_-]?0*(\d+).*$/i, "$1")
            .replace(/^cam[_-]?0*(\d+)$/i, "$1")
            .replace(/^0+(\d+)$/, "$1");
          const label = `${camId} #${mk.trackId}`;
          const fontPx = Math.max(11, mw / 52);
          ctx.font = `700 ${fontPx}px "IBM Plex Mono", monospace`;
          const tw = ctx.measureText(label).width;
          const lx = u + r + 4;
          const ly = v - r;
          ctx.fillStyle = "rgba(0,0,0,0.72)";
          ctx.fillRect(lx, ly - fontPx - 4, tw + 6, fontPx + 6);
          ctx.fillStyle = "#fff";
          ctx.fillText(label, lx + 3, ly - 5);
          ctx.globalAlpha = 1;
        }
      } else if (!H || !tracking) {
        return;
      } else {
      const t =
        tracking.fps > 0
          ? currentFrameRef.current / tracking.fps
          : (videoRefProp?.current?.currentTime ?? 0);
      const frameFloat = frameAtTime(t, tracking.fps, tracking.frame_count);
      const keyframes = buildTrackKeyframes(tracking);
      const every = resolveDetectEveryN(tracking);
      const dets = detectionsAtFrame(keyframes, frameFloat, every);
      const focus = focusTrackIds != null ? new Set(focusTrackIds) : null;
      const hasFocus = focus != null && focus.size > 0;

      const ordered =
        !hasFocus
          ? dets
          : [...dets.filter((d) => !focus.has(d.track_id)), ...dets.filter((d) => focus.has(d.track_id))];

      const markerR = Math.max(14, mw / 55);
      const trailW = Math.max(4, mw / 280);

      for (const det of ordered) {
        const camKey = activeCameraKey ?? homography?.camera_key ?? "";
        const projected = resolveFeetOnMap(det, frameFloat, H, {
          cameraKey: camKey,
          homography,
          trackingSize: tracking ? [tracking.width, tracking.height] : null,
          feetDoc,
          feetIndex,
          detectEveryN: every,
          personHeightM: feetDoc?.person_height_m,
        });
        if (!projected) continue;
        const [u, v] = projected.map;
        if (u < -mw * 0.2 || v < -mh * 0.2 || u > mw * 1.2 || v > mh * 1.2) continue;

        let fragTrackId = det.track_id;
        let gid = groupByTrack[det.track_id];
        if (mergeTimeline && mergeTimeline.tracks.length) {
          const match = mergeTimeline.tracks.find(
            (tr) =>
              (tr.global_id === det.track_id || tr.track_id === det.track_id) &&
              t >= tr.t0 - 0.5 &&
              t <= tr.t1 + 0.5,
          );
          if (match) {
            fragTrackId = match.track_id;
            if (match.group_id != null) gid = match.group_id;
          }
        }

        const dimmed = hasFocus && !focus.has(det.track_id) && !focus.has(fragTrackId);
        const color = dimmed ? "#9aa0a6" : colorForTrackId(fragTrackId);
        const r = dimmed ? markerR * 0.7 : markerR;

        let hist = trailRef.current.get(det.track_id);
        if (!hist) {
          hist = [];
          trailRef.current.set(det.track_id, hist);
        }
        const last = hist[hist.length - 1];
        if (!last || Math.hypot(last[0] - u, last[1] - v) > 1.5) {
          hist.push([u, v]);
          if (hist.length > 50) hist.shift();
        }

        ctx.globalAlpha = dimmed ? 0.38 : 1;
        if (showTrails && hist.length > 1) {
          ctx.beginPath();
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = trailW + 3;
          ctx.lineJoin = "round";
          ctx.lineCap = "round";
          ctx.moveTo(hist[0]![0], hist[0]![1]);
          for (let i = 1; i < hist.length; i++) ctx.lineTo(hist[i]![0], hist[i]![1]);
          ctx.stroke();
          ctx.beginPath();
          ctx.strokeStyle = color;
          ctx.lineWidth = trailW;
          ctx.moveTo(hist[0]![0], hist[0]![1]);
          for (let i = 1; i < hist.length; i++) ctx.lineTo(hist[i]![0], hist[i]![1]);
          ctx.stroke();
        }

        ctx.beginPath();
        ctx.arc(u, v, r + 4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0,0,0,0.35)";
        ctx.fill();

        ctx.beginPath();
        ctx.arc(u, v, r, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = Math.max(3, mw / 400);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(u, v, Math.max(3, r * 0.28), 0, Math.PI * 2);
        ctx.fillStyle = "#fff";
        ctx.fill();

        const label = typeof gid === "number" ? `t${fragTrackId} g${gid}` : `t${fragTrackId}`;
        const fontPx = Math.max(16, mw / 42);
        ctx.font = `800 ${fontPx}px "IBM Plex Mono", monospace`;
        const tw = ctx.measureText(label).width;
        const padX = 8;
        const padY = 5;
        const lx = u + r + 6;
        const ly = v - r - 2;
        const bw = tw + padX * 2;
        const bh = fontPx + padY * 2;
        ctx.fillStyle = "rgba(0,0,0,0.72)";
        if (typeof ctx.roundRect === "function") {
          ctx.beginPath();
          ctx.roundRect(lx, ly - bh, bw, bh, 6);
          ctx.fill();
        } else {
          ctx.fillRect(lx, ly - bh, bw, bh);
        }
        ctx.fillStyle = "#fff";
        ctx.fillText(label, lx + padX, ly - padY - 2);
        ctx.globalAlpha = 1;
      }

      for (const id of [...trailRef.current.keys()]) {
        if (!ordered.some((d) => d.track_id === id)) trailRef.current.delete(id);
      }
      }
    };

    drawRef.current = draw;

    const loop = () => {
      draw();
      const video = videoRefProp?.current;
      if (video && !video.paused && !video.ended) raf = requestAnimationFrame(loop);
    };
    const kick = () => {
      cancelAnimationFrame(raf);
      draw();
      const video = videoRefProp?.current;
      if (video && !video.paused && !video.ended) raf = requestAnimationFrame(loop);
    };

    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => kick()) : null;
    if (stage) ro?.observe(stage);

    img?.addEventListener("load", kick);
    window.addEventListener("resize", kick);
    const video = videoRefProp?.current;
    video?.addEventListener("play", kick);
    video?.addEventListener("pause", kick);
    video?.addEventListener("seeked", kick);
    kick();
    return () => {
      cancelAnimationFrame(raf);
      ro?.disconnect();
      img?.removeEventListener("load", kick);
      window.removeEventListener("resize", kick);
      video?.removeEventListener("play", kick);
      video?.removeEventListener("pause", kick);
      video?.removeEventListener("seeked", kick);
    };
  }, [
    floorplanUrl,
    useGrid,
    homography,
    tracking,
    videoRefProp,
    videoActive,
    showTrails,
    focusTrackIds,
    groupByTrack,
    camerasOnMap,
    activeCameraKey,
    counters,
    rotDeg,
    markers,
    mergeTimeline,
    feetDoc,
    feetIndex,
  ]);

  useEffect(() => {
    const v = videoRefProp?.current;
    if (v && !v.paused && !v.ended) return;
    drawRef.current();
  }, [currentFrame, videoRefProp]);

  const hasAnyCam =
    !!normalizePlacement(homography?.placement) ||
    camerasOnMap.some((c) => !!normalizePlacement(c.placement));

  return (
    <div className={`map-floor${compact ? " is-compact" : ""}`} ref={stageRef}>
      {useGrid ? (
        <canvas ref={gridRef} className="map-floor-img map-floor-grid" />
      ) : (
        <img ref={imgRef} className="map-floor-img" src={floorplanUrl} alt="План помещения" draggable={false} />
      )}
      <canvas ref={canvasRef} className="map-floor-canvas" />
      <div className="map-floor-rot">
        <button type="button" title="Повернуть −45°" onClick={() => setRotDeg((r) => normRot(r - 45))}>
          ↶ 45°
        </button>
        <span>{rotDeg}°</span>
        <button type="button" title="Повернуть +45°" onClick={() => setRotDeg((r) => normRot(r + 45))}>
          ↷ 45°
        </button>
      </div>
      {!markers.length && !homography?.H && !hasAnyCam && (
        <p className="map-floor-hint">Нет гомографии — откройте вкладку «Карта» и задайте ≥4 точки на сетке</p>
      )}
    </div>
  );
}

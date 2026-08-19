import type { Detection, TrackingData, TrackKeyframe } from "./types";

const STORAGE_KEY = "ai-video-pilot-viewer-v1";

export type FloatVideoGeom = {
  x: number;
  y: number;
  w: number;
  h: number;
};

export type ViewerPrefs = {
  libraryVideo: string | null;
  librarySession: string | null;
  floatVideo: FloatVideoGeom;
  floatVideoMinimized: boolean;
  showVideo: boolean;
  showMap: boolean;
  floatMap: FloatVideoGeom;
  floatMapMinimized: boolean;
};

const FLOAT_MIN_W = 280;
export const FLOAT_MINI_W = 220;
export const FLOAT_BAR_H = 28;
export const FLOAT_PLAYER_CONTROLS_H = 111;
export const VIDEO_ASPECT = 16 / 9;

export function floatHeightForWidth(w: number, aspect = VIDEO_ASPECT, extraH = 0): number {
  return Math.round(FLOAT_BAR_H + extraH + w / aspect);
}

export function defaultFloatMap(): FloatVideoGeom {
  if (typeof window === "undefined") {
    const w = 640;
    return { x: 24, y: 72, w, h: floatHeightForWidth(w, 4800 / 3200) };
  }
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const aspect = 4800 / 3200;
  const w = Math.min(720, Math.max(FLOAT_MIN_W, Math.round(vw * 0.42)));
  const h = floatHeightForWidth(w, aspect);
  return {
    x: 16,
    y: Math.max(8, Math.min(56, vh - h - 8)),
    w,
    h,
  };
}

export function clampFloatVideo(
  g: FloatVideoGeom,
  aspect = VIDEO_ASPECT,
  minW = FLOAT_MIN_W,
  extraH = 0,
): FloatVideoGeom {
  if (typeof window === "undefined") {
    const w = Math.max(minW, Math.round(g.w));
    return { ...g, w, h: floatHeightForWidth(w, aspect, extraH) };
  }
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const maxW = Math.max(minW, vw - 16);
  const maxH = Math.max(floatHeightForWidth(minW, aspect, extraH), vh - 16);
  let w = Math.min(Math.max(Math.round(g.w), minW), maxW);
  let h = floatHeightForWidth(w, aspect, extraH);
  if (h > maxH) {
    w = Math.max(minW, Math.floor((maxH - FLOAT_BAR_H - extraH) * aspect));
    h = floatHeightForWidth(w, aspect, extraH);
  }
  const x = Math.min(Math.max(Math.round(g.x), 8 - w + 88), Math.max(0, vw - 88));
  const y = Math.min(Math.max(Math.round(g.y), 0), Math.max(0, vh - 36));
  return { x, y, w, h };
}

function parseFloatVideo(v: unknown): FloatVideoGeom | null {
  if (!v || typeof v !== "object") return null;
  const o = v as Record<string, unknown>;
  if (![o.x, o.y, o.w, o.h].every((n) => typeof n === "number" && Number.isFinite(n))) return null;
  return { x: o.x as number, y: o.y as number, w: o.w as number, h: o.h as number };
}

export type MediaItem = {
  video: string;
  json: string;
  hasJson: boolean;
  videoUrl: string;
  jsonUrl: string;
  feetUrl?: string;
  /** track_id → кропы tracklet ReID (если save_crops) */
  crops?: Record<string, CropShot[]>;
  /** group_id / track_id → кропы лиц InsightFace */
  faces?: Record<string, FaceShot[]>;
  /** track_id → skip-рёбра tracklet_link */
  similar?: Record<string, SimilarHit[]>;
  /** track_id → skip-рёбра на итоговых ID */
  merge?: Record<string, SimilarHit[]>;
};

export type FaceShot = {
  url: string;
  rank: number;
  score: number | null;
  pose_score?: number | null;
  quality?: number | null;
  frame: number;
  t: number | null;
  bbox: number[] | null;
  model?: string | null;
  track_id?: number | null;
  entity?: string | null;
  solo?: boolean;
};

export type FaceGallery = {
  models: string[];
  primary: Record<string, FaceShot[]>;
  byModel: Record<string, Record<string, FaceShot[]>>;
};

export type SimilarHit = {
  track_id: number;
  score: number;
  reid?: number | null;
  motion?: number | null;
  size?: number | null;
  gap?: number | null;
  dist?: number | null;
  space?: string | null;
  reason?: string | null;
  pass?: number | null;
  group_id?: number | null;
  t0?: number | null;
  t1?: number | null;
};

export type MergeTimelineTrack = {
  track_id: number;
  t0: number;
  t1: number;
  group_id: number | null;
  /** Итоговый track_id в tracking.json после tracklet_link. */
  global_id?: number | null;
};

export type MergeTimelineGroup = {
  group_id: number;
  track_ids: number[];
  score: number | null;
  reason: string | null;
  /** 1, 2, 10 или оба, если группа собрана разными проходами. */
  passes?: number[];
};

export type MergeTimelinePair = {
  a: number;
  b: number;
  score: number;
  reason: string | null;
  reid: number | null;
  face?: number | null;
  face_scores?: Record<string, number> | null;
  pose_face?: number | null;
  motion?: number | null;
  size?: number | null;
  gap?: number | null;
  dist?: number | null;
  space?: string | null;
  pass?: number | null;
  pass2?: boolean | null;
};

export type MergeTimelineSummary = {
  method: string | null;
  model: string | null;
  min_score: number | null;
  n_pairs: number | null;
  n_groups: number | null;
  complete_link: boolean | null;
};

export type MergeTimeline = {
  duration_sec: number;
  groups: MergeTimelineGroup[];
  tracks: MergeTimelineTrack[];
  pairs: MergeTimelinePair[];
  summary: MergeTimelineSummary;
};

export type VideoInfoParsed = {
  ok: boolean;
  camera?: string | null;
  camera_index?: number | null;
  ip?: string | null;
  peer_ip?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
  recording_id?: string | null;
  duration_sec?: number | null;
};

export type VideoInfo = {
  name: string;
  stem: string;
  path: string;
  url: string;
  size_bytes: number;
  mtime?: string | null;
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  duration_sec: number | null;
  codec?: string | null;
  parsed?: VideoInfoParsed | null;
};

export type CropShot = {
  url: string;
  rank: number;
  score: number | null;
  frame: number | null;
  t?: number | null;
  conf?: number | null;
  bbox?: number[] | null;
};

export function colorForTrackId(trackId: number): string {
  const hue = (trackId * 47) % 360;
  return `hsl(${hue} 72% 48%)`;
}

/** Три оттенка для разметки трека (триадные hue). */
export function trackBoxColors(trackId: number): [string, string, string] {
  const hue = (trackId * 47) % 360;
  return [
    `hsl(${hue} 88% 58%)`,
    `hsl(${(hue + 120) % 360} 82% 54%)`,
    `hsl(${(hue + 240) % 360} 78% 52%)`,
  ];
}

export function bboxWh(bbox: Detection["bbox"]): { w: number; h: number; area: number } {
  const w = Math.max(0, bbox[2] - bbox[0]);
  const h = Math.max(0, bbox[3] - bbox[1]);
  return { w, h, area: w * h };
}

/** Bbox трека в координатах JPEG-кропа (roi = crop_roi из tracklet ReID). */
export function bboxInCrop(
  bbox: number[] | null | undefined,
  roi: number[] | null | undefined,
): [number, number, number, number] | null {
  if (!bbox || bbox.length < 4 || !roi || roi.length < 4) return null;
  const rx1 = roi[0];
  const ry1 = roi[1];
  const bx1 = bbox[0];
  const by1 = bbox[1];
  const bx2 = bbox[2];
  const by2 = bbox[3];
  return [bx1 - rx1, by1 - ry1, bx2 - rx1, by2 - ry1];
}

export function frameIndexAtTime(currentTime: number, fps: number, frameCount: number): number {
  return Math.floor(frameAtTime(currentTime, fps, frameCount));
}

/** Непрерывный 1-based индекс кадра (для сглаживания bbox между детекциями). */
export function frameAtTime(currentTime: number, fps: number, frameCount: number): number {
  if (fps <= 0) return 1;
  const raw = currentTime * fps + 1;
  return Math.min(Math.max(raw, 1), Math.max(frameCount, 1));
}

export function resolveDetectEveryN(data: TrackingData): number {
  const stored = data.detect_every_n;
  if (typeof stored === "number" && Number.isFinite(stored) && stored >= 1) {
    return Math.max(1, Math.round(stored));
  }
  const idxs = data.frames
    .filter((f) => f.detections.length > 0)
    .map((f) => f.frame_index);
  if (idxs.length < 4) return 1;
  const gaps: number[] = [];
  for (let i = 1; i < idxs.length; i++) {
    const g = idxs[i] - idxs[i - 1];
    if (g > 0) gaps.push(g);
  }
  if (!gaps.length) return 1;
  gaps.sort((a, b) => a - b);
  const median = gaps[Math.floor(gaps.length / 2)];
  return median >= 2 ? median : 1;
}

export function buildTrackKeyframes(data: TrackingData): Map<number, TrackKeyframe[]> {
  const map = new Map<number, TrackKeyframe[]>();
  for (const frame of data.frames) {
    for (const det of frame.detections) {
      let arr = map.get(det.track_id);
      if (!arr) {
        arr = [];
        map.set(det.track_id, arr);
      }
      arr.push({ frame: frame.frame_index, det });
    }
  }
  return map;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpDet(a: Detection, b: Detection, t: number): Detection {
  const bbox: Detection["bbox"] = [
    lerp(a.bbox[0], b.bbox[0], t),
    lerp(a.bbox[1], b.bbox[1], t),
    lerp(a.bbox[2], b.bbox[2], t),
    lerp(a.bbox[3], b.bbox[3], t),
  ];
  return {
    track_id: a.track_id,
    confidence: lerp(a.confidence, b.confidence, t),
    bbox,
  };
}

function spanAtFrame(
  keys: TrackKeyframe[],
  frame: number,
): { exact: TrackKeyframe } | { prev: TrackKeyframe; next: TrackKeyframe } | null {
  if (!keys.length) return null;
  if (frame < keys[0].frame || frame > keys[keys.length - 1].frame) return null;
  let lo = 0;
  let hi = keys.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const mf = keys[mid].frame;
    if (mf === frame) return { exact: keys[mid] };
    if (mf < frame) lo = mid + 1;
    else hi = mid - 1;
  }
  if (hi < 0 || lo >= keys.length) return null;
  return { prev: keys[hi], next: keys[lo] };
}

/** Детекции на кадре: при detect_every_n > 1 — линейная интерполяция bbox между соседними хитами. */
export function detectionsAtFrame(
  keyframes: Map<number, TrackKeyframe[]>,
  frame: number,
  detectEveryN: number,
): Detection[] {
  const maxGap = Math.max(1, detectEveryN);
  const out: Detection[] = [];
  for (const keys of keyframes.values()) {
    const span = spanAtFrame(keys, frame);
    if (!span) continue;
    if ("exact" in span) {
      out.push(span.exact.det);
      continue;
    }
    const gap = span.next.frame - span.prev.frame;
    if (gap > maxGap) continue;
    const t = (frame - span.prev.frame) / gap;
    out.push(lerpDet(span.prev.det, span.next.det, t));
  }
  return out;
}

export function parseTrackingData(text: string): TrackingData {
  const data = JSON.parse(text) as TrackingData;
  if (!data || !Array.isArray(data.frames) || typeof data.fps !== "number") {
    throw new Error("Неверный формат tracking JSON");
  }
  return data;
}

export async function fetchTrackingJson(url: string): Promise<TrackingData> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`JSON не найден (${res.status}): ${url}`);
  }
  return parseTrackingData(await res.text());
}

export function feetJsonUrl(trackingJsonUrl: string): string {
  return trackingJsonUrl.replace(/tracking\.json(\?.*)?$/, "feet.json$1");
}

export async function fetchFeetJson(url: string): Promise<import("./types").FeetDoc | null> {
  const res = await fetch(url);
  if (res.status === 404) return null;
  if (!res.ok) return null;
  try {
    const data = (await res.json()) as unknown;
    if (!data || typeof data !== "object" || !Array.isArray((data as { frames?: unknown }).frames)) {
      return null;
    }
    return data as import("./types").FeetDoc;
  } catch {
    return null;
  }
}

export async function fetchMediaLibrary(): Promise<MediaItem[]> {
  const res = await fetch("/api/media/list");
  if (!res.ok) throw new Error("Не удалось получить список видео");
  const data = (await res.json()) as { items: MediaItem[] };
  return data.items ?? [];
}

export type PipelineJob = {
  id: string;
  status: "pending" | "running" | "done" | "failed" | "stopped";
  input: string;
  cmd: string[];
  created_at: string;
  stage?: string | null;
  stage_from?: string | null;
  stage_to?: string | null;
  pid?: number | null;
  started_at?: string | null;
  ended_at?: string | null;
  exit_code?: number | null;
  error?: string | null;
  log_path?: string;
};

export type CreateJobBody = {
  input: string;
  stage?: string | null;
  from?: string | null;
  to?: string | null;
};

function jobsHeaders(): HeadersInit {
  const token = (import.meta as { env?: { VITE_JOBS_API_TOKEN?: string } }).env?.VITE_JOBS_API_TOKEN;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function jobsJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...jobsHeaders(),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const err = (await res.json()) as { detail?: string };
      if (err.detail) detail = err.detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function fetchJobs(limit = 50): Promise<PipelineJob[]> {
  const data = await jobsJson<{ jobs: PipelineJob[] }>(`/api/jobs?limit=${limit}`);
  return data.jobs ?? [];
}

export async function fetchActiveJob(): Promise<PipelineJob | null> {
  const data = await jobsJson<{ job: PipelineJob | null }>("/api/jobs/active");
  return data.job ?? null;
}

export async function createJob(body: CreateJobBody): Promise<PipelineJob> {
  return jobsJson<PipelineJob>("/api/jobs", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function stopJob(id: string): Promise<PipelineJob> {
  return jobsJson<PipelineJob>(`/api/jobs/${encodeURIComponent(id)}/stop`, { method: "POST" });
}

export async function restartJob(id: string): Promise<PipelineJob> {
  return jobsJson<PipelineJob>(`/api/jobs/${encodeURIComponent(id)}/restart`, { method: "POST" });
}

export async function fetchJobLog(
  id: string,
  offset = 0,
): Promise<{ offset: number; size: number; text: string }> {
  return jobsJson(`/api/jobs/${encodeURIComponent(id)}/log?offset=${offset}`);
}

export async function fetchMediaSessions(): Promise<import("./session").MediaSession[]> {
  const res = await fetch("/api/media/sessions");
  if (!res.ok) throw new Error("Не удалось получить список sessions");
  const data = (await res.json()) as { sessions: import("./session").MediaSession[] };
  return data.sessions ?? [];
}

export type PipelineStageInfo = {
  stage: string;
  file: string;
  exists: boolean;
  file_version: number | null;
  written_at: string | null;
  size?: number | null;
  mtime?: string | null;
  stale: boolean;
  reason: string | null;
};

export type PipelineExtraInfo = {
  key: string;
  label: string;
  path: string;
  kind: "dir" | "file";
  exists: boolean;
  files?: number | null;
  size: number | null;
  mtime: string | null;
  note?: string | null;
};

export type PipelineDirInfo = {
  path: string;
  exists: boolean;
  files: number;
  size: number | null;
  mtime: string | null;
  note?: string | null;
};

export type PipelineStaleReport = {
  stages: Record<string, PipelineStageInfo>;
  stale: string[];
  recompute_from: string | null;
  recompute_to: string | null;
  cli: string | null;
  dirs?: Record<string, PipelineDirInfo>;
  extras?: Record<string, PipelineExtraInfo[]>;
};

export async function fetchMediaMeta(
  baseOrSession: string,
  opts?: { session?: boolean },
): Promise<{
  info: VideoInfo | null;
  crops: Record<string, CropShot[]>;
  faces: Record<string, FaceShot[]>;
  facesByModel: Record<string, Record<string, FaceShot[]>>;
  faceModels: string[];
  cameraLink: {
    face_models?: string[];
    edges?: MergeTimelinePair[];
    candidate_edges?: MergeTimelinePair[];
  } | null;
  similar: Record<string, SimilarHit[]>;
  merge: Record<string, SimilarHit[]>;
  mergeTimeline: MergeTimeline | null;
  pipeline: PipelineStaleReport | null;
}> {
  const isSession = opts?.session ?? /^\d{2}_\d{8}$/.test(baseOrSession);
  const q = isSession
    ? `session=${encodeURIComponent(baseOrSession)}`
    : `base=${encodeURIComponent(baseOrSession.replace(/\.[^.]+$/, ""))}`;
  const res = await fetch(`/api/media/meta?${q}`);
  if (!res.ok) {
    return {
      info: null,
      crops: {},
      faces: {},
      facesByModel: {},
      faceModels: [],
      cameraLink: null,
      similar: {},
      merge: {},
      mergeTimeline: null,
      pipeline: null,
    };
  }
  const data = (await res.json()) as {
    info?: VideoInfo | null;
    crops?: Record<string, CropShot[]>;
    faces?: Record<string, FaceShot[]>;
    facesByModel?: Record<string, Record<string, FaceShot[]>>;
    faceModels?: string[];
    cameraLink?: {
      face_models?: string[];
      edges?: MergeTimelinePair[];
      candidate_edges?: MergeTimelinePair[];
    } | null;
    similar?: Record<string, SimilarHit[]>;
    merge?: Record<string, SimilarHit[]>;
    stitch?: Record<string, SimilarHit[]>;
    mergeTimeline?: MergeTimeline | null;
    pipeline?: PipelineStaleReport | null;
  };
  return {
    info: data.info ?? null,
    crops: data.crops ?? {},
    faces: data.faces ?? {},
    facesByModel: data.facesByModel ?? {},
    faceModels: data.faceModels ?? [],
    cameraLink: data.cameraLink ?? null,
    similar: data.similar ?? {},
    merge: data.merge ?? data.stitch ?? {},
    mergeTimeline: data.mergeTimeline ?? null,
    pipeline: data.pipeline ?? null,
  };
}

export type MapsConfig = {
  floorplan: string;
  floorplans: { name: string; url: string }[];
  cameras: {
    key: string;
    camera_index: number | null;
    video: string | null;
    videoUrl: string | null;
    pairs: number;
    hasH: boolean;
    hasPlacement: boolean;
    placement: import("./homography").CameraPlacement | null;
    map_points: { index: number; map: [number, number] }[];
  }[];
};

export async function fetchMapsConfig(): Promise<MapsConfig> {
  const res = await fetch("/api/maps/config");
  if (!res.ok) throw new Error("Не удалось загрузить конфиг карты");
  return (await res.json()) as MapsConfig;
}

export async function fetchHomography(cameraKey: string): Promise<import("./homography").HomographyDoc> {
  const res = await fetch(`/api/maps/homography?camera=${encodeURIComponent(cameraKey)}`);
  if (!res.ok) throw new Error("Не удалось загрузить гомографию");
  return (await res.json()) as import("./homography").HomographyDoc;
}

export async function saveHomography(
  cameraKey: string,
  doc: import("./homography").HomographyDoc,
): Promise<import("./homography").HomographyDoc> {
  const res = await fetch(`/api/maps/homography?camera=${encodeURIComponent(cameraKey)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(doc),
  });
  if (!res.ok) throw new Error("Не удалось сохранить гомографию");
  return (await res.json()) as import("./homography").HomographyDoc;
}

export async function fetchCounters(): Promise<import("./counters").CountersDoc> {
  const res = await fetch("/api/maps/counters");
  if (!res.ok) throw new Error("Не удалось загрузить прилавки");
  return (await res.json()) as import("./counters").CountersDoc;
}

export async function saveCounters(
  doc: import("./counters").CountersDoc,
): Promise<import("./counters").CountersDoc> {
  const res = await fetch("/api/maps/counters", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(doc),
  });
  if (!res.ok) throw new Error("Не удалось сохранить прилавки");
  return (await res.json()) as import("./counters").CountersDoc;
}

export function cameraKeyFromVideo(videoName: string, cameraIndex?: number | null): string {
  if (typeof cameraIndex === "number" && Number.isFinite(cameraIndex)) {
    return String(cameraIndex).padStart(3, "0");
  }
  const m = /Cam(?:era)?[_-]?(\d+)/i.exec(videoName);
  if (m) return String(Number(m[1])).padStart(3, "0");
  return videoName.replace(/\.[^.]+$/, "") || "000";
}

export function defaultFloatVideo(aspect = VIDEO_ASPECT): FloatVideoGeom {
  if (typeof window === "undefined") {
    const w = 560;
    return { x: 24, y: 72, w, h: floatHeightForWidth(w, aspect, FLOAT_PLAYER_CONTROLS_H) };
  }
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const w = Math.min(620, Math.max(FLOAT_MIN_W, Math.round(vw * 0.4)));
  const h = floatHeightForWidth(w, aspect, FLOAT_PLAYER_CONTROLS_H);
  return {
    x: Math.max(8, vw - w - 16),
    y: Math.max(8, Math.min(56, vh - h - 8)),
    w,
    h,
  };
}

export function formatDuration(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(sec) || sec < 0) return "—";
  const s = Math.round(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h) return `${h}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
  return `${m}:${String(r).padStart(2, "0")}`;
}

export function parseTimecode(str: string): number | null {
  if (!str) return null;
  const parts = str.trim().split(":").map(Number);
  if (parts.some(Number.isNaN)) return null;
  if (parts.length === 3) return parts[0]! * 3600 + parts[1]! * 60 + parts[2]!;
  if (parts.length === 2) return parts[0]! * 60 + parts[1]!;
  if (parts.length === 1 && Number.isFinite(parts[0])) return parts[0]!;
  return null;
}

export type ActivityCurveData = {
  svgPath: string;
  strokePath: string;
  maxCount: number;
  effDuration: number;
  bins: Array<{ timeSec: number; count: number }>;
};

/** Построение полигона активности (количества людей/треков в кадре по времени). */
export function buildActivityCurve(
  tracking: { frames: Array<{ frame_index: number; timestamp_sec?: number; detections: unknown[] }>; fps?: number; frame_count?: number } | null,
  durationSec: number,
  binsCount = 180,
): ActivityCurveData {
  if (!tracking || !tracking.frames || !tracking.frames.length) {
    return { svgPath: "", strokePath: "", maxCount: 0, effDuration: 0, bins: [] };
  }

  const fps = tracking.fps || 25;
  const lastFrame = tracking.frames[tracking.frames.length - 1];
  const lastSec = typeof lastFrame?.timestamp_sec === "number"
    ? lastFrame.timestamp_sec
    : (lastFrame?.frame_index ?? tracking.frames.length) / fps;
  const effDuration = durationSec > 0 ? durationSec : (tracking.frame_count ? tracking.frame_count / fps : lastSec);

  if (effDuration <= 0) {
    return { svgPath: "", strokePath: "", maxCount: 0, effDuration: 0, bins: [] };
  }

  const bins = new Array<number>(binsCount).fill(0);

  for (const f of tracking.frames) {
    const t = typeof f.timestamp_sec === "number" ? f.timestamp_sec : f.frame_index / fps;
    const binIdx = Math.min(binsCount - 1, Math.max(0, Math.floor((t / effDuration) * binsCount)));
    const personCount = f.detections?.length ?? 0;
    if (personCount > bins[binIdx]!) {
      bins[binIdx] = personCount;
    }
  }

  const maxCount = Math.max(1, ...bins);
  const resultBins = bins.map((count, i) => ({
    timeSec: (i / binsCount) * effDuration,
    count,
  }));

  // Строим SVG path для заливки и контурной линии
  const strokePoints: string[] = [];
  const fillPoints: string[] = ["0,32"];

  for (let i = 0; i < binsCount; i++) {
    const x = ((i / (binsCount - 1)) * 100).toFixed(2);
    const height = Math.max(1, (bins[i]! / maxCount) * 26); // 1..26px
    const y = (30 - height).toFixed(2);
    fillPoints.push(`${x},${y}`);
    strokePoints.push(`${x},${y}`);
  }
  fillPoints.push("100,32");

  return {
    svgPath: `M ${fillPoints.join(" L ")} Z`,
    strokePath: `M ${strokePoints.join(" L ")}`,
    maxCount,
    effDuration,
    bins: resultBins,
  };
}

export function formatBytes(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n) || n < 0) return "—";
  if (n < 1024) return `${Math.round(n)} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function formatTs(iso: string | null | undefined): string {
  if (!iso) return "—";
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})/.exec(iso);
  if (!m) return iso;
  return `${m[3]}.${m[2]}.${m[1]} ${m[4]}:${m[5]}:${m[6]}`;
}

function defaultPrefs(): ViewerPrefs {
  return {
    libraryVideo: null,
    librarySession: null,
    floatVideo: defaultFloatVideo(),
    floatVideoMinimized: false,
    showVideo: true,
    showMap: false,
    floatMap: defaultFloatMap(),
    floatMapMinimized: false,
  };
}

export function loadPrefs(): ViewerPrefs {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultPrefs();
    const parsed = JSON.parse(raw) as Partial<ViewerPrefs>;
    return {
      libraryVideo: parsed.librarySession ?? parsed.libraryVideo ?? null,
      librarySession: parsed.librarySession ?? parsed.libraryVideo ?? null,
      floatVideo: clampFloatVideo(parseFloatVideo(parsed.floatVideo) ?? defaultFloatVideo()),
      floatVideoMinimized: false,
      showVideo: parsed.showVideo ?? true,
      showMap: parsed.showMap ?? false,
      floatMap: clampFloatVideo(parseFloatVideo(parsed.floatMap) ?? defaultFloatMap(), 4800 / 3200),
      floatMapMinimized: parsed.floatMapMinimized ?? false,
    };
  } catch {
    return defaultPrefs();
  }
}

export function savePrefs(prefs: ViewerPrefs): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
}

/** Верхняя граница шкалы Pass N → цвет (Pass 10 = красный). */
export const PASS_BADGE_MAX = 10;

export function resolveLinkPass(p: { pass?: number | null; pass2?: boolean | null }): number | null {
  if (typeof p.pass === "number" && p.pass >= 0) return p.pass;
  if (p.pass2) return 2;
  return null;
}

/** Градация зелёный → жёлтый → красный по уровню pass. */
export function passBadgeColors(
  pass: number,
  maxPass = PASS_BADGE_MAX,
): { bg: string; fg: string; border: string } {
  const t = Math.min(1, Math.max(0, pass / maxPass));
  let r: number;
  let g: number;
  let b: number;
  if (t <= 0.5) {
    const u = t / 0.5;
    r = Math.round(15 + (202 - 15) * u);
    g = Math.round(110 + (138 - 110) * u);
    b = Math.round(86 + (4 - 86) * u);
  } else {
    const u = (t - 0.5) / 0.5;
    r = Math.round(202 + (220 - 202) * u);
    g = Math.round(138 + (38 - 138) * u);
    b = Math.round(4 + (38 - 4) * u);
  }
  return {
    bg: `rgba(${r}, ${g}, ${b}, 0.14)`,
    fg: `rgb(${r}, ${g}, ${b})`,
    border: `rgba(${r}, ${g}, ${b}, 0.32)`,
  };
}

/** Camera-day sessions: prod-имена и группировка (зеркало app/session/discover.py). */

export type ParsedPart = {
  stem: string;
  name: string;
  camera_index: number;
  started_raw: string;
  ended_raw: string;
  started_at: string;
  ended_at: string;
  day: string;
  session_key: string;
};

export type SessionPart = {
  name: string;
  stem: string;
  videoUrl: string;
  started_at: string;
  ended_at: string;
  frame_offset: number;
  frame_count: number;
  time_offset_sec: number;
};

export type MediaSession = {
  key: string;
  camera: string;
  camera_index: number;
  day: string;
  parts: SessionPart[];
  hasJson: boolean;
  jsonUrl: string;
  feetUrl?: string;
  duration_sec?: number;
  fps?: number;
};

/** Camera_01_<source>_<started>_<ended>_<seg>; source = nvr_local | IP | другое */
const PROD_STEM =
  /^Camera_(\d+)_(.+)_(\d{14})_(\d{14})_(.+)$/i;

function isoFromNvr(raw: string): string | null {
  if (raw.length !== 14) return null;
  const y = raw.slice(0, 4);
  const mo = raw.slice(4, 6);
  const d = raw.slice(6, 8);
  const h = raw.slice(8, 10);
  const mi = raw.slice(10, 12);
  const s = raw.slice(12, 14);
  return `${y}-${mo}-${d}T${h}:${mi}:${s}`;
}

function dayFromStarted(startedRaw: string): string | null {
  if (startedRaw.length < 8) return null;
  const y = startedRaw.slice(0, 4);
  const mo = startedRaw.slice(4, 6);
  const d = startedRaw.slice(6, 8);
  return `${y}-${mo}-${d}`;
}

export function sessionKeyFromPart(cameraIndex: number, startedRaw: string): string {
  return `${String(cameraIndex).padStart(3, "0")}_${startedRaw.slice(0, 8)}`;
}

export function parseProdStem(stem: string): ParsedPart | null {
  const m = PROD_STEM.exec(stem);
  if (!m) return null;
  const camera_index = Number(m[1]);
  const started_raw = m[3]!;
  const ended_raw = m[4]!;
  const started_at = isoFromNvr(started_raw);
  const ended_at = isoFromNvr(ended_raw);
  const day = dayFromStarted(started_raw);
  if (!started_at || !ended_at || !day) return null;
  return {
    stem,
    name: `${stem}.mp4`,
    camera_index,
    started_raw,
    ended_raw,
    started_at,
    ended_at,
    day,
    session_key: sessionKeyFromPart(camera_index, started_raw),
  };
}

export function groupBySessionKey(parts: ParsedPart[]): Map<string, ParsedPart[]> {
  const byKey = new Map<string, ParsedPart[]>();
  for (const p of parts) {
    const list = byKey.get(p.session_key) ?? [];
    list.push(p);
    byKey.set(p.session_key, list);
  }
  for (const [key, list] of byKey) {
    list.sort((a, b) => a.started_raw.localeCompare(b.started_raw));
    byKey.set(key, list);
  }
  return byKey;
}

/** Длительность части: frame_count/fps или стенка started_at…ended_at. */
export function partDurationSec(part: SessionPart, fps: number): number {
  if (part.frame_count > 0) return part.frame_count / Math.max(fps, 1e-6);
  const a = Date.parse(part.started_at);
  const b = Date.parse(part.ended_at);
  if (Number.isFinite(a) && Number.isFinite(b) && b > a) return (b - a) / 1000;
  return 0;
}

/** Суммарная длительность session (или fallback из tracking). */
export function sessionDurationSec(
  parts: SessionPart[],
  fps: number,
  fallback?: number | null,
): number {
  if (!parts.length) return Math.max(0, fallback ?? 0);
  const last = parts[parts.length - 1]!;
  return last.time_offset_sec + partDurationSec(last, fps);
}

/** global time (sec) → part + local time within part video. */
export function partAtTime(
  parts: SessionPart[],
  tSec: number,
  fps: number,
): { part: SessionPart; localTime: number } | null {
  if (!parts.length) return null;
  const t = Math.max(0, tSec);
  for (const part of parts) {
    const dur = partDurationSec(part, fps);
    const t0 = part.time_offset_sec;
    const t1 = t0 + dur;
    if (t >= t0 && t < t1 - 1e-6) {
      return { part, localTime: t - t0 };
    }
  }
  const last = parts[parts.length - 1]!;
  const dur = partDurationSec(last, fps);
  return { part: last, localTime: Math.min(Math.max(0, tSec - last.time_offset_sec), dur) };
}

export function partIndexOf(parts: SessionPart[], part: SessionPart | null): number {
  if (!part || !parts.length) return 0;
  const i = parts.findIndex((p) => p.name === part.name);
  return i >= 0 ? i : 0;
}

export function globalFrameAtTime(tSec: number, fps: number): number {
  return Math.max(0, Math.floor(tSec * fps));
}

export function sessionLabel(session: MediaSession): string {
  return `${String(session.camera_index).padStart(2, "0")} · ${session.day}`;
}

/** Дни по убыванию (новые сверху). */
export function sessionDays(sessions: MediaSession[]): string[] {
  return [...new Set(sessions.map((s) => s.day))].sort((a, b) => b.localeCompare(a));
}

export function sessionsForDay(sessions: MediaSession[], day: string): MediaSession[] {
  return sessions
    .filter((s) => s.day === day)
    .sort((a, b) => a.camera_index - b.camera_index || a.key.localeCompare(b.key));
}

/** 2026-06-01 → 01.06.2026 */
export function formatSessionDay(day: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(day);
  if (!m) return day;
  return `${m[3]}.${m[2]}.${m[1]}`;
}

export function cameraLabel(session: MediaSession): string {
  const cam = session.camera || `Camera_${String(session.camera_index).padStart(2, "0")}`;
  const parts = session.parts.length > 1 ? ` · ${session.parts.length} ч.` : "";
  const res = session.hasJson ? "" : " · нет результатов";
  return `${cam}${parts}${res}`;
}


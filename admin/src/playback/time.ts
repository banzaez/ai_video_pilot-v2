import type { TimeBounds } from "./types";

export function makeTimeBounds(minT: number, maxT: number): TimeBounds {
  const lo = Number.isFinite(minT) ? minT : 0;
  let hi = Number.isFinite(maxT) ? maxT : lo + 300;
  if (hi <= lo) hi = lo + 300;
  return { minT: lo, maxT: hi, span: Math.max(1e-6, hi - lo) };
}

export function clampTime(t: number, bounds: TimeBounds): number {
  return Math.max(bounds.minT, Math.min(bounds.maxT, t));
}

export function formatTimeOfDay(sec: number, withHundredths = true): string {
  const s = Math.floor(sec) % 86400;
  const hh = String(Math.floor(s / 3600)).padStart(2, "0");
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  if (!withHundredths) return `${hh}:${mm}:${ss}`;
  const ms = Math.floor((sec - Math.floor(sec)) * 100);
  return `${hh}:${mm}:${ss}.${String(ms).padStart(2, "0")}`;
}

export function formatHms(sec: number): string {
  return formatTimeOfDay(sec, false);
}

export function formatDurationClock(sec: number): string {
  const t = Math.max(0, sec);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = Math.floor(t % 60);
  const frac = Math.floor((t - Math.floor(t)) * 10);
  if (h > 0) {
    return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${frac}`;
  }
  return `${m}:${String(s).padStart(2, "0")}.${frac}`;
}

import { calibFingerprint, DEFAULT_PERSON_H_M } from "./calibFingerprint";
import { projectFeetToMap, type FeetSource } from "./feet";
import type { HomographyDoc, Mat3, Pt } from "./homography";
import type { Detection, FeetDoc } from "./types";

export type FeetSample = {
  frame: number;
  map: Pt;
  source: FeetSource;
  confidence: number;
};

export type FeetIndex = Map<number, FeetSample[]>;

export function buildFeetIndex(doc: FeetDoc | null | undefined): FeetIndex {
  const idx: FeetIndex = new Map();
  if (!doc?.frames) return idx;
  for (const fr of doc.frames) {
    const frame = Number(fr.frame_index);
    if (!Number.isFinite(frame)) continue;
    for (const p of fr.points || []) {
      const tid = Number(p.track_id);
      if (!Number.isFinite(tid) || !Array.isArray(p.map) || p.map.length < 2) continue;
      let arr = idx.get(tid);
      if (!arr) {
        arr = [];
        idx.set(tid, arr);
      }
      arr.push({
        frame,
        map: [Number(p.map[0]), Number(p.map[1])],
        source: (p.source as FeetSource) || "ray",
        confidence: Number(p.confidence) || 0,
      });
    }
  }
  for (const arr of idx.values()) arr.sort((a, b) => a.frame - b.frame);
  return idx;
}

function spanAtFrame(
  keys: FeetSample[],
  frame: number,
): { exact: FeetSample } | { prev: FeetSample; next: FeetSample } | null {
  if (!keys.length) return null;
  if (frame < keys[0]!.frame || frame > keys[keys.length - 1]!.frame) return null;
  let lo = 0;
  let hi = keys.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const mf = keys[mid]!.frame;
    if (mf === frame) return { exact: keys[mid]! };
    if (mf < frame) lo = mid + 1;
    else hi = mid - 1;
  }
  if (hi < 0 || lo >= keys.length) return null;
  return { prev: keys[hi]!, next: keys[lo]! };
}

export function feetAtFrame(
  index: FeetIndex,
  trackId: number,
  frameFloat: number,
  detectEveryN: number,
): FeetSample | null {
  const keys = index.get(trackId);
  if (!keys?.length) return null;
  const maxGap = Math.max(1, detectEveryN);
  const span = spanAtFrame(keys, frameFloat);
  if (!span) return null;
  if ("exact" in span) return span.exact;
  const gap = span.next.frame - span.prev.frame;
  if (gap > maxGap) return null;
  const t = (frameFloat - span.prev.frame) / gap;
  return {
    frame: frameFloat,
    map: [
      span.prev.map[0] + (span.next.map[0] - span.prev.map[0]) * t,
      span.prev.map[1] + (span.next.map[1] - span.prev.map[1]) * t,
    ],
    source: span.prev.source,
    confidence: span.prev.confidence + (span.next.confidence - span.prev.confidence) * t,
  };
}

export function liveCalibFingerprint(opts: {
  cameraKey: string;
  homography?: HomographyDoc | null;
  trackingSize?: [number, number] | null;
  torsoHeightM?: number;
  personHeightM?: number;
}): string {
  return calibFingerprint({
    cameraKey: opts.cameraKey,
    cameraDoc: opts.homography as unknown as Record<string, unknown> | null,
    torsoHeightM: opts.torsoHeightM ?? 0,
    trackingSize: opts.trackingSize ?? null,
    personHeightM: opts.personHeightM ?? DEFAULT_PERSON_H_M,
  });
}

export function resolveFeetOnMap(
  det: Detection,
  frameFloat: number,
  H: Mat3 | null,
  opts: {
    cameraKey: string;
    homography?: HomographyDoc | null;
    torsoHeightM?: number;
    personHeightM?: number;
    trackingSize?: [number, number] | null;
    feetDoc?: FeetDoc | null;
    feetIndex?: FeetIndex | null;
    detectEveryN?: number;
  },
): { map: Pt; source: FeetSource; confidence: number } | null {
  const stored = opts.feetDoc?.calibration?.fingerprint;
  const index = opts.feetIndex;
  if (stored && index && index.size) {
    const live = liveCalibFingerprint({
      cameraKey: opts.cameraKey,
      homography: opts.homography,
      trackingSize: opts.trackingSize,
      torsoHeightM: opts.torsoHeightM,
      personHeightM: opts.personHeightM ?? opts.feetDoc?.person_height_m ?? DEFAULT_PERSON_H_M,
    });
    if (stored === live) {
      const got = feetAtFrame(index, det.track_id, frameFloat, opts.detectEveryN ?? 1);
      if (!got) return null;
      return { map: got.map, source: got.source, confidence: got.confidence };
    }
  }
  return projectFeetToMap(det, Math.round(frameFloat), H, {
    cameraKey: opts.cameraKey,
    homography: opts.homography,
    torsoHeightM: opts.torsoHeightM,
    personHeightM: opts.personHeightM ?? opts.feetDoc?.person_height_m ?? DEFAULT_PERSON_H_M,
    trackingSize: opts.trackingSize,
  });
}

export function parseFeetDoc(raw: unknown): FeetDoc | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  if (!Array.isArray(o.frames)) return null;
  return o as FeetDoc;
}

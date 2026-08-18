import fs from "node:fs";
import path from "node:path";
import { resultsDir } from "./config.js";
import { readJsonFile, resultsUrl, sessionFps, workFile } from "./common.js";

export type SessionPartRef = { path?: string; frame_offset?: number; frame_count?: number };

export type CropShotOut = {
  url: string;
  rank: number;
  score: number | null;
  frame: number | null;
  t: number | null;
  conf: number | null;
  bbox: number[] | null;
};

/** Имя кропа tl_*_fN — N 1-based (как frame_index). Видео и parts — 0-based. */
export function videoLocalFrame(
  info: { parts?: SessionPartRef[]; fps?: number } | null,
  frame1: number,
): { videoRelPath: string | undefined; frame0: number } {
  const global0 = Math.max(0, Math.trunc(frame1) - 1);
  const parts = info?.parts ?? [];
  for (const part of parts) {
    const offset = part.frame_offset ?? 0;
    const count = part.frame_count ?? Infinity;
    if (global0 >= offset && global0 < offset + count) {
      return { videoRelPath: part.path, frame0: global0 - offset };
    }
  }
  return { videoRelPath: parts[0]?.path, frame0: global0 };
}

export function findTrackletBbox(base: string, trackletId: number, targetFrame: number): number[] | null {
  // 1. Проверяем tracklet_frames.json (наиболее точный источник для tracklet_id до объединения)
  const tfData = readJsonFile(workFile(base, "tracklet_frames.json")) as {
    frames?: Array<{ frame_index: number; detections?: Array<{ tracklet_id: number; bbox: number[] }> }>;
  } | null;
  if (tfData?.frames?.length) {
    let bestBbox: number[] | null = null;
    let minDiff = Infinity;
    for (const f of tfData.frames) {
      const det = f.detections?.find((d) => d.tracklet_id === trackletId);
      if (det?.bbox) {
        const diff = Math.abs(f.frame_index - targetFrame);
        if (diff < minDiff) {
          minDiff = diff;
          bestBbox = det.bbox;
          if (diff === 0) break;
        }
      }
    }
    if (bestBbox && minDiff <= 60) {
      return bestBbox;
    }
  }

  // 2. Проверяем tracklets.json (границы треклета и ключевые боксы)
  const tdata = readJsonFile(workFile(base, "tracklets.json")) as {
    tracklets?: Array<{ tracklet_id: number; f0: number; f1: number; bbox0?: number[]; bbox1?: number[] }>;
  } | null;
  const tr = tdata?.tracklets?.find((t) => t.tracklet_id === trackletId);
  if (tr) {
    if (Math.abs(tr.f0 - targetFrame) <= Math.abs(tr.f1 - targetFrame) && tr.bbox0) {
      return tr.bbox0;
    }
    if (tr.bbox1) return tr.bbox1;
    if (tr.bbox0) return tr.bbox0;
  }

  // 3. tracking.json — массив кадров с frame_index (1-based), не словарь
  const linksData = readJsonFile(workFile(base, "tracklet_links.json")) as {
    tracklet_to_global?: Record<string, number>;
  } | null;
  const globalId = linksData?.tracklet_to_global?.[String(trackletId)] ?? trackletId;
  const tracking = readJsonFile(workFile(base, "tracking.json")) as {
    frames?: Array<{ frame_index: number; detections?: Array<{ track_id?: number; bbox?: number[] }> }>;
  } | null;
  if (tracking?.frames?.length) {
    let bestBbox: number[] | null = null;
    let minDiff = Infinity;
    for (const f of tracking.frames) {
      const match = f.detections?.find((d) => d.track_id === globalId || d.track_id === trackletId);
      if (match?.bbox) {
        const diff = Math.abs(f.frame_index - targetFrame);
        if (diff < minDiff) {
          minDiff = diff;
          bestBbox = match.bbox;
          if (diff === 0) break;
        }
      }
    }
    if (bestBbox && minDiff <= 60) {
      return bestBbox;
    }
  }

  return null;
}

export function cropUrlsFor(base: string): Record<string, CropShotOut[]> {
  const buckets: Record<string, (CropShotOut & { file: string })[]> = {};
  const fps = sessionFps(base);
  const patternTl = /^tl_(\d+)_k(\d+)_f(\d+)\.(jpe?g|png|webp)$/i;
  const patternLegacy = /^id_(\d+)_k(\d+)_s(\d+)_f(\d+)\.(jpe?g|png|webp)$/i;
  const seen = new Set<string>();

  const pushShot = (key: string, shot: CropShotOut & { file: string }) => {
    const sig = `${key}:${shot.file}`;
    if (seen.has(sig)) return;
    seen.add(sig);
    (buckets[key] ??= []).push(shot);
  };

  const trackletCropsDir = path.join(resultsDir, base, "tracklet_crops");
  const reidData = readJsonFile(workFile(base, "tracklet_reid.json")) as {
    tracklets?: Array<{ tracklet_id: number; crop_paths?: string[] }>;
  } | null;

  // Только tracklet_id. Global_id сюда нельзя: склейки тогда показывают чужие кропы группы.
  if (reidData?.tracklets?.length) {
    for (const t of reidData.tracklets) {
      const tl = String(t.tracklet_id);
      const tid = Number(t.tracklet_id);
      for (const cp of t.crop_paths ?? []) {
        const f = path.basename(cp);
        const m = patternTl.exec(f);
        if (!m) continue;
        if (Number(m[1]) !== tid) continue;
        const rank = Number(m[2]) + 1;
        const frame = Number(m[3]);
        const onDiskPath = path.join(trackletCropsDir, f);
        let url: string;
        if (fs.existsSync(onDiskPath)) {
          const st = fs.statSync(onDiskPath);
          url = `${resultsUrl(base, "tracklet_crops", f)}?t=${Math.round(st.mtimeMs)}`;
        } else {
          url = `/api/crop/${encodeURIComponent(base)}/${encodeURIComponent(f)}`;
        }
        pushShot(tl, {
          rank,
          file: f,
          url,
          score: null,
          frame,
          t: frame / fps,
          conf: null,
          bbox: null,
        });
      }
    }
  } else if (fs.existsSync(trackletCropsDir) && fs.statSync(trackletCropsDir).isDirectory()) {
    for (const f of fs.readdirSync(trackletCropsDir)) {
      const m = patternTl.exec(f);
      if (!m) continue;
      const tl = String(Number(m[1]));
      const rank = Number(m[2]) + 1;
      const frame = Number(m[3]);
      const st = fs.statSync(path.join(trackletCropsDir, f));
      const url = `${resultsUrl(base, "tracklet_crops", f)}?t=${Math.round(st.mtimeMs)}`;
      pushShot(tl, {
        rank,
        file: f,
        url,
        score: null,
        frame,
        t: frame / fps,
        conf: null,
        bbox: null,
      });
    }
  }

  const legacyCropsDir = path.join(resultsDir, base, "crops");
  if (!Object.keys(buckets).length && fs.existsSync(legacyCropsDir) && fs.statSync(legacyCropsDir).isDirectory()) {
    for (const f of fs.readdirSync(legacyCropsDir)) {
      const m = patternLegacy.exec(f);
      if (!m) continue;
      const trackId = String(Number(m[1]));
      const rank = Number(m[2]);
      const frame = Number(m[4]);
      const st = fs.statSync(path.join(legacyCropsDir, f));
      const url = `${resultsUrl(base, "crops", f)}?t=${Math.round(st.mtimeMs)}`;
      pushShot(trackId, {
        rank,
        file: f,
        url,
        score: null,
        frame,
        t: frame / fps,
        conf: null,
        bbox: null,
      });
    }
  }

  const out: Record<string, CropShotOut[]> = {};
  for (const [id, items] of Object.entries(buckets)) {
    items.sort((a, b) => (a.frame ?? 0) - (b.frame ?? 0) || a.rank - b.rank || a.file.localeCompare(b.file));
    out[id] = items.map(({ file: _file, ...shot }) => shot);
  }
  return out;
}

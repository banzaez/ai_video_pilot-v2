import fs from "node:fs";
import path from "node:path";
import { camerasDir, mapsDir, videoDir } from "./config.js";
import { readJsonFile } from "./common.js";

export function listFloorplans(): { name: string; url: string }[] {
  if (!fs.existsSync(mapsDir)) return [];
  return fs
    .readdirSync(mapsDir)
    .filter((f) => /\.(svg|png|jpe?g|webp)$/i.test(f))
    .sort()
    .map((name) => ({ name, url: `/maps/${encodeURIComponent(name)}` }));
}

export function cameraHomoPath(key: string): string {
  const direct = path.join(camerasDir, `${key}.json`);
  if (fs.existsSync(direct)) return direct;
  const num = Number(key);
  if (!Number.isNaN(num)) {
    const pad3 = path.join(camerasDir, `${String(num).padStart(3, "0")}.json`);
    if (fs.existsSync(pad3)) return pad3;
    const pad2 = path.join(camerasDir, `${String(num).padStart(2, "0")}.json`);
    if (fs.existsSync(pad2)) return pad2;
  }
  return direct;
}

export function cameraKeyFromName(video: string): string {
  const m = /Cam(?:era)?[_-]?(\d+)/i.exec(video);
  if (m) {
    const num = Number(m[1]);
    const pad3 = `${String(num).padStart(3, "0")}.json`;
    if (fs.existsSync(path.join(camerasDir, pad3))) return String(num).padStart(3, "0");
    const pad2 = `${String(num).padStart(2, "0")}.json`;
    if (fs.existsSync(path.join(camerasDir, pad2))) return String(num).padStart(2, "0");
    return String(num).padStart(3, "0");
  }
  return video.replace(/\.[^.]+$/, "") || "000";
}

export function mapsConfig() {
  const preferred = "grid";
  const floorplans = [
    { name: "grid", url: "" },
    ...listFloorplans().filter((f) => f.name !== "grid"),
  ];

  const homoMeta = new Map<
    string,
    {
      pairs: number;
      hasH: boolean;
      placement: Record<string, unknown> | null;
      map_points: { index: number; map: [number, number] }[];
    }
  >();
  if (fs.existsSync(camerasDir)) {
    for (const f of fs.readdirSync(camerasDir).sort()) {
      if (!/\.json$/i.test(f)) continue;
      const key = f.replace(/\.json$/i, "");
      const data = readJsonFile(cameraHomoPath(key)) as {
        pairs?: { map?: unknown }[];
        H?: unknown;
        placement?: Record<string, unknown> | null;
      } | null;
      const pl = data?.placement;
      const hasPl =
        pl &&
        typeof pl === "object" &&
        Array.isArray((pl as { position?: unknown }).position) &&
        ((pl as { position: unknown[] }).position?.length ?? 0) >= 2;
      const map_points: { index: number; map: [number, number] }[] = [];
      if (Array.isArray(data?.pairs)) {
        data!.pairs!.forEach((p, index) => {
          const m = p?.map;
          if (!Array.isArray(m) || m.length < 2) return;
          const x = Number(m[0]);
          const y = Number(m[1]);
          if (!Number.isFinite(x) || !Number.isFinite(y)) return;
          map_points.push({ index, map: [x, y] });
        });
      }
      homoMeta.set(key, {
        pairs: Array.isArray(data?.pairs) ? data!.pairs!.length : 0,
        hasH: Array.isArray(data?.H) && (data!.H as unknown[]).length === 9,
        placement: hasPl ? pl! : null,
        map_points,
      });
    }
  }

  const byKey = new Map<
    string,
    {
      key: string;
      camera_index: number | null;
      video: string | null;
      videoUrl: string | null;
      pairs: number;
      hasH: boolean;
      hasPlacement: boolean;
      placement: Record<string, unknown> | null;
      map_points: { index: number; map: [number, number] }[];
    }
  >();

  const files = fs.existsSync(videoDir) ? fs.readdirSync(videoDir) : [];
  for (const video of files.filter((f) => /\.(mp4|webm|mov|mkv)$/i.test(f)).sort()) {
    const key = cameraKeyFromName(video);
    if (byKey.has(key)) continue;
    const meta = homoMeta.get(key);
    const idx = Number(key);
    byKey.set(key, {
      key,
      camera_index: Number.isFinite(idx) ? idx : null,
      video,
      videoUrl: `/media/${encodeURIComponent(video)}`,
      pairs: meta?.pairs ?? 0,
      hasH: meta?.hasH ?? false,
      hasPlacement: !!meta?.placement,
      placement: meta?.placement ?? null,
      map_points: meta?.map_points ?? [],
    });
  }

  for (const [key, meta] of homoMeta) {
    if (byKey.has(key)) continue;
    const idx = Number(key);
    byKey.set(key, {
      key,
      camera_index: Number.isFinite(idx) ? idx : null,
      video: null,
      videoUrl: null,
      pairs: meta.pairs,
      hasH: meta.hasH,
      hasPlacement: !!meta.placement,
      placement: meta.placement,
      map_points: meta.map_points,
    });
  }

  const cameras = [...byKey.values()].sort((a, b) => a.key.localeCompare(b.key, undefined, { numeric: true }));

  return {
    floorplan: preferred,
    floorplans,
    cameras,
  };
}

import fs from "node:fs";
import path from "node:path";
import { resultsDir, videoDir } from "./config.js";
import { groupBySessionKey, parseProdStem, type MediaSession, type SessionPart } from "../src/session.js";
import { readJsonFile, resultsUrl, workFile } from "./common.js";

export function discoverSessionsFromVideoDir(): MediaSession[] {
  if (!fs.existsSync(videoDir)) return [];
  const files = fs.readdirSync(videoDir);
  const parts = files
    .filter((f) => /\.(mp4|webm|mov|mkv)$/i.test(f) && f !== "lite")
    .map((f) => parseProdStem(f.replace(/\.[^.]+$/, "")))
    .filter((p): p is NonNullable<typeof p> => p != null);
  const grouped = groupBySessionKey(parts);
  const sessions: MediaSession[] = [];
  for (const [key, group] of grouped) {
    const infoPath = path.join(resultsDir, key, "info.json");
    const info = readJsonFile(infoPath) as {
      parts?: SessionPart[];
      duration_sec?: number;
      fps?: number;
    } | null;
    const manifestParts = info?.parts;
    const sessionParts: SessionPart[] = group.map((p, i) => {
      const mp = manifestParts?.[i];
      return {
        name: p.name,
        stem: p.stem,
        videoUrl: `/media/${encodeURIComponent(p.name)}`,
        started_at: p.started_at,
        ended_at: p.ended_at,
        frame_offset: mp?.frame_offset ?? 0,
        frame_count: mp?.frame_count ?? 0,
        time_offset_sec: mp?.time_offset_sec ?? 0,
      };
    });
    const hasJson = fs.existsSync(workFile(key, "tracking.json"));
    sessions.push({
      key,
      camera: `Camera_${String(group[0]!.camera_index).padStart(2, "0")}`,
      camera_index: group[0]!.camera_index,
      day: group[0]!.day,
      parts: sessionParts,
      hasJson,
      jsonUrl: resultsUrl(key, "tracking.json"),
      feetUrl: resultsUrl(key, "feet.json"),
      duration_sec: typeof info?.duration_sec === "number" ? info.duration_sec : undefined,
      fps: typeof info?.fps === "number" ? info.fps : undefined,
    });
  }
  return sessions.sort((a, b) => a.key.localeCompare(b.key));
}

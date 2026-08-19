import fs from "node:fs";
import path from "node:path";
import { camerasDir } from "./config.js";
import { calibFingerprint } from "../src/calibFingerprint.js";
import { readArtifact, readJsonFile, refMatches, workFile } from "./common.js";

export const STAGE_ORDER = [
  "info",
  "detect",
  "tracklets",
  "pose",
  "feet",
  "tracklet_reid",
  "tracklet_link",
  "track",
] as const;

export type StageName = (typeof STAGE_ORDER)[number];

export const STAGE_FILES: Record<StageName, string> = {
  info: "info.json",
  detect: "detections.json",
  tracklets: "tracklet_frames.json",
  pose: "poses.json",
  feet: "feet.json",
  tracklet_reid: "tracklet_reid.json",
  tracklet_link: "tracklet_links.json",
  track: "tracking.json",
};

export const STAGE_PARENT: Record<StageName, StageName | null> = {
  info: null,
  detect: null,
  tracklets: "detect",
  pose: "tracklets",
  feet: "pose",
  tracklet_reid: "tracklets",
  tracklet_link: "tracklet_reid",
  track: "tracklet_link",
};

export type ExtraArtifact = {
  key: string;
  label: string;
  path: string;
  kind: "dir" | "file";
  exists: boolean;
  files: number | null;
  size: number | null;
  mtime: string | null;
  note: string | null;
};

export function fileArtifactStats(fp: string): {
  exists: boolean;
  size: number | null;
  mtime: string | null;
} {
  if (!fs.existsSync(fp)) return { exists: false, size: null, mtime: null };
  const st = fs.statSync(fp);
  if (!st.isFile()) return { exists: false, size: null, mtime: null };
  return { exists: true, size: st.size, mtime: new Date(st.mtimeMs).toISOString() };
}

export function extraFile(base: string, name: string, label = name): ExtraArtifact {
  const st = fileArtifactStats(workFile(base, name));
  return {
    key: name,
    label,
    path: name,
    kind: "file",
    exists: st.exists,
    files: null,
    size: st.size,
    mtime: st.mtime,
    note: st.exists ? null : "нет файла",
  };
}

export function resolveParent(stage: StageName, base: string): StageName | null {
  let parent = STAGE_PARENT[stage];
  if (stage === "track" && parent === "tracklet_link") {
    if (!fs.existsSync(workFile(base, STAGE_FILES.tracklet_link))) parent = "detect";
  }
  return parent;
}

export function dirArtifactStats(dirPath: string): {
  exists: boolean;
  files: number;
  size: number | null;
  mtime: string | null;
} {
  if (!fs.existsSync(dirPath)) {
    return { exists: false, files: 0, size: null, mtime: null };
  }
  const root = fs.statSync(dirPath);
  if (!root.isDirectory()) {
    return { exists: false, files: 0, size: null, mtime: null };
  }
  let files = 0;
  let size = 0;
  for (const name of fs.readdirSync(dirPath)) {
    try {
      const st = fs.statSync(path.join(dirPath, name));
      if (st.isFile()) {
        files += 1;
        size += st.size;
      }
    } catch {
      /* skip */
    }
  }
  return {
    exists: true,
    files,
    size,
    mtime: new Date(root.mtimeMs).toISOString(),
  };
}

export function cameraKeyFromWorkDir(base: string): string {
  const info = readJsonFile(workFile(base, "info.json")) as { camera_index?: number } | null;
  if (info?.camera_index != null && Number.isFinite(Number(info.camera_index))) {
    return String(Number(info.camera_index)).padStart(2, "0");
  }
  const m = /^(\d{2})_/.exec(base);
  return m?.[1] ?? "01";
}

export function liveFeetFingerprint(
  base: string,
  feetData: { torso_height_m?: number; person_height_m?: number; tracking_size?: number[] } | null,
): string | null {
  const cam = cameraKeyFromWorkDir(base);
  const cameraDoc = readJsonFile(path.join(camerasDir, `${cam}.json`)) as Record<string, unknown> | null;
  const tracking = readJsonFile(workFile(base, "tracking.json")) as { width?: number; height?: number } | null;
  const trackingSize =
    Array.isArray(feetData?.tracking_size) && feetData.tracking_size.length >= 2
      ? feetData.tracking_size
      : tracking?.width && tracking?.height
        ? [tracking.width, tracking.height]
        : null;
  return calibFingerprint({
    cameraKey: cam,
    cameraDoc,
    torsoHeightM: Number(feetData?.torso_height_m ?? 0),
    trackingSize,
    personHeightM: Number(feetData?.person_height_m ?? 1.7),
  });
}

export function staleStagesReport(base: string) {
  const stages: Record<
    string,
    {
      stage: string;
      file: string;
      exists: boolean;
      file_version: number | null;
      written_at: string | null;
      size: number | null;
      mtime: string | null;
      stale: boolean;
      reason: string | null;
    }
  > = {};
  const stale: string[] = [];

  for (const stage of STAGE_ORDER) {
    const file = STAGE_FILES[stage];
    const fp = workFile(base, file);
    const exists = fs.existsSync(fp);
    const art = exists ? readArtifact(fp) : null;
    const st = exists ? fs.statSync(fp) : null;
    const entry = {
      stage,
      file,
      exists,
      file_version: art?.file_version ?? null,
      written_at: art?.written_at ?? null,
      size: st?.size ?? null,
      mtime: st ? new Date(st.mtimeMs).toISOString() : null,
      stale: false,
      reason: null as string | null,
    };
    if (!exists) {
      stages[stage] = entry;
      continue;
    }
    const parent = resolveParent(stage, base);
    if (parent) {
      const parentPath = workFile(base, STAGE_FILES[parent]);
      if (!fs.existsSync(parentPath)) {
        entry.stale = true;
        entry.reason = `нет родителя ${STAGE_FILES[parent]}`;
      } else {
        const parentArt = readArtifact(parentPath);
        const recorded = art?.inputs?.[parent];
        if (recorded && refMatches(recorded, parentPath, parentArt)) {
          // ok
        } else if (!art) {
          const parentNewer = fs.statSync(parentPath).mtimeMs > fs.statSync(fp).mtimeMs + 10;
          if (parentNewer) {
            entry.stale = true;
            entry.reason = `${STAGE_FILES[parent]} новее (mtime)`;
          }
        } else {
          entry.stale = true;
          entry.reason = `не совпадает с ${STAGE_FILES[parent]} — пересчитать`;
        }
      }
    }
    if (stage === "feet" && !entry.stale) {
      const feetData = readJsonFile(fp) as {
        calibration?: { fingerprint?: string };
        torso_height_m?: number;
        tracking_size?: number[];
      } | null;
      const stored = feetData?.calibration?.fingerprint;
      const live = liveFeetFingerprint(base, feetData);
      if (stored && live && stored !== live) {
        entry.stale = true;
        entry.reason = "калибровка изменилась — пересчитать feet";
      }
    }
    if (entry.stale) stale.push(stage);
    stages[stage] = entry;
  }

  const staleSet = new Set(stale);
  for (const stage of STAGE_ORDER) {
    const parent = resolveParent(stage, base);
    if (parent && staleSet.has(parent) && stages[stage].exists && !staleSet.has(stage)) {
      stages[stage].stale = true;
      stages[stage].reason = stages[stage].reason || `устарел родитель ${parent}`;
      staleSet.add(stage);
    }
  }

  const tlCrops = dirArtifactStats(workFile(base, "tracklet_crops"));

  const staleOrdered = STAGE_ORDER.filter((s) => stages[s].stale);

  const recompute_from = staleOrdered[0] ?? null;
  const recompute_to = staleOrdered.length ? staleOrdered[staleOrdered.length - 1]! : null;
  let cli: string | null = null;
  if (recompute_from && recompute_to) {
    const sessionKeyRe = /^\d{2}_\d{8}$/;
    const inputArg = sessionKeyRe.test(base) ? ` --input session:${base}` : "";
    cli =
      recompute_from === recompute_to
        ? `python -m app.main${inputArg} --stage ${recompute_from}`
        : `python -m app.main${inputArg} --from ${recompute_from} --to ${recompute_to}`;
  }

  const extras: Record<string, ExtraArtifact[]> = {
    tracklets: [extraFile(base, "tracklets.json")],
    tracklet_reid: [
      extraFile(base, "tracklet_reid.npz"),
      extraFile(base, "tracklet_pose_cache.json"),
      {
        key: "tracklet_crops",
        label: "tracklet_crops/",
        path: "tracklet_crops/",
        kind: "dir",
        exists: tlCrops.exists,
        files: tlCrops.files,
        size: tlCrops.size,
        mtime: tlCrops.mtime,
        note: tlCrops.exists ? null : "JPG только при save_crops",
      },
    ],
    track: [extraFile(base, "tracks.json")],
  };

  return { stages, stale: [...staleOrdered], recompute_from, recompute_to, cli, extras };
}

import { formatEntityId, groupId, parseEntityIdOptional } from "../src/entityId.js";
import { readJsonFile, workFile } from "./common.js";
import type { MergeTimelinePair } from "./merges.js";

type CameraLinkEdgeRaw = {
  from?: number;
  to?: number;
  a?: number;
  b?: number;
  score?: number;
  face?: number | null;
  face_scores?: Record<string, number> | null;
  pose_face?: number | null;
  reid?: number | null;
  motion?: number | null;
  dist_m?: number | null;
  dist?: number | null;
  gap_sec?: number | null;
  gap?: number | null;
  pass?: number | null;
  reason?: string | null;
  space?: string | null;
};

type FaceEntryRaw = {
  frame_index?: number;
  rank?: number;
  det_score?: number;
  pose_face_score?: number;
  pose_conf?: number;
  quality?: number;
  face_bbox?: number[];
  crop_file?: string;
  track_id?: number;
  entity?: string;
  group_id?: number;
  solo?: boolean;
};

function numOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function normalizeCameraLinkEdge(e: CameraLinkEdgeRaw): MergeTimelinePair | null {
  const a = typeof e.a === "number" ? e.a : e.from;
  const b = typeof e.b === "number" ? e.b : e.to;
  if (typeof a !== "number" || typeof b !== "number") return null;
  const faceScores =
    e.face_scores && typeof e.face_scores === "object" && !Array.isArray(e.face_scores)
      ? Object.fromEntries(
          Object.entries(e.face_scores).filter(
            (entry): entry is [string, number] => typeof entry[1] === "number" && Number.isFinite(entry[1]),
          ),
        )
      : null;
  return {
    a,
    b,
    score: typeof e.score === "number" && Number.isFinite(e.score) ? e.score : 0,
    reason: typeof e.reason === "string" && e.reason.trim() ? e.reason : null,
    reid: numOrNull(e.reid),
    face: numOrNull(e.face),
    face_scores: faceScores && Object.keys(faceScores).length ? faceScores : null,
    pose_face: numOrNull(e.pose_face),
    motion: numOrNull(e.motion),
    gap: numOrNull(e.gap_sec ?? e.gap),
    dist: numOrNull(e.dist_m ?? e.dist),
    space: typeof e.space === "string" ? e.space : "map",
    pass: typeof e.pass === "number" && e.pass >= 0 ? e.pass : null,
    pass2: null,
  };
}

function normalizeCameraLinkEdges(raw: CameraLinkEdgeRaw[] | undefined): MergeTimelinePair[] {
  if (!raw?.length) return [];
  const out: MergeTimelinePair[] = [];
  for (const e of raw) {
    const pair = normalizeCameraLinkEdge(e);
    if (pair) out.push(pair);
  }
  return out;
}

export type FaceShotOut = {
  url: string;
  rank: number;
  score: number | null;
  pose_score: number | null;
  quality: number | null;
  frame: number;
  t: number | null;
  bbox: number[] | null;
  model?: string | null;
  track_id?: number | null;
  entity?: string | null;
  solo?: boolean;
};

function fpsFor(base: string): number {
  const info = readJsonFile(workFile(base, "info.json")) as { fps?: number } | null;
  const fps = Number(info?.fps);
  return Number.isFinite(fps) && fps > 0 ? fps : 25;
}

function bucketKey(raw: string): string {
  const parsed = parseEntityIdOptional(raw);
  if (parsed) return formatEntityId(parsed);
  const n = Number(raw);
  if (Number.isInteger(n) && n > 0) return formatEntityId(groupId(n));
  return raw;
}

function toShots(
  base: string,
  entries: FaceEntryRaw[],
  fps: number,
  model?: string,
): FaceShotOut[] {
  return entries.map((e, i) => {
    const frame = e.frame_index ?? 0;
    const entity =
      parseEntityIdOptional(e.entity)?.space === "t"
        ? e.entity!
        : typeof e.track_id === "number" && e.track_id > 0
          ? formatEntityId({ space: "t", n: e.track_id })
          : null;
    return {
      url: e.crop_file
        ? `/api/face_crop/${encodeURIComponent(base)}/${encodeURIComponent(e.crop_file)}`
        : "",
      rank: typeof e.rank === "number" ? e.rank : i,
      score: typeof e.det_score === "number" ? e.det_score : null,
      pose_score: typeof e.pose_face_score === "number" ? e.pose_face_score : null,
      quality: typeof e.quality === "number" ? e.quality : null,
      frame,
      t: fps > 0 ? frame / fps : null,
      bbox: e.face_bbox ?? null,
      model: model ?? null,
      track_id: typeof e.track_id === "number" ? e.track_id : null,
      entity,
      solo: Boolean(e.solo),
    };
  });
}

function putBucket(
  dest: Record<string, FaceShotOut[]>,
  rawKey: string,
  shots: FaceShotOut[],
): void {
  const key = bucketKey(rawKey);
  if (!shots.length) return;
  dest[key] = dest[key] ? dest[key].concat(shots) : shots;
}

function mapPrefixed(
  base: string,
  fps: number,
  raw: Record<string, FaceEntryRaw[]>,
  dest: Record<string, FaceShotOut[]>,
  model?: string,
): void {
  for (const [id, entries] of Object.entries(raw)) {
    putBucket(dest, id, toShots(base, entries, fps, model));
  }
}

export function faceGalleryFor(base: string): {
  faces: Record<string, FaceShotOut[]>;
  facesByModel: Record<string, Record<string, FaceShotOut[]>>;
  faceModels: string[];
} {
  const empty = {
    faces: {},
    facesByModel: {},
    faceModels: [] as string[],
  };
  const cfPath = workFile(base, "camera_face.json");
  const cf = readJsonFile(cfPath) as {
    models?: string[];
    faces?: Record<string, FaceEntryRaw[]>;
    faces_by_model?: Record<string, Record<string, FaceEntryRaw[]>>;
    groups?: Record<string, FaceEntryRaw[]>;
    tracks?: Record<string, FaceEntryRaw[]>;
    groups_by_model?: Record<string, Record<string, FaceEntryRaw[]>>;
    tracks_by_model?: Record<string, Record<string, FaceEntryRaw[]>>;
  } | null;
  if (!cf) return empty;

  const fps = fpsFor(base);
  const models = cf.models ?? [];
  const faces: Record<string, FaceShotOut[]> = {};
  const facesByModel: Record<string, Record<string, FaceShotOut[]>> = {};

  if (cf.faces && Object.keys(cf.faces).length) {
    mapPrefixed(base, fps, cf.faces, faces);
  } else {
    mapPrefixed(base, fps, cf.groups ?? {}, faces);
    mapPrefixed(base, fps, cf.tracks ?? {}, faces);
  }

  if (cf.faces_by_model && Object.keys(cf.faces_by_model).length) {
    for (const [model, bucket] of Object.entries(cf.faces_by_model)) {
      facesByModel[model] = {};
      mapPrefixed(base, fps, bucket, facesByModel[model], model);
    }
  } else {
    const modelsSrc = new Set([
      ...Object.keys(cf.groups_by_model ?? {}),
      ...Object.keys(cf.tracks_by_model ?? {}),
    ]);
    for (const model of modelsSrc) {
      facesByModel[model] = {};
      mapPrefixed(base, fps, cf.groups_by_model?.[model] ?? {}, facesByModel[model], model);
      mapPrefixed(base, fps, cf.tracks_by_model?.[model] ?? {}, facesByModel[model], model);
    }
  }

  return { faces, facesByModel, faceModels: models };
}

export function cameraLinkFor(base: string): {
  face_models?: string[];
  edges?: MergeTimelinePair[];
  candidate_edges?: MergeTimelinePair[];
} | null {
  const clPath = workFile(base, "camera_links.json");
  const cl = readJsonFile(clPath) as {
    face_models?: string[];
    edges?: CameraLinkEdgeRaw[];
    candidate_edges?: CameraLinkEdgeRaw[];
  } | null;
  if (!cl) return null;
  return {
    face_models: cl.face_models,
    edges: normalizeCameraLinkEdges(cl.edges),
    candidate_edges: normalizeCameraLinkEdges(cl.candidate_edges),
  };
}

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
  solo?: boolean;
};

function toShots(
  base: string,
  entries: FaceEntryRaw[],
  model?: string,
): FaceShotOut[] {
  return entries.map((e, i) => ({
    url: e.crop_file
      ? `/api/face_crop/${encodeURIComponent(base)}/${encodeURIComponent(e.crop_file)}`
      : "",
    rank: typeof e.rank === "number" ? e.rank : i,
    score: typeof e.det_score === "number" ? e.det_score : null,
    pose_score: typeof e.pose_face_score === "number" ? e.pose_face_score : null,
    quality: typeof e.quality === "number" ? e.quality : null,
    frame: e.frame_index ?? 0,
    t: null,
    bbox: e.face_bbox ?? null,
    model: model ?? null,
    track_id: typeof e.track_id === "number" ? e.track_id : null,
    solo: Boolean(e.solo),
  }));
}

function mapFaceBuckets(
  base: string,
  groups: Record<string, FaceEntryRaw[]>,
  byModel: Record<string, Record<string, FaceEntryRaw[]>>,
): {
  faces: Record<string, FaceShotOut[]>;
  facesByModel: Record<string, Record<string, FaceShotOut[]>>;
} {
  const faces: Record<string, FaceShotOut[]> = {};
  for (const [id, entries] of Object.entries(groups)) {
    faces[id] = toShots(base, entries);
  }
  const facesByModel: Record<string, Record<string, FaceShotOut[]>> = {};
  for (const [model, bucket] of Object.entries(byModel)) {
    facesByModel[model] = {};
    for (const [id, entries] of Object.entries(bucket)) {
      facesByModel[model][id] = toShots(base, entries, model);
    }
  }
  return { faces, facesByModel };
}

export function faceGalleryFor(base: string): {
  faces: Record<string, FaceShotOut[]>;
  facesByModel: Record<string, Record<string, FaceShotOut[]>>;
  groupFaces: Record<string, FaceShotOut[]>;
  trackFaces: Record<string, FaceShotOut[]>;
  groupFacesByModel: Record<string, Record<string, FaceShotOut[]>>;
  trackFacesByModel: Record<string, Record<string, FaceShotOut[]>>;
  faceModels: string[];
} {
  const empty = {
    faces: {},
    facesByModel: {},
    groupFaces: {},
    trackFaces: {},
    groupFacesByModel: {},
    trackFacesByModel: {},
    faceModels: [] as string[],
  };
  const cfPath = workFile(base, "camera_face.json");
  const cf = readJsonFile(cfPath) as {
    models?: string[];
    groups?: Record<string, FaceEntryRaw[]>;
    tracks?: Record<string, FaceEntryRaw[]>;
    groups_by_model?: Record<string, Record<string, FaceEntryRaw[]>>;
    tracks_by_model?: Record<string, Record<string, FaceEntryRaw[]>>;
  } | null;
  if (!cf) return empty;

  const models = cf.models ?? [];
  const groupRaw = cf.groups ?? {};
  const trackRaw = cf.tracks ?? {};
  const groupByModelRaw = cf.groups_by_model ?? {};
  const trackByModelRaw = cf.tracks_by_model ?? {};

  const groupBucket = mapFaceBuckets(base, groupRaw, groupByModelRaw);
  const trackBucket = mapFaceBuckets(base, trackRaw, trackByModelRaw);

  const faces: Record<string, FaceShotOut[]> = { ...groupBucket.faces, ...trackBucket.faces };
  const facesByModel: Record<string, Record<string, FaceShotOut[]>> = {};
  for (const model of models) {
    facesByModel[model] = {
      ...(groupBucket.facesByModel[model] ?? {}),
      ...(trackBucket.facesByModel[model] ?? {}),
    };
  }

  return {
    faces,
    facesByModel,
    groupFaces: groupBucket.faces,
    trackFaces: trackBucket.faces,
    groupFacesByModel: groupBucket.facesByModel,
    trackFacesByModel: trackBucket.facesByModel,
    faceModels: models,
  };
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

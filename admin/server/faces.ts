import { readJsonFile, workFile } from "./common.js";

export type FaceShotOut = {
  url: string;
  rank: number;
  score: number | null;
  frame: number;
  t: number | null;
  bbox: number[] | null;
  model?: string | null;
};

export function faceGalleryFor(base: string): {
  faces: Record<string, FaceShotOut[]>;
  facesByModel: Record<string, Record<string, FaceShotOut[]>>;
  faceModels: string[];
} {
  const cfPath = workFile(base, "camera_face.json");
  const cf = readJsonFile(cfPath) as {
    models?: string[];
    groups?: Record<string, { frame_index?: number; det_score?: number; face_bbox?: number[]; crop_file?: string }[]>;
    groups_by_model?: Record<string, Record<string, { frame_index?: number; det_score?: number; face_bbox?: number[]; crop_file?: string }[]>>;
  } | null;
  if (!cf) return { faces: {}, facesByModel: {}, faceModels: [] };

  const models = cf.models ?? [];
  const primaryGroups = cf.groups ?? {};
  const byModelRaw = cf.groups_by_model ?? {};

  const toShots = (
    _gid: string,
    entries: { frame_index?: number; det_score?: number; face_bbox?: number[]; crop_file?: string }[],
    model?: string,
  ): FaceShotOut[] =>
    entries.map((e, i) => ({
      url: e.crop_file
        ? `/api/face_crop/${encodeURIComponent(base)}/${encodeURIComponent(e.crop_file)}`
        : "",
      rank: i,
      score: typeof e.det_score === "number" ? e.det_score : null,
      frame: e.frame_index ?? 0,
      t: null,
      bbox: e.face_bbox ?? null,
      model: model ?? null,
    }));

  const faces: Record<string, FaceShotOut[]> = {};
  for (const [gid, entries] of Object.entries(primaryGroups)) {
    faces[gid] = toShots(gid, entries);
  }

  const facesByModel: Record<string, Record<string, FaceShotOut[]>> = {};
  for (const [model, groups] of Object.entries(byModelRaw)) {
    facesByModel[model] = {};
    for (const [gid, entries] of Object.entries(groups)) {
      facesByModel[model][gid] = toShots(gid, entries, model);
    }
  }

  return { faces, facesByModel, faceModels: models };
}

export function cameraLinkFor(base: string): {
  face_models?: string[];
  edges?: Record<string, unknown>[];
  candidate_edges?: Record<string, unknown>[];
} | null {
  const clPath = workFile(base, "camera_links.json");
  const cl = readJsonFile(clPath) as {
    face_models?: string[];
    edges?: Record<string, unknown>[];
    candidate_edges?: Record<string, unknown>[];
  } | null;
  if (!cl) return null;
  return {
    face_models: cl.face_models,
    edges: cl.edges,
    candidate_edges: cl.candidate_edges,
  };
}

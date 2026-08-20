/** 3D-поза камеры: луч через пиксель → пол (ground plane). */

import type { CameraPlacement, HomoPair, Mat3, Pt } from "./homography";
import { applyHomography, feetFromBbox, preferHomographyOverRay } from "./homography";
import { FLOOR_ORIGIN, METER_PX } from "./mapGrid";

export type CameraPose = CameraPlacement & {
  height_m: number;
  pitch_deg: number;
};

export type RayFeetSource =
  | "ray"
  | "ray_head"
  | "ray_feet"
  | "h_bbox"
  | "kpt_lsq"
  | "kpt_ankle"
  | "kpt_hip"
  | "kpt_head"
  | "kpt_interp";

export type RayProjectResult = {
  map: Pt;
  source: RayFeetSource;
  confidence: number;
};

export type RayPairStats = {
  rmsPx: number;
  projected: number;
  total: number;
};

export type FitRayPoseResult = {
  height_m: number;
  pitch_deg: number;
  fov_deg: number;
  yaw_deg: number;
  position: Pt;
  rmsPx: number;
  projected: number;
  total: number;
};

const DEFAULT_HEIGHT_M = 3.0;
const DEFAULT_PITCH_DEG = 35.0;
/** z=0 — пол; калибровка пар и проекция ног на одной плоскости */
const DEFAULT_TORSO_H_M = 0;
export const DEFAULT_PERSON_H_M = 1.7;
export const MISS_PENALTY_PX = 5000;
export const KPT_MIN_DEFAULT = 0.25;
const COCO_NOSE = 0;
const COCO_L_HIP = 11;
const COCO_R_HIP = 12;
const COCO_L_ANKLE = 15;
const COCO_R_ANKLE = 16;
const Z_FRAC_ANKLE = 0.03;
const Z_FRAC_HIP = 0.55;
const Z_FRAC_NOSE = 0.94;
const TRUNC_ABS_M = 0.4;
const TRUNC_REL = 0.15;

const FIT_H_MIN = 1.0;
const FIT_H_MAX = 6.0;
const FIT_PITCH_MIN = 0;
const FIT_PITCH_MAX = 85;
const FIT_FOV_MIN = 40;
const FIT_FOV_MAX = 140;
const FIT_YAW_SPAN = 20;
const FIT_POS_SPAN = 600;
const FIT_MAX_ITER = 200;

export function normalizeCameraPose(
  raw: unknown,
  fallback?: Partial<CameraPose>,
): CameraPose | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const pos = o.position;
  if (!Array.isArray(pos) || pos.length < 2) return null;
  const x = Number(pos[0]);
  const y = Number(pos[1]);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  const yaw = Number(o.yaw_deg);
  const fov = Number(o.fov_deg);
  const height = Number(o.height_m ?? fallback?.height_m ?? DEFAULT_HEIGHT_M);
  const pitch = Number(o.pitch_deg ?? fallback?.pitch_deg ?? DEFAULT_PITCH_DEG);
  return {
    position: [x, y],
    yaw_deg: Number.isFinite(yaw) ? ((yaw % 360) + 360) % 360 : 0,
    fov_deg: Number.isFinite(fov) ? Math.min(160, Math.max(20, fov)) : 70,
    height_m: Number.isFinite(height) ? Math.max(0.5, height) : DEFAULT_HEIGHT_M,
    pitch_deg: Number.isFinite(pitch) ? Math.min(89, Math.max(0, pitch)) : DEFAULT_PITCH_DEG,
  };
}

export function pinholeK(imageSize: [number, number], fovDeg: number): [number, number, number, number] {
  const [w, h] = imageSize;
  const fov = (Math.max(20, Math.min(160, fovDeg)) * Math.PI) / 180;
  const fx = w / 2 / Math.tan(fov / 2);
  const fy = fx;
  return [fx, fy, w / 2, h / 2];
}

function mapPxToMeters(map: Pt): Pt {
  const [ox, oy] = FLOOR_ORIGIN;
  return [(map[0] - ox) / METER_PX, (map[1] - oy) / METER_PX];
}

function metersToMapPx(mx: number, my: number): Pt {
  const [ox, oy] = FLOOR_ORIGIN;
  return [ox + mx * METER_PX, oy + my * METER_PX];
}

function lookBasis(yawDeg: number, pitchDeg: number): { look: Pt3; right: Pt3; down: Pt3 } {
  const yaw = (yawDeg * Math.PI) / 180;
  const pitch = (pitchDeg * Math.PI) / 180;
  const fwdX = Math.cos(yaw);
  const fwdY = Math.sin(yaw);
  const look: Pt3 = [Math.cos(pitch) * fwdX, Math.cos(pitch) * fwdY, -Math.sin(pitch)];
  const right: Pt3 = [-Math.sin(yaw), Math.cos(yaw), 0];
  const down: Pt3 = [
    right[1] * look[2] - right[2] * look[1],
    right[2] * look[0] - right[0] * look[2],
    right[0] * look[1] - right[1] * look[0],
  ];
  const dl = Math.hypot(down[0], down[1], down[2]) || 1;
  return { look, right, down: [down[0] / dl, down[1] / dl, down[2] / dl] };
}

type Pt3 = [number, number, number];

function normalize3(v: Pt3): Pt3 | null {
  const l = Math.hypot(v[0], v[1], v[2]);
  if (l < 1e-9) return null;
  return [v[0] / l, v[1] / l, v[2] / l];
}

/** Луч из камеры через пиксель → точка на z=0 (метры пола → px плана). */
export function rayToGroundMap(
  px: number,
  py: number,
  pose: CameraPose,
  imageSize: [number, number],
  opts?: { torsoHeightM?: number },
): Pt | null {
  const torsoH = opts?.torsoHeightM ?? DEFAULT_TORSO_H_M;
  const [fx, fy, cx, cy] = pinholeK(imageSize, pose.fov_deg);
  const [camMx, camMy] = mapPxToMeters(pose.position);
  const camZ = pose.height_m;
  const { look, right, down } = lookBasis(pose.yaw_deg, pose.pitch_deg);
  const u = (px - cx) / fx;
  const v = (py - cy) / fy;
  const dir = normalize3([
    look[0] + u * right[0] + v * down[0],
    look[1] + u * right[1] + v * down[1],
    look[2] + u * right[2] + v * down[2],
  ]);
  if (!dir || dir[2] >= -1e-6) return null;

  const tTorso = (torsoH - camZ) / dir[2];
  if (tTorso <= 0) return null;
  const tx = camMx + tTorso * dir[0];
  const ty = camMy + tTorso * dir[1];
  return metersToMapPx(tx, ty);
}

export function rayPairStats(
  pose: CameraPose,
  pairs: HomoPair[],
  imageSize: [number, number],
): RayPairStats | null {
  if (!pairs.length || !imageSize[0] || !imageSize[1]) return null;
  let total = 0;
  let projected = 0;
  let sumSq = 0;
  for (const pair of pairs) {
    total += 1;
    const mapped = rayToGroundMap(pair.image[0], pair.image[1], pose, imageSize, { torsoHeightM: 0 });
    const err = mapped
      ? Math.hypot(mapped[0] - pair.map[0], mapped[1] - pair.map[1])
      : MISS_PENALTY_PX;
    if (mapped) projected += 1;
    sumSq += err * err;
  }
  if (total < 2) return null;
  return { rmsPx: Math.sqrt(sumSq / total), projected, total };
}

function wrapDeg(deg: number): number {
  return ((deg % 360) + 360) % 360;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function yawDelta(a: number, b: number): number {
  let d = (((a - b) % 360) + 360) % 360;
  if (d > 180) d -= 360;
  return d;
}

function clampPose(trial: CameraPose, origin: CameraPose, fitPose: boolean): CameraPose {
  let yaw = origin.yaw_deg;
  let pos: Pt = origin.position;
  if (fitPose) {
    const dYaw = clamp(yawDelta(trial.yaw_deg, origin.yaw_deg), -FIT_YAW_SPAN, FIT_YAW_SPAN);
    yaw = wrapDeg(origin.yaw_deg + dYaw);
    pos = [
      clamp(trial.position[0], origin.position[0] - FIT_POS_SPAN, origin.position[0] + FIT_POS_SPAN),
      clamp(trial.position[1], origin.position[1] - FIT_POS_SPAN, origin.position[1] + FIT_POS_SPAN),
    ];
  }
  return {
    position: pos,
    yaw_deg: yaw,
    fov_deg: clamp(trial.fov_deg, FIT_FOV_MIN, FIT_FOV_MAX),
    height_m: clamp(trial.height_m, FIT_H_MIN, FIT_H_MAX),
    pitch_deg: clamp(trial.pitch_deg, FIT_PITCH_MIN, FIT_PITCH_MAX),
  };
}

type FitAxis = "height_m" | "pitch_deg" | "fov_deg" | "yaw_deg" | "x" | "y";

function nudge(pose: CameraPose, axis: FitAxis, delta: number): CameraPose {
  if (axis === "height_m") return { ...pose, height_m: pose.height_m + delta };
  if (axis === "pitch_deg") return { ...pose, pitch_deg: pose.pitch_deg + delta };
  if (axis === "fov_deg") return { ...pose, fov_deg: pose.fov_deg + delta };
  if (axis === "yaw_deg") return { ...pose, yaw_deg: wrapDeg(pose.yaw_deg + delta) };
  if (axis === "x") return { ...pose, position: [pose.position[0] + delta, pose.position[1]] };
  return { ...pose, position: [pose.position[0], pose.position[1] + delta] };
}

export function fitRayPose(
  pose: CameraPlacement,
  pairs: HomoPair[],
  imageSize: [number, number],
  opts?: { fitPose?: boolean },
): FitRayPoseResult | null {
  if (!pairs.length || !imageSize[0] || !imageSize[1]) return null;
  const origin = normalizeCameraPose(pose);
  if (!origin) return null;
  const fitPose = opts?.fitPose ?? true;

  let best: CameraPose = origin;
  let bestStats = rayPairStats(best, pairs, imageSize);
  if (!bestStats) return null;

  for (let fov = 60; fov <= 120; fov += 5) {
    for (let hi = 3; hi <= 10; hi++) {
      const h = hi * 0.5;
      for (let p = 10; p <= 75; p += 5) {
        const trial: CameraPose = {
          ...origin,
          height_m: h,
          pitch_deg: p,
          fov_deg: fov,
        };
        const stats = rayPairStats(trial, pairs, imageSize);
        if (stats && stats.rmsPx < bestStats.rmsPx) {
          best = trial;
          bestStats = stats;
        }
      }
    }
  }

  const axes: FitAxis[] = fitPose
    ? ["height_m", "pitch_deg", "fov_deg", "yaw_deg", "x", "y"]
    : ["height_m", "pitch_deg", "fov_deg"];
  const steps: Record<FitAxis, number> = {
    height_m: 0.4,
    pitch_deg: 4,
    fov_deg: 4,
    yaw_deg: 4,
    x: 160,
    y: 160,
  };
  const tols: Record<FitAxis, number> = {
    height_m: 0.02,
    pitch_deg: 0.25,
    fov_deg: 0.5,
    yaw_deg: 0.25,
    x: 10,
    y: 10,
  };

  for (let iter = 0; iter < FIT_MAX_ITER; iter++) {
    if (axes.every((a) => steps[a] <= tols[a])) break;
    let improved = false;
    for (const axis of axes) {
      for (const sign of [1, -1] as const) {
        const trial = clampPose(nudge(best, axis, sign * steps[axis]), origin, fitPose);
        const stats = rayPairStats(trial, pairs, imageSize);
        if (stats && stats.rmsPx < bestStats.rmsPx) {
          best = trial;
          bestStats = stats;
          improved = true;
        }
      }
    }
    if (!improved) {
      for (const a of axes) steps[a] *= 0.5;
    }
  }

  return {
    height_m: best.height_m,
    pitch_deg: best.pitch_deg,
    fov_deg: best.fov_deg,
    yaw_deg: best.yaw_deg,
    position: best.position,
    rmsPx: bestStats.rmsPx,
    projected: bestStats.projected,
    total: bestStats.total,
  };
}

export function camDistM(map: Pt, pose: CameraPose): number {
  return Math.hypot(map[0] - pose.position[0], map[1] - pose.position[1]) / METER_PX;
}

export function isTruncatedDual(pHead: Pt | null, pFeet: Pt | null, pose: CameraPose): boolean {
  if (!pHead) return false;
  if (!pFeet) return true;
  const dHead = camDistM(pHead, pose);
  const dFeet = camDistM(pFeet, pose);
  return dFeet > dHead + Math.max(TRUNC_ABS_M, TRUNC_REL * dHead);
}

export function dualPlaneFromBbox(
  bbox: number[],
  pose: CameraPose,
  imageSize: [number, number],
  personHeightM = DEFAULT_PERSON_H_M,
): { map: Pt | null; source: RayFeetSource; truncated: boolean; pHead: Pt | null; pFeet: Pt | null } {
  const [x1, y1, x2, y2] = bbox;
  const xMid = (x1 + x2) * 0.5;
  const yTop = Math.min(y1, y2);
  const yBot = Math.max(y1, y2);
  const pHead = rayToGroundMap(xMid, yTop, pose, imageSize, { torsoHeightM: personHeightM });
  const pFeet = rayToGroundMap(xMid, yBot, pose, imageSize, { torsoHeightM: 0 });
  const truncated = isTruncatedDual(pHead, pFeet, pose);
  if (truncated && pHead) return { map: pHead, source: "ray_head", truncated, pHead, pFeet };
  if (pFeet) return { map: pFeet, source: "ray_feet", truncated, pHead, pFeet };
  if (pHead) return { map: pHead, source: "ray_head", truncated, pHead, pFeet };
  return { map: null, source: "ray", truncated, pHead, pFeet };
}

function agreeConfidence(pHead: Pt | null, pFeet: Pt | null, pose: CameraPose): number {
  if (!pHead || !pFeet) return 0.55;
  const dHead = Math.max(camDistM(pHead, pose), 1);
  const dFeet = camDistM(pFeet, pose);
  const agree = 1 - Math.min(1, Math.abs(dFeet - dHead) / dHead);
  return Math.max(0.35, Math.min(1, agree));
}

function kptOk(kxy: number[][] | undefined, kcf: number[] | undefined, idx: number, kptMin: number): boolean {
  return !!kxy && !!kcf && idx < kxy.length && idx < kcf.length && kcf[idx]! >= kptMin && kxy[idx]!.length >= 2;
}

export function imageFeetFromKpts(
  kxy: number[][] | undefined,
  kcf: number[] | undefined,
  kptMin = KPT_MIN_DEFAULT,
): Pt | null {
  const ankles: Pt[] = [];
  for (const idx of [COCO_L_ANKLE, COCO_R_ANKLE]) {
    if (kptOk(kxy, kcf, idx, kptMin)) ankles.push([kxy![idx]![0], kxy![idx]![1]]);
  }
  if (ankles.length === 2) return [(ankles[0]![0] + ankles[1]![0]) / 2, (ankles[0]![1] + ankles[1]![1]) / 2];
  if (ankles.length === 1) return ankles[0]!;
  const hips: Pt[] = [];
  for (const idx of [COCO_L_HIP, COCO_R_HIP]) {
    if (kptOk(kxy, kcf, idx, kptMin)) hips.push([kxy![idx]![0], kxy![idx]![1]]);
  }
  if (hips.length === 2) return [(hips[0]![0] + hips[1]![0]) / 2, (hips[0]![1] + hips[1]![1]) / 2];
  if (hips.length === 1) return hips[0]!;
  return null;
}

export function projectKeypointsToMap(
  kxy: number[][] | undefined,
  kcf: number[] | undefined,
  pose: CameraPose,
  imageSize: [number, number],
  personHeightM = DEFAULT_PERSON_H_M,
  kptMin = KPT_MIN_DEFAULT,
): RayProjectResult | null {
  if (!kxy || !kcf) return null;
  const samples: { map: Pt; w: number; kind: string }[] = [];
  const add = (idx: number, zFrac: number, kind: string) => {
    if (!kptOk(kxy, kcf, idx, kptMin)) return;
    const mapped = rayToGroundMap(kxy[idx]![0], kxy[idx]![1], pose, imageSize, {
      torsoHeightM: zFrac * personHeightM,
    });
    if (mapped) samples.push({ map: mapped, w: kcf[idx]!, kind });
  };
  add(COCO_L_ANKLE, Z_FRAC_ANKLE, "ankle");
  add(COCO_R_ANKLE, Z_FRAC_ANKLE, "ankle");
  add(COCO_L_HIP, Z_FRAC_HIP, "hip");
  add(COCO_R_HIP, Z_FRAC_HIP, "hip");
  add(COCO_NOSE, Z_FRAC_NOSE, "nose");
  if (!samples.length) return null;
  const wsum = samples.reduce((s, x) => s + x.w, 0);
  if (wsum <= 0) return null;
  const mx = samples.reduce((s, x) => s + x.map[0] * x.w, 0) / wsum;
  const my = samples.reduce((s, x) => s + x.map[1] * x.w, 0) / wsum;
  const nAnkle = samples.filter((s) => s.kind === "ankle").length;
  const kinds = new Set(samples.map((s) => s.kind));
  let source: RayFeetSource = "kpt_head";
  let base = 0.72;
  if (nAnkle >= 2 && kinds.size > 1) {
    source = "kpt_lsq";
    base = 0.92;
  } else if (nAnkle >= 1) {
    source = "kpt_ankle";
    base = 0.88;
  } else if (kinds.has("hip")) {
    source = "kpt_hip";
    base = 0.78;
  }
  return { map: [mx, my], source, confidence: base };
}

export function projectBboxFeetToMap(
  bbox: number[],
  pose: CameraPose | null,
  imageSize: [number, number] | null,
  H: Mat3 | null,
  opts?: {
    torsoHeightM?: number;
    personHeightM?: number;
    hLooRms?: number | null;
    rayRms?: number | null;
    kxy?: number[][];
    kcf?: number[];
    kptMin?: number;
  },
): RayProjectResult | null {
  const personH = opts?.personHeightM ?? DEFAULT_PERSON_H_M;
  const kptMin = opts?.kptMin ?? KPT_MIN_DEFAULT;
  if (pose && imageSize && imageSize[0] > 0 && imageSize[1] > 0) {
    const kpt = projectKeypointsToMap(opts?.kxy, opts?.kcf, pose, imageSize, personH, kptMin);
    if (kpt) return kpt;
    const dual = dualPlaneFromBbox(bbox, pose, imageSize, personH);
    const agree = agreeConfidence(dual.pHead, dual.pFeet, pose);
    const useH =
      !dual.truncated && !!H && preferHomographyOverRay(opts?.hLooRms ?? null, opts?.rayRms ?? null);
    if (useH && H) {
      const feet = feetFromBbox(bbox);
      const mapped = applyHomography(H, feet[0], feet[1]);
      if (mapped) return { map: mapped, source: "h_bbox", confidence: 0.55 * agree };
    }
    if (dual.map) {
      const base = dual.source === "ray_feet" ? 0.85 : 0.72;
      return { map: dual.map, source: dual.source, confidence: base * agree };
    }
    const torso = opts?.torsoHeightM ?? DEFAULT_TORSO_H_M;
    if (torso && Math.abs(torso) > 1e-9) {
      const feet = feetFromBbox(bbox);
      const mapped = rayToGroundMap(feet[0], feet[1], pose, imageSize, { torsoHeightM: torso });
      if (mapped) return { map: mapped, source: "ray", confidence: 0.85 };
    }
  }
  if (H?.length === 9) {
    const feet = feetFromBbox(bbox);
    const mapped = applyHomography(H, feet[0], feet[1]);
    if (mapped) return { map: mapped, source: "h_bbox", confidence: 0.55 };
  }
  return null;
}


import type { CounterPoly } from "./counters";
import {
  DEFAULT_PERSON_H_M,
  normalizeCameraPose,
  projectBboxFeetToMap,
  type CameraPose,
  type RayFeetSource,
} from "./cameraPose";
import { applyHomography, feetFromBbox, type HomographyDoc, type Mat3, type Pt } from "./homography";
import type { Detection } from "./types";

/** px-отношение плечи→ноги / плечи→бёдра; оставлено для чтения старого body_calib. */
const FALLBACK_K = 2.0;
const MIN_BODY_CALIB_SAMPLES = 30;

export type BodyCalibBand = { y_max?: number; k: number };

export type BodyCalib = {
  source?: string;
  n_samples?: number;
  k_shoulder_to_feet?: number;
  bands?: {
    near?: BodyCalibBand;
    mid?: BodyCalibBand;
    far?: BodyCalibBand;
  };
  fallback_k?: number;
  updated_at?: string;
};

export type FeetSource = RayFeetSource | "bbox" | "none";

export type FeetResult = {
  pt: Pt;
  source: FeetSource;
  /** 0…1 — насколько доверять точке на плане */
  confidence: number;
  /** bbox-низ попал в зону прилавка на кадре */
  counterBlocked: boolean;
};

export function normalizeBodyCalib(raw: unknown): BodyCalib | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const band = (v: unknown): BodyCalibBand | undefined => {
    if (!v || typeof v !== "object") return undefined;
    const b = v as Record<string, unknown>;
    const k = Number(b.k);
    if (!Number.isFinite(k)) return undefined;
    const yMax = b.y_max != null ? Number(b.y_max) : undefined;
    return { k, ...(Number.isFinite(yMax!) ? { y_max: yMax } : {}) };
  };
  const bandsRaw = o.bands;
  let bands: BodyCalib["bands"];
  if (bandsRaw && typeof bandsRaw === "object") {
    const br = bandsRaw as Record<string, unknown>;
    bands = {
      near: band(br.near),
      mid: band(br.mid),
      far: band(br.far),
    };
  }
  const k = Number(o.k_shoulder_to_feet);
  const fb = Number(o.fallback_k);
  const n = Number(o.n_samples);
  return {
    source: typeof o.source === "string" ? o.source : undefined,
    n_samples: Number.isFinite(n) ? n : undefined,
    k_shoulder_to_feet: Number.isFinite(k) ? k : undefined,
    bands,
    fallback_k: Number.isFinite(fb) ? fb : FALLBACK_K,
    updated_at: typeof o.updated_at === "string" ? o.updated_at : undefined,
  };
}

export function bodyCalibTrusted(calib: BodyCalib | null | undefined): boolean {
  const n = calib?.n_samples;
  return typeof n === "number" && Number.isFinite(n) && n >= MIN_BODY_CALIB_SAMPLES;
}

export function pickKForShoulder(yShoulder: number, calib: BodyCalib | null | undefined): number {
  if (!bodyCalibTrusted(calib)) {
    return calib?.fallback_k ?? FALLBACK_K;
  }
  const c = calib!;
  const bands = c.bands;
  if (bands) {
    if (bands.near?.y_max != null && yShoulder <= bands.near.y_max) return bands.near.k;
    if (bands.mid?.y_max != null && yShoulder <= bands.mid.y_max) return bands.mid.k;
    if (bands.far) return bands.far.k;
  }
  if (c.k_shoulder_to_feet != null && c.k_shoulder_to_feet > 0) return c.k_shoulder_to_feet;
  return c.fallback_k ?? FALLBACK_K;
}

/** Ray casting — точка внутри полигона (кадр или план). */
export function pointInPolygon(p: Pt, poly: Pt[]): boolean {
  if (poly.length < 3) return false;
  const [x, y] = p;
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i]![0];
    const yi = poly[i]![1];
    const xj = poly[j]![0];
    const yj = poly[j]![1];
    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

/** Низ bbox попадает в контур прилавка на кадре этой камеры. */
export function bboxFeetInCounterZone(
  bbox: number[],
  counters: CounterPoly[],
  cameraKey: string,
): boolean {
  if (!counters.length || !cameraKey) return false;
  const feet = feetFromBbox(bbox);
  for (const c of counters) {
    const poly = c.image_by_camera?.[cameraKey];
    if (!poly || poly.length < 3) continue;
    if (pointInPolygon(feet, poly)) return true;
    const [x1, , x2, y2] = bbox;
    const mid = [(x1 + x2) / 2, y2] as Pt;
    if (pointInPolygon(mid, poly)) return true;
  }
  return false;
}

/** Ближайшая точка на ребре полигона (для выталкивания с прилавка на плане). */
export function nearestPointOnPolygonBoundary(p: Pt, poly: Pt[]): Pt {
  if (poly.length < 2) return p;
  let best: Pt = poly[0]!;
  let bestD = Infinity;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i]!;
    const b = poly[(i + 1) % poly.length]!;
    const q = nearestPointOnSegment(p, a, b);
    const d = Math.hypot(q[0] - p[0], q[1] - p[1]);
    if (d < bestD) {
      bestD = d;
      best = q;
    }
  }
  return best;
}

function nearestPointOnSegment(p: Pt, a: Pt, b: Pt): Pt {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  if (len2 < 1e-9) return a;
  let t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  return [a[0] + t * dx, a[1] + t * dy];
}

/** Если точка на плане внутри map-контура прилавка — сдвинуть к ближайшему краю. */
export function adjustMapPointForCounters(mapPt: Pt, counters: CounterPoly[]): Pt {
  for (const c of counters) {
    if (c.map.length < 3) continue;
    if (!pointInPolygon(mapPt, c.map)) continue;
    return nearestPointOnPolygonBoundary(mapPt, c.map);
  }
  return mapPt;
}

export function resolveFeetPoint(opts: {
  bbox: number[];
  cameraKey?: string;
}): FeetResult | null {
  const { bbox } = opts;
  return {
    pt: feetFromBbox(bbox),
    source: "bbox",
    confidence: 0.4,
    counterBlocked: false,
  };
}

export function scaleBboxToImageSize(
  bbox: number[],
  trackingSize: [number, number] | null | undefined,
  imageSize: [number, number] | null | undefined,
): number[] {
  if (!trackingSize || !imageSize) return bbox;
  const [tw, th] = trackingSize;
  const [iw, ih] = imageSize;
  if (tw <= 0 || th <= 0 || (tw === iw && th === ih)) return bbox;
  const sx = iw / tw;
  const sy = ih / th;
  return [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy];
}

export function projectFeetToMap(
  det: Detection,
  _frameIndex: number,
  H: Mat3 | null,
  opts: {
    cameraKey: string;
    homography?: HomographyDoc | null;
    torsoHeightM?: number;
    personHeightM?: number;
    trackingSize?: [number, number] | null;
    kxy?: number[][];
    kcf?: number[];
    kptMin?: number;
    hLooRms?: number | null;
    rayRms?: number | null;
  },
): { map: Pt; source: FeetSource; confidence: number } | null {
  const doc = opts.homography;
  const imageSize = doc?.image_size ?? null;
  const bbox = scaleBboxToImageSize(det.bbox, opts.trackingSize, imageSize);
  const feet = resolveFeetPoint({ bbox, cameraKey: opts.cameraKey });
  if (!feet) return null;

  const pose: CameraPose | null =
    doc?.placement != null ? normalizeCameraPose(doc.placement) : null;

  const projected = projectBboxFeetToMap(bbox, pose, imageSize, H, {
    torsoHeightM: opts.torsoHeightM,
    personHeightM: opts.personHeightM ?? DEFAULT_PERSON_H_M,
    kxy: opts.kxy,
    kcf: opts.kcf,
    kptMin: opts.kptMin,
    hLooRms: opts.hLooRms,
    rayRms: opts.rayRms,
  });

  let mapped: Pt | null = null;
  let source: FeetSource = feet.source;
  let confidence = feet.confidence;

  if (projected) {
    mapped = projected.map;
    source = projected.source;
    confidence = projected.confidence;
  } else if (H?.length === 9) {
    mapped = applyHomography(H, feet.pt[0], feet.pt[1]);
    source = "h_bbox";
    confidence = 0.55;
  }

  if (!mapped) return null;
  return { map: mapped, source, confidence };
}

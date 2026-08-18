/** Канонический фингерпринт калибровки камеры (зеркало app/global_id/calib_fingerprint.py). */

const FNV_OFFSET = 2166136261;
const FNV_PRIME = 16777619;
export const DEFAULT_PERSON_H_M = 1.7;

function fmt(v: unknown): string {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(6) : "0.000000";
}

function fmtPt(pt: unknown): string {
  if (!Array.isArray(pt) || pt.length < 2) return "0.000000,0.000000";
  return `${fmt(pt[0])},${fmt(pt[1])}`;
}

function fmtSize(size: unknown): string {
  if (!Array.isArray(size) || size.length < 2) return "0,0";
  const w = Number(size[0]);
  const h = Number(size[1]);
  if (!Number.isFinite(w) || !Number.isFinite(h)) return "0,0";
  return `${Math.trunc(w)},${Math.trunc(h)}`;
}

export function canonicalCalibString(opts: {
  cameraKey: string;
  cameraDoc: Record<string, unknown> | null | undefined;
  torsoHeightM: number;
  trackingSize: [number, number] | number[] | null | undefined;
  personHeightM?: number;
}): string {
  const doc = opts.cameraDoc && typeof opts.cameraDoc === "object" ? opts.cameraDoc : {};
  const plRaw = doc.placement;
  const pl = plRaw && typeof plRaw === "object" ? (plRaw as Record<string, unknown>) : {};
  const pos = Array.isArray(pl.position) ? pl.position : [0, 0];
  const personH = opts.personHeightM ?? DEFAULT_PERSON_H_M;
  const lines = [
    "v2",
    `camera_key=${opts.cameraKey}`,
    `image_size=${fmtSize(doc.image_size)}`,
    `tracking_size=${fmtSize(opts.trackingSize)}`,
    `torso_height_m=${fmt(opts.torsoHeightM)}`,
    `person_height_m=${fmt(personH)}`,
    "placement=" +
      [fmtPt(pos), fmt(pl.yaw_deg ?? 0), fmt(pl.fov_deg ?? 70), fmt(pl.height_m ?? 3), fmt(pl.pitch_deg ?? 35)].join(
        "|",
      ),
  ];
  const h = doc.H;
  if (Array.isArray(h) && h.length >= 9) {
    lines.push("H=" + h.slice(0, 9).map(fmt).join(","));
  } else {
    lines.push("H=");
  }
  return lines.join("\n");
}

export function fnv1a32(text: string): string {
  let h = FNV_OFFSET >>> 0;
  const bytes = new TextEncoder().encode(text);
  for (const b of bytes) {
    h ^= b;
    h = Math.imul(h, FNV_PRIME) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}

export function calibFingerprint(opts: {
  cameraKey: string;
  cameraDoc: Record<string, unknown> | null | undefined;
  torsoHeightM: number;
  trackingSize: [number, number] | number[] | null | undefined;
  personHeightM?: number;
}): string {
  return fnv1a32(canonicalCalibString(opts));
}

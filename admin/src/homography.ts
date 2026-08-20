/** Гомография image → map (3×3). */

export type Pt = [number, number];
export type Mat3 = [number, number, number, number, number, number, number, number, number];

export type HomoPair = {
  image: Pt;
  map: Pt;
};

/** Расположение камеры на плане: позиция + направление взгляда. */
export type CameraPlacement = {
  /** Пиксели плана */
  position: Pt;
  /**
   * Угол взгляда в градусах: 0 = вправо (+x), 90 = вниз (+y), как на canvas.
   * По часовой стрелке.
   */
  yaw_deg: number;
  /** Горизонтальный угол обзора конуса (по умолчанию 70). */
  fov_deg: number;
  /** Высота камеры над полом, м */
  height_m?: number;
  /** Наклон вниз от горизонта, градусы */
  pitch_deg?: number;
};

export type HomographyDoc = {
  camera_key: string;
  floorplan: string;
  image_size: [number, number] | null;
  map_size: [number, number] | null;
  pairs: HomoPair[];
  H: Mat3 | null;
  /** Где стоит камера и куда смотрит на плане */
  placement: CameraPlacement | null;
  /** Калибровка роста (историческое поле cameras/*.json, больше не считается из pose) */
  body_calib?: Record<string, unknown> | null;
  updated_at?: string;
};

export function normalizePlacement(raw: unknown): CameraPlacement | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const pos = o.position;
  if (!Array.isArray(pos) || pos.length < 2) return null;
  const x = Number(pos[0]);
  const y = Number(pos[1]);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  const yaw = Number(o.yaw_deg);
  const fov = Number(o.fov_deg);
  const height = Number(o.height_m);
  const pitch = Number(o.pitch_deg);
  return {
    position: [x, y],
    yaw_deg: Number.isFinite(yaw) ? ((yaw % 360) + 360) % 360 : 0,
    fov_deg: Number.isFinite(fov) ? Math.min(160, Math.max(20, fov)) : 70,
    ...(Number.isFinite(height) ? { height_m: Math.max(0.5, height) } : {}),
    ...(Number.isFinite(pitch) ? { pitch_deg: Math.min(89, Math.max(0, pitch)) } : {}),
  };
}

export function yawFromPoints(from: Pt, to: Pt): number {
  const deg = (Math.atan2(to[1] - from[1], to[0] - from[0]) * 180) / Math.PI;
  return ((deg % 360) + 360) % 360;
}

const CAM_DOT_COLORS = ["#0f6e56", "#2a5a8a", "#8a5a2b", "#6b3d7a", "#1a6b6b", "#8a3d4a", "#4a6b1a"];

export function colorForCameraKey(key: string): string {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) | 0;
  return CAM_DOT_COLORS[Math.abs(h) % CAM_DOT_COLORS.length]!;
}

function hexToRgba(hex: string, alpha: number): string {
  const raw = hex.replace("#", "");
  const full =
    raw.length === 3
      ? raw
          .split("")
          .map((c) => c + c)
          .join("")
      : raw;
  const n = Number.parseInt(full, 16);
  if (!Number.isFinite(n)) return `rgba(42, 90, 138, ${alpha})`;
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** Рисует иконку камеры + конус обзора на canvas плана. */
export function drawCameraPlacement(
  ctx: CanvasRenderingContext2D,
  placement: CameraPlacement,
  label: string,
  opts?: {
    active?: boolean;
    dimmed?: boolean;
    mapW?: number;
    color?: string;
    cameraKey?: string;
    /** Конус обзора (по умолчанию true) */
    cone?: boolean;
    /** Круг + луч (по умолчанию true) */
    body?: boolean;
    /** Подпись (по умолчанию true; всегда читаемая даже при dimmed) */
    label?: boolean;
  },
): void {
  const [x, y] = placement.position;
  const yaw = (placement.yaw_deg * Math.PI) / 180;
  const half = ((placement.fov_deg / 2) * Math.PI) / 180;
  const mw = opts?.mapW ?? 4800;
  const range = Math.max(120, mw * 0.09);
  const r = Math.max(8, mw / 90);
  const active = opts?.active ?? false;
  const dimmed = opts?.dimmed ?? false;
  const drawCone = opts?.cone !== false;
  const drawBody = opts?.body !== false;
  const drawLabel = opts?.label !== false;
  const color =
    opts?.color ??
    (opts?.cameraKey ? colorForCameraKey(opts.cameraKey) : null) ??
    colorForCameraKey(label.replace(/^cam\s+/i, "").trim() || "00");
  const fill = color;
  const cone = hexToRgba(color, active ? 0.28 : 0.16);
  const stroke = color;

  ctx.save();
  ctx.globalAlpha = dimmed ? 0.42 : 1;

  if (drawCone) {
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.arc(x, y, range, yaw - half, yaw + half);
    ctx.closePath();
    ctx.fillStyle = cone;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = Math.max(active ? 2 : 1.5, mw / 600);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + Math.cos(yaw) * range * 0.92, y + Math.sin(yaw) * range * 0.92);
    ctx.strokeStyle = stroke;
    ctx.lineWidth = Math.max(active ? 2.5 : 2, mw / 450);
    ctx.stroke();
  }

  if (drawBody) {
    ctx.beginPath();
    ctx.arc(x, y, r * (active ? 1.08 : 1), 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = Math.max(1.5, mw / 500);
    ctx.stroke();
    if (active) {
      ctx.beginPath();
      ctx.arc(x, y, r * 1.35, 0, Math.PI * 2);
      ctx.strokeStyle = hexToRgba(color, 0.45);
      ctx.lineWidth = Math.max(2, mw / 400);
      ctx.stroke();
    }
  }

  ctx.restore();

  if (drawLabel) {
    const fontPx = Math.max(13, mw / 50);
    ctx.save();
    ctx.globalAlpha = 1;
    ctx.font = `700 ${fontPx}px "IBM Plex Mono", monospace`;
    const tw = ctx.measureText(label).width;
    const padX = Math.max(4, fontPx * 0.28);
    const padY = Math.max(3, fontPx * 0.2);
    const lx = x + r + 6;
    const ly = y - r - fontPx * 0.35;
    ctx.fillStyle = "rgba(255, 255, 255, 0.94)";
    ctx.strokeStyle = hexToRgba(color, 0.55);
    ctx.lineWidth = Math.max(1.5, mw / 700);
    ctx.beginPath();
    const bw = tw + padX * 2;
    const bh = fontPx + padY * 2;
    const bx = lx - padX;
    const by = ly - fontPx + padY * 0.2;
    ctx.roundRect?.(bx, by, bw, bh, 4);
    if (!ctx.roundRect) {
      ctx.rect(bx, by, bw, bh);
    }
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = fill;
    ctx.fillText(label, lx, ly);
    ctx.restore();
  }
}

function matMul3(a: Mat3, b: Mat3): Mat3 {
  const out = new Array(9).fill(0) as number[];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] =
        a[r * 3 + 0]! * b[c]! + a[r * 3 + 1]! * b[3 + c]! + a[r * 3 + 2]! * b[6 + c]!;
    }
  }
  return out as Mat3;
}

function solveLinear(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]!]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r]![col]!) > Math.abs(M[pivot]![col]!)) pivot = r;
    }
    if (Math.abs(M[pivot]![col]!) < 1e-12) return null;
    if (pivot !== col) {
      const tmp = M[col]!;
      M[col] = M[pivot]!;
      M[pivot] = tmp;
    }
    const div = M[col]![col]!;
    for (let c = col; c <= n; c++) M[col]![c]! /= div;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r]![col]!;
      for (let c = col; c <= n; c++) M[r]![c]! -= f * M[col]![c]!;
    }
  }
  return M.map((row) => row[n]!);
}

/** Нормализация точек (Hartley) для устойчивости DLT. */
function normalizePoints(pts: Pt[]): { npts: Pt[]; T: Mat3 } {
  const n = pts.length;
  let cx = 0;
  let cy = 0;
  for (const [x, y] of pts) {
    cx += x;
    cy += y;
  }
  cx /= n;
  cy /= n;
  let meanDist = 0;
  for (const [x, y] of pts) meanDist += Math.hypot(x - cx, y - cy);
  meanDist /= n;
  const s = meanDist > 1e-9 ? Math.SQRT2 / meanDist : 1;
  const T: Mat3 = [s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1];
  const npts: Pt[] = pts.map(([x, y]) => [s * (x - cx), s * (y - cy)]);
  return { npts, T };
}

function invertMat3(m: Mat3): Mat3 | null {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h;
  const B = c * h - b * i;
  const C = b * f - c * e;
  const D = f * g - d * i;
  const E = a * i - c * g;
  const F = c * d - a * f;
  const G = d * h - e * g;
  const H = b * g - a * h;
  const I = a * e - b * d;
  const det = a * A + b * D + c * G;
  if (Math.abs(det) < 1e-12) return null;
  const invDet = 1 / det;
  return [A * invDet, B * invDet, C * invDet, D * invDet, E * invDet, F * invDet, G * invDet, H * invDet, I * invDet];
}

/** DLT: ≥4 пар image→map. */
export function computeHomography(pairs: HomoPair[]): Mat3 | null {
  if (pairs.length < 4) return null;
  const src = pairs.map((p) => p.image);
  const dst = pairs.map((p) => p.map);
  const ns = normalizePoints(src);
  const nd = normalizePoints(dst);

  const rows: number[][] = [];
  for (let i = 0; i < pairs.length; i++) {
    const [x, y] = ns.npts[i]!;
    const [u, v] = nd.npts[i]!;
    rows.push([-x, -y, -1, 0, 0, 0, u * x, u * y, u]);
    rows.push([0, 0, 0, -x, -y, -1, v * x, v * y, v]);
  }

  // Нормальные уравнения для h0..h7 при h8=1 (аффинная фиксация масштаба)
  // Надёжнее: SVD через ATA, берём минимальный собственный вектор — упростим до 8×8 с h8=1
  const AtA = Array.from({ length: 8 }, () => Array(8).fill(0) as number[]);
  const Atb = Array(8).fill(0) as number[];
  for (const row of rows) {
    const rhs = -row[8]!;
    for (let i = 0; i < 8; i++) {
      Atb[i]! += row[i]! * rhs;
      for (let j = 0; j < 8; j++) AtA[i]![j]! += row[i]! * row[j]!;
    }
  }
  const h8 = solveLinear(AtA, Atb);
  if (!h8) return null;
  const Hn: Mat3 = [h8[0]!, h8[1]!, h8[2]!, h8[3]!, h8[4]!, h8[5]!, h8[6]!, h8[7]!, 1];

  const TdstInv = invertMat3(nd.T);
  if (!TdstInv) return null;
  // H = Tdst^{-1} * Hn * Tsrc
  const H = matMul3(TdstInv, matMul3(Hn, ns.T));
  // Нормируем
  const scale = H[8] !== 0 ? H[8] : 1;
  return H.map((v) => v / scale) as Mat3;
}

export function applyHomography(H: Mat3, x: number, y: number): Pt | null {
  const w = H[6]! * x + H[7]! * y + H[8]!;
  if (Math.abs(w) < 1e-12) return null;
  return [(H[0]! * x + H[1]! * y + H[2]!) / w, (H[3]! * x + H[4]! * y + H[5]!) / w];
}

export function invertHomography(H: Mat3): Mat3 | null {
  return invertMat3(H);
}

export type PairError = {
  index: number;
  errPx: number;
  projected: Pt | null;
};

/** Ошибки репроекции image→map: |H(image) − map| в пикселях плана. */
export function reprojectionErrors(pairs: HomoPair[], H: Mat3 | null): PairError[] {
  if (!H) return pairs.map((_, index) => ({ index, errPx: NaN, projected: null }));
  return pairs.map((p, index) => {
    const projected = applyHomography(H, p.image[0], p.image[1]);
    if (!projected) return { index, errPx: NaN, projected: null };
    const errPx = Math.hypot(projected[0] - p.map[0], projected[1] - p.map[1]);
    return { index, errPx, projected };
  });
}

export function rmsError(errors: PairError[]): number | null {
  const vals = errors.map((e) => e.errPx).filter((v) => Number.isFinite(v));
  if (!vals.length) return null;
  return Math.sqrt(vals.reduce((s, v) => s + v * v, 0) / vals.length);
}

/** Минимум пар, чтобы H была переопределена и leave-one-out имел смысл (8 dof). */
export const H_LOO_MIN_PAIRS = 6;

/**
 * Leave-one-out RMS гомографии: для каждой пары фит H по остальным, ошибка на отложенной.
 * При <6 парах H почти/точно интерполирует свои точки — возвращаем null («не проверена»).
 */
export function leaveOneOutRms(pairs: HomoPair[]): number | null {
  if (pairs.length < H_LOO_MIN_PAIRS) return null;
  const errs: number[] = [];
  for (let i = 0; i < pairs.length; i++) {
    const rest = pairs.filter((_, j) => j !== i);
    const H = computeHomography(rest);
    if (!H) continue;
    const held = pairs[i]!;
    const proj = applyHomography(H, held.image[0], held.image[1]);
    if (!proj) continue;
    errs.push(Math.hypot(proj[0] - held.map[0], proj[1] - held.map[1]));
  }
  if (errs.length < 2) return null;
  return Math.sqrt(errs.reduce((s, v) => s + v * v, 0) / errs.length);
}

/** H вместо 3D-луча только если честный LOO H заметно лучше RMS луча. */
export function preferHomographyOverRay(hLooRms: number | null, rayRms: number | null): boolean {
  if (hLooRms == null || rayRms == null) return false;
  if (!(hLooRms < 25)) return false;
  return rayRms > Math.max(2.5 * hLooRms, hLooRms + 15);
}

function centroid(pts: Pt[]): Pt | null {
  if (!pts.length) return null;
  let sx = 0;
  let sy = 0;
  for (const [x, y] of pts) {
    sx += x;
    sy += y;
  }
  return [sx / pts.length, sy / pts.length];
}

/**
 * Оценка позиции камеры на плане по H (image→пол):
 * «низ» кадра ≈ ближняя зона, «середина» ≈ дальше; камера — сзади ближней зоны.
 */
export function estimatePlacementFromHomography(
  H: Mat3,
  imageSize: [number, number],
  opts?: { fov_deg?: number; snap?: (p: Pt) => Pt },
): CameraPlacement | null {
  const [w, h] = imageSize;
  if (w < 8 || h < 8) return null;

  const sampleRow = (yRatio: number): Pt[] => {
    const pts: Pt[] = [];
    for (let i = 0; i <= 10; i++) {
      const p = applyHomography(H, (w * i) / 10, h * yRatio);
      if (p && Number.isFinite(p[0]) && Number.isFinite(p[1])) pts.push(p);
    }
    return pts;
  };

  const near = centroid(sampleRow(0.92));
  const far = centroid(sampleRow(0.38));
  if (!near || !far) return null;

  const dx = far[0] - near[0];
  const dy = far[1] - near[1];
  const len = Math.hypot(dx, dy);
  if (len < 1e-3) return null;

  const back = Math.max(160, len * 0.4);
  let position: Pt = [near[0] - (dx / len) * back, near[1] - (dy / len) * back];
  if (opts?.snap) position = opts.snap(position);

  return {
    position,
    yaw_deg: yawFromPoints(position, far),
    fov_deg: opts?.fov_deg ?? 70,
  };
}

export type AutoCalibrateResult = {
  pairs: HomoPair[];
  removed: number;
  snapped: number;
  H: Mat3 | null;
  rmsBefore: number | null;
  rmsAfter: number | null;
  placement: CameraPlacement | null;
};

/**
 * Авто: snap map→сетка, RANSAC/выбросы по err, пересчёт H, оценка placement.
 */
export function autoCalibrate(
  pairs: HomoPair[],
  imageSize: [number, number] | null,
  opts?: {
    snapMap?: (p: Pt) => Pt;
    maxErrPx?: number;
    minPairs?: number;
    ransacIters?: number;
    fov_deg?: number;
    keepPlacement?: CameraPlacement | null;
  },
): AutoCalibrateResult {
  const minPairs = opts?.minPairs ?? 4;
  const maxErrPx = opts?.maxErrPx ?? 40;
  const ransacIters = opts?.ransacIters ?? 80;
  const empty: AutoCalibrateResult = {
    pairs,
    removed: 0,
    snapped: 0,
    H: null,
    rmsBefore: null,
    rmsAfter: null,
    placement: opts?.keepPlacement ?? null,
  };
  if (pairs.length < minPairs) return empty;

  let snapped = 0;
  const snappedPairs: HomoPair[] = pairs.map((p) => {
    const map = opts?.snapMap ? opts.snapMap(p.map) : ([...p.map] as Pt);
    if (opts?.snapMap && (map[0] !== p.map[0] || map[1] !== p.map[1])) snapped += 1;
    return { image: [...p.image] as Pt, map };
  });

  const H0 = computeHomography(snappedPairs);
  const rmsBefore = rmsError(reprojectionErrors(snappedPairs, H0));

  // RANSAC: лучшее ядро из 4 точек по числу инлаеров
  let bestInliers: number[] = snappedPairs.map((_, i) => i);
  let bestScore = -1;
  const n = snappedPairs.length;
  const trials = Math.min(ransacIters, n <= 4 ? 1 : 200);
  for (let t = 0; t < trials; t++) {
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = idx[i]!;
      idx[i] = idx[j]!;
      idx[j] = tmp;
    }
    const seed = idx.slice(0, 4).map((i) => snappedPairs[i]!);
    const Hs = computeHomography(seed);
    if (!Hs) continue;
    const inliers: number[] = [];
    for (let i = 0; i < n; i++) {
      const p = snappedPairs[i]!;
      const proj = applyHomography(Hs, p.image[0], p.image[1]);
      if (!proj) continue;
      const err = Math.hypot(proj[0] - p.map[0], proj[1] - p.map[1]);
      if (err <= maxErrPx) inliers.push(i);
    }
    if (inliers.length > bestScore) {
      bestScore = inliers.length;
      bestInliers = inliers;
    }
  }

  let kept =
    bestInliers.length >= minPairs
      ? bestInliers.map((i) => snappedPairs[i]!)
      : snappedPairs.slice();

  while (kept.length > minPairs) {
    const Hk = computeHomography(kept);
    if (!Hk) break;
    const errs = reprojectionErrors(kept, Hk);
    let worstI = -1;
    let worstE = -Infinity;
    for (let i = 0; i < errs.length; i++) {
      const e = errs[i]!.errPx;
      if (Number.isFinite(e) && e > worstE) {
        worstE = e;
        worstI = i;
      }
    }
    if (worstI < 0 || worstE <= maxErrPx) break;
    kept = kept.filter((_, i) => i !== worstI);
  }

  const removed = snappedPairs.length - kept.length;
  const H = computeHomography(kept);
  const rmsAfter = rmsError(reprojectionErrors(kept, H));

  let placement: CameraPlacement | null = opts?.keepPlacement ?? null;
  if (H && imageSize) {
    const estimated = estimatePlacementFromHomography(H, imageSize, {
      fov_deg: opts?.fov_deg ?? placement?.fov_deg ?? 70,
      snap: opts?.snapMap,
    });
    if (estimated) placement = estimated;
  }

  return {
    pairs: kept,
    removed,
    snapped,
    H,
    rmsBefore,
    rmsAfter,
    placement,
  };
}

/** Низ центра bbox → точка «ног» в кадре. */
export function feetFromBbox(bbox: number[]): Pt {
  const [x1, y1, x2, y2] = bbox;
  return [(x1 + x2) / 2, Math.max(y1, y2)];
}

export function emptyHomographyDoc(cameraKey: string, floorplan = "grid"): HomographyDoc {
  return {
    camera_key: cameraKey,
    floorplan,
    image_size: null,
    map_size: null,
    pairs: [],
    H: null,
    placement: null,
  };
}

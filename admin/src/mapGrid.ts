/** План-сетка: клетка = 0.5 м. */

import type { Pt } from "./homography";

export const CELL_M = 0.5;
/** Пикселей на клетку 0.5 м */
export const CELL_PX = 80;
/** 1 м в пикселях */
export const METER_PX = CELL_PX * 2;

/** Отступ «стен» до пола */
export const FLOOR_ORIGIN: Pt = [120, 120];
export const FLOOR_CELLS_X = 57; // 28.5 м
export const FLOOR_CELLS_Y = 37; // 18.5 м
export const FLOOR_W = FLOOR_CELLS_X * CELL_PX;
export const FLOOR_H = FLOOR_CELLS_Y * CELL_PX;

export const MAP_W = FLOOR_ORIGIN[0] * 2 + FLOOR_W; // 4800
export const MAP_H = FLOOR_ORIGIN[1] * 2 + FLOOR_H; // 3200
export const MAP_SIZE: [number, number] = [MAP_W, MAP_H];

export const GRID_FLOORPLAN = "grid";

export function isGridFloorplan(name: string | null | undefined): boolean {
  if (!name) return true;
  const n = name.replace(/^\/maps\//, "").replace(/\/$/, "");
  return n === GRID_FLOORPLAN || n === "__grid__" || /^grid(\.|$)/i.test(n);
}

export function snapToGrid(pt: Pt): Pt {
  const [ox, oy] = FLOOR_ORIGIN;
  let gx = Math.round((pt[0] - ox) / CELL_PX) * CELL_PX + ox;
  let gy = Math.round((pt[1] - oy) / CELL_PX) * CELL_PX + oy;
  gx = Math.max(ox, Math.min(ox + FLOOR_W, gx));
  gy = Math.max(oy, Math.min(oy + FLOOR_H, gy));
  return [gx, gy];
}

export function gridLabel(pt: Pt): string {
  const [ox, oy] = FLOOR_ORIGIN;
  const cx = Math.round((pt[0] - ox) / CELL_PX);
  const cy = Math.round((pt[1] - oy) / CELL_PX);
  const mx = (cx * CELL_M).toFixed(1);
  const my = (cy * CELL_M).toFixed(1);
  return `${mx}×${my} м`;
}

export function toCell(pt: Pt): [number, number] {
  const [ox, oy] = FLOOR_ORIGIN;
  return [Math.round((pt[0] - ox) / CELL_PX), Math.round((pt[1] - oy) / CELL_PX)];
}

/** Расстояние между точками в клетках сетки (0.5 м). */
export function tilesBetween(
  a: Pt,
  b: Pt,
): { dx: number; dy: number; tiles: number; meters: number; label: string } {
  const [ax, ay] = toCell(a);
  const [bx, by] = toCell(b);
  const dx = Math.abs(bx - ax);
  const dy = Math.abs(by - ay);
  const tiles = Math.hypot(dx, dy);
  const meters = tiles * CELL_M;
  let label: string;
  if (dx === 0 && dy === 0) label = "0 пл";
  else if (dy === 0) label = `${dx} пл · ${(dx * CELL_M).toFixed(1)} м`;
  else if (dx === 0) label = `${dy} пл · ${(dy * CELL_M).toFixed(1)} м`;
  else label = `${tiles.toFixed(1)} пл (Δ${dx}×${dy}) · ${meters.toFixed(1)} м`;
  return { dx, dy, tiles, meters, label };
}

export type WallDist = {
  id: "W" | "E" | "N" | "S";
  name: string;
  foot: Pt;
  tiles: number;
  meters: number;
};

/** Ортогональные расстояния до стен пола (в клетках 0.5 м). */
export function distancesToWalls(pt: Pt): WallDist[] {
  const [ox, oy] = FLOOR_ORIGIN;
  const [cx, cy] = toCell(pt);
  const x = ox + cx * CELL_PX;
  const y = oy + cy * CELL_PX;
  return [
    {
      id: "W",
      name: "W стена",
      foot: [ox, y],
      tiles: cx,
      meters: cx * CELL_M,
    },
    {
      id: "E",
      name: "E стена",
      foot: [ox + FLOOR_W, y],
      tiles: FLOOR_CELLS_X - cx,
      meters: (FLOOR_CELLS_X - cx) * CELL_M,
    },
    {
      id: "N",
      name: "N стена",
      foot: [x, oy],
      tiles: cy,
      meters: cy * CELL_M,
    },
    {
      id: "S",
      name: "S стена",
      foot: [x, oy + FLOOR_H],
      tiles: FLOOR_CELLS_Y - cy,
      meters: (FLOOR_CELLS_Y - cy) * CELL_M,
    },
  ];
}

/** Точка на отрезке + перпендикулярный сдвиг (чтобы подписи не наезжали). */
export function offsetAlongSegment(from: Pt, to: Pt, t: number, sidePx: number): Pt {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const len = Math.hypot(dx, dy) || 1;
  const mx = from[0] + dx * t;
  const my = from[1] + dy * t;
  return [mx + (-dy / len) * sidePx, my + (dx / len) * sidePx];
}

/**
 * Раздвигает близкие подписи на плане (простой итеративный push).
 * minDist — минимальное расстояние между центрами в px плана.
 */
export function deconflictLabelPositions(points: Pt[], minDist: number, iterations = 12): Pt[] {
  if (points.length < 2) return points.map((p) => [...p] as Pt);
  const out = points.map((p) => [...p] as Pt);
  const minD = Math.max(8, minDist);
  for (let iter = 0; iter < iterations; iter++) {
    for (let i = 0; i < out.length; i++) {
      for (let j = i + 1; j < out.length; j++) {
        const a = out[i]!;
        const b = out[j]!;
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let d = Math.hypot(dx, dy);
        if (d >= minD) continue;
        if (d < 1e-6) {
          dx = (j - i) * 0.7 + 0.3;
          dy = (i - j) * 0.5 - 0.2;
          d = Math.hypot(dx, dy);
        }
        const push = (minD - d) / 2;
        const nx = dx / d;
        const ny = dy / d;
        a[0] -= nx * push;
        a[1] -= ny * push;
        b[0] += nx * push;
        b[1] += ny * push;
      }
    }
  }
  return out;
}

/** Рисует пол + сетку 0.5 м на canvas в координатах плана. */
export function drawFloorGrid(
  ctx: CanvasRenderingContext2D,
  w = MAP_W,
  h = MAP_H,
): void {
  const [ox, oy] = FLOOR_ORIGIN;
  ctx.clearRect(0, 0, w, h);

  // фон / стены
  ctx.fillStyle = "#e8ece7";
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "#2a3832";
  ctx.fillRect(ox - 72, oy - 72, FLOOR_W + 144, FLOOR_H + 144);

  // пол
  ctx.fillStyle = "#f4f6f2";
  ctx.fillRect(ox, oy, FLOOR_W, FLOOR_H);

  // клетки 0.5 м
  ctx.save();
  ctx.beginPath();
  ctx.rect(ox, oy, FLOOR_W, FLOOR_H);
  ctx.clip();

  for (let x = ox; x <= ox + FLOOR_W + 0.5; x += CELL_PX) {
    const i = Math.round((x - ox) / CELL_PX);
    const meter = i % 2 === 0;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, oy);
    ctx.lineTo(x + 0.5, oy + FLOOR_H);
    ctx.strokeStyle = meter ? "#9aa89e" : "#d0d8d1";
    ctx.lineWidth = meter ? 1.5 : 1;
    ctx.stroke();
  }
  for (let y = oy; y <= oy + FLOOR_H + 0.5; y += CELL_PX) {
    const i = Math.round((y - oy) / CELL_PX);
    const meter = i % 2 === 0;
    ctx.beginPath();
    ctx.moveTo(ox, y + 0.5);
    ctx.lineTo(ox + FLOOR_W, y + 0.5);
    ctx.strokeStyle = meter ? "#9aa89e" : "#d0d8d1";
    ctx.lineWidth = meter ? 1.5 : 1;
    ctx.stroke();
  }

  // пересечения (лёгкие точки)
  ctx.fillStyle = "rgba(21, 32, 24, 0.35)";
  for (let x = ox; x <= ox + FLOOR_W + 0.5; x += CELL_PX) {
    for (let y = oy; y <= oy + FLOOR_H + 0.5; y += CELL_PX) {
      ctx.beginPath();
      ctx.arc(x, y, 1.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();

  // рамка пола
  ctx.strokeStyle = "#152018";
  ctx.lineWidth = 3;
  ctx.strokeRect(ox + 1.5, oy + 1.5, FLOOR_W - 3, FLOOR_H - 3);

  // заголовок
  ctx.fillStyle = "#0f6e56";
  ctx.font = `600 ${Math.max(22, w / 90)}px "IBM Plex Mono", monospace`;
  ctx.textAlign = "center";
  ctx.fillText(
    `GRID 0.5 m · ${(FLOOR_W / METER_PX).toFixed(1)} × ${(FLOOR_H / METER_PX).toFixed(1)} m · snap to intersections`,
    w / 2,
    Math.max(36, oy * 0.55),
  );

  // шкала 0.5 м
  const lx = ox + 40;
  const ly = oy + FLOOR_H + 48;
  ctx.strokeStyle = "#152018";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(lx, ly);
  ctx.lineTo(lx + CELL_PX, ly);
  ctx.moveTo(lx, ly - 10);
  ctx.lineTo(lx, ly + 10);
  ctx.moveTo(lx + CELL_PX, ly - 10);
  ctx.lineTo(lx + CELL_PX, ly + 10);
  ctx.stroke();
  ctx.fillStyle = "#5a675e";
  ctx.font = `500 ${Math.max(16, w / 120)}px "IBM Plex Mono", monospace`;
  ctx.textAlign = "center";
  ctx.fillText("0.5 m", lx + CELL_PX / 2, ly + 28);

  // N
  const nx = ox + FLOOR_W - 40;
  const ny = oy + 80;
  ctx.beginPath();
  ctx.arc(nx, ny, 36, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();
  ctx.strokeStyle = "#0f6e56";
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(nx, ny + 22);
  ctx.lineTo(nx, ny - 22);
  ctx.lineTo(nx - 8, ny - 8);
  ctx.moveTo(nx, ny - 22);
  ctx.lineTo(nx + 8, ny - 8);
  ctx.stroke();
  ctx.fillStyle = "#0f6e56";
  ctx.font = `700 20px "IBM Plex Mono", monospace`;
  ctx.fillText("N", nx, ny + 8);
}

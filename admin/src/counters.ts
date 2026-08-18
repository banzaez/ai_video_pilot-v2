import { applyHomography, type Mat3, type Pt } from "./homography";
import { MAP_SIZE } from "./mapGrid";

export type CounterPoly = {
  id: string;
  name: string;
  /** Вершины на плане (px) */
  map: Pt[];
  /** Опционально контур на кадре по id камеры */
  image_by_camera?: Record<string, Pt[]>;
};

export type CountersDoc = {
  floorplan: string;
  map_size: [number, number];
  counters: CounterPoly[];
  updated_at?: string;
};

export function emptyCountersDoc(floorplan = "grid"): CountersDoc {
  return {
    floorplan,
    map_size: MAP_SIZE,
    counters: [],
  };
}

export function newCounterId(existing: CounterPoly[]): string {
  let n = existing.length + 1;
  const ids = new Set(existing.map((c) => c.id));
  while (ids.has(`c${n}`)) n += 1;
  return `c${n}`;
}

export function normalizeCountersDoc(raw: unknown, floorplan = "grid"): CountersDoc {
  const base = emptyCountersDoc(floorplan);
  if (!raw || typeof raw !== "object") return base;
  const o = raw as Record<string, unknown>;
  const counters: CounterPoly[] = [];
  if (Array.isArray(o.counters)) {
    for (const item of o.counters) {
      if (!item || typeof item !== "object") continue;
      const c = item as Record<string, unknown>;
      const map = Array.isArray(c.map)
        ? (c.map as unknown[])
            .filter((p) => Array.isArray(p) && p.length >= 2)
            .map((p) => [Number((p as number[])[0]), Number((p as number[])[1])] as Pt)
            .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
        : [];
      if (map.length < 3) continue;
      const image_by_camera: Record<string, Pt[]> = {};
      if (c.image_by_camera && typeof c.image_by_camera === "object") {
        for (const [cam, pts] of Object.entries(c.image_by_camera as Record<string, unknown>)) {
          if (!Array.isArray(pts)) continue;
          const arr = pts
            .filter((p) => Array.isArray(p) && p.length >= 2)
            .map((p) => [Number((p as number[])[0]), Number((p as number[])[1])] as Pt)
            .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
          if (arr.length >= 3) image_by_camera[cam] = arr;
        }
      }
      counters.push({
        id: typeof c.id === "string" && c.id ? c.id : newCounterId(counters),
        name: typeof c.name === "string" && c.name ? c.name : `Прилавок ${counters.length + 1}`,
        map,
        image_by_camera: Object.keys(image_by_camera).length ? image_by_camera : undefined,
      });
    }
  }
  const ms = Array.isArray(o.map_size) && o.map_size.length >= 2 ? o.map_size : MAP_SIZE;
  return {
    floorplan: typeof o.floorplan === "string" ? o.floorplan : floorplan,
    map_size: [Number(ms[0]) || MAP_SIZE[0], Number(ms[1]) || MAP_SIZE[1]],
    counters,
    updated_at: typeof o.updated_at === "string" ? o.updated_at : undefined,
  };
}

export function countersFingerprint(doc: CountersDoc): string {
  return JSON.stringify({
    floorplan: doc.floorplan,
    map_size: doc.map_size,
    counters: doc.counters,
  });
}

/**
 * После изменения H: у прилавков с контуром на кадре `cameraKey`
 * пересчитываем вершины на плане (image → map).
 */
export function reprojectCountersFromImage(
  doc: CountersDoc,
  cameraKey: string,
  H: Mat3,
  projectMap?: (pt: Pt) => Pt,
): { doc: CountersDoc; updated: number } {
  let updated = 0;
  const counters = doc.counters.map((c) => {
    const img = c.image_by_camera?.[cameraKey];
    if (!img || img.length < 3) return c;
    const map: Pt[] = [];
    for (const p of img) {
      const m = applyHomography(H, p[0], p[1]);
      if (!m) return c;
      map.push(projectMap ? projectMap(m) : m);
    }
    if (map.length !== img.length) return c;
    const same =
      map.length === c.map.length &&
      map.every((p, i) => Math.hypot(p[0] - c.map[i]![0], p[1] - c.map[i]![1]) < 0.05);
    if (same) return c;
    updated += 1;
    return { ...c, map };
  });
  if (!updated) return { doc, updated: 0 };
  return { doc: { ...doc, counters }, updated };
}

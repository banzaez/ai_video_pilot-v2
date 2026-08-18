import { describe, expect, it } from "vitest";
import { buildFeetIndex, feetAtFrame } from "./feetIndex";
import type { FeetDoc } from "./types";

const doc: FeetDoc = {
  calibration: { fingerprint: "deadbeef" },
  frames: [
    { frame_index: 1, points: [{ track_id: 7, map: [100, 200], source: "ray", confidence: 0.85 }] },
    { frame_index: 4, points: [{ track_id: 7, map: [130, 260], source: "ray", confidence: 0.85 }] },
  ],
};

describe("feetIndex", () => {
  it("returns exact sample on keyframes", () => {
    const idx = buildFeetIndex(doc);
    const a = feetAtFrame(idx, 7, 1, 3);
    const b = feetAtFrame(idx, 7, 4, 3);
    expect(a?.map).toEqual([100, 200]);
    expect(b?.map).toEqual([130, 260]);
  });

  it("interpolates monotonically between keyframes", () => {
    const idx = buildFeetIndex(doc);
    const mid = feetAtFrame(idx, 7, 2.5, 3);
    expect(mid).not.toBeNull();
    expect(mid!.map[0]).toBeGreaterThan(100);
    expect(mid!.map[0]).toBeLessThan(130);
    expect(mid!.map[1]).toBeGreaterThan(200);
    expect(mid!.map[1]).toBeLessThan(260);
    const t = (2.5 - 1) / 3;
    expect(mid!.map[0]).toBeCloseTo(100 + 30 * t, 5);
  });

  it("skips gaps larger than detectEveryN", () => {
    const idx = buildFeetIndex(doc);
    expect(feetAtFrame(idx, 7, 2.5, 1)).toBeNull();
  });
});

import { describe, expect, it } from "vitest";
import { adjustMapPointForCounters, pointInPolygon, scaleBboxToImageSize } from "../src/feet";
import type { CounterPoly } from "../src/counters";
import type { Pt } from "../src/homography";

describe("feet geometry and scaling", () => {
  it("pointInPolygon correctly detects interior points", () => {
    const poly: Pt[] = [
      [0, 0],
      [100, 0],
      [100, 100],
      [0, 100],
    ];
    expect(pointInPolygon([50, 50], poly)).toBe(true);
    expect(pointInPolygon([150, 50], poly)).toBe(false);
    expect(pointInPolygon([-10, 50], poly)).toBe(false);
  });

  it("scaleBboxToImageSize scales bbox correctly", () => {
    const bbox = [100, 100, 200, 300];
    const scaled = scaleBboxToImageSize(bbox, [1000, 1000], [2000, 2000]);
    expect(scaled).toEqual([200, 200, 400, 600]);
  });

  it("adjustMapPointForCounters pushes point to boundary", () => {
    const counters: CounterPoly[] = [
      {
        id: "c1",
        map: [
          [100, 100],
          [200, 100],
          [200, 200],
          [100, 200],
        ],
      },
    ];
    const inside: Pt = [150, 150];
    const adjusted = adjustMapPointForCounters(inside, counters);
    // Distance from center [150, 150] to boundary must be 50
    expect(Math.hypot(adjusted[0] - inside[0], adjusted[1] - inside[1])).toBeCloseTo(50, 1);
  });
});


import { describe, expect, it } from "vitest";
import {
  applyHomography,
  computeHomography,
  invertHomography,
  leaveOneOutRms,
  normalizePlacement,
  preferHomographyOverRay,
  reprojectionErrors,
  yawFromPoints,
  type HomoPair,
  type Mat3,
} from "./homography";

describe("homography unit tests", () => {
  it("computes identity-like homography for identical points", () => {
    const pairs: HomoPair[] = [
      { image: [0, 0], map: [0, 0] },
      { image: [100, 0], map: [100, 0] },
      { image: [100, 100], map: [100, 100] },
      { image: [0, 100], map: [0, 100] },
    ];
    const H = computeHomography(pairs);
    expect(H).not.toBeNull();
    if (!H) return;

    const p = applyHomography(H, 50, 50);
    expect(p).not.toBeNull();
    expect(p![0]).toBeCloseTo(50, 2);
    expect(p![1]).toBeCloseTo(50, 2);

    const errors = reprojectionErrors(pairs, H);
    expect(errors.length).toBe(4);
    for (const e of errors) {
      expect(e.errPx).toBeCloseTo(0, 2);
    }
  });

  it("inverts homography matrix accurately", () => {
    // Матрица сдвига и масштабирования
    const H: Mat3 = [2, 0, 10, 0, 3, 20, 0, 0, 1];
    const inv = invertHomography(H);
    expect(inv).not.toBeNull();
    if (!inv) return;

    // Прямое и обратное преобразование
    const orig = [15, 25];
    const mapped = applyHomography(H, orig[0], orig[1])!;
    expect(mapped[0]).toBeCloseTo(40, 4);
    expect(mapped[1]).toBeCloseTo(95, 4);

    const back = applyHomography(inv, mapped[0], mapped[1])!;
    expect(back[0]).toBeCloseTo(orig[0], 4);
    expect(back[1]).toBeCloseTo(orig[1], 4);
  });

  it("normalizes placement and angles correctly", () => {
    expect(normalizePlacement(null)).toBeNull();
    expect(normalizePlacement({ position: [10] })).toBeNull();

    const valid = normalizePlacement({
      position: [120, 240],
      yaw_deg: 450, // 450 mod 360 = 90
      fov_deg: 180, // clamped to 160
    });
    expect(valid).toEqual({
      position: [120, 240],
      yaw_deg: 90,
      fov_deg: 160,
    });
  });

  it("calculates yaw angle between points", () => {
    // Вправо
    expect(yawFromPoints([0, 0], [10, 0])).toBeCloseTo(0, 4);
    // Вниз (+y на canvas)
    expect(yawFromPoints([0, 0], [0, 10])).toBeCloseTo(90, 4);
    // Влево
    expect(yawFromPoints([0, 0], [-10, 0])).toBeCloseTo(180, 4);
    // Вверх
    expect(yawFromPoints([0, 0], [0, -10])).toBeCloseTo(270, 4);
  });

  it("leave-one-out RMS is null for 4 pairs and differs on 6", () => {
    const four: HomoPair[] = [
      { image: [0, 0], map: [0, 0] },
      { image: [100, 0], map: [200, 0] },
      { image: [100, 100], map: [200, 150] },
      { image: [0, 100], map: [0, 150] },
    ];
    expect(leaveOneOutRms(four)).toBeNull();
    const six: HomoPair[] = [
      { image: [0, 0], map: [10, 10] },
      { image: [100, 0], map: [210, 12] },
      { image: [100, 100], map: [205, 190] },
      { image: [0, 100], map: [8, 188] },
      { image: [50, 40], map: [108, 82] },
      { image: [80, 70], map: [170, 140] },
    ];
    const loo = leaveOneOutRms(six);
    expect(loo).not.toBeNull();
    expect(loo!).toBeGreaterThan(0);
    expect(preferHomographyOverRay(loo, 40)).toBe(false);
    expect(preferHomographyOverRay(10, 40)).toBe(true);
  });
});

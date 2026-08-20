import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { dualPlaneFromBbox, fitRayPose, normalizeCameraPose, projectKeypointsToMap, rayPairStats, rayToGroundMap } from "./cameraPose";

type FeetCase = {
  name: string;
  fn: string;
  expect?: number;
  pose?: {
    position: [number, number];
    yaw_deg: number;
    fov_deg: number;
    height_m: number;
    pitch_deg: number;
  };
  image_size?: [number, number];
  pair?: { image: [number, number]; map: [number, number] };
  max_err_px?: number;
};

const fixturePath = join(dirname(fileURLToPath(import.meta.url)), "../../tests/fixtures/feet_cases.json");
const cases = JSON.parse(readFileSync(fixturePath, "utf8")) as FeetCase[];

type RayFitCase = FeetCase & {
  pairs: Array<{ image: [number, number]; map: [number, number] }>;
  fit_pose?: boolean;
  expect: {
    height_m: number;
    pitch_deg: number;
    fov_deg: number;
    yaw_deg: number;
    position: [number, number];
    rms_px: number;
    projected: number;
    total: number;
  };
  tol?: Record<string, number>;
};

const rayFit = cases.find((c) => c.fn === "ray_fit") as RayFitCase | undefined;
if (!rayFit) throw new Error("нет кейса ray_fit в feet_cases.json");

const PAIRS = rayFit.pairs;

describe("camera pose fixtures (mirror Python)", () => {
  for (const c of cases) {
    if (c.fn === "ray_pair_err") {
      it(c.name, () => {
        const pose = normalizeCameraPose(c.pose)!;
        const mapped = rayToGroundMap(c.pair!.image[0], c.pair!.image[1], pose, c.image_size!, {
          torsoHeightM: 0,
        });
        expect(mapped).not.toBeNull();
        const err = Math.hypot(mapped![0] - c.pair!.map[0], mapped![1] - c.pair!.map[1]);
        expect(err).toBeLessThan(c.max_err_px ?? 500);
      });
    }
  }
});

describe("fitRayPose", () => {
  const origin = normalizeCameraPose(rayFit.pose)!;
  const imageSize = rayFit.image_size as [number, number];

  it("matches fixture and covers all pairs", () => {
    const before = rayPairStats(origin, PAIRS, imageSize)!;
    const est = fitRayPose(origin, PAIRS, imageSize, { fitPose: rayFit.fit_pose ?? true });
    expect(est).not.toBeNull();
    expect(est!.projected).toBe(est!.total);
    expect(est!.rmsPx).toBeLessThan(before.rmsPx);
    const exp = rayFit.expect;
    expect(est!.height_m).toBeCloseTo(exp.height_m, 2);
    expect(est!.pitch_deg).toBeCloseTo(exp.pitch_deg, 1);
    expect(est!.fov_deg).toBeCloseTo(exp.fov_deg, 1);
    expect(est!.yaw_deg).toBeCloseTo(exp.yaw_deg, 1);
    expect(est!.position[0]).toBeCloseTo(exp.position[0], 0);
    expect(est!.position[1]).toBeCloseTo(exp.position[1], 0);
    expect(est!.rmsPx).toBeCloseTo(exp.rms_px, 1);
    expect(est!.projected).toBe(exp.projected);
    expect(est!.total).toBe(exp.total);
  });

  it("is deterministic", () => {
    const a = fitRayPose(origin, PAIRS, imageSize, { fitPose: true })!;
    const b = fitRayPose(origin, PAIRS, imageSize, { fitPose: true })!;
    expect(a.height_m).toBe(b.height_m);
    expect(a.pitch_deg).toBe(b.pitch_deg);
    expect(a.fov_deg).toBe(b.fov_deg);
    expect(a.yaw_deg).toBe(b.yaw_deg);
    expect(a.position).toEqual(b.position);
    expect(a.rmsPx).toBe(b.rmsPx);
  });

  it("penalizes poses that miss the ground", () => {
    const ok = rayPairStats(
      { ...origin, fov_deg: 88, height_m: 3.25, pitch_deg: 31 },
      PAIRS,
      imageSize,
    )!;
    const sky = rayPairStats(
      { ...origin, fov_deg: 88, height_m: 3.25, pitch_deg: 0 },
      PAIRS,
      imageSize,
    )!;
    expect(sky.rmsPx).toBeGreaterThan(ok.rmsPx);
  });
});

describe("dual-plane and keypoints", () => {
  const pose = normalizeCameraPose({
    position: [1550, 2490],
    yaw_deg: 0.15,
    fov_deg: 82.5,
    height_m: 3.15,
    pitch_deg: 35.5,
  })!;
  const imageSize: [number, number] = [1920, 1080];

  it("uses head plane for a truncated bbox", () => {
    const truncated = dualPlaneFromBbox([802, 154, 964, 281], pose, imageSize, 1.7);
    expect(truncated.map).not.toBeNull();
    expect(truncated.pHead).not.toBeNull();
    expect(truncated.pFeet).not.toBeNull();
    expect(truncated.truncated).toBe(true);
    expect(truncated.source).toBe("ray_head");
    const dHead = Math.hypot(truncated.pHead![0] - pose.position[0], truncated.pHead![1] - pose.position[1]);
    const dFeet = Math.hypot(truncated.pFeet![0] - pose.position[0], truncated.pFeet![1] - pose.position[1]);
    expect(dHead).toBeLessThan(dFeet);
  });

  it("projects ankles closer than bbox bottom", () => {
    const kxy: number[][] = Array.from({ length: 17 }, () => [0, 0]);
    const kcf = Array(17).fill(0);
    kxy[15] = [850, 400];
    kxy[16] = [890, 400];
    kcf[15] = 0.9;
    kcf[16] = 0.9;
    const kpt = projectKeypointsToMap(kxy, kcf, pose, imageSize, 1.7, 0.25);
    const dual = dualPlaneFromBbox([800, 150, 960, 281], pose, imageSize, 1.7);
    expect(kpt).not.toBeNull();
    expect(kpt!.source.startsWith("kpt")).toBe(true);
    expect(dual.pFeet).not.toBeNull();
    const dK = Math.hypot(kpt!.map[0] - pose.position[0], kpt!.map[1] - pose.position[1]);
    const dB = Math.hypot(dual.pFeet![0] - pose.position[0], dual.pFeet![1] - pose.position[1]);
    expect(dK).toBeLessThan(dB);
  });
});

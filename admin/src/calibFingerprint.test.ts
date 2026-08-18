import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { calibFingerprint, canonicalCalibString } from "./calibFingerprint";

const fixturePath = join(
  dirname(fileURLToPath(import.meta.url)),
  "../../tests/fixtures/calib_fingerprint_cases.json",
);
const data = JSON.parse(readFileSync(fixturePath, "utf8")) as {
  cases: Array<{
    name: string;
    expect?: string;
    camera_key: string;
    torso_height_m: number;
    tracking_size: [number, number];
    camera_doc: Record<string, unknown>;
    counters: { counters: unknown[] };
  }>;
};

describe("calibFingerprint (mirror Python)", () => {
  it("is stable and changes when height changes", () => {
    const c = data.cases[0]!;
    const fp = calibFingerprint({
      cameraKey: c.camera_key,
      cameraDoc: c.camera_doc,
      torsoHeightM: c.torso_height_m,
      trackingSize: c.tracking_size,
    });
    expect(fp).toBe(c.expect);
    const again = calibFingerprint({
      cameraKey: c.camera_key,
      cameraDoc: c.camera_doc,
      torsoHeightM: c.torso_height_m,
      trackingSize: c.tracking_size,
    });
    expect(again).toBe(fp);
    const mutated = structuredClone(c.camera_doc) as Record<string, unknown>;
    const pl = { ...(mutated.placement as Record<string, unknown>), height_m: 4.0 };
    mutated.placement = pl;
    const other = calibFingerprint({
      cameraKey: c.camera_key,
      cameraDoc: mutated,
      torsoHeightM: c.torso_height_m,
      trackingSize: c.tracking_size,
    });
    expect(other).not.toBe(fp);
    const text = canonicalCalibString({
      cameraKey: c.camera_key,
      cameraDoc: c.camera_doc,
      torsoHeightM: c.torso_height_m,
      trackingSize: c.tracking_size,
    });
    expect(text).toContain("v2");
    expect(text).toContain("camera_key=01");
    expect(text).toContain("person_height_m=");
    expect(text).not.toContain("counters=");
  });
});

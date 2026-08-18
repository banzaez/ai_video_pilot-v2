import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { normalizeBodyCalib, pickKForShoulder } from "../src/feet";

type FeetCase = {
  name: string;
  fn: string;
  args: { y_shoulder: number; calib: unknown };
  expect: number;
};

const fixturePath = join(dirname(fileURLToPath(import.meta.url)), "../../tests/fixtures/feet_cases.json");
const cases = JSON.parse(readFileSync(fixturePath, "utf8")) as FeetCase[];

describe("feet fixtures (mirror Python)", () => {
  for (const c of cases) {
    if (c.fn !== "pick_k") continue;
    it(c.name, () => {
      const calib = normalizeBodyCalib(c.args.calib);
      const got = pickKForShoulder(c.args.y_shoulder, calib);
      expect(got).toBeCloseTo(c.expect, 4);
    });
  }
});

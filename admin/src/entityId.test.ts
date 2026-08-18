import { describe, expect, it } from "vitest";
import {
  formatEntityId,
  groupId,
  parseEntityId,
  parseEntityIdOptional,
  personId,
  trackletId,
} from "./entityId";

describe("entityId", () => {
  it("formats t/g/p", () => {
    expect(formatEntityId(trackletId(12))).toBe("t12");
    expect(formatEntityId(groupId(3))).toBe("g3");
    expect(formatEntityId(personId(1))).toBe("p1");
  });

  it("parses canonical tokens", () => {
    expect(parseEntityId("g1")).toEqual({ space: "g", n: 1 });
    expect(parseEntityId("T12")).toEqual({ space: "t", n: 12 });
    expect(parseEntityId("p1")).toEqual({ space: "p", n: 1 });
  });

  it("rejects a bare number", () => {
    expect(() => parseEntityId("1")).toThrow();
    expect(parseEntityIdOptional("1")).toBeNull();
    expect(parseEntityIdOptional("")).toBeNull();
  });
});

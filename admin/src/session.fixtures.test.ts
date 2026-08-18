import { describe, expect, it } from "vitest";
import {
  groupBySessionKey,
  parseProdStem,
  partAtTime,
  sessionDurationSec,
  sessionKeyFromPart,
  sessionLabel,
  sessionDays,
  sessionsForDay,
  formatSessionDay,
  cameraLabel,
  type MediaSession,
} from "./session";

const PROD = [
  "Camera_01_nvr_local_20260601095444_20260601111302_tid1401s20260601095444",
  "Camera_01_nvr_local_20260601111302_20260601123000_tid1402s20260601111302",
  "Camera_01_nvr_local_20260601123000_20260601140000_tid1403s20260601123000",
];

describe("session.ts", () => {
  it("parses prod stem", () => {
    const p = parseProdStem(PROD[0]!);
    expect(p?.camera_index).toBe(1);
    expect(p?.session_key).toBe("01_20260601");
  });

  it("parses IP-source stem", () => {
    const p = parseProdStem(
      "Camera_01_10.12.0.35_10.12.0.235_20260401113050_20260401113550_3084159",
    );
    expect(p?.camera_index).toBe(1);
    expect(p?.session_key).toBe("01_20260401");
    expect(p?.started_raw).toBe("20260401113050");
    expect(p?.day).toBe("2026-04-01");
  });

  it("groups three parts", () => {
    const parts = PROD.map((s) => parseProdStem(s)).filter(Boolean) as NonNullable<
      ReturnType<typeof parseProdStem>
    >[];
    const grouped = groupBySessionKey(parts);
    expect(grouped.size).toBe(1);
    expect(grouped.get("01_20260601")?.length).toBe(3);
  });

  it("session key from part", () => {
    expect(sessionKeyFromPart(2, "20260601095444")).toBe("02_20260601");
  });

  it("session label", () => {
    expect(
      sessionLabel({
        key: "01_20260601",
        camera: "Camera_01",
        camera_index: 1,
        day: "2026-06-01",
        parts: [],
        hasJson: false,
        jsonUrl: "",
      }),
    ).toBe("01 · 2026-06-01");
  });

  it("session days then cameras", () => {
    const items: MediaSession[] = [
      {
        key: "02_20260602",
        camera: "Camera_02",
        camera_index: 2,
        day: "2026-06-02",
        parts: [],
        hasJson: true,
        jsonUrl: "",
      },
      {
        key: "01_20260601",
        camera: "Camera_01",
        camera_index: 1,
        day: "2026-06-01",
        parts: [],
        hasJson: false,
        jsonUrl: "",
      },
      {
        key: "03_20260601",
        camera: "Camera_03",
        camera_index: 3,
        day: "2026-06-01",
        parts: [],
        hasJson: true,
        jsonUrl: "",
      },
    ];
    expect(sessionDays(items)).toEqual(["2026-06-02", "2026-06-01"]);
    expect(formatSessionDay("2026-06-01")).toBe("01.06.2026");
    expect(sessionsForDay(items, "2026-06-01").map((s) => s.key)).toEqual([
      "01_20260601",
      "03_20260601",
    ]);
    expect(cameraLabel(items[1]!)).toBe("Camera_01 · нет результатов");
  });

  it("session duration and partAtTime across boundary", () => {
    const parts = [
      {
        name: "a.mp4",
        stem: "a",
        videoUrl: "/media/a.mp4",
        started_at: "2026-06-01T09:00:00",
        ended_at: "2026-06-01T10:00:00",
        frame_offset: 0,
        frame_count: 2500,
        time_offset_sec: 0,
      },
      {
        name: "b.mp4",
        stem: "b",
        videoUrl: "/media/b.mp4",
        started_at: "2026-06-01T10:00:00",
        ended_at: "2026-06-01T11:00:00",
        frame_offset: 2500,
        frame_count: 2500,
        time_offset_sec: 100,
      },
    ];
    expect(sessionDurationSec(parts, 25)).toBeCloseTo(200, 5);
    const hit = partAtTime(parts, 100.5, 25);
    expect(hit?.part.name).toBe("b.mp4");
    expect(hit?.localTime).toBeCloseTo(0.5, 5);
  });
});

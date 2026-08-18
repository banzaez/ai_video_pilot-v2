import { colorForCameraKey } from "./homography";

export type GlobalMember = {
  key: string;
  camera?: string;
  group_id?: number | null;
  track_ids: number[];
  t0?: number;
  t1?: number;
  tracks?: Array<{ track_id: number; t0: number; t1: number }>;
};

export type PeopleTimelineBar = {
  key: string;
  memberKey: string;
  camera: string;
  trackId: number;
  trackIds: number[];
  t0: number;
  t1: number;
};

export type PeopleTimelineSection = {
  member: GlobalMember;
  memberKey: string;
  camera: string;
  label: string;
  summaryT0: number;
  summaryT1: number;
  tracks: PeopleTimelineBar[];
};

export type PeopleTimelineCameraGroup = {
  camera: string;
  color: string;
  sections: PeopleTimelineSection[];
};

export function memberLabel(m: GlobalMember): string {
  if (typeof m.group_id === "number") return `g${m.group_id}`;
  const tid = m.track_ids[0];
  return tid != null ? `#${tid}` : m.key;
}

export function buildPeopleTimeline(members: GlobalMember[]): {
  duration: number;
  sections: PeopleTimelineSection[];
  cameraGroups: PeopleTimelineCameraGroup[];
} {
  let duration = 0.001;
  const sections: PeopleTimelineSection[] = [];

  const sorted = [...members].sort((a, b) => {
    const ca = a.camera ?? "";
    const cb = b.camera ?? "";
    if (ca !== cb) return ca.localeCompare(cb);
    const ga = a.group_id ?? 9999;
    const gb = b.group_id ?? 9999;
    if (ga !== gb) return ga - gb;
    return a.key.localeCompare(b.key);
  });

  for (const m of sorted) {
    const spans: Array<{ track_id: number; t0: number; t1: number }> = (
      m.tracks?.length
        ? m.tracks
        : m.track_ids.map((track_id: number) => ({
            track_id,
            t0: m.t0 ?? 0,
            t1: m.t1 ?? m.t0 ?? 0,
          }))
    ).filter((t: { t0: number; t1: number }) => Number.isFinite(t.t0) && Number.isFinite(t.t1));

    if (!spans.length) continue;

    const summaryT0 = Math.min(...spans.map((t: { t0: number }) => t.t0));
    const summaryT1 = Math.max(...spans.map((t: { t1: number }) => t.t1));
    duration = Math.max(duration, summaryT1);

    const tracks: PeopleTimelineBar[] = spans
      .slice()
      .sort((a: { t0: number; track_id: number }, b: { t0: number; track_id: number }) => a.t0 - b.t0 || a.track_id - b.track_id)
      .map((t: { track_id: number; t0: number; t1: number }) => ({
        key: `${m.key}#${t.track_id}`,
        memberKey: m.key,
        camera: m.camera ?? "??",
        trackId: t.track_id,
        trackIds: [t.track_id],
        t0: t.t0,
        t1: t.t1,
      }));

    sections.push({
      member: m,
      memberKey: m.key,
      camera: m.camera ?? "??",
      label: memberLabel(m),
      summaryT0,
      summaryT1,
      tracks,
    });
  }

  const byCam = new Map<string, PeopleTimelineSection[]>();
  for (const s of sections) {
    const list = byCam.get(s.camera) ?? [];
    list.push(s);
    byCam.set(s.camera, list);
  }

  const cameraGroups: PeopleTimelineCameraGroup[] = [...byCam.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([camera, camSections]) => ({
      camera,
      color: colorForCameraKey(camera),
      sections: camSections,
    }));

  return { duration, sections, cameraGroups };
}

export function liveTrackIdsAt(members: GlobalMember[], sec: number): number[] {
  const ids: number[] = [];
  for (const m of members) {
    const spans = m.tracks?.length
      ? m.tracks
      : m.track_ids.map((track_id: number) => ({
          track_id,
          t0: m.t0 ?? 0,
          t1: m.t1 ?? m.t0 ?? 0,
        }));
    for (const t of spans) {
      if (sec >= t.t0 && sec <= t.t1) ids.push(t.track_id);
    }
  }
  return [...new Set(ids)];
}

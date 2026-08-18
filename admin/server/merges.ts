import { readJsonFile, workFile } from "./common.js";

export type SimilarHit = {
  track_id: number;
  score: number;
  reid?: number | null;
  motion?: number | null;
  size?: number | null;
  gap?: number | null;
  dist?: number | null;
  space?: string | null;
  reason?: string | null;
  pass?: number | null;
  group_id?: number | null;
  t0?: number | null;
  t1?: number | null;
};

export type LinkEdge = {
  from?: number;
  to?: number;
  score?: number;
  reid?: number;
  motion?: number;
  size?: number;
  gap?: number;
  dist?: number;
  space?: string;
  reason?: string;
  pass?: number;
  pass2?: boolean;
};

export type SimilarRaw = {
  track_id?: number;
  score?: number;
  reid?: number | null;
  motion?: number | null;
  size?: number | null;
  gap?: number | null;
  dist?: number | null;
  space?: string | null;
  reason?: string | null;
};

export type MergeTimelineTrack = {
  track_id: number;
  t0: number;
  t1: number;
  group_id: number | null;
  global_id?: number | null;
};

export type MergeTimelineGroup = {
  group_id: number;
  track_ids: number[];
  score: number | null;
  reason: string | null;
  passes?: number[];
};

export type MergeTimelinePair = {
  a: number;
  b: number;
  score: number;
  reason: string | null;
  reid: number | null;
  face?: number | null;
  face_scores?: Record<string, number> | null;
  pose_face?: number | null;
  motion?: number | null;
  size?: number | null;
  gap?: number | null;
  dist?: number | null;
  space?: string | null;
  pass?: number | null;
  pass2?: boolean | null;
};

export type MergeTimelineSummary = {
  method: string | null;
  model: string | null;
  min_score: number | null;
  n_pairs: number | null;
  n_groups: number | null;
  complete_link: boolean | null;
};

export type MergeTimeline = {
  duration_sec: number;
  groups: MergeTimelineGroup[];
  tracks: MergeTimelineTrack[];
  pairs: MergeTimelinePair[];
  summary: MergeTimelineSummary;
};

export type MergePairRaw = {
  a?: number;
  b?: number;
  score?: number;
  reason?: string;
  reid?: number;
};

export type MergeJsonRaw = {
  method?: string;
  model?: string;
  min_score?: number;
  pass1_min_score?: number;
  auto_min_score?: number;
  n_pairs?: number;
  n_groups?: number;
  complete_link?: boolean;
  group_mode?: string;
  pairs?: MergePairRaw[];
  auto_pairs?: MergePairRaw[];
  auto_groups?: {
    track_ids?: number[];
    score?: number;
    reason?: string;
  }[];
  tracks?: {
    track_id?: number;
    group_id?: number | null;
    t0?: number;
    t1?: number;
  }[];
  groups?: {
    group_id?: number;
    track_ids?: number[];
    score?: number;
    reason?: string;
  }[];
};

export function numOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function linkPass(e: { pass?: number | null; pass2?: boolean | null }): number | null {
  if (typeof e.pass === "number" && e.pass >= 0) return e.pass;
  if (e.pass2) return 2;
  return null;
}

export function edgeMetrics(e: LinkEdge): Omit<SimilarHit, "track_id" | "group_id" | "t0" | "t1"> {
  const reason = typeof e.reason === "string" && e.reason.trim() ? e.reason : "skip";
  const pass = linkPass(e);
  return {
    score: typeof e.score === "number" ? e.score : typeof e.reid === "number" ? e.reid : 0,
    reid: numOrNull(e.reid),
    motion: numOrNull(e.motion),
    size: numOrNull(e.size),
    gap: numOrNull(e.gap),
    dist: numOrNull(e.dist),
    space: typeof e.space === "string" ? e.space : null,
    reason,
    pass,
  };
}

export function bestEdgesByPair(
  edges: LinkEdge[],
  pairOf: (e: LinkEdge) => [number, number] | null,
): LinkEdge[] {
  const best = new Map<string, LinkEdge>();
  for (const e of edges) {
    const ids = pairOf(e);
    if (!ids) continue;
    const key = ids[0] < ids[1] ? `${ids[0]}:${ids[1]}` : `${ids[1]}:${ids[0]}`;
    const combo = typeof e.score === "number" ? e.score : -1;
    const prev = best.get(key);
    const prevCombo = prev && typeof prev.score === "number" ? prev.score : -1;
    if (!prev || combo > prevCombo) best.set(key, e);
  }
  return [...best.values()];
}

export function hitsByTrack(base: string, file: string): Record<string, SimilarHit[]> {
  const data = readJsonFile(workFile(base, file)) as {
    tracks?: {
      track_id: number;
      group_id?: number | null;
      t0?: number;
      t1?: number;
      similar?: SimilarRaw[];
    }[];
    groups?: { track_ids?: number[]; score?: number; reason?: string }[];
  } | null;
  const out: Record<string, SimilarHit[]> = {};
  if (!data) return out;

  const byId = new Map((data.tracks ?? []).map((t) => [t.track_id, t]));

  for (const t of data.tracks ?? []) {
    const hits = (t.similar ?? [])
      .filter(
        (x): x is SimilarRaw & { track_id: number; score: number } =>
          typeof x.track_id === "number" && typeof x.score === "number" && Number.isFinite(x.score),
      )
      .map((x) => {
        const nb = byId.get(x.track_id);
        return {
          track_id: x.track_id,
          score: x.score,
          reid: numOrNull(x.reid),
          motion: numOrNull(x.motion),
          size: numOrNull(x.size),
          gap: numOrNull(x.gap),
          dist: numOrNull(x.dist),
          space: typeof x.space === "string" ? x.space : null,
          reason: typeof x.reason === "string" && x.reason.trim() ? x.reason : null,
          group_id: typeof t.group_id === "number" ? t.group_id : null,
          t0: numOrNull(nb?.t0),
          t1: numOrNull(nb?.t1),
        };
      });
    if (hits.length) out[String(t.track_id)] = hits;
  }

  if (Object.keys(out).length === 0) {
    for (const cl of data.groups ?? []) {
      const ids = (cl.track_ids ?? []).filter((n) => typeof n === "number");
      const score = typeof cl.score === "number" && Number.isFinite(cl.score) ? cl.score : NaN;
      const reason = typeof cl.reason === "string" && cl.reason.trim() ? cl.reason : null;
      for (const id of ids) {
        out[String(id)] = ids
          .filter((other) => other !== id)
          .map((track_id) => ({
            track_id,
            score,
            reason,
          }));
      }
    }
  }
  return out;
}

/** Skip-рёбра tracklet_link, ключи — итоговые track_id (вкладка Треки). */
export function skipHitsByGlobalId(base: string): Record<string, SimilarHit[]> | null {
  const links = readJsonFile(workFile(base, "tracklet_links.json")) as {
    edges?: LinkEdge[];
    tracklet_to_global?: Record<string, number>;
  } | null;
  const mapping = links?.tracklet_to_global;
  if (!mapping || !Object.keys(mapping).length) return null;

  const tracks = readJsonFile(workFile(base, "tracks.json")) as {
    tracks?: { track_id?: number; t0?: number; t1?: number }[];
  } | null;
  const spans = new Map<number, { t0: number; t1: number }>();
  for (const t of tracks?.tracks ?? []) {
    if (typeof t.track_id !== "number") continue;
    spans.set(t.track_id, {
      t0: typeof t.t0 === "number" ? t.t0 : 0,
      t1: typeof t.t1 === "number" ? t.t1 : 0,
    });
  }

  const out: Record<string, SimilarHit[]> = {};
  const edges = bestEdgesByPair(links?.edges ?? [], (e) => {
    if (typeof e.from !== "number" || typeof e.to !== "number") return null;
    const gFrom = mapping[String(e.from)];
    const gTo = mapping[String(e.to)];
    if (typeof gFrom !== "number" || typeof gTo !== "number" || gFrom === gTo) return null;
    return [gFrom, gTo];
  });
  for (const e of edges) {
    if (typeof e.from !== "number" || typeof e.to !== "number") continue;
    const gFrom = mapping[String(e.from)]!;
    const gTo = mapping[String(e.to)]!;
    const metrics = edgeMetrics(e);
    const push = (src: number, dst: number) => {
      const sp = spans.get(dst);
      (out[String(src)] ??= []).push({
        track_id: dst,
        ...metrics,
        group_id: null,
        t0: sp?.t0 ?? null,
        t1: sp?.t1 ?? null,
      });
    };
    push(gFrom, gTo);
    push(gTo, gFrom);
  }
  for (const hits of Object.values(out)) hits.sort((a, b) => b.score - a.score);
  return out;
}

export function similarFromTrackletLinks(base: string): Record<string, SimilarHit[]> | null {
  const links = readJsonFile(workFile(base, "tracklet_links.json")) as {
    edges?: LinkEdge[];
    tracklet_to_global?: Record<string, number>;
  } | null;
  const mapping = links?.tracklet_to_global;
  if (!mapping || !Object.keys(mapping).length) return null;
  const tls = readJsonFile(workFile(base, "tracklets.json")) as {
    tracklets?: { tracklet_id?: number; t0?: number; t1?: number }[];
  } | null;
  const spans = new Map<number, { t0: number; t1: number }>();
  for (const t of tls?.tracklets ?? []) {
    if (typeof t.tracklet_id !== "number") continue;
    spans.set(t.tracklet_id, {
      t0: typeof t.t0 === "number" ? t.t0 : 0,
      t1: typeof t.t1 === "number" ? t.t1 : 0,
    });
  }
  const out: Record<string, SimilarHit[]> = {};
  const edges = bestEdgesByPair(links?.edges ?? [], (e) => {
    if (typeof e.from !== "number" || typeof e.to !== "number") return null;
    if (mapping[String(e.from)] === mapping[String(e.to)]) return null;
    return [e.from, e.to];
  });
  for (const e of edges) {
    if (typeof e.from !== "number" || typeof e.to !== "number") continue;
    const metrics = edgeMetrics(e);
    const push = (src: number, dst: number) => {
      const sp = spans.get(dst);
      (out[String(src)] ??= []).push({
        track_id: dst,
        ...metrics,
        group_id: typeof mapping[String(dst)] === "number" ? mapping[String(dst)] : null,
        t0: sp?.t0 ?? null,
        t1: sp?.t1 ?? null,
      });
    };
    push(e.from, e.to);
    push(e.to, e.from);
  }
  for (const hits of Object.values(out)) hits.sort((a, b) => b.score - a.score);
  return out;
}

export function mergeHitsFor(base: string): Record<string, SimilarHit[]> {
  const remapped = skipHitsByGlobalId(base);
  if (remapped) return remapped;
  return hitsByTrack(base, "link.json");
}

export function mergeTimelineFromTracklets(base: string): MergeTimeline | null {
  const links = readJsonFile(workFile(base, "tracklet_links.json")) as {
    solver?: string;
    groups?: number[][];
    edges?: { from?: number; to?: number; score?: number; reid?: number }[];
    tracklet_to_global?: Record<string, number>;
    pass2_merged?: number;
  } | null;
  const tls = readJsonFile(workFile(base, "tracklets.json")) as {
    tracklets?: { tracklet_id?: number; t0?: number; t1?: number }[];
  } | null;
  const mapping = links?.tracklet_to_global;
  const rawTracklets = tls?.tracklets;
  if (!mapping || !Object.keys(mapping).length || !rawTracklets?.length) return null;

  const spans = new Map<number, { t0: number; t1: number }>();
  let maxT = 0;
  for (const t of rawTracklets) {
    if (typeof t.tracklet_id !== "number") continue;
    const t0 = typeof t.t0 === "number" && Number.isFinite(t.t0) ? t.t0 : 0;
    const t1 = typeof t.t1 === "number" && Number.isFinite(t.t1) ? t.t1 : t0;
    spans.set(t.tracklet_id, { t0, t1 });
    maxT = Math.max(maxT, t1);
  }

  const members = new Map<number, number[]>();
  for (const [key, gid] of Object.entries(mapping)) {
    const tid = Number(key);
    if (!Number.isFinite(tid) || typeof gid !== "number") continue;
    const list = members.get(gid) ?? [];
    list.push(tid);
    members.set(gid, list);
  }
  for (const ids of members.values()) ids.sort((a, b) => a - b);

  const tracks: MergeTimelineTrack[] = [];
  for (const [tid, span] of spans) {
    const gid = mapping[String(tid)];
    const grouped = typeof gid === "number" && (members.get(gid)?.length ?? 0) >= 2;
    tracks.push({
      track_id: tid,
      t0: span.t0,
      t1: span.t1,
      group_id: grouped ? gid : null,
      global_id: typeof gid === "number" ? gid : null,
    });
  }
  tracks.sort((a, b) => a.track_id - b.track_id);

  const edges = (links?.edges ?? []) as LinkEdge[];
  const groups: MergeTimelineGroup[] = [];
  for (const [gid, ids] of [...members.entries()].sort((a, b) => a[0] - b[0])) {
    if (ids.length < 2) continue;
    const idSet = new Set(ids);
    const intra = edges.filter(
      (e) => typeof e.from === "number" && typeof e.to === "number" && idSet.has(e.from) && idSet.has(e.to),
    );
    const used = intra.filter((e) => linkPass(e) != null);
    const source = used.length ? used : intra;
    const scores = source
      .map((e) => (typeof e.reid === "number" ? e.reid : typeof e.score === "number" ? e.score : null))
      .filter((n): n is number => n != null);
    const rawReason = source
      .map((e) => e.reason)
      .find((r): r is string => typeof r === "string" && r.length > 0);
    const metrics = rawReason ? rawReason.replace(/^Pass \d+:\s*/i, "") : "";
    groups.push({
      group_id: gid,
      track_ids: ids,
      score: scores.length ? Math.max(...scores) : null,
      reason: metrics || "Hungarian / MCF",
      passes: [...new Set(used.map((e) => linkPass(e) ?? 2))].sort((a, b) => a - b),
    });
  }

  const pairKey = (a: number, b: number) => (a < b ? `${a}:${b}` : `${b}:${a}`);
  const seenPairs = new Set<string>();
  const pairs: MergeTimelinePair[] = [];
  for (const e of edges) {
    if (typeof e.from !== "number" || typeof e.to !== "number") continue;
    const key = pairKey(e.from, e.to);
    if (seenPairs.has(key)) continue;
    seenPairs.add(key);
    const same = mapping[String(e.from)] === mapping[String(e.to)];
    const pass = linkPass(e);
    pairs.push({
      a: e.from,
      b: e.to,
      score: typeof e.score === "number" ? e.score : typeof e.reid === "number" ? e.reid : 0,
      reason: typeof e.reason === "string" ? e.reason : (same ? "merged" : "skip"),
      reid: typeof e.reid === "number" ? e.reid : null,
      motion: typeof e.motion === "number" ? e.motion : null,
      size: typeof e.size === "number" ? e.size : null,
      gap: typeof e.gap === "number" ? e.gap : null,
      dist: typeof e.dist === "number" ? e.dist : null,
      space: typeof e.space === "string" ? e.space : null,
      pass,
      pass2: pass === 2,
    });
  }

  const info = readJsonFile(workFile(base, "info.json")) as { duration_sec?: number } | null;
  const duration =
    typeof info?.duration_sec === "number" && info.duration_sec > 0 ? info.duration_sec : Math.max(maxT, 1);
  const summary: MergeTimelineSummary = {
    method: "tracklet_link",
    model: typeof links?.solver === "string" ? links.solver : null,
    min_score: null,
    n_pairs: pairs.length,
    n_groups: groups.length,
    complete_link: true,
  };
  return { duration_sec: duration, groups, tracks, pairs, summary };
}

export function mergeTimelineFromLinkJson(base: string): MergeTimeline | null {
  let data: MergeJsonRaw | null = null;

  data = readJsonFile(workFile(base, "link.json")) as MergeJsonRaw | null;
  if (!data?.tracks?.length && !data?.groups?.length && !data?.auto_groups?.length) {
    return null;
  }

  const tracks: MergeTimelineTrack[] = [];
  let maxT = 0;
  for (const t of data.tracks ?? []) {
    if (typeof t.track_id !== "number") continue;
    const t0 = typeof t.t0 === "number" && Number.isFinite(t.t0) ? t.t0 : 0;
    const t1 = typeof t.t1 === "number" && Number.isFinite(t.t1) ? t.t1 : t0;
    maxT = Math.max(maxT, t1);
    tracks.push({
      track_id: t.track_id,
      t0,
      t1,
      group_id: typeof t.group_id === "number" ? t.group_id : null,
    });
  }

  let groups: MergeTimelineGroup[] = (data.groups ?? [])
    .filter((g): g is { group_id: number; track_ids?: number[]; score?: number; reason?: string } => typeof g.group_id === "number")
    .map((g) => ({
      group_id: g.group_id,
      track_ids: (g.track_ids ?? []).filter((n) => typeof n === "number"),
      score: typeof g.score === "number" ? g.score : null,
      reason: typeof g.reason === "string" && g.reason.trim() ? g.reason : null,
    }))
    .sort((a, b) => a.group_id - b.group_id);

  if (!groups.length) {
    groups = (data.auto_groups ?? [])
      .map((g, i) => ({
        group_id: i + 1,
        track_ids: (g.track_ids ?? []).filter((n) => typeof n === "number"),
        score: typeof g.score === "number" ? g.score : null,
        reason: typeof g.reason === "string" && g.reason.trim() ? g.reason : "reid_auto",
      }))
      .filter((g) => g.track_ids.length >= 2);
  }

  const tidToGid = new Map<number, number>();
  for (const g of groups) {
    for (const tid of g.track_ids) tidToGid.set(tid, g.group_id);
  }
  for (const t of tracks) {
    if (t.group_id == null) t.group_id = tidToGid.get(t.track_id) ?? null;
  }

  if (!tracks.length && !groups.length) return null;

  const pairKey = (a: number, b: number) => (a < b ? `${a}:${b}` : `${b}:${a}`);
  const seenPairs = new Set<string>();
  const pairs: MergeTimelinePair[] = [];
  for (const p of [...(data.auto_pairs ?? []), ...(data.pairs ?? [])]) {
    if (typeof p.a !== "number" || typeof p.b !== "number") continue;
    const key = pairKey(p.a, p.b);
    if (seenPairs.has(key)) continue;
    seenPairs.add(key);
    pairs.push({
      a: p.a,
      b: p.b,
      score: typeof p.score === "number" ? p.score : 0,
      reason: typeof p.reason === "string" && p.reason.trim() ? p.reason : null,
      reid: typeof p.reid === "number" ? p.reid : null,
    });
  }

  const completeLink =
    typeof data.complete_link === "boolean"
      ? data.complete_link
      : data.group_mode === "complete_link"
        ? true
        : data.group_mode
          ? false
          : null;
  const minScore =
    typeof data.min_score === "number"
      ? data.min_score
      : typeof data.pass1_min_score === "number"
        ? data.pass1_min_score
        : typeof data.auto_min_score === "number"
          ? data.auto_min_score
          : null;
  const summary: MergeTimelineSummary = {
    method: typeof data.method === "string" ? data.method : "reid_auto",
    model: typeof data.model === "string" ? data.model : null,
    min_score: minScore,
    n_pairs: typeof data.n_pairs === "number" ? data.n_pairs : pairs.length || null,
    n_groups: typeof data.n_groups === "number" ? data.n_groups : groups.length || null,
    complete_link: completeLink,
  };

  const info = readJsonFile(workFile(base, "info.json")) as { duration_sec?: number } | null;
  const duration =
    typeof info?.duration_sec === "number" && info.duration_sec > 0 ? info.duration_sec : Math.max(maxT, 1);

  return { duration_sec: duration, groups, tracks, pairs, summary };
}

export function mergeTimelineFor(base: string): MergeTimeline | null {
  return mergeTimelineFromTracklets(base) ?? mergeTimelineFromLinkJson(base);
}

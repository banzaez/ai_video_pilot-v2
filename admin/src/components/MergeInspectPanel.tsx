import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TrackingData } from "../types";
import {
  buildTrackKeyframes,
  colorForTrackId,
  detectionsAtFrame,
  formatDuration,
  resolveDetectEveryN,
  type CropShot,
  type FaceShot,
  type MergeTimeline,
  type MergeTimelineGroup,
  type MergeTimelinePair,
  type MergeTimelineSummary,
  type MergeTimelineTrack,
  type PipelineStaleReport,
  type SimilarHit,
} from "../utils";

export type MergeListFilter = "all" | "grouped" | "solo" | "weak";
export type MergeListSort = "group_id" | "score" | "tracks" | "span";

export type MergeListRow =
  | { kind: "group"; group: MergeTimelineGroup; tracks: MergeTimelineTrack[]; span: number }
  | { kind: "solo"; track: MergeTimelineTrack };

type Props = {
  activeVideo: string;
  mergeTimeline: MergeTimeline | null;
  cropUrls: Record<string, CropShot[]>;
  faceUrls?: Record<string, FaceShot[]>;
  faceUrlsByModel?: Record<string, Record<string, FaceShot[]>>;
  faceModels?: string[];
  cameraLink?: {
    face_models?: string[];
    edges?: MergeTimelinePair[];
    candidate_edges?: MergeTimelinePair[];
  } | null;
  similarByTrack: Record<string, SimilarHit[]>;
  pipeline: PipelineStaleReport | null;
  tracking: TrackingData | null;
  currentSec: number;
  currentFrame: number;
  selectedTrackId: number | null;
  onSelectTrackId: (trackId: number | null) => void;
  onSeekToSec: (sec: number) => void;
  onFocusTracks: (ids: number[] | null) => void;
};

function formatScore(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toFixed(2);
}

function copyText(text: string) {
  navigator.clipboard?.writeText(text).catch(() => {});
}

/** Начало / середина / конец, чтобы в длинном треклете была видна смена человека. */
function cropPreview(crops: CropShot[]): CropShot[] {
  const ordered = [...crops].sort((a, b) => (a.frame ?? 0) - (b.frame ?? 0) || a.rank - b.rank);
  if (ordered.length <= 4) return ordered;
  const last = ordered.length - 1;
  const a = Math.floor(last / 3);
  const b = Math.floor((2 * last) / 3);
  const idxs = [...new Set([0, a, b, last])];
  return idxs.map((i) => ordered[i]);
}

function spanSec(tracks: MergeTimelineTrack[]): number {
  if (!tracks.length) return 0;
  const t0 = Math.min(...tracks.map((t) => t.t0));
  const t1 = Math.max(...tracks.map((t) => t.t1));
  return Math.max(0, t1 - t0);
}

function overlaps(a: MergeTimelineTrack, b: MergeTimelineTrack): boolean {
  return a.t0 < b.t1 && b.t0 < a.t1;
}

function sanityForGroup(
  group: MergeTimelineGroup,
  tracks: MergeTimelineTrack[],
  minScore: number | null,
): string[] {
  const warnings: string[] = [];
  for (let i = 0; i < tracks.length; i++) {
    for (let j = i + 1; j < tracks.length; j++) {
      if (overlaps(tracks[i], tracks[j])) {
        warnings.push("overlap");
        break;
      }
    }
    if (warnings.includes("overlap")) break;
  }
  if (minScore != null && group.score != null && group.score < minScore && group.track_ids.length >= 2) {
    warnings.push("low_score");
  }
  return [...new Set(warnings)];
}

function duplicateTrackGroups(timeline: MergeTimeline): Set<number> {
  const seen = new Map<number, number>();
  const dup = new Set<number>();
  for (const g of timeline.groups) {
    for (const tid of g.track_ids) {
      const prev = seen.get(tid);
      if (prev != null) {
        dup.add(tid);
        dup.add(prev);
      } else {
        seen.set(tid, g.group_id);
      }
    }
  }
  return dup;
}

function pairsForGroup(
  pairs: MergeTimelinePair[],
  trackIds: number[],
  tracks: MergeTimelineTrack[] = [],
): MergeTimelinePair[] {
  const set = new Set(trackIds);
  const trackMap = new Map(tracks.map((t) => [t.track_id, t]));
  const intra = pairs.filter((p) => set.has(p.a) && set.has(p.b));
  const used = intra.filter((p) => pairPass(p) != null);
  const filtered = used.length ? used : intra;
  return filtered.sort((p1, p2) => {
    const t1 = trackMap.get(p1.a)?.t0 ?? 0;
    const t2 = trackMap.get(p2.a)?.t0 ?? 0;
    return t1 - t2 || p1.a - p2.a;
  });
}

function pairPass(p: { pass?: number | null; pass2?: boolean | null }): number | null {
  if (typeof p.pass === "number" && p.pass >= 1) return p.pass;
  if (p.pass2) return 2;
  return null;
}

function stripPassPrefix(reason: string): string {
  return reason.replace(/^Pass \d+:\s*/i, "");
}

function PassBadge({ pass }: { pass: number | null }) {
  if (typeof pass !== "number" || pass < 1) return null;
  if (pass === 10) {
    return <span className="badge-pass badge-pass10" title="Pass 10 (Camera Link / Face + Feet)">Pass 10 · Face + Feet</span>;
  }
  return <span className="badge-pass">Pass {pass}</span>;
}

function GroupFaceThumb({
  faces,
  facesByModel,
  faceModels,
  groupKey,
}: {
  faces?: FaceShot[];
  facesByModel?: Record<string, Record<string, FaceShot[]>>;
  faceModels?: string[];
  groupKey: string;
}) {
  const models = faceModels?.length ? faceModels : ["buffalo_l"];
  const [modelIdx, setModelIdx] = useState(0);
  const [idx, setIdx] = useState(0);
  const activeModel = models[Math.min(modelIdx, models.length - 1)] ?? models[0];
  const modelFaces = facesByModel?.[activeModel]?.[groupKey] ?? faces ?? [];
  if (!modelFaces.length) return null;
  const safeIdx = Math.min(Math.max(0, idx), modelFaces.length - 1);
  const currentFace = modelFaces[safeIdx];

  return (
    <div
      className="merge-inspect-face-box"
      title={`${activeModel}: лицо ${safeIdx + 1}/${modelFaces.length} (det ${currentFace.score?.toFixed(2) ?? "—"}, f${currentFace.frame})`}
      onClick={(e) => {
        if (modelFaces.length > 1) {
          e.stopPropagation();
          setIdx((prev) => (prev + 1) % modelFaces.length);
        }
      }}
    >
      {models.length > 1 && (
        <div className="merge-inspect-face-models" onClick={(e) => e.stopPropagation()}>
          {models.map((model, i) => (
            <button
              key={model}
              type="button"
              className={i === modelIdx ? "on" : ""}
              title={`Кропы ${model}`}
              onClick={(e) => {
                e.stopPropagation();
                setModelIdx(i);
                setIdx(0);
              }}
            >
              {model}
            </button>
          ))}
        </div>
      )}
      <img src={currentFace.url} alt="" loading="lazy" />
      {modelFaces.length > 1 && (
        <div className="merge-inspect-face-nav" onClick={(e) => e.stopPropagation()}>
          <button
            type="button"
            className="merge-inspect-face-btn"
            title="Пред. лицо"
            onClick={(e) => {
              e.stopPropagation();
              setIdx((prev) => (prev <= 0 ? modelFaces.length - 1 : prev - 1));
            }}
          >
            ‹
          </button>
          <span className="merge-inspect-face-idx">{safeIdx + 1}/{modelFaces.length}</span>
          <button
            type="button"
            className="merge-inspect-face-btn"
            title="След. лицо"
            onClick={(e) => {
              e.stopPropagation();
              setIdx((prev) => (prev + 1) % modelFaces.length);
            }}
          >
            ›
          </button>
        </div>
      )}
    </div>
  );
}

type RejectedHit = SimilarHit & { from_id: number };

function similarRejected(
  trackIds: number[],
  similarByTrack: Record<string, SimilarHit[]>,
  groupTrackSet: Set<number>,
): RejectedHit[] {
  const out: RejectedHit[] = [];
  const seen = new Set<string>();
  for (const tid of trackIds) {
    for (const hit of similarByTrack[String(tid)] ?? []) {
      if (groupTrackSet.has(hit.track_id)) continue;
      const key = `${Math.min(tid, hit.track_id)}:${Math.max(tid, hit.track_id)}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ ...hit, from_id: tid });
    }
  }
  return out.sort((a, b) => b.score - a.score);
}

function motionLabel(p: {
  motion?: number | null;
  dist?: number | null;
  space?: string | null;
}): string {
  const distUnit = p.space === "map" ? "м" : "px";
  if (p.motion != null) {
    return `${formatScore(p.motion)}${p.dist != null ? ` (${p.dist.toFixed(1)}${distUnit})` : ""}`;
  }
  if (p.dist != null) return `${p.dist.toFixed(1)}${distUnit}`;
  return "—";
}

function gapLabel(gap: number | null | undefined): string {
  return gap != null && Number.isFinite(gap) ? `${gap.toFixed(1)}с` : "—";
}

function faceModelLabels(models: string[]): string[] {
  return models.length ? models : ["buffalo_l"];
}

function faceScoreForModel(
  p: { face_scores?: Record<string, number> | null; face?: number | null },
  model: string,
): string {
  const fromMap = p.face_scores?.[model];
  if (fromMap != null && Number.isFinite(fromMap)) return formatScore(fromMap);
  if (model === faceModelLabels([])[0] && p.face != null) return formatScore(p.face);
  return "—";
}

function videoTrackId(t: MergeTimelineTrack): number {
  return t.global_id ?? t.track_id;
}

function fragmentLive(t: MergeTimelineTrack, currentSec: number): boolean {
  return currentSec >= t.t0 && currentSec <= t.t1;
}

export function MergeInspectPanel({
  activeVideo,
  mergeTimeline,
  cropUrls,
  faceUrls,
  faceUrlsByModel,
  faceModels = [],
  cameraLink,
  similarByTrack,
  pipeline,
  tracking,
  currentSec,
  currentFrame,
  selectedTrackId,
  onSelectTrackId,
  onSeekToSec,
  onFocusTracks,
}: Props) {
  const [query, setQuery] = useState("");
  const [listFilter, setListFilter] = useState<MergeListFilter>("all");
  const [sortBy, setSortBy] = useState<MergeListSort>("group_id");
  const [selectedRowKey, setSelectedRowKey] = useState<string | null>(null);
  const [selectedFragmentId, setSelectedFragmentId] = useState<number | null>(null);
  const [atCurrentFrame, setAtCurrentFrame] = useState(false);
  const selectedGroupRef = useRef<HTMLDivElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const selectedBarRef = useRef<HTMLButtonElement>(null);

  const minScore = mergeTimeline?.summary?.min_score ?? null;
  const dupTracks = useMemo(
    () => (mergeTimeline ? duplicateTrackGroups(mergeTimeline) : new Set<number>()),
    [mergeTimeline],
  );

  const soloTracks = useMemo(() => {
    if (!mergeTimeline) return [];
    return mergeTimeline.tracks.filter((t) => t.group_id == null);
  }, [mergeTimeline]);

  const rows = useMemo((): MergeListRow[] => {
    if (!mergeTimeline) return [];
    const list: MergeListRow[] = mergeTimeline.groups.map((group) => {
      const tracks = mergeTimeline.tracks
        .filter((t) => t.group_id === group.group_id || group.track_ids.includes(t.track_id))
        .sort((a, b) => a.t0 - b.t0 || a.track_id - b.track_id);
      return { kind: "group", group, tracks, span: spanSec(tracks) };
    });
    for (const t of soloTracks) list.push({ kind: "solo", track: t });
    return list;
  }, [mergeTimeline, soloTracks]);

  const filteredRows = useMemo(() => {
    let list = rows;
    const q = query.trim().toLowerCase();
    const cleanQ = q.replace(/^#/, "");
    if (q) {
      list = list.filter((row) => {
        if (row.kind === "solo") {
          return (
            String(row.track.track_id).includes(cleanQ) ||
            String(videoTrackId(row.track)).includes(cleanQ) ||
            q === "solo"
          );
        }
        const gq = `g${row.group.group_id}`;
        if (
          gq.includes(q) ||
          String(row.group.group_id).includes(cleanQ) ||
          `t${row.group.group_id}`.includes(q)
        )
          return true;
        if (row.group.reason?.toLowerCase().includes(q)) return true;
        return row.tracks.some(
          (t) => String(t.track_id).includes(cleanQ) || String(videoTrackId(t)).includes(cleanQ),
        );
      });
    }
    if (listFilter === "grouped") list = list.filter((r) => r.kind === "group");
    else if (listFilter === "solo") list = list.filter((r) => r.kind === "solo");
    else if (listFilter === "weak") {
      list = list.filter((r) => {
        if (r.kind === "solo") return false;
        return minScore != null && r.group.score != null && r.group.score < minScore;
      });
    }
    const sorted = [...list];
    sorted.sort((a, b) => {
      if (sortBy === "score") {
        const sa = a.kind === "group" ? (a.group.score ?? -1) : -1;
        const sb = b.kind === "group" ? (b.group.score ?? -1) : -1;
        return sb - sa;
      }
      if (sortBy === "tracks") {
        const na = a.kind === "group" ? a.tracks.length : 1;
        const nb = b.kind === "group" ? b.tracks.length : 1;
        return nb - na;
      }
      if (sortBy === "span") {
        const sa = a.kind === "group" ? a.span : a.track.t1 - a.track.t0;
        const sb = b.kind === "group" ? b.span : b.track.t1 - b.track.t0;
        return sb - sa;
      }
      const ga = a.kind === "group" ? a.group.group_id : 1e9;
      const gb = b.kind === "group" ? b.group.group_id : 1e9;
      return ga - gb;
    });
    return sorted;
  }, [rows, query, listFilter, sortBy, minScore]);

  const activeRowKey = selectedRowKey;

  const selectedRow = useMemo(() => {
    if (!activeRowKey || !mergeTimeline) return null;
    return filteredRows.find((r) => rowKey(r) === activeRowKey) ?? rows.find((r) => rowKey(r) === activeRowKey) ?? null;
  }, [activeRowKey, filteredRows, rows, mergeTimeline]);

  const focusTrackIds = useMemo((): number[] | null => {
    if (!selectedRow) {
      if (selectedTrackId != null) return [selectedTrackId];
      return null;
    }
    if (selectedRow.kind === "solo") return [videoTrackId(selectedRow.track)];
    const ids = [...new Set(selectedRow.tracks.map((t) => videoTrackId(t)))];
    return ids;
  }, [selectedRow, selectedTrackId]);

  const detectEveryN = tracking ? resolveDetectEveryN(tracking) : 1;
  const inFrameIds = useMemo(() => {
    if (!tracking) return new Set<number>();
    const keyframes = buildTrackKeyframes(tracking);
    const dets = detectionsAtFrame(keyframes, currentFrame, detectEveryN);
    return new Set(dets.map((d) => d.track_id));
  }, [tracking, currentFrame, detectEveryN]);

  const activeAtFrame = useMemo(() => {
    if (!atCurrentFrame) return null;
    return inFrameIds;
  }, [atCurrentFrame, inFrameIds]);

  const effectiveFocusIds = useMemo(() => {
    if (!focusTrackIds) return null;
    if (!activeAtFrame) return focusTrackIds;
    return focusTrackIds.filter((id) => activeAtFrame.has(id));
  }, [focusTrackIds, activeAtFrame]);

  useEffect(() => {
    onFocusTracks(effectiveFocusIds);
  }, [effectiveFocusIds, onFocusTracks]);

  // Synchronize when external selectedTrackId changes (e.g. clicking on video)
  useEffect(() => {
    if (selectedTrackId == null || !mergeTimeline) return;
    // Don't disturb if already in the same group or track
    if (selectedRowKey) {
      const curRow = rows.find((r) => rowKey(r) === selectedRowKey);
      if (curRow) {
        if (
          curRow.kind === "group" &&
          curRow.tracks.some(
            (t) => t.track_id === selectedFragmentId || videoTrackId(t) === selectedTrackId,
          )
        ) {
          return;
        }
        if (
          curRow.kind === "solo" &&
          (curRow.track.track_id === selectedFragmentId ||
            videoTrackId(curRow.track) === selectedTrackId)
        ) {
          return;
        }
      }
    }
    const tr =
      mergeTimeline.tracks.find((t) => t.track_id === selectedTrackId) ??
      mergeTimeline.tracks.find((t) => t.global_id === selectedTrackId);
    if (!tr) return;
    const key = tr.group_id != null ? `g:${tr.group_id}` : `solo:${tr.track_id}`;
    setSelectedRowKey(key);
    setSelectedFragmentId(tr.track_id);
  }, [selectedTrackId, mergeTimeline, selectedRowKey, selectedFragmentId, rows]);

  useEffect(() => {
    selectedBarRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    if (selectedRow) {
      selectedGroupRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [selectedTrackId, selectedRow]);

  const clearSelection = useCallback(() => {
    setSelectedRowKey(null);
    setSelectedFragmentId(null);
    onSelectTrackId(null);
  }, [onSelectTrackId]);

  const chronTracks = useMemo(() => {
    if (!selectedRow || selectedRow.kind !== "group") return [];
    return [...selectedRow.tracks].sort((a, b) => a.t0 - b.t0 || a.track_id - b.track_id);
  }, [selectedRow]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "Escape") {
        clearSelection();
        return;
      }
      if (!chronTracks.length) return;
      const idx = selectedFragmentId != null ? chronTracks.findIndex((t) => t.track_id === selectedFragmentId) : -1;
      if (e.key === "ArrowRight") {
        e.preventDefault();
        const nextIdx = idx < 0 ? 0 : Math.min(idx + 1, chronTracks.length - 1);
        const tr = chronTracks[nextIdx];
        pickTrack(tr.track_id, tr.t0);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        const nextIdx = idx <= 0 ? 0 : idx - 1;
        const tr = chronTracks[nextIdx];
        pickTrack(tr.track_id, tr.t0);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [chronTracks, selectedFragmentId, clearSelection]);

  function rowKey(row: MergeListRow): string {
    return row.kind === "group" ? `g:${row.group.group_id}` : `solo:${row.track.track_id}`;
  }

  function pickRow(row: MergeListRow) {
    const key = rowKey(row);
    if (activeRowKey === key) {
      clearSelection();
      return;
    }
    setSelectedRowKey(key);
    if (row.kind === "solo") {
      setSelectedFragmentId(row.track.track_id);
      onSelectTrackId(videoTrackId(row.track));
      onSeekToSec(row.track.t0);
    } else if (row.tracks.length) {
      const sorted = [...row.tracks].sort((a, b) => a.t0 - b.t0);
      const first = sorted[0];
      setSelectedFragmentId(first.track_id);
      onSelectTrackId(videoTrackId(first));
      onSeekToSec(first.t0);
    }
  }

  function trackT0(trackId: number): number {
    const tr = mergeTimeline?.tracks.find((t) => t.track_id === trackId);
    return tr?.t0 ?? 0;
  }

  function pickTrack(trackId: number, sec: number) {
    const tr = mergeTimeline?.tracks.find((t) => t.track_id === trackId);
    if (tr) {
      setSelectedRowKey(tr.group_id != null ? `g:${tr.group_id}` : `solo:${trackId}`);
      setSelectedFragmentId(trackId);
      onSelectTrackId(videoTrackId(tr));
    } else {
      setSelectedFragmentId(trackId);
      onSelectTrackId(trackId);
    }
    onSeekToSec(sec);
  }

  const duration = Math.max(mergeTimeline?.duration_sec ?? 1, 0.001);
  const pairs = mergeTimeline?.pairs ?? [];
  const summary: MergeTimelineSummary = mergeTimeline?.summary ?? {
    method: null,
    model: null,
    min_score: null,
    n_pairs: null,
    n_groups: null,
    complete_link: null,
  };
  const nGrouped = mergeTimeline?.tracks.filter((t) => t.group_id != null).length ?? 0;
  const nSolo = soloTracks.length;
  const playhead = Math.min(100, Math.max(0, (currentSec / duration) * 100));

  const allTracksSorted = useMemo(() => {
    if (!mergeTimeline) return [];
    return [...mergeTimeline.tracks].sort((a, b) => {
      const ga = a.group_id ?? 1e9;
      const gb = b.group_id ?? 1e9;
      if (ga !== gb) return ga - gb;
      return a.t0 - b.t0 || a.track_id - b.track_id;
    });
  }, [mergeTimeline]);

  const filteredTracks = useMemo(() => {
    if (!activeAtFrame) return allTracksSorted;
    return allTracksSorted.filter((t) => fragmentLive(t, currentSec) && activeAtFrame.has(videoTrackId(t)));
  }, [allTracksSorted, activeAtFrame, currentSec]);

  const timelineSections = useMemo(() => {
    const byGroup = new Map<number | "solo", MergeTimelineTrack[]>();
    for (const t of filteredTracks) {
      const key = t.group_id ?? "solo";
      const arr = byGroup.get(key) ?? [];
      arr.push(t);
      byGroup.set(key, arr);
    }
    const keys = [...byGroup.keys()].sort((a, b) => {
      if (a === "solo") return 1;
      if (b === "solo") return -1;
      return a - b;
    });
    return keys.map((key) => ({
      key,
      label: key === "solo" ? "без склейки" : `трек #${key}`,
      groupId: key === "solo" ? null : key,
      reason:
        key === "solo"
          ? null
          : mergeTimeline?.groups.find((g) => g.group_id === key)?.reason ?? null,
      score:
        key === "solo" ? null : mergeTimeline?.groups.find((g) => g.group_id === key)?.score ?? null,
      tracks: byGroup.get(key) ?? [],
    }));
  }, [filteredTracks, mergeTimeline]);

  const visibleSections = useMemo(() => {
    if (!activeRowKey) return timelineSections;
    if (activeRowKey.startsWith("g:")) {
      const gid = Number(activeRowKey.slice(2));
      const groupRow = rows.find((r) => r.kind === "group" && r.group.group_id === gid);
      const allGroupTracks = groupRow && groupRow.kind === "group" ? [...groupRow.tracks].sort((a, b) => a.t0 - b.t0) : [];
      return [
        {
          key: gid,
          label: `трек #${gid}`,
          groupId: gid,
          reason: mergeTimeline?.groups.find((g) => g.group_id === gid)?.reason ?? null,
          score: mergeTimeline?.groups.find((g) => g.group_id === gid)?.score ?? null,
          tracks: allGroupTracks,
        },
      ];
    }
    if (activeRowKey.startsWith("solo:")) {
      const tid = Number(activeRowKey.slice(5));
      const soloTrack = soloTracks.find((t) => t.track_id === tid);
      return soloTrack
        ? [
            {
              key: "solo",
              label: "без склейки",
              groupId: null,
              reason: null,
              score: null,
              tracks: [soloTrack],
            },
          ]
        : [];
    }
    return timelineSections;
  }, [timelineSections, activeRowKey, rows, soloTracks, mergeTimeline]);

  const groupTrackSet = useMemo(() => {
    if (!selectedRow || selectedRow.kind !== "group") return new Set<number>();
    return new Set(selectedRow.tracks.map((t) => t.track_id));
  }, [selectedRow]);

  const selectedGroupPairs =
    selectedRow?.kind === "group"
      ? pairsForGroup(pairs, selectedRow.group.track_ids, selectedRow.tracks)
      : [];
  const compareFaceModels = useMemo(
    () => faceModelLabels(cameraLink?.face_models?.length ? cameraLink.face_models : faceModels),
    [cameraLink?.face_models, faceModels],
  );
  const cameraLinkPairs = useMemo(() => {
    const rows = cameraLink?.candidate_edges?.length
      ? cameraLink.candidate_edges
      : cameraLink?.edges ?? [];
    if (!selectedRow || selectedRow.kind !== "group") return rows;
    const gid = selectedRow.group.group_id;
    return rows.filter((p) => p.a === gid || p.b === gid);
  }, [cameraLink, selectedRow]);
  const rejectedSimilar: RejectedHit[] =
    selectedRow?.kind === "group"
      ? similarRejected(selectedRow.group.track_ids, similarByTrack, groupTrackSet)
      : selectedRow?.kind === "solo"
        ? (similarByTrack[String(selectedRow.track.track_id)] ?? []).map((h) => ({
            ...h,
            from_id: selectedRow.track.track_id,
          }))
        : [];

  const staleMerge = pipeline?.stale?.includes("tracklet_link");
  const fromTracklets = summary.method === "tracklet_link";

  if (!activeVideo) {
    return <p className="merge-inspect-empty">Выберите видео в шапке</p>;
  }

  if (!mergeTimeline) {
    return (
      <div className="merge-inspect-panel">
        <p className="merge-inspect-empty">
          Нет tracklet_links.json — сначала стадии tracklets → tracklet_link
        </p>
      </div>
    );
  }

  return (
    <div className="merge-inspect-panel">
      <div className="merge-inspect-summary">
        <span>
          {mergeTimeline.groups.length} склеек · {nGrouped} фрагментов · {nSolo} без склейки · {pairs.length} рёбер
        </span>
        {summary.model && <span>{summary.model}</span>}
        {minScore != null && <span>min {formatScore(minScore)}</span>}
        {summary.complete_link != null && <span>{summary.complete_link ? "complete_link" : "partial"}</span>}
      </div>

      {staleMerge && pipeline?.cli && (
        <div className="merge-inspect-stale">
          склейки устарели —{" "}
          <code title="Скопировать" onClick={() => copyText(pipeline.cli!)}>
            {pipeline.cli}
          </code>
        </div>
      )}

      <div className="merge-inspect-grid">
        {/* List column */}
        <aside className="merge-inspect-list">
          <input
            type="search"
            className="merge-inspect-search"
            placeholder="g3, #15, reason…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="merge-inspect-filters">
            {(["all", "grouped", "solo", "weak"] as const).map((f) => (
              <button key={f} type="button" className={listFilter === f ? "on" : ""} onClick={() => setListFilter(f)}>
                {f === "all" ? "все" : f === "grouped" ? "склейки" : f === "solo" ? "без склейки" : "слабые"}
              </button>
            ))}
          </div>
          <label className="merge-inspect-sort">
            <span>сорт</span>
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value as MergeListSort)}>
              <option value="group_id">группа</option>
              <option value="score">score</option>
              <option value="tracks">число треков</option>
              <option value="span">длительность</option>
            </select>
          </label>

          <ul className="merge-inspect-rows">
            {filteredRows.map((row) => {
              const key = rowKey(row);
              const isSelected = activeRowKey === key;
              const hasWarn =
                row.kind === "group"
                  ? sanityForGroup(row.group, row.tracks, minScore).length > 0 ||
                    row.tracks.some((t) => dupTracks.has(t.track_id))
                  : false;
              const groupKey =
                row.kind === "group" ? String(row.group.group_id) : String(row.track.track_id);
              const faces =
                row.kind === "group"
                  ? faceUrls?.[groupKey] ?? faceUrls?.[String(row.tracks[0]?.track_id)]
                  : faceUrls?.[groupKey];
              return (
                <li key={key}>
                  <button
                    type="button"
                    className={`merge-inspect-row${isSelected ? " on" : ""}`}
                    onClick={() => pickRow(row)}
                    title={row.kind === "group" ? row.group.reason ?? undefined : undefined}
                  >
                    <div className="merge-inspect-row-content">
                      {row.kind === "group" ? (
                        <>
                          <div className="merge-inspect-row-head">
                            <span className="merge-inspect-row-title">
                              <strong>{fromTracklets ? `трек #${row.group.group_id}` : `группа ${row.group.group_id}`}</strong>
                              {hasWarn && <span className="merge-inspect-warn-dot" title="Overlap или дубликат" />}
                            </span>
                            {row.group.score != null && (
                              <span className="merge-inspect-badge-score">{formatScore(row.group.score)}</span>
                            )}
                          </div>
                          <div className="merge-inspect-row-meta">
                            <span>{row.tracks.length} фрагм.</span>
                            <span>·</span>
                            <span>{formatDuration(row.span)}</span>
                          </div>
                          <div className="merge-inspect-ids" title={row.tracks.map((t) => `#${t.track_id}`).join(" ")}>
                            {row.tracks.map((t) => `#${t.track_id}`).join(" ")}
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="merge-inspect-row-head">
                            <span className="merge-inspect-row-title">
                              <strong>без склейки #{row.track.track_id}</strong>
                            </span>
                          </div>
                          <div className="merge-inspect-row-meta">
                            <span>{formatDuration(row.track.t0)}–{formatDuration(row.track.t1)}</span>
                            <span>·</span>
                            <span>{formatDuration(row.track.t1 - row.track.t0)}</span>
                          </div>
                        </>
                      )}
                    </div>
                    {(faces?.length || faceUrlsByModel) && (
                      <GroupFaceThumb
                        faces={faces}
                        facesByModel={faceUrlsByModel}
                        faceModels={faceModels}
                        groupKey={groupKey}
                      />
                    )}
                  </button>
                </li>
              );
            })}
            {!filteredRows.length && <li className="merge-inspect-empty">Нет строк по фильтру</li>}
          </ul>
        </aside>

        {/* Center column */}
        <section className="merge-inspect-center">
          <div className="merge-inspect-center-toolbar">
            <label className="toggle sidebar-toggle">
              <input type="checkbox" checked={atCurrentFrame} onChange={(e) => setAtCurrentFrame(e.target.checked)} />
              только текущий кадр
            </label>
            {activeRowKey && (
              <button type="button" className="merge-clear-sel" onClick={clearSelection}>
                сбросить выбор
              </button>
            )}
            {selectedRow?.kind === "group" && chronTracks.length > 1 && (
              <div className="merge-nav">
                <button
                  type="button"
                  title="Предыдущий фрагмент (стрелка влево)"
                  onClick={() => {
                    const idx = selectedFragmentId != null ? chronTracks.findIndex((t) => t.track_id === selectedFragmentId) : 0;
                    const nextIdx = idx <= 0 ? chronTracks.length - 1 : idx - 1;
                    const tr = chronTracks[nextIdx];
                    pickTrack(tr.track_id, tr.t0);
                  }}
                >
                  ← пред
                </button>
                <button
                  type="button"
                  title="Следующий фрагмент (стрелка вправо)"
                  onClick={() => {
                    const idx = selectedFragmentId != null ? chronTracks.findIndex((t) => t.track_id === selectedFragmentId) : -1;
                    const nextIdx = idx < 0 || idx >= chronTracks.length - 1 ? 0 : idx + 1;
                    const tr = chronTracks[nextIdx];
                    pickTrack(tr.track_id, tr.t0);
                  }}
                >
                  след →
                </button>
              </div>
            )}
          </div>

          {selectedRow?.kind === "group" && (
            <div className="merge-mini-scale">
              <span className="merge-mini-label">
                {fromTracklets ? `фрагменты трека #${selectedRow.group.group_id}` : `шкала группы ${selectedRow.group.group_id}`}
              </span>
              <div className="merge-mini-lane">
                {chronTracks.map((t, idx) => {
                  const left = (t.t0 / duration) * 100;
                  const width = Math.max(0.5, ((t.t1 - t.t0) / duration) * 100);
                  return (
                    <button
                      key={t.track_id}
                      type="button"
                      className={`merge-mini-bar${selectedFragmentId === t.track_id ? " on" : ""}`}
                      style={{ left: `${left}%`, width: `${width}%`, backgroundColor: colorForTrackId(t.track_id) }}
                      title={`[${idx + 1}/${chronTracks.length}] #${t.track_id} ${t.t0.toFixed(1)}–${t.t1.toFixed(1)}s → tracking #${videoTrackId(t)}`}
                      onClick={() => pickTrack(t.track_id, t.t0)}
                    />
                  );
                })}
              </div>
            </div>
          )}

          <div className="merge-inspect-axis">
            <span>0</span>
            <span>{formatDuration(duration / 2)}</span>
            <span>{formatDuration(duration)}</span>
          </div>

          <div className="merge-inspect-timeline" ref={timelineRef}>
            {visibleSections.length === 0 ? (
              <p className="merge-inspect-empty">Нет треков по фильтру</p>
            ) : (
              visibleSections.map((section) => {
                const sectionKey = section.groupId != null ? `g:${section.groupId}` : "solo";
                const sectionSelected = activeRowKey === sectionKey || activeRowKey?.startsWith("solo:");
                return (
                  <div
                    key={String(section.key)}
                    ref={sectionSelected ? selectedGroupRef : undefined}
                    className={`merge-inspect-group${sectionSelected ? " on" : ""}`}
                  >
                    <div className="merge-inspect-group-head">
                      <button
                        type="button"
                        className="merge-inspect-group-label"
                        onClick={() => {
                          if (section.groupId == null) return;
                          const row = rows.find(
                            (r) => r.kind === "group" && r.group.group_id === section.groupId,
                          );
                          if (row) pickRow(row);
                        }}
                      >
                        <strong>{section.label}</strong>
                        {section.score != null && <span>{formatScore(section.score)}</span>}
                        <span>{section.tracks.length} фрагм.</span>
                      </button>
                      {section.reason ? <em title={section.reason}>{section.reason}</em> : null}
                    </div>
                    {section.tracks.map((t) => {
                      const left = (t.t0 / duration) * 100;
                      const width = Math.max(0.4, ((t.t1 - t.t0) / duration) * 100);
                      const color = colorForTrackId(t.track_id);
                      const selected = selectedFragmentId === t.track_id;
                      const live = fragmentLive(t, currentSec);
                      const gid = t.group_id;
                      return (
                        <div
                          key={t.track_id}
                          className={`merge-inspect-track-row${selected ? " on" : ""}${live ? " is-live" : ""}`}
                        >
                          <button
                            type="button"
                            className="merge-inspect-track-id"
                            style={{ backgroundColor: color }}
                            onClick={() => pickTrack(t.track_id, t.t0)}
                          >
                            #{t.track_id}
                          </button>
                          <div className="merge-inspect-lane">
                            <div className="merge-inspect-playhead" style={{ left: `${playhead}%` }} />
                            <button
                              ref={selected ? selectedBarRef : undefined}
                              type="button"
                              className="merge-inspect-bar"
                              style={{ left: `${left}%`, width: `${width}%`, backgroundColor: color }}
                              title={`#${t.track_id}  ${t.t0.toFixed(1)}–${t.t1.toFixed(1)}s${gid != null ? `  g${gid}` : ""}`}
                              onClick={() => pickTrack(t.track_id, t.t0)}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                );
              })
            )}
          </div>

          {selectedRow && (
            <div className="merge-inspect-dossier">
              <h3>
                {selectedRow.kind === "group"
                  ? fromTracklets
                    ? `Трек #${selectedRow.group.group_id} · ${selectedRow.tracks.length} фрагментов (${chronTracks.map((t) => `#${t.track_id}`).join(", ")})`
                    : `Группа ${selectedRow.group.group_id} · ${selectedRow.tracks.length} треков`
                  : `Без склейки #${selectedRow.track.track_id}`}
              </h3>
              {selectedRow.kind === "group" && selectedRow.group.reason && (
                <p className="merge-inspect-reason">{selectedRow.group.reason}</p>
              )}
              <div className="merge-crop-compare">
                {(selectedRow.kind === "group" ? chronTracks : [selectedRow.track]).map((t, idx) => {
                  const crops = cropPreview(cropUrls[String(t.track_id)] ?? []);
                  const live = fragmentLive(t, currentSec);
                  const isCurFrag = selectedFragmentId === t.track_id;
                  return (
                    <div
                      key={t.track_id}
                      className={`merge-crop-col${live ? " is-live" : ""}${isCurFrag ? " is-current" : ""}`}
                      style={isCurFrag || live ? { borderColor: colorForTrackId(t.track_id) } : undefined}
                    >
                      <header>
                        <button type="button" onClick={() => pickTrack(t.track_id, t.t0)}>
                          {selectedRow.kind === "group" ? `[${idx + 1}/${chronTracks.length}] ` : ""}#{t.track_id}
                        </button>
                        <span>
                          {formatDuration(t.t0)}–{formatDuration(t.t1)}
                        </span>
                      </header>
                      <div className="merge-crop-strip">
                        {crops.length ? (
                          crops.map((c, i) => (
                            <button
                              key={i}
                              type="button"
                              className="merge-crop-thumb"
                              onClick={() => {
                                const fps = tracking?.fps && tracking.fps > 0 ? tracking.fps : 25;
                                pickTrack(t.track_id, c.t ?? (c.frame != null ? c.frame / fps : t.t0));
                              }}
                            >
                              <img src={c.url} alt="" loading="lazy" />
                            </button>
                          ))
                        ) : (
                          <span className="merge-inspect-empty">нет кропов</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </section>

        {/* Why column */}
        <aside className="merge-inspect-why">
          <h3>Почему</h3>
          {!selectedRow ? (
            <p className="merge-inspect-empty">Выберите группу или solo-трек</p>
          ) : (
            <>
              {selectedRow.kind === "group" && selectedRow.group.reason && (
                <blockquote className="merge-why-quote">
                  <button
                    type="button"
                    className="merge-copy"
                    title="Копировать reason"
                    onClick={() => copyText(selectedRow.group.reason!)}
                  >
                    ⎘
                  </button>
                  {selectedRow.group.reason}
                </blockquote>
              )}

              {selectedGroupPairs.length > 0 && (
                <>
                  <h4>Рёбра в склейке (Метрики и оценки)</h4>
                  <table className="merge-pairs-table">
                    <thead>
                      <tr>
                        <th>A↔B</th>
                        <th title="Комбинированный общий скор">Combo</th>
                        <th title="ReID сходство внешности">ReID</th>
                        <th title="Оценка непрерывности движения и расстояние">Motion / Дист.</th>
                        <th title="Похожесть размера bbox">Size</th>
                        <th title="Временной разрыв между фрагментами">Δt</th>
                        <th title="Тип склейки и обоснование">Обоснование</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedGroupPairs.map((p) => {
                        const ta = mergeTimeline.tracks.find((t) => t.track_id === p.a);
                        const tb = mergeTimeline.tracks.find((t) => t.track_id === p.b);
                        const live = (ta != null && fragmentLive(ta, currentSec)) || (tb != null && fragmentLive(tb, currentSec));
                        return (
                        <tr key={`${p.a}-${p.b}`} className={live ? "is-live" : undefined}>
                          <td>
                            <button type="button" onClick={() => pickTrack(p.a, trackT0(p.a))}>
                              #{p.a}
                            </button>
                            ↔
                            <button type="button" onClick={() => pickTrack(p.b, trackT0(p.b))}>
                              #{p.b}
                            </button>
                          </td>
                          <td><strong>{formatScore(p.score)}</strong></td>
                          <td>{formatScore(p.reid)}</td>
                          <td>{motionLabel(p)}</td>
                          <td>{formatScore(p.size)}</td>
                          <td>{gapLabel(p.gap)}</td>
                          <td className="merge-pair-why" title={p.reason ?? undefined}>
                            <PassBadge pass={pairPass(p)} />
                            {p.reason ? (
                              <span className="merge-pair-reason">{stripPassPrefix(p.reason)}</span>
                            ) : pairPass(p) == null ? (
                              <span>—</span>
                            ) : null}
                          </td>
                        </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </>
              )}

              {rejectedSimilar.length > 0 && (
                <>
                  <h4>Похожие, но не склеены</h4>
                  <table className="merge-pairs-table">
                    <thead>
                      <tr>
                        <th>A↔B</th>
                        <th title="Комбинированный общий скор">Combo</th>
                        <th title="ReID сходство внешности">ReID</th>
                        <th title="Оценка непрерывности движения и расстояние">Motion / Дист.</th>
                        <th title="Похожесть размера bbox">Size</th>
                        <th title="Временной разрыв между фрагментами">Δt</th>
                        <th title="Почему не взяли в склейку">Обоснование</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rejectedSimilar.slice(0, 20).map((h) => {
                        const live = inFrameIds.has(h.track_id) || inFrameIds.has(h.from_id);
                        return (
                          <tr key={`${h.from_id}-${h.track_id}`} className={live ? "is-live" : undefined}>
                            <td>
                              <button type="button" onClick={() => pickTrack(h.from_id, trackT0(h.from_id))}>
                                #{h.from_id}
                              </button>
                              ↔
                              <button type="button" onClick={() => pickTrack(h.track_id, h.t0 ?? 0)}>
                                #{h.track_id}
                              </button>
                            </td>
                            <td><strong>{formatScore(h.score)}</strong></td>
                            <td>{formatScore(h.reid)}</td>
                            <td>{motionLabel(h)}</td>
                            <td>{formatScore(h.size)}</td>
                            <td>{gapLabel(h.gap)}</td>
                            <td className="merge-pair-why" title={h.reason ?? undefined}>
                              {h.reason ? <span className="merge-pair-reason">{h.reason}</span> : <span>—</span>}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </>
              )}

              {cameraLinkPairs.length > 0 && (
                <>
                  <h4>Pass 10 · Camera Link (лица по моделям)</h4>
                  <table className="merge-pairs-table">
                    <thead>
                      <tr>
                        <th>Группа A↔B</th>
                        <th title="Комбинированный общий скор">Combo</th>
                        {compareFaceModels.map((model) => (
                          <th key={model} title={`Сходство лиц (${model})`}>
                            Face {model}
                          </th>
                        ))}
                        <th title="ReID тела">ReID</th>
                        <th title="Motion / дистанция ног">Motion</th>
                        <th title="Временной разрыв">Δt</th>
                        <th>Обоснование</th>
                      </tr>
                    </thead>
                    <tbody>
                      {cameraLinkPairs.map((p) => (
                        <tr key={`cl-${p.a}-${p.b}`}>
                          <td>
                            g{p.a} ↔ g{p.b}
                          </td>
                          <td><strong>{formatScore(p.score)}</strong></td>
                          {compareFaceModels.map((model) => (
                            <td key={`${p.a}-${p.b}-${model}`}>{faceScoreForModel(p, model)}</td>
                          ))}
                          <td>{formatScore(p.reid)}</td>
                          <td>{motionLabel(p)}</td>
                          <td>{gapLabel(p.gap)}</td>
                          <td className="merge-pair-why" title={p.reason ?? undefined}>
                            <PassBadge pass={pairPass(p)} />
                            {p.reason ? (
                              <span className="merge-pair-reason">{stripPassPrefix(p.reason)}</span>
                            ) : null}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
            </>
          )}
        </aside>
      </div>
    </div>
  );
}

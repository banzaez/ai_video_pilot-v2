import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { TrackingData } from "../types";
import {
  PlaybackToolbar,
  PlayheadTimeline,
  formatDurationClock,
  makeTimeBounds,
  usePlaybackClock,
  type PlaybackSink,
  type TimelineLane,
} from "../playback";
import {
  faceBucketKeys,
  formatEntityId,
  groupId,
  trackletId,
} from "../entityId";
import {
  buildTrackKeyframes,
  colorForTrackId,
  detectionsAtFrame,
  formatDuration,
  passBadgeColors,
  resolveDetectEveryN,
  resolveLinkPass,
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
  playbackSink: PlaybackSink;
  onFocusTracks: (ids: number[] | null) => void;
  jumpTo?: { trackId: number; sec: number } | null;
  onJumpConsumed?: () => void;
};

type MergeSeg = { trackId: number; t0: number };

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
  const used = intra.filter((p) => resolveLinkPass(p) != null);
  const filtered = used.length ? used : intra;
  return filtered.sort((p1, p2) => {
    const t1 = trackMap.get(p1.a)?.t0 ?? 0;
    const t2 = trackMap.get(p2.a)?.t0 ?? 0;
    return t1 - t2 || p1.a - p2.a;
  });
}

function pairPass(p: { pass?: number | null; pass2?: boolean | null }): number | null {
  return resolveLinkPass(p);
}

function PassBadge({ pass }: { pass: number | null }) {
  if (typeof pass !== "number" || pass < 0) return <span>—</span>;
  const { bg, fg, border } = passBadgeColors(pass);
  return (
    <span
      className="badge-pass"
      style={{ background: bg, color: fg, border: `1px solid ${border}` }}
    >
      Pass {pass}
    </span>
  );
}

function pairKey(a: number, b: number): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

function pairsForGroupId(pairs: MergeTimelinePair[], groupId: number): MergeTimelinePair[] {
  return pairs.filter((p) => p.a === groupId || p.b === groupId);
}

function pairPassFromReason(reason: string | null | undefined): number | null {
  if (!reason) return null;
  const m = reason.match(/^Pass\s+(\d+)/i);
  return m ? Number(m[1]) : null;
}

function entityRefLabel(kind: "track" | "group", id: number): string {
  return formatEntityId(kind === "group" ? groupId(id) : trackletId(id));
}

function trackIdsForEntity(
  kind: "track" | "group",
  id: number,
  mergeTimeline: MergeTimeline | null,
): number[] {
  if (kind === "track") return [id];
  return (mergeTimeline?.tracks ?? [])
    .filter((t) => t.group_id === id)
    .map((t) => t.track_id)
    .sort((a, b) => a - b);
}

function spanForEntity(
  kind: "track" | "group",
  id: number,
  mergeTimeline: MergeTimeline | null,
): string | null {
  if (kind === "track") {
    const tr = mergeTimeline?.tracks.find((t) => t.track_id === id);
    return tr ? `${formatDuration(tr.t0)}–${formatDuration(tr.t1)}` : null;
  }
  const tracks = mergeTimeline?.tracks.filter((t) => t.group_id === id) ?? [];
  if (!tracks.length) return null;
  const t0 = Math.min(...tracks.map((t) => t.t0));
  const t1 = Math.max(...tracks.map((t) => t.t1));
  return `${formatDuration(t0)}–${formatDuration(t1)} · ${tracks.length} фрагм.`;
}

function filterFaceShotsForTrack(shots: FaceShot[], trackId: number, tr?: MergeTimelineTrack | null): FaceShot[] {
  const want = formatEntityId(trackletId(trackId));
  const exact = shots.filter((s) => s.entity === want || s.track_id === trackId);
  if (exact.length) return exact;
  if (!tr) return [];
  const inSpan = shots.filter((s) => {
    if (s.t == null || !Number.isFinite(s.t)) return false;
    return s.t >= tr.t0 && s.t <= tr.t1;
  });
  return inSpan;
}

function shotsFromFaces(
  key: string,
  faceUrls?: Record<string, FaceShot[]>,
  faceUrlsByModel?: Record<string, Record<string, FaceShot[]>>,
  model?: string,
): FaceShot[] {
  if (model) {
    const byModel = faceUrlsByModel?.[model];
    if (byModel?.[key]?.length) return byModel[key];
  }
  return faceUrls?.[key] ?? [];
}

function faceShotsForEntity(
  kind: "track" | "group",
  id: number,
  mergeTimeline: MergeTimeline | null,
  faceUrls?: Record<string, FaceShot[]>,
  faceUrlsByModel?: Record<string, Record<string, FaceShot[]>>,
  model?: string,
): FaceShot[] {
  const fromGroup = (gid: number) => {
    for (const key of faceBucketKeys(groupId(gid))) {
      const shots = shotsFromFaces(key, faceUrls, faceUrlsByModel, model);
      if (shots.length) return shots;
    }
    return [] as FaceShot[];
  };

  if (kind === "group") return fromGroup(id);

  const tr = mergeTimeline?.tracks.find((t) => t.track_id === id) ?? null;
  const groupN = tr?.group_id ?? tr?.global_id ?? null;
  if (groupN == null) return [];
  if (tr?.group_id != null) return filterFaceShotsForTrack(fromGroup(tr.group_id), id, tr);
  return fromGroup(groupN);
}

function facesForEntity(
  kind: "track" | "group",
  id: number,
  mergeTimeline: MergeTimeline | null,
  faceUrls?: Record<string, FaceShot[]>,
  faceUrlsByModel?: Record<string, Record<string, FaceShot[]>>,
  faceModels?: string[],
): { model: string; shots: FaceShot[] }[] {
  const models = faceModels?.length ? faceModels : ["buffalo_l"];
  return models
    .map((model) => ({
      model,
      shots: faceShotsForEntity(kind, id, mergeTimeline, faceUrls, faceUrlsByModel, model).slice(0, 6),
    }))
    .filter((row) => row.shots.length > 0);
}

function cropsForEntity(
  kind: "track" | "group",
  id: number,
  mergeTimeline: MergeTimeline | null,
  cropUrls: Record<string, CropShot[]>,
): CropShot[] {
  const trackIds = trackIdsForEntity(kind, id, mergeTimeline);
  const out: CropShot[] = [];
  for (const tid of trackIds) {
    out.push(...cropPreview(cropUrls[String(tid)] ?? []));
  }
  return out.slice(0, 8);
}

function MergeEntityRef({
  kind,
  id,
  label,
  mergeTimeline,
  cropUrls,
  faceUrls,
  faceUrlsByModel,
  faceModels,
  onPickTrack,
  onPickGroup,
  variant = "default",
}: {
  kind: "track" | "group";
  id: number;
  label?: string;
  mergeTimeline: MergeTimeline | null;
  cropUrls: Record<string, CropShot[]>;
  faceUrls?: Record<string, FaceShot[]>;
  faceUrlsByModel?: Record<string, Record<string, FaceShot[]>>;
  faceModels?: string[];
  onPickTrack: (trackId: number, sec: number) => void;
  onPickGroup: (groupId: number) => void;
  variant?: "default" | "timeline";
}) {
  const btnRef = useRef<HTMLButtonElement>(null);
  const popRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [geom, setGeom] = useState({ left: 0, top: 0 });
  const [ready, setReady] = useState(0);

  const display = label ?? entityRefLabel(kind, id);
  const trackIds = trackIdsForEntity(kind, id, mergeTimeline);
  const colorId = kind === "track" ? id : (trackIds[0] ?? id);
  const color = colorForTrackId(colorId);
  const span = spanForEntity(kind, id, mergeTimeline);
  const crops = cropsForEntity(kind, id, mergeTimeline, cropUrls);
  const faceRows = facesForEntity(kind, id, mergeTimeline, faceUrls, faceUrlsByModel, faceModels);

  useLayoutEffect(() => {
    if (!open || !btnRef.current) return;
    const br = btnRef.current.getBoundingClientRect();
    const w = popRef.current?.offsetWidth ?? 280;
    const h = popRef.current?.offsetHeight ?? 420;
    let left = br.left;
    let top = br.bottom + 8;
    left = Math.min(Math.max(8, left), Math.max(8, window.innerWidth - w - 8));
    if (top + h > window.innerHeight - 8) {
      top = Math.max(8, br.top - h - 8);
    }
    setGeom({ left, top });
  }, [open, crops.length, faceRows.length, ready]);

  function handlePick() {
    if (kind === "group") {
      onPickGroup(id);
      return;
    }
    const tr = mergeTimeline ? resolveMergeTrack(mergeTimeline, id, null) : null;
    if (tr?.group_id != null) {
      onPickTrack(tr.track_id, tr.t0);
      return;
    }
    onPickTrack(id, tr?.t0 ?? 0);
  }

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        className={`merge-ref-chip${variant === "timeline" ? " merge-ref-chip-timeline" : ""}${open ? " on" : ""}`}
        style={
          variant === "timeline"
            ? { backgroundColor: color, borderColor: color }
            : { borderColor: color, color }
        }
        onMouseEnter={() => {
          const br = btnRef.current?.getBoundingClientRect();
          if (br) setGeom({ left: br.left, top: br.bottom + 8 });
          setOpen(true);
        }}
        onMouseLeave={() => setOpen(false)}
        onClick={(e) => {
          e.stopPropagation();
          handlePick();
        }}
      >
        {display}
      </button>
      {open &&
        createPortal(
          <div
            ref={popRef}
            className="similar-pop merge-entity-pop"
            style={{ left: geom.left, top: geom.top }}
          >
            <div className="similar-pop-head">
              <span className="track-id" style={{ backgroundColor: color }}>
                {display}
              </span>
              {span ? <span className="pop-span">{span}</span> : null}
            </div>
            {faceRows.length > 0 && (
              <div className="merge-entity-pop-faces">
                {faceRows.map(({ model, shots }) => (
                  <div key={model} className="merge-entity-pop-face-row">
                    <span className="merge-entity-pop-label">{model}</span>
                    <div className="pop-crops">
                      {shots.map((s, i) => (
                        <div key={`${model}-${i}`} className="pop-crop">
                          <img src={s.url} alt="" loading="lazy" onLoad={() => setReady((n) => n + 1)} />
                          {s.track_id != null && <span>t{s.track_id}</span>}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {crops.length > 0 ? (
              <div className="pop-crops">
                {crops.map((s, i) => (
                  <div key={`${s.rank}-${s.frame ?? i}`} className="pop-crop">
                    <img
                      src={s.url}
                      alt=""
                      loading="lazy"
                      onLoad={() => setReady((n) => n + 1)}
                    />
                    <span>k{s.rank}</span>
                  </div>
                ))}
              </div>
            ) : (
              !faceRows.length && <p className="similar-pop-empty">Нет кропов и лиц</p>
            )}
          </div>,
          document.body,
        )}
    </>
  );
}

function GroupFaceThumb({ faces }: { faces?: FaceShot[] }) {
  const [idx, setIdx] = useState(0);
  const modelFaces = faces ?? [];
  if (!modelFaces.length) return null;
  const safeIdx = Math.min(Math.max(0, idx), modelFaces.length - 1);
  const currentFace = modelFaces[safeIdx];

  return (
    <div className="merge-inspect-face-wrap" onClick={(e) => e.stopPropagation()}>
      <div
        className="merge-inspect-face-box"
        title={`лицо ${safeIdx + 1}/${modelFaces.length} (t${currentFace.track_id ?? "—"}, det ${currentFace.score?.toFixed(2) ?? "—"}, pose ${currentFace.pose_score?.toFixed(2) ?? "—"}, q ${currentFace.quality?.toFixed(2) ?? "—"}, f${currentFace.frame})`}
        onClick={(e) => {
          if (modelFaces.length > 1) {
            e.stopPropagation();
            setIdx((prev) => (prev + 1) % modelFaces.length);
          }
        }}
      >
        <img src={currentFace.url} alt="" loading="lazy" />
        {currentFace.track_id != null && (
          <span className="merge-inspect-face-tid">t{currentFace.track_id}</span>
        )}
        {modelFaces.length > 1 && (
          <div className="merge-inspect-face-nav">
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

/** Фрагмент по track_id / global_id; если задано время — живой или ближайший. */
function resolveMergeTrack(
  timeline: MergeTimeline,
  trackId: number,
  sec?: number | null,
): MergeTimelineTrack | null {
  const hits = timeline.tracks.filter((t) => t.track_id === trackId || t.global_id === trackId);
  if (!hits.length) return null;
  if (sec == null || !Number.isFinite(sec)) return hits[0] ?? null;
  const live = hits.find((t) => sec >= t.t0 - 0.05 && sec <= t.t1 + 0.05);
  if (live) return live;
  return [...hits].sort((a, b) => Math.abs(a.t0 - sec) - Math.abs(b.t0 - sec))[0] ?? null;
}

function rowKeyForTrack(tr: MergeTimelineTrack): string {
  return tr.group_id != null ? `g:${tr.group_id}` : `solo:${tr.track_id}`;
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
  playbackSink,
  onFocusTracks,
  jumpTo = null,
  onJumpConsumed,
}: Props) {
  const [query, setQuery] = useState("");
  const [listFilter, setListFilter] = useState<MergeListFilter>("all");
  const [sortBy, setSortBy] = useState<MergeListSort>("group_id");
  const [selectedRowKey, setSelectedRowKey] = useState<string | null>(null);
  const [selectedFragmentId, setSelectedFragmentId] = useState<number | null>(null);
  const [atCurrentFrame, setAtCurrentFrame] = useState(false);
  const selectedRowBtnRef = useRef<HTMLButtonElement>(null);

  const timeBounds = useMemo(
    () => makeTimeBounds(0, Math.max(mergeTimeline?.duration_sec ?? 1, 0.001)),
    [mergeTimeline?.duration_sec],
  );
  const clock = usePlaybackClock({
    bounds: timeBounds,
    sink: playbackSink,
    defaultRate: 2,
    hotkeys: { space: true, arrows: false },
    enabled: Boolean(activeVideo && mergeTimeline),
  });
  const playheadSec = clock.currentSec;

  useEffect(() => {
    clock.noteExternalTime(currentSec);
  }, [currentSec, clock.noteExternalTime]);

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
    const cleanQ = q.replace(/^#/, "").replace(/^[gt]/i, "");
    if (q) {
      list = list.filter((row) => {
        if (row.kind === "solo") {
          const tq = `t${row.track.track_id}`;
          return (
            tq.includes(q) ||
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
    if (jumpTo || selectedTrackId == null || !mergeTimeline) return;
    const tr = resolveMergeTrack(mergeTimeline, selectedTrackId, playheadSec);
    if (!tr) return;
    const key = rowKeyForTrack(tr);
    if (selectedRowKey === key && selectedFragmentId === tr.track_id) return;
    setSelectedRowKey(key);
    setSelectedFragmentId(tr.track_id);
  }, [jumpTo, selectedTrackId, mergeTimeline, playheadSec, selectedRowKey, selectedFragmentId]);

  useEffect(() => {
    selectedRowBtnRef.current?.scrollIntoView({ block: "nearest" });
  }, [selectedRowKey]);

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
    selectRow(row);
  }

  function selectRow(row: MergeListRow) {
    setSelectedRowKey(rowKey(row));
    if (row.kind === "solo") {
      setSelectedFragmentId(row.track.track_id);
      onSelectTrackId(videoTrackId(row.track));
      clock.seek(row.track.t0);
    } else if (row.tracks.length) {
      const sorted = [...row.tracks].sort((a, b) => a.t0 - b.t0);
      const first = sorted[0];
      setSelectedFragmentId(first.track_id);
      onSelectTrackId(videoTrackId(first));
      clock.seek(first.t0);
    }
  }

  function pickTrack(trackId: number, sec: number) {
    const tr = mergeTimeline ? resolveMergeTrack(mergeTimeline, trackId, sec) : null;
    if (tr) {
      setSelectedRowKey(rowKeyForTrack(tr));
      setSelectedFragmentId(tr.track_id);
      onSelectTrackId(videoTrackId(tr));
    } else {
      setSelectedFragmentId(trackId);
      onSelectTrackId(trackId);
    }
    clock.seek(sec);
  }

  useEffect(() => {
    if (!jumpTo) return;
    if (!mergeTimeline) {
      onJumpConsumed?.();
      return;
    }
    setListFilter("all");
    setQuery("");
    const tr = resolveMergeTrack(mergeTimeline, jumpTo.trackId, jumpTo.sec);
    if (tr) {
      setSelectedRowKey(rowKeyForTrack(tr));
      setSelectedFragmentId(tr.track_id);
      onSelectTrackId(videoTrackId(tr));
      clock.seek(jumpTo.sec);
    }
    onJumpConsumed?.();
  }, [jumpTo, mergeTimeline, clock.seek, onSelectTrackId, onJumpConsumed]);

  function pickGroup(groupId: number) {
    const row = rows.find((r) => r.kind === "group" && r.group.group_id === groupId);
    if (row) {
      selectRow(row);
      return;
    }
    const tr = mergeTimeline?.tracks.find((t) => t.group_id === groupId);
    if (tr) pickTrack(tr.track_id, tr.t0);
  }

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
    return allTracksSorted.filter((t) => fragmentLive(t, playheadSec) && activeAtFrame.has(videoTrackId(t)));
  }, [allTracksSorted, activeAtFrame, playheadSec]);

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
      label: key === "solo" ? "без склейки" : `g${key}`,
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
          label: `g${gid}`,
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

  const timelineLanes = useMemo((): TimelineLane<MergeSeg>[] => {
    return visibleSections.map((section) => ({
      id: String(section.key),
      label: section.label,
      highlight:
        section.groupId != null
          ? activeRowKey === `g:${section.groupId}`
          : Boolean(activeRowKey?.startsWith("solo:")),
      segments: section.tracks.map((t) => {
        const selected = selectedFragmentId === t.track_id;
        const score = section.score != null ? ` · ${formatScore(section.score)}` : "";
        return {
          id: String(t.track_id),
          t0: t.t0,
          t1: t.t1,
          color: colorForTrackId(t.track_id),
          label: selected ? `t${t.track_id}` : undefined,
          title: `${section.label} t${t.track_id}  ${t.t0.toFixed(1)}–${t.t1.toFixed(1)}s${score}`,
          selected,
          dimmed: selectedFragmentId != null && !selected && section.tracks.length > 1,
          data: { trackId: t.track_id, t0: t.t0 },
        };
      }),
    }));
  }, [visibleSections, activeRowKey, selectedFragmentId]);

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
  const cameraLinkMergedPairs = useMemo(() => {
    const rows = cameraLink?.edges ?? [];
    if (!selectedRow || selectedRow.kind !== "group") return rows;
    return pairsForGroupId(rows, selectedRow.group.group_id);
  }, [cameraLink, selectedRow]);

  const cameraLinkSimilarPairs = useMemo(() => {
    const mergedKeys = new Set((cameraLink?.edges ?? []).map((p) => pairKey(p.a, p.b)));
    const rows = (cameraLink?.candidate_edges ?? []).filter((p) => !mergedKeys.has(pairKey(p.a, p.b)));
    if (!selectedRow || selectedRow.kind !== "group") return rows;
    return pairsForGroupId(rows, selectedRow.group.group_id);
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
  const entityRefCommon = {
    mergeTimeline,
    cropUrls,
    faceUrls,
    faceUrlsByModel,
    faceModels,
    onPickTrack: pickTrack,
    onPickGroup: pickGroup,
  };

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
            placeholder="g3, t15…"
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
              const primaryFaceModel =
                compareFaceModels[0] ?? faceModels[0] ?? "buffalo_l";
              const listFaces = faceShotsForEntity(
                row.kind === "group" ? "group" : "track",
                row.kind === "group" ? row.group.group_id : row.track.track_id,
                mergeTimeline,
                faceUrls,
                faceUrlsByModel,
                primaryFaceModel,
              );
              return (
                <li key={key}>
                  <button
                    type="button"
                    className={`merge-inspect-row${isSelected ? " on" : ""}`}
                    ref={isSelected ? selectedRowBtnRef : undefined}
                    onClick={() => pickRow(row)}
                  >
                    <div className="merge-inspect-row-content">
                      {row.kind === "group" ? (
                        <>
                          <div className="merge-inspect-row-head">
                            <span className="merge-inspect-row-title">
                              <strong>{entityRefLabel("group", row.group.group_id)}</strong>
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
                          <div className="merge-inspect-ids">
                            {row.tracks.map((t) => (
                              <MergeEntityRef key={t.track_id} kind="track" id={t.track_id} {...entityRefCommon} />
                            ))}
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="merge-inspect-row-head">
                            <span className="merge-inspect-row-title">
                              <strong>{entityRefLabel("track", row.track.track_id)}</strong>
                            </span>
                          </div>
                          <div className="merge-inspect-row-meta">
                            <span>без склейки</span>
                            <span>·</span>
                            <span>{formatDuration(row.track.t0)}–{formatDuration(row.track.t1)}</span>
                            <span>·</span>
                            <span>{formatDuration(row.track.t1 - row.track.t0)}</span>
                          </div>
                        </>
                      )}
                    </div>
                    {listFaces?.length ? <GroupFaceThumb faces={listFaces} /> : null}
                  </button>
                </li>
              );
            })}
            {!filteredRows.length && <li className="merge-inspect-empty">Нет строк по фильтру</li>}
          </ul>
        </aside>

        {/* Center column */}
        <section className="merge-inspect-center">
          <PlaybackToolbar
            clock={clock}
            bounds={timeBounds}
            formatCurrent={formatDurationClock}
            extras={
              <>
                <label className="toggle sidebar-toggle">
                  <input type="checkbox" checked={atCurrentFrame} onChange={(e) => setAtCurrentFrame(e.target.checked)} />
                  только текущий кадр
                </label>
                {activeRowKey && (
                  <button type="button" className="playhead-btn" onClick={clearSelection}>
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
              </>
            }
          />

          {visibleSections.length === 0 ? (
            <p className="merge-inspect-empty">Нет треков по фильтру</p>
          ) : (
            <PlayheadTimeline
              lanes={timelineLanes}
              bounds={timeBounds}
              currentSec={playheadSec}
              zoom={clock.zoom}
              formatTick={formatDurationClock}
              onSeek={(sec) => clock.seek(sec)}
              onScrubbing={clock.setScrubbing}
              onSelect={(seg) => pickTrack(seg.data.trackId, seg.data.t0)}
              onLaneClick={(lane) => {
                if (/^\d+$/.test(lane.id)) {
                  pickGroup(Number(lane.id));
                  return;
                }
                const first = lane.segments[0];
                if (first) pickTrack(first.data.trackId, first.data.t0);
              }}
            />
          )}

          {selectedRow && (
            <div className="merge-inspect-dossier">
              <h3>
                {selectedRow.kind === "group"
                  ? `${entityRefLabel("group", selectedRow.group.group_id)} · ${selectedRow.tracks.length} фрагм. (${chronTracks.map((t) => entityRefLabel("track", t.track_id)).join(", ")})`
                  : `${entityRefLabel("track", selectedRow.track.track_id)} · без склейки`}
              </h3>
              <div className="merge-crop-compare">
                {(selectedRow.kind === "group" ? chronTracks : [selectedRow.track]).map((t, idx) => {
                  const crops = cropPreview(cropUrls[String(t.track_id)] ?? []);
                  const live = fragmentLive(t, playheadSec);
                  const isCurFrag = selectedFragmentId === t.track_id;
                  return (
                    <div
                      key={t.track_id}
                      className={`merge-crop-col${live ? " is-live" : ""}${isCurFrag ? " is-current" : ""}`}
                      style={isCurFrag || live ? { borderColor: colorForTrackId(t.track_id) } : undefined}
                    >
                      <header>
                        <button type="button" onClick={() => pickTrack(t.track_id, t.t0)}>
                          {selectedRow.kind === "group" ? `[${idx + 1}/${chronTracks.length}] ` : ""}{entityRefLabel("track", t.track_id)}
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
                        <th>Pass</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedGroupPairs.map((p) => {
                        const ta = mergeTimeline.tracks.find((t) => t.track_id === p.a);
                        const tb = mergeTimeline.tracks.find((t) => t.track_id === p.b);
                        const live = (ta != null && fragmentLive(ta, playheadSec)) || (tb != null && fragmentLive(tb, playheadSec));
                        return (
                        <tr key={`${p.a}-${p.b}`} className={live ? "is-live" : undefined}>
                          <td className="merge-pair-ids">
                            <MergeEntityRef kind="track" id={p.a} {...entityRefCommon} />
                            <span className="merge-pair-sep">↔</span>
                            <MergeEntityRef kind="track" id={p.b} {...entityRefCommon} />
                          </td>
                          <td><strong>{formatScore(p.score)}</strong></td>
                          <td>{formatScore(p.reid)}</td>
                          <td>{motionLabel(p)}</td>
                          <td>{formatScore(p.size)}</td>
                          <td>{gapLabel(p.gap)}</td>
                          <td><PassBadge pass={pairPass(p)} /></td>
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
                        <th>Pass</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rejectedSimilar.slice(0, 20).map((h) => {
                        const live = inFrameIds.has(h.track_id) || inFrameIds.has(h.from_id);
                        return (
                          <tr key={`${h.from_id}-${h.track_id}`} className={live ? "is-live" : undefined}>
                            <td className="merge-pair-ids">
                              <MergeEntityRef kind="track" id={h.from_id} {...entityRefCommon} />
                              <span className="merge-pair-sep">↔</span>
                              <MergeEntityRef kind="track" id={h.track_id} {...entityRefCommon} />
                            </td>
                            <td><strong>{formatScore(h.score)}</strong></td>
                            <td>{formatScore(h.reid)}</td>
                            <td>{motionLabel(h)}</td>
                            <td>{formatScore(h.size)}</td>
                            <td>{gapLabel(h.gap)}</td>
                            <td><PassBadge pass={h.pass ?? pairPassFromReason(h.reason)} /></td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </>
              )}

              {cameraLinkSimilarPairs.length > 0 && (
                <>
                  <h4>Pass 10 · похожие, не склеены</h4>
                  <table className="merge-pairs-table merge-pairs-table-face">
                    <thead>
                      <tr>
                        <th>A↔B</th>
                        <th title="Комбинированный общий скор">Combo</th>
                        {compareFaceModels.map((model) => (
                          <th key={`sim-${model}`} title={`Сходство лиц (${model})`}>
                            Face {model}
                          </th>
                        ))}
                        <th title="Уверенность лица из позы (одна на ребро)">Pose</th>
                        <th title="ReID тела">ReID</th>
                        <th title="Motion / дистанция ног">Motion</th>
                        <th title="Временной разрыв">Δt</th>
                        <th>Pass</th>
                      </tr>
                    </thead>
                    <tbody>
                      {cameraLinkSimilarPairs.map((p) => (
                        <tr key={`cls-${p.a}-${p.b}`}>
                          <td className="merge-pair-ids">
                            <MergeEntityRef kind="group" id={p.a} {...entityRefCommon} />
                            <span className="merge-pair-sep">↔</span>
                            <MergeEntityRef kind="group" id={p.b} {...entityRefCommon} />
                          </td>
                          <td><strong>{formatScore(p.score)}</strong></td>
                          {compareFaceModels.map((model) => (
                            <td key={`${p.a}-${p.b}-sim-${model}`}>{faceScoreForModel(p, model)}</td>
                          ))}
                          <td>{formatScore(p.pose_face)}</td>
                          <td>{formatScore(p.reid)}</td>
                          <td>{motionLabel(p)}</td>
                          <td>{gapLabel(p.gap)}</td>
                          <td><PassBadge pass={pairPass(p)} /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}

              {cameraLinkMergedPairs.length > 0 && (
                <>
                  <h4>Pass 10 · Camera Link (склеено)</h4>
                  <table className="merge-pairs-table merge-pairs-table-face">
                    <thead>
                      <tr>
                        <th>A↔B</th>
                        <th title="Комбинированный общий скор">Combo</th>
                        {compareFaceModels.map((model) => (
                          <th key={model} title={`Сходство лиц (${model})`}>
                            Face {model}
                          </th>
                        ))}
                        <th title="Уверенность лица из позы (одна на ребро)">Pose</th>
                        <th title="ReID тела">ReID</th>
                        <th title="Motion / дистанция ног">Motion</th>
                        <th title="Временной разрыв">Δt</th>
                        <th>Pass</th>
                      </tr>
                    </thead>
                    <tbody>
                      {cameraLinkMergedPairs.map((p) => (
                        <tr key={`cl-${p.a}-${p.b}`}>
                          <td className="merge-pair-ids">
                            <MergeEntityRef kind="group" id={p.a} {...entityRefCommon} />
                            <span className="merge-pair-sep">↔</span>
                            <MergeEntityRef kind="group" id={p.b} {...entityRefCommon} />
                          </td>
                          <td><strong>{formatScore(p.score)}</strong></td>
                          {compareFaceModels.map((model) => (
                            <td key={`${p.a}-${p.b}-${model}`}>{faceScoreForModel(p, model)}</td>
                          ))}
                          <td>{formatScore(p.pose_face)}</td>
                          <td>{formatScore(p.reid)}</td>
                          <td>{motionLabel(p)}</td>
                          <td>{gapLabel(p.gap)}</td>
                          <td><PassBadge pass={pairPass(p)} /></td>
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

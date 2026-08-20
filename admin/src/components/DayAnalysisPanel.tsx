import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  buildTrackKeyframes,
  cameraKeyFromVideo,
  colorForTrackId,
  detectionsAtFrame,
  fetchHomography,
  fetchTrackingJson,
  formatDuration,
  frameAtTime,
  resolveDetectEveryN,
} from "../utils";
import { TrackingPlayer, type TrackingPlayerHandle } from "./TrackingPlayer";
import { sessionDurationSec, type MediaSession } from "../session";
import type { TrackingData } from "../types";
import type { HomographyDoc, Mat3 } from "../homography";
import { resolveFeetOnMap } from "../feetIndex";
import type { MapLiveMarker } from "./MapFloorView";
import {
  PlaybackToolbar,
  PlayheadTimeline,
  formatHms as formatShortTime,
  formatTimeOfDay,
  makeTimeBounds,
  usePlaybackClock,
  type PlaybackSink,
  type TimeBounds,
  type TimelineLane,
} from "../playback";

export interface CropRef {
  session_key: string;
  file: string;
}

export interface DayTrack {
  uid: string;
  session_key: string;
  camera: string;
  camera_index: number;
  track_id: number;
  t0: number;
  t1: number;
  p0: [number, number] | null;
  p1: [number, number] | null;
  n_frames: number;
  has_reid: boolean;
  crops: string[];
}

export type OpenTrackInMerge = (sessionKey: string, trackId: number, t0: number) => void;

const DAY_CAM_COLS_KEY = "ai-video-pilot-day-cam-cols";
const DAY_CAM_COLS_DEFAULT = 3;
const DAY_CAM_COLS_MAX = 6;

function readDayCamCols(): number {
  try {
    const n = Number(localStorage.getItem(DAY_CAM_COLS_KEY));
    if (Number.isInteger(n) && n >= 1 && n <= DAY_CAM_COLS_MAX) return n;
  } catch {
    /* ignore */
  }
  return DAY_CAM_COLS_DEFAULT;
}

export interface DayTransitionEdge {
  from: string;
  to: string;
  from_session: string;
  from_camera: string;
  from_track: number;
  to_session: string;
  to_camera: string;
  to_track: number;
  is_same_camera: boolean;
  is_overlap: boolean;
  score: number;
  reid?: number | null;
  motion: number;
  size?: number | null;
  dist_m: number;
  gap_sec: number;
  speed_mps: number;
  reason: string;
  pass?: number;
}

export interface GlobalPerson {
  person_id: number;
  label: string;
  t0: number;
  t1: number;
  duration_sec: number;
  n_tracks: number;
  n_cameras: number;
  n_transitions: number;
  cameras: string[];
  best_crop: string | CropRef | null;
  crops: Array<string | CropRef>;
  tracks: DayTrack[];
}

export interface DayLinksData {
  has_links: boolean;
  day: string;
  day_clean: string;
  cameras: string[];
  sessions: string[];
  camera_sessions?: Array<MediaSession & { t0_abs?: number }>;
  track_to_person?: Record<string, Record<number, number>>;
  solver?: string;
  n_persons: number;
  persons: GlobalPerson[];
  edges: DayTransitionEdge[];
  candidate_edges: DayTransitionEdge[];
  stats: {
    n_tracks_total?: number;
    n_persons?: number;
    n_multi_cam_persons?: number;
    n_solo_persons?: number;
    n_merges_total?: number;
    pass0_merges?: number;
    pass1_merges?: number;
    pass2_merges?: number;
    pass4_merges?: number;
  };
}

export interface DaySummaryItem {
  day: string;
  day_clean: string;
  has_links: boolean;
  sessions: string[];
  cameras: string[];
  stats?: DayLinksData["stats"];
}

type FilterMode = "all" | "multi_cam" | "pass0" | "pass1" | "solo";
type SortMode = "person_id" | "tracks" | "span" | "time";
type RightTab = "edges" | "tracks" | "inspector";
type DayCameraSession = MediaSession & { t0_abs?: number };

function sessionFileDuration(sess: DayCameraSession): number {
  if (typeof sess.duration_sec === "number" && sess.duration_sec > 0) return sess.duration_sec;
  if (sess.parts?.length) {
    const d = sessionDurationSec(sess.parts, sess.fps ?? 25);
    if (d > 0) return d;
  }
  return 0;
}

function sessionCoverage(sess: DayCameraSession): { t0: number; t1: number } | undefined {
  const t0 = sess.t0_abs;
  if (t0 == null || !Number.isFinite(t0)) return undefined;
  const dur = sessionFileDuration(sess);
  if (dur > 0) return { t0, t1: t0 + dur };
  return undefined;
}
type DaySeg = { personId: number; sessionKey: string; trackId: number; t0: number };

const trackingCache = new Map<string, Record<string, TrackingData>>();
const homographyCache = new Map<string, Record<string, HomographyDoc>>();
const EMPTY_FOCUS: number[] = [];
const EMPTY_GROUP: Record<number, number> = {};

function groupCropUrl(sessionKey: string, file: string): string {
  return `/api/group_crop/${encodeURIComponent(sessionKey)}/${encodeURIComponent(file)}`;
}

function cropSrc(crop: string | CropRef | null | undefined, sessionKey?: string): string | null {
  if (!crop) return null;
  if (typeof crop === "string") {
    if (crop.startsWith("/")) return crop;
    if (crop.includes("/") && !sessionKey) {
      const slash = crop.indexOf("/");
      return groupCropUrl(crop.slice(0, slash), crop.slice(slash + 1));
    }
    return sessionKey ? groupCropUrl(sessionKey, crop) : crop;
  }
  return groupCropUrl(crop.session_key, crop.file);
}

function formatScore(v: number | null | undefined, digits: number = 4): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

function scoreClass(score: number): string {
  if (score >= 0.85) return "day-score high";
  if (score >= 0.7) return "day-score mid";
  return "day-score low";
}

function passBadge(pass?: number) {
  switch (pass) {
    case 0:
      return <span className="pass-pill pass0">Pass 0</span>;
    case 1:
      return <span className="pass-pill pass1">Pass 1</span>;
    case 2:
      return <span className="pass-pill pass2">Pass 2</span>;
    case 4:
      return <span className="pass-pill pass4">Pass 4</span>;
    default:
      return <span className="pass-pill">Link</span>;
  }
}

function cameraShort(name: string): string {
  return name.replace(/^Camera_0?/, "C");
}

function CropThumb({
  crop,
  sessionKey,
  color,
  label,
  className,
}: {
  crop: string | CropRef | null | undefined;
  sessionKey?: string;
  color: string;
  label: string;
  className?: string;
}) {
  const src = cropSrc(crop, sessionKey);
  return (
    <div className={`day-crop ${className ?? ""}`} style={{ borderColor: color }}>
      {src ? (
        <img src={src} alt={label} loading="lazy" />
      ) : (
        <div className="day-crop-fallback" style={{ background: color }}>
          {label}
        </div>
      )}
      <span className="day-crop-id">{label}</span>
    </div>
  );
}

const DayPersonList = memo(function DayPersonList({
  persons,
  counts,
  filterMode,
  sortMode,
  searchQuery,
  selectedPersonId,
  onFilter,
  onSort,
  onSearch,
  onSelect,
}: {
  persons: GlobalPerson[];
  counts: { all: number; multi: number; solo: number };
  filterMode: FilterMode;
  sortMode: SortMode;
  searchQuery: string;
  selectedPersonId: number | null;
  onFilter: (mode: FilterMode) => void;
  onSort: (mode: SortMode) => void;
  onSearch: (q: string) => void;
  onSelect: (person: GlobalPerson) => void;
}) {
  return (
    <section className="day-list" aria-label="Список персон">
      <input
        type="search"
        className="day-search"
        placeholder="Поиск по ID, треку, камере…"
        value={searchQuery}
        onChange={(e) => onSearch(e.target.value)}
      />
      <div className="day-filters">
        <button type="button" className={filterMode === "all" ? "on" : ""} onClick={() => onFilter("all")}>
          Все ({counts.all})
        </button>
        <button type="button" className={filterMode === "multi_cam" ? "on" : ""} onClick={() => onFilter("multi_cam")}>
          Мультикам ({counts.multi})
        </button>
        <button type="button" className={filterMode === "pass0" ? "on" : ""} onClick={() => onFilter("pass0")}>
          Pass 0
        </button>
        <button type="button" className={filterMode === "pass1" ? "on" : ""} onClick={() => onFilter("pass1")}>
          Pass 1
        </button>
        <button type="button" className={filterMode === "solo" ? "on" : ""} onClick={() => onFilter("solo")}>
          Соло ({counts.solo})
        </button>
      </div>
      <div className="day-sort">
        <span>Сортировка</span>
        <select value={sortMode} onChange={(e) => onSort(e.target.value as SortMode)}>
          <option value="person_id">По ID</option>
          <option value="tracks">По трекам</option>
          <option value="span">По длительности</option>
          <option value="time">По времени</option>
        </select>
      </div>
      {persons.length === 0 ? (
        <p className="day-empty">Нет персон по текущему фильтру.</p>
      ) : (
        <ul className="day-rows">
          {persons.map((p) => {
            const isSelected = p.person_id === selectedPersonId;
            const pColor = colorForTrackId(p.person_id);
            return (
              <li key={p.person_id}>
                <button
                  type="button"
                  className={`day-row ${isSelected ? "on" : ""}`}
                  onClick={() => onSelect(p)}
                >
                  <CropThumb crop={p.best_crop} color={pColor} label={`#${p.person_id}`} />
                  <div className="day-row-body">
                    <div className="day-row-head">
                      <div className="day-row-title">
                        <strong>{p.label}</strong>
                        {p.n_cameras > 1 && <span className="day-tag">{p.n_cameras} кам.</span>}
                      </div>
                      <span className={scoreClass(p.best_crop ? 0.9 : 0.7)}>{p.tracks.length} тр.</span>
                    </div>
                    <div className="day-row-meta">
                      <span>
                        {formatShortTime(p.t0)}–{formatShortTime(p.t1)}
                      </span>
                      <span>{formatDuration(p.duration_sec)}</span>
                    </div>
                    <div className="day-row-route">{p.cameras.join(" → ")}</div>
                    <div className="day-row-ids">
                      {p.tracks.map((t) => (
                        <span key={t.uid} style={{ borderLeft: `2px solid ${colorForTrackId(t.track_id)}`, paddingLeft: 4 }}>
                          {cameraShort(t.camera)} #{t.track_id}
                        </span>
                      ))}
                    </div>
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
});

const DayInspector = memo(function DayInspector({
  person,
  edges,
  selectedEdge,
  rightTab,
  onTab,
  onSelectEdge,
  onSeek,
  onOpenTrackInMerge,
}: {
  person: GlobalPerson | null;
  edges: DayTransitionEdge[];
  selectedEdge: DayTransitionEdge | null;
  rightTab: RightTab;
  onTab: (tab: RightTab) => void;
  onSelectEdge: (edge: DayTransitionEdge) => void;
  onSeek: (sec: number) => void;
  onOpenTrackInMerge?: OpenTrackInMerge;
}) {
  if (!person) {
    return (
      <section className="day-inspect" aria-label="Инспектор персоны">
        <p className="day-empty">Выберите персону в списке или на таймлайне.</p>
      </section>
    );
  }

  const pColor = colorForTrackId(person.person_id);
  const crops = person.crops ?? [];

  return (
    <section className="day-inspect" aria-label="Инспектор персоны">
      <div className="day-inspect-stack">
        <div className="day-person-card">
          <CropThumb crop={person.best_crop} color={pColor} label={`#${person.person_id}`} />
          <div className="day-person-meta">
            <div className="day-person-meta-top">
              <strong>{person.label}</strong>
              <span className="day-score high">{person.n_tracks} треков</span>
            </div>
            <div className="day-row-meta">
              {formatShortTime(person.t0)}–{formatShortTime(person.t1)} ({formatDuration(person.duration_sec)})
            </div>
            <div className="day-row-route">{person.cameras.join(" → ")}</div>
          </div>
        </div>

        {crops.length > 0 && (
          <div>
            <div className="day-gallery-label">Кропы тела</div>
            <div className="day-gallery">
              {crops.map((crop, i) => {
                const src = cropSrc(crop);
                if (!src) return null;
                return (
                  <a key={i} href={src} target="_blank" rel="noreferrer">
                    <img src={src} alt="" />
                  </a>
                );
              })}
            </div>
          </div>
        )}

        <div className="day-tabs">
          <button type="button" className={rightTab === "edges" ? "on" : ""} onClick={() => onTab("edges")}>
            Склейки ({edges.length})
          </button>
          <button type="button" className={rightTab === "tracks" ? "on" : ""} onClick={() => onTab("tracks")}>
            Треки ({person.tracks.length})
          </button>
          {selectedEdge && (
            <button type="button" className={rightTab === "inspector" ? "on" : ""} onClick={() => onTab("inspector")}>
              Пара
            </button>
          )}
        </div>

        {rightTab === "edges" && (
          <div className="day-inspect-stack">
            {edges.length === 0 ? (
              <p className="day-empty">Один трек — межкамерных склеек нет.</p>
            ) : (
              edges.map((edge) => {
                const on = selectedEdge?.from === edge.from && selectedEdge?.to === edge.to;
                return (
                  <div
                    key={`${edge.from}->${edge.to}`}
                    className={`day-edge ${on ? "on" : ""}`}
                    title="Клик — к склейке на таймлайне · Shift+клик — вкладка Склейки"
                    onClick={(e) => {
                      if (e.shiftKey) {
                        const tr = person.tracks.find((t) => t.uid === edge.from);
                        if (tr) onOpenTrackInMerge?.(tr.session_key, tr.track_id, tr.t0);
                        return;
                      }
                      onSelectEdge(edge);
                    }}
                  >
                    <div className="day-edge-head">
                      <div className="day-edge-title">
                        {cameraShort(edge.from_camera)} #{edge.from_track} → {cameraShort(edge.to_camera)} #{edge.to_track}
                      </div>
                      {passBadge(edge.pass)}
                    </div>
                    <div className="day-edge-metrics">
                      <span className={scoreClass(edge.score)}>Скор {formatScore(edge.score)}</span>
                      {edge.reid != null && <span className="pass-pill">ReID {formatScore(edge.reid)}</span>}
                      <span className="pass-pill">Motion {formatScore(edge.motion)}</span>
                      <span className="pass-pill">Δt {edge.gap_sec.toFixed(1)}с</span>
                    </div>
                    <div className="day-edge-reason">{edge.reason}</div>
                  </div>
                );
              })
            )}
          </div>
        )}

        {rightTab === "inspector" && selectedEdge && (
          <div className="day-inspect-stack">
            <div className="day-edge-head">
              <strong>
                {cameraShort(selectedEdge.from_camera)} #{selectedEdge.from_track} → {cameraShort(selectedEdge.to_camera)} #
                {selectedEdge.to_track}
              </strong>
              {passBadge(selectedEdge.pass)}
            </div>
            <div className="day-pair">
              {(["from", "to"] as const).map((side) => {
                const tr = person.tracks.find((t) => t.uid === (side === "from" ? selectedEdge.from : selectedEdge.to));
                const isFrom = side === "from";
                return (
                  <div
                    key={side}
                    className="day-pair-col"
                    onClick={(e) => {
                      if (!tr) return;
                      if (e.shiftKey) {
                        onOpenTrackInMerge?.(tr.session_key, tr.track_id, isFrom ? tr.t1 : tr.t0);
                        return;
                      }
                      onSeek(isFrom ? tr.t1 : tr.t0);
                    }}
                    title={
                      isFrom
                        ? "Клик — к точке выхода · Shift+клик — Склейки"
                        : "Клик — к точке входа · Shift+клик — Склейки"
                    }
                  >
                    <strong>
                      {isFrom ? "До" : "После"} {cameraShort(isFrom ? selectedEdge.from_camera : selectedEdge.to_camera)} #
                      {isFrom ? selectedEdge.from_track : selectedEdge.to_track}
                    </strong>
                    {tr?.crops[0] && cropSrc(tr.crops[0], tr.session_key) && (
                      <img src={cropSrc(tr.crops[0], tr.session_key)!} alt="" />
                    )}
                    <p>{isFrom ? "Конец" : "Старт"}: {formatShortTime(isFrom ? tr?.t1 ?? 0 : tr?.t0 ?? 0)}</p>
                    <p>
                      Точка:{" "}
                      {isFrom
                        ? tr?.p1
                          ? `(${tr.p1[0].toFixed(0)}, ${tr.p1[1].toFixed(0)})`
                          : "—"
                        : tr?.p0
                          ? `(${tr.p0[0].toFixed(0)}, ${tr.p0[1].toFixed(0)})`
                          : "—"}
                    </p>
                  </div>
                );
              })}
            </div>
            <table className="day-metrics">
              <tbody>
                <tr>
                  <td>Скор</td>
                  <td>
                    <span className={scoreClass(selectedEdge.score)}>{formatScore(selectedEdge.score)}</span>
                  </td>
                </tr>
                <tr>
                  <td>ReID</td>
                  <td>
                    <strong>{formatScore(selectedEdge.reid)}</strong>
                  </td>
                </tr>
                <tr>
                  <td>Motion</td>
                  <td>
                    <strong>{formatScore(selectedEdge.motion)}</strong>
                  </td>
                </tr>
                <tr>
                  <td>Δd</td>
                  <td>
                    <strong>{selectedEdge.dist_m.toFixed(2)} м</strong>
                  </td>
                </tr>
                <tr>
                  <td>Δt</td>
                  <td>
                    <strong>{selectedEdge.gap_sec.toFixed(2)} с</strong>
                  </td>
                </tr>
                <tr>
                  <td>Солвер</td>
                  <td className="day-muted">{selectedEdge.reason}</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}

        {rightTab === "tracks" && (
          <table className="day-metrics">
            <thead>
              <tr>
                <th>Камера</th>
                <th>ID</th>
                <th>Интервал</th>
                <th>Кадров</th>
                <th>ReID</th>
              </tr>
            </thead>
            <tbody>
              {person.tracks.map((t) => (
                <tr
                  key={t.uid}
                  onClick={(e) => {
                    if (e.shiftKey) {
                      onOpenTrackInMerge?.(t.session_key, t.track_id, t.t0);
                      return;
                    }
                    onSeek(t.t0);
                  }}
                  title="Клик — к началу трека · Shift+клик — вкладка Склейки"
                >
                  <td>
                    <strong>{t.camera}</strong>
                  </td>
                  <td>#{t.track_id}</td>
                  <td>
                    {formatShortTime(t.t0)}–{formatShortTime(t.t1)}
                  </td>
                  <td>{t.n_frames}</td>
                  <td>{t.has_reid ? "да" : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
});

export function DayAnalysisPanel({
  selectedDay,
  onOpenTrackInMerge,
  onLiveMarkersChange,
}: {
  selectedDay: string;
  onOpenTrackInMerge?: OpenTrackInMerge;
  onLiveMarkersChange?: (markers: MapLiveMarker[]) => void;
}) {
  const [dayData, setDayData] = useState<DayLinksData | null>(null);
  const [loading, setLoading] = useState(false);
  const [runningSolver, setRunningSolver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cameraTracking, setCameraTracking] = useState<Record<string, TrackingData>>({});
  const [cameraHomographies, setCameraHomographies] = useState<Record<string, HomographyDoc>>({});

  const [filterMode, setFilterMode] = useState<FilterMode>("multi_cam");
  const [sortMode, setSortMode] = useState<SortMode>("person_id");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedPersonId, setSelectedPersonId] = useState<number | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<DayTransitionEdge | null>(null);
  const [rightTab, setRightTab] = useState<RightTab>("edges");
  const [cameraCols, setCameraCols] = useState(readDayCamCols);

  useEffect(() => {
    try {
      localStorage.setItem(DAY_CAM_COLS_KEY, String(cameraCols));
    } catch {
      /* ignore */
    }
  }, [cameraCols]);

  const playerRefs = useRef<Record<string, TrackingPlayerHandle | null>>({});
  const playerRefCbs = useRef<Record<string, (r: TrackingPlayerHandle | null) => void>>({});
  const playbackSpeedRef = useRef(2);
  const timeBoundsRef = useRef<TimeBounds>(makeTimeBounds(0, 300));
  const cameraSessionsRef = useRef<DayCameraSession[]>([]);
  const startedKeysRef = useRef<Set<string>>(new Set());

  const timeBounds = useMemo((): TimeBounds => {
    let minT = Infinity;
    let maxT = -Infinity;
    const bump = (a: number, b: number) => {
      if (a < minT) minT = a;
      if (b > maxT) maxT = b;
    };
    for (const p of dayData?.persons ?? []) {
      if (p.tracks?.length) {
        for (const t of p.tracks) bump(t.t0, t.t1);
      } else {
        bump(p.t0, p.t1);
      }
    }
    for (const sess of dayData?.camera_sessions ?? []) {
      const cov = sessionCoverage(sess);
      if (cov) bump(cov.t0, cov.t1);
    }
    if (!Number.isFinite(minT) || !Number.isFinite(maxT)) return makeTimeBounds(0, 300);
    return makeTimeBounds(minT, maxT);
  }, [dayData?.persons, dayData?.camera_sessions]);

  timeBoundsRef.current = timeBounds;
  cameraSessionsRef.current = dayData?.camera_sessions ?? [];

  const playerRefFor = (key: string) =>
    (playerRefCbs.current[key] ??= (r) => {
      playerRefs.current[key] = r;
    });

  const activeDaySecRef = useRef<number>(timeBounds.minT);
  const lastSampleRef = useRef<number | null>(null);

  const sampleDaySecFromPlayers = () => {
    const sessions = cameraSessionsRef.current;
    const minT = timeBoundsRef.current.minT;
    const currentT = activeDaySecRef.current;

    for (const sess of sessions) {
      const handle = playerRefs.current[sess.key];
      if (!handle) continue;
      const t0Abs = sess.t0_abs ?? minT;
      const duration = sessionFileDuration(sess);

      // Проверяем, активна ли эта камера в текущее время дня
      const isActiveNow =
        currentT >= t0Abs - 0.5 && (duration <= 0 || currentT <= t0Abs + duration + 0.5);
      if (!isActiveNow) continue;

      if (handle.seeking() || handle.paused()) continue;

      const g = handle.getGlobalSec();
      if (g == null) continue;
      const daySec = t0Abs + g;
      if (daySec < t0Abs - 0.05) continue;
      if (duration > 0 && daySec > t0Abs + duration + 0.05) continue;

      // Убеждаемся, что время камеры не из старого буфера
      if (Math.abs(daySec - currentT) <= 2.5) {
        lastSampleRef.current = daySec;
        return daySec;
      }
    }
    return null;
  };
  const sampleDaySecRef = useRef(sampleDaySecFromPlayers);
  sampleDaySecRef.current = sampleDaySecFromPlayers;

  const syncPlayersToTime = useCallback((tDay: number, playAfter = false, hard = true) => {
    const sessions = cameraSessionsRef.current;
    if (!sessions.length) return;
    if (hard) lastSampleRef.current = tDay;
    const minT = timeBoundsRef.current.minT;
    const baseRate = playbackSpeedRef.current;
    const nextStarted = new Set<string>();

    for (const sess of sessions) {
      const handle = playerRefs.current[sess.key];
      if (!handle) continue;
      const t0Abs = sess.t0_abs ?? minT;
      const duration = sessionFileDuration(sess);
      const local = tDay - t0Abs;

      // Если текущее время дня вне интервала записи камеры
      if (local < -0.05 || (duration > 0 && local > duration + 0.05)) {
        if (!handle.paused()) handle.pause();
        continue;
      }

      nextStarted.add(sess.key);
      const seekLocal = Math.max(0, local);
      const already = startedKeysRef.current.has(sess.key);

      if (hard || !already) {
        handle.setPlaybackRate(baseRate);
        if (!playAfter) handle.pause();
        handle.seekToGlobal(seekLocal, playAfter);
        continue;
      }

      if (!playAfter) {
        if (!handle.paused()) handle.pause();
        continue;
      }

      if (handle.seeking()) continue;
      const g = handle.getGlobalSec();
      if (g == null) continue;

      const drift = g - seekLocal;
      // Если расхождение критическое (> 2.5с) — делаем сик
      if (Math.abs(drift) > 2.5) {
        handle.setPlaybackRate(baseRate);
        handle.seekToGlobal(seekLocal, true);
      } else if (Math.abs(drift) > 0.35) {
        // Мягкая подстройка скорости без прерывания декодера
        const adjRate = drift < 0 ? baseRate * 1.12 : baseRate * 0.88;
        handle.setPlaybackRate(adjRate);
        if (handle.paused()) handle.play();
      } else {
        // В синхроне — возвращаем стандартную скорость
        handle.setPlaybackRate(baseRate);
        if (handle.paused()) handle.play();
      }
    }
    startedKeysRef.current = nextStarted;
  }, []);

  const playbackSink = useMemo<PlaybackSink>(
    () => ({
      sampleTime: () => sampleDaySecRef.current(),
      apply: (t, play, mode) => syncPlayersToTime(t, play, mode === "hard"),
      setRate: (next) => {
        playbackSpeedRef.current = next;
        for (const handle of Object.values(playerRefs.current)) {
          handle?.setPlaybackRate(next);
        }
      },
    }),
    [syncPlayersToTime],
  );

  const clock = usePlaybackClock({
    bounds: timeBounds,
    sink: playbackSink,
    defaultRate: 2,
  });
  const seekToDaySec = clock.seek;
  const currentDaySec = clock.currentSec;
  const currentDaySecRef = clock.currentSecRef;
  activeDaySecRef.current = currentDaySec;

  useEffect(() => {
    if (!selectedDay) return;
    let cancelled = false;

    const load = async () => {
      try {
        setLoading(true);
        setError(null);
        const res = await fetch(`/api/day/meta?day=${encodeURIComponent(selectedDay)}`);
        const json = (await res.json()) as DayLinksData;
        if (cancelled) return;
        setDayData(json);
        if (json.persons?.length) {
          const multi = json.persons.find((p) => p.n_cameras > 1);
          setFilterMode(multi ? "multi_cam" : "all");
          const first = multi ?? json.persons[0]!;
          setSelectedPersonId(first.person_id);
        } else {
          setSelectedPersonId(null);
        }
        setSelectedEdge(null);
        startedKeysRef.current = new Set();

        const cached = trackingCache.get(selectedDay);
        if (cached) {
          setCameraTracking(cached);
        } else if (json.camera_sessions?.length) {
          const loaded: Record<string, TrackingData> = {};
          await Promise.all(
            json.camera_sessions.map(async (sess) => {
              if (!sess.jsonUrl) return;
              try {
                loaded[sess.key] = await fetchTrackingJson(sess.jsonUrl);
              } catch {
                /* нет tracking.json */
              }
            }),
          );
          if (!cancelled) {
            trackingCache.set(selectedDay, loaded);
            setCameraTracking(loaded);
          }
        }

        const homoCached = homographyCache.get(selectedDay);
        if (homoCached) {
          setCameraHomographies(homoCached);
        } else if (json.camera_sessions?.length) {
          const loadedHomos: Record<string, HomographyDoc> = {};
          await Promise.all(
            json.camera_sessions.map(async (sess) => {
              const videoName = sess.parts?.[0]?.name || sess.camera || sess.key;
              const camKey = cameraKeyFromVideo(videoName, sess.camera_index);
              try {
                const doc = await fetchHomography(camKey);
                if (doc) {
                  loadedHomos[camKey] = doc;
                  loadedHomos[sess.camera] = doc;
                  loadedHomos[sess.key] = doc;
                }
              } catch {
                /* нет homography */
              }
            }),
          );
          if (!cancelled) {
            homographyCache.set(selectedDay, loadedHomos);
            setCameraHomographies(loadedHomos);
          }
        }
      } catch (e) {
        if (!cancelled) setError(`Ошибка загрузки дня ${selectedDay}: ${e}`);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, [selectedDay]);

  const initDayRef = useRef("");
  useEffect(() => {
    if (loading || !dayData || initDayRef.current === selectedDay) return;
    const dayKey = dayData.day || dayData.day_clean;
    if (dayKey && dayKey !== selectedDay && dayData.day_clean !== selectedDay) return;
    initDayRef.current = selectedDay;
    clock.pause();
    startedKeysRef.current = new Set();
    if (dayData.persons?.length) {
      const first = dayData.persons.find((p) => p.n_cameras > 1) ?? dayData.persons[0]!;
      clock.seek(first.t0, false);
    } else {
      clock.seek(timeBounds.minT, false);
    }
  }, [loading, dayData, selectedDay, timeBounds.minT, clock.pause, clock.seek]);

  const handleRunDayLink = async () => {
    if (!selectedDay || runningSolver) return;
    try {
      setRunningSolver(true);
      setError(null);
      const res = await fetch("/api/day/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ day: selectedDay }),
      });
      const json = (await res.json()) as { success: boolean; output: string; error?: string };
      if (!json.success) {
        setError(`Склейка дня завершилась с ошибкой: ${json.output || json.error}`);
      } else {
        trackingCache.delete(selectedDay);
        const metaRes = await fetch(`/api/day/meta?day=${encodeURIComponent(selectedDay)}`);
        const metaJson = (await metaRes.json()) as DayLinksData;
        setDayData(metaJson);
      }
    } catch (e) {
      setError(`Ошибка запуска склейки дня: ${e}`);
    } finally {
      setRunningSolver(false);
    }
  };

  const selectedPerson = useMemo(() => {
    if (!dayData?.persons || selectedPersonId == null) return null;
    return dayData.persons.find((p) => p.person_id === selectedPersonId) || null;
  }, [dayData?.persons, selectedPersonId]);

  const personEdges = useMemo(() => {
    if (!selectedPerson || !dayData?.edges) return [];
    const uids = new Set(selectedPerson.tracks.map((t) => t.uid));
    return dayData.edges.filter((e) => uids.has(e.from) && uids.has(e.to));
  }, [selectedPerson, dayData?.edges]);

  const focusTrackIdsByCamera = useMemo(() => {
    const map: Record<string, number[]> = {};
    if (!selectedPerson) return map;
    for (const tr of selectedPerson.tracks) {
      (map[tr.camera] ??= []).push(tr.track_id);
    }
    return map;
  }, [selectedPerson]);

  const displayPersons = useMemo(() => {
    if (!dayData?.persons) return [];
    let list = dayData.persons.filter((p) => {
      if (filterMode === "multi_cam" && p.n_cameras < 2) return false;
      if (filterMode === "solo" && (p.n_cameras > 1 || p.n_tracks > 1)) return false;
      if (filterMode === "pass0" || filterMode === "pass1") {
        const passN = Number(filterMode.slice(4));
        const uids = new Set(p.tracks.map((t) => t.uid));
        if (!dayData.edges.some((e) => e.pass === passN && uids.has(e.from) && uids.has(e.to))) return false;
      }
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        const hit =
          String(p.person_id).includes(q) ||
          p.label.toLowerCase().includes(q) ||
          p.cameras.some((c) => c.toLowerCase().includes(q)) ||
          p.tracks.some((t) => String(t.track_id).includes(q) || t.uid.toLowerCase().includes(q));
        if (!hit) return false;
      }
      return true;
    });
    list = [...list].sort((a, b) => {
      if (sortMode === "person_id") return a.person_id - b.person_id;
      if (sortMode === "tracks") return b.n_tracks - a.n_tracks;
      if (sortMode === "span") return b.duration_sec - a.duration_sec;
      return a.t0 - b.t0;
    });
    return list;
  }, [dayData?.persons, dayData?.edges, filterMode, sortMode, searchQuery]);

  const cameraSessionsList = dayData?.camera_sessions ?? [];
  const timelinePersons = selectedPerson ? dayData?.persons ?? [] : displayPersons;

  const timelineLanes = useMemo((): TimelineLane<DaySeg>[] => {
    return cameraSessionsList.map((sess) => {
      const segments = timelinePersons.flatMap((p) =>
        p.tracks
          .filter((t) => t.session_key === sess.key || t.camera === sess.camera)
          .map((t) => {
            const selected = selectedPersonId === p.person_id;
            return {
              id: t.uid,
              t0: t.t0,
              t1: t.t1,
              color: colorForTrackId(t.track_id),
              label: selected ? `P${p.person_id} T${t.track_id}` : undefined,
              title: `${p.label} · ${sess.camera} T${t.track_id}  ${formatShortTime(t.t0)}–${formatShortTime(t.t1)} · Shift+клик — Склейки`,
              selected,
              dimmed: selectedPersonId != null && !selected,
              data: { personId: p.person_id, sessionKey: t.session_key, trackId: t.track_id, t0: t.t0 },
            };
          }),
      );
      const highlight = selectedPerson?.tracks.some((t) => t.session_key === sess.key || t.camera === sess.camera);
      return { id: sess.key, label: sess.camera, highlight, coverage: sessionCoverage(sess), segments };
    });
  }, [cameraSessionsList, timelinePersons, selectedPersonId, selectedPerson]);

  const liveMarkers = useMemo((): MapLiveMarker[] => {
    if (!dayData?.camera_sessions?.length) return [];
    const minT = timeBounds.minT;
    const markers: MapLiveMarker[] = [];
    const focusIds = selectedPerson ? focusTrackIdsByCamera : null;

    for (const sess of dayData.camera_sessions) {
      const tracking = cameraTracking[sess.key];
      if (!tracking) continue;
      const videoName = sess.parts?.[0]?.name || sess.camera || sess.key;
      const camKey = cameraKeyFromVideo(videoName, sess.camera_index);
      const homo =
        cameraHomographies[sess.camera] ??
        cameraHomographies[sess.key] ??
        cameraHomographies[camKey];
      const H = (homo?.H?.length === 9 ? (homo.H as Mat3) : null) ?? null;
      if (!H && !homo?.placement) continue;

      const t0Abs = sess.t0_abs ?? minT;
      const duration = sessionFileDuration(sess);
      const localSec = currentDaySec - t0Abs;
      if (localSec < -0.05 || (duration > 0 && localSec > duration + 0.05)) continue;

      const fps = tracking.fps || sess.fps || 25;
      const frameFloat = frameAtTime(Math.max(0, localSec), fps, tracking.frame_count);
      const keyframes = buildTrackKeyframes(tracking);
      const every = resolveDetectEveryN(tracking);
      const dets = detectionsAtFrame(keyframes, frameFloat, every);

      const sessFocus = focusIds?.[sess.camera];

      for (const det of dets) {
        const projected = resolveFeetOnMap(det, frameFloat, H, {
          cameraKey: camKey,
          homography: homo,
          trackingSize: [tracking.width, tracking.height],
          detectEveryN: every,
        });
        if (!projected) continue;

        const isFocus = sessFocus != null && sessFocus.includes(det.track_id);
        const dimmed = selectedPerson != null && !isFocus;

        markers.push({
          camera: sess.camera,
          trackId: det.track_id,
          map: projected.map,
          live: true,
          dimmed,
          feetSource: projected.source,
          confidence: projected.confidence,
        });
      }
    }
    return markers;
  }, [
    dayData?.camera_sessions,
    timeBounds.minT,
    selectedPerson,
    focusTrackIdsByCamera,
    cameraTracking,
    cameraHomographies,
    currentDaySec,
  ]);

  useEffect(() => {
    onLiveMarkersChange?.(liveMarkers);
  }, [liveMarkers, onLiveMarkersChange]);

  useEffect(() => {
    return () => {
      onLiveMarkersChange?.([]);
    };
  }, [onLiveMarkersChange]);

  const selectPerson = useCallback(
    (person: GlobalPerson) => {
      setSelectedPersonId(person.person_id);
      setSelectedEdge(null);
      setRightTab("edges");
      seekToDaySec(person.t0);
    },
    [seekToDaySec],
  );

  const jumpToTransition = (direction: "prev" | "next") => {
    if (!personEdges.length || !selectedPerson) return;
    const sorted = [...personEdges].sort((a, b) => {
      const trA = selectedPerson.tracks.find((t) => t.uid === a.from);
      const trB = selectedPerson.tracks.find((t) => t.uid === b.from);
      return (trA?.t1 ?? 0) - (trB?.t1 ?? 0);
    });
    const cur = currentDaySecRef.current;
    const edge =
      direction === "next"
        ? sorted.find((e) => (selectedPerson.tracks.find((t) => t.uid === e.from)?.t1 ?? 0) > cur + 0.5) ?? sorted[0]
        : [...sorted].reverse().find((e) => (selectedPerson.tracks.find((t) => t.uid === e.from)?.t1 ?? 0) < cur - 0.5) ??
          sorted[sorted.length - 1];
    if (!edge) return;
    setSelectedEdge(edge);
    setRightTab("inspector");
    const tr = selectedPerson.tracks.find((t) => t.uid === edge.from);
    if (tr) seekToDaySec(tr.t1);
  };

  const onSelectEdge = useCallback(
    (edge: DayTransitionEdge) => {
      setSelectedEdge(edge);
      setRightTab("inspector");
      const trFrom = selectedPerson?.tracks.find((t) => t.uid === edge.from);
      if (trFrom) seekToDaySec(trFrom.t1);
    },
    [selectedPerson, seekToDaySec],
  );

  if (!selectedDay) {
    return (
      <div className="day-panel">
        <p className="day-empty">Выберите день в шапке, чтобы открыть межкамерную склейку.</p>
      </div>
    );
  }

  const stats = dayData?.stats;
  const camCount = dayData?.cameras?.length ?? cameraSessionsList.length;
  const sessionTitle = dayData?.sessions?.join(", ") || "";

  return (
    <div className="day-panel">
      <div className="day-header">
        <div className="day-kpi" title={sessionTitle}>
          <span>
            Камеры <b>{camCount || "—"}</b>
          </span>
          {stats && (
            <>
              <span>
                Персоны <b>{stats.n_persons ?? 0}</b>
              </span>
              <span className="day-kpi-accent">
                Мультикам <b>{stats.n_multi_cam_persons ?? 0}</b>
              </span>
              <span>
                Склейки <b>{stats.n_merges_total ?? 0}</b>
              </span>
              <span className="pass-pill pass0">P0 {stats.pass0_merges ?? 0}</span>
              <span className="pass-pill pass1">P1 {stats.pass1_merges ?? 0}</span>
            </>
          )}
        </div>
        <div className="day-header-actions">
          {loading && <span className="day-muted">Загрузка…</span>}
          <button
            type="button"
            className="day-btn day-btn-primary"
            onClick={() => void handleRunDayLink()}
            disabled={runningSolver || !selectedDay}
          >
            {runningSolver ? "Склейка…" : "Склеить день"}
          </button>
        </div>
      </div>

      {error && <div className="day-error">{error}</div>}

      {!loading && dayData && !dayData.has_links && !(dayData.persons?.length) ? (
        <div className="day-empty">
          Нет day_links для этого дня.
          <div className="day-empty-actions">
            <button type="button" className="day-btn day-btn-primary" onClick={() => void handleRunDayLink()} disabled={runningSolver}>
              Склеить день
            </button>
          </div>
        </div>
      ) : (
        <div className="day-grid">
          <DayPersonList
            persons={displayPersons}
            counts={{
              all: dayData?.persons?.length ?? 0,
              multi: stats?.n_multi_cam_persons ?? 0,
              solo: stats?.n_solo_persons ?? 0,
            }}
            filterMode={filterMode}
            sortMode={sortMode}
            searchQuery={searchQuery}
            selectedPersonId={selectedPersonId}
            onFilter={setFilterMode}
            onSort={setSortMode}
            onSearch={setSearchQuery}
            onSelect={selectPerson}
          />

          <section className="day-center" aria-label="Таймлайн и видео дня">
            <PlaybackToolbar
              clock={clock}
              bounds={timeBounds}
              formatCurrent={formatTimeOfDay}
              formatBound={formatShortTime}
              extras={
                <>
                  <span className="day-cam-scale" title="Камер в ряду">
                    <span className="playhead-clock-span">В ряд</span>
                    {Array.from({ length: DAY_CAM_COLS_MAX }, (_, i) => i + 1).map((n) => (
                      <button
                        key={n}
                        type="button"
                        className={`playhead-btn ${cameraCols === n ? "on" : ""}`}
                        onClick={() => setCameraCols(n)}
                      >
                        {n}
                      </button>
                    ))}
                  </span>
                  {selectedPerson ? (
                    <>
                      <button
                        type="button"
                        className="playhead-btn"
                        onClick={() => {
                          setSelectedPersonId(null);
                          setSelectedEdge(null);
                        }}
                      >
                        Все персоны
                      </button>
                      {personEdges.length > 0 && (
                        <>
                          <button type="button" className="playhead-btn" onClick={() => jumpToTransition("prev")}>
                            ← Склейка
                          </button>
                          <button type="button" className="playhead-btn" onClick={() => jumpToTransition("next")}>
                            Склейка →
                          </button>
                        </>
                      )}
                    </>
                  ) : null}
                </>
              }
            />

            <PlayheadTimeline
              lanes={timelineLanes}
              bounds={timeBounds}
              currentSec={currentDaySec}
              zoom={clock.zoom}
              formatTick={formatShortTime}
              onSeek={(sec) => clock.seek(sec)}
              onScrubbing={clock.setScrubbing}
              onSelect={(seg) => {
                setSelectedPersonId(seg.data.personId);
                setSelectedEdge(null);
                setRightTab("edges");
                clock.seek(seg.data.t0);
              }}
              onShiftSelect={(seg) => {
                onOpenTrackInMerge?.(seg.data.sessionKey, seg.data.trackId, seg.data.t0);
              }}
            />

            <div
              className="day-cameras"
              style={{
                gridTemplateColumns: `repeat(${Math.max(1, Math.min(cameraCols, Math.max(1, cameraSessionsList.length)))}, minmax(0, 1fr))`,
              }}
            >
              {cameraSessionsList.map((sess) => {
                const t0Abs = sess.t0_abs ?? timeBounds.minT;
                const duration = sessionFileDuration(sess) || 300;
                const isBefore = currentDaySec < t0Abs - 0.05;
                const isAfter = duration > 0 && currentDaySec > t0Abs + duration + 0.05;
                const isActiveNow = !isBefore && !isAfter;
                const localSec = Math.max(0, Math.min(currentDaySec - t0Abs, duration));
                const focusIds = focusTrackIdsByCamera[sess.camera] ?? EMPTY_FOCUS;
                const isPersonInCamera = focusIds.length > 0 && isActiveNow;

                let standbyTitle = "";
                let standbySub = "";
                if (isBefore) {
                  standbyTitle = "Ожидание начала записи";
                  standbySub = `Старт в ${formatTimeOfDay(t0Abs, false)}`;
                } else if (isAfter) {
                  standbyTitle = "Запись завершена";
                  standbySub = `Конец в ${formatTimeOfDay(t0Abs + duration, false)}`;
                }

                return (
                  <div
                    key={sess.key}
                    className={`day-camera-card ${isPersonInCamera ? "has-person" : ""} ${isActiveNow ? "" : "is-idle"}`}
                  >
                    <div className="day-camera-head">
                      <div className="day-camera-head-left">
                        <span className="day-camera-name">{sess.camera}</span>
                        {isActiveNow ? <span className="day-live">В эфире</span> : <span className="day-idle">Нет записи</span>}
                      </div>
                      <div className="day-camera-head-right">
                        {selectedPerson && isPersonInCamera && (
                          <span className="day-focus-tag">
                            {selectedPerson.label} · T{focusIds.join(",")}
                          </span>
                        )}
                        <span className="day-camera-local">{formatDuration(localSec)}</span>
                      </div>
                    </div>
                    <div className="day-camera-body">
                      <TrackingPlayer
                        ref={playerRefFor(sess.key)}
                        videoUrl={sess.parts[0]?.videoUrl ?? null}
                        tracking={cameraTracking[sess.key] ?? null}
                        sessionParts={sess.parts.length ? sess.parts : null}
                        fps={sess.fps}
                        durationSec={sess.duration_sec}
                        showLabels={true}
                        showTrails={true}
                        trailLength={25}
                        groupByTrack={dayData?.track_to_person?.[sess.key] ?? EMPTY_GROUP}
                        highlightIds={selectedPerson ? focusIds : []}
                        focusTrackIds={selectedPerson ? focusIds : null}
                        compact={true}
                        inactive={!isActiveNow}
                      />
                      {!isActiveNow && (
                        <div className="day-camera-standby-overlay">
                          <span className="day-camera-standby-title">{standbyTitle}</span>
                          <span className="day-camera-standby-sub">{standbySub}</span>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <DayInspector
            person={selectedPerson}
            edges={personEdges}
            selectedEdge={selectedEdge}
            rightTab={rightTab}
            onTab={setRightTab}
            onSelectEdge={onSelectEdge}
            onSeek={seekToDaySec}
            onOpenTrackInMerge={onOpenTrackInMerge}
          />
        </div>
      )}
    </div>
  );
}

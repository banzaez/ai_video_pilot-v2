import { memo, useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { colorForTrackId, fetchTrackingJson, formatDuration } from "../utils";
import { TrackingPlayer, type TrackingPlayerHandle } from "./TrackingPlayer";
import type { MediaSession } from "../session";
import type { TrackingData } from "../types";

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

type FilterMode = "all" | "multi_cam" | "pass1" | "pass2" | "pass4" | "solo";
type SortMode = "person_id" | "tracks" | "span" | "time";
type RightTab = "edges" | "tracks" | "inspector";
type TimeBounds = { minT: number; maxT: number; span: number };
type DayCameraSession = MediaSession & { t0_abs?: number };

const trackingCache = new Map<string, Record<string, TrackingData>>();
const UI_HZ_MS = 100;

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

function formatSecToTimeOfDay(sec: number): string {
  const s = Math.floor(sec) % 86400;
  const ms = Math.floor((sec - Math.floor(sec)) * 100);
  const hh = String(Math.floor(s / 3600)).padStart(2, "0");
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}.${String(ms).padStart(2, "0")}`;
}

function formatShortTime(sec: number): string {
  const s = Math.floor(sec) % 86400;
  const hh = String(Math.floor(s / 3600)).padStart(2, "0");
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function formatScore(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toFixed(2);
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
        <button type="button" className={filterMode === "pass1" ? "on" : ""} onClick={() => onFilter("pass1")}>
          Pass 1
        </button>
        <button type="button" className={filterMode === "pass2" ? "on" : ""} onClick={() => onFilter("pass2")}>
          Pass 2
        </button>
        <button type="button" className={filterMode === "pass4" ? "on" : ""} onClick={() => onFilter("pass4")}>
          Pass 4
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
}: {
  person: GlobalPerson | null;
  edges: DayTransitionEdge[];
  selectedEdge: DayTransitionEdge | null;
  rightTab: RightTab;
  onTab: (tab: RightTab) => void;
  onSelectEdge: (edge: DayTransitionEdge) => void;
  onSeek: (sec: number) => void;
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
                    onClick={() => onSelectEdge(edge)}
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
                    onClick={() => {
                      if (tr) onSeek(isFrom ? tr.t1 : tr.t0);
                    }}
                    title={isFrom ? "К точке выхода" : "К точке входа"}
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
                <tr key={t.uid} onClick={() => onSeek(t.t0)} title="К началу трека">
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

const DayTimeline = memo(function DayTimeline({
  cameras,
  persons,
  selectedPersonId,
  timeBounds,
  zoom,
  currentDaySec,
  onSeek,
  onSelectPerson,
  onScrubbing,
}: {
  cameras: DayCameraSession[];
  persons: GlobalPerson[];
  selectedPersonId: number | null;
  timeBounds: TimeBounds;
  zoom: number;
  currentDaySec: number;
  onSeek: (sec: number) => void;
  onSelectPerson: (id: number, t0: number) => void;
  onScrubbing?: (active: boolean) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const shellRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const downPos = useRef({ x: 0, y: 0 });
  const barHit = useRef<{ id: number; t0: number } | null>(null);

  const seekFromClientX = useCallback(
    (clientX: number) => {
      const plot = plotRef.current;
      if (!plot) return;
      const rect = plot.getBoundingClientRect();
      const pct = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(1, rect.width)));
      onSeek(timeBounds.minT + pct * timeBounds.span);
    },
    [onSeek, timeBounds.minT, timeBounds.span],
  );

  const endDrag = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (!dragging.current) return;
      dragging.current = false;
      onScrubbing?.(false);
      e.currentTarget.classList.remove("is-dragging");
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
      const dx = Math.abs(e.clientX - downPos.current.x);
      const dy = Math.abs(e.clientY - downPos.current.y);
      if (dx < 5 && dy < 5 && barHit.current) {
        onSelectPerson(barHit.current.id, barHit.current.t0);
      } else {
        seekFromClientX(e.clientX);
      }
      barHit.current = null;
    },
    [onSelectPerson, onScrubbing, seekFromClientX],
  );

  useEffect(() => {
    if (dragging.current) return;
    const scroller = scrollRef.current;
    const plot = plotRef.current;
    if (!scroller || !plot || timeBounds.span <= 0) return;
    const pct = (currentDaySec - timeBounds.minT) / timeBounds.span;
    const plotRect = plot.getBoundingClientRect();
    const scrollRect = scroller.getBoundingClientRect();
    const x = plotRect.left - scrollRect.left + scroller.scrollLeft + pct * plot.offsetWidth;
    const view = scroller.clientWidth;
    const sl = scroller.scrollLeft;
    if (x < sl + 48 || x > sl + view - 48) {
      scroller.scrollLeft = Math.max(0, x - view * 0.45);
    }
  }, [currentDaySec, zoom, timeBounds.minT, timeBounds.span]);

  const playheadPct = timeBounds.span > 0 ? ((currentDaySec - timeBounds.minT) / timeBounds.span) * 100 : 0;

  return (
    <div className="day-timeline-scroll" ref={scrollRef}>
      <div
        ref={shellRef}
        className="day-timeline-inner"
        style={{ width: `${Math.max(100, 100 * zoom)}%` }}
        onPointerDown={(e) => {
          if (e.button !== 0) return;
          e.preventDefault();
          e.currentTarget.setPointerCapture(e.pointerId);
          e.currentTarget.classList.add("is-dragging");
          dragging.current = true;
          onScrubbing?.(true);
          downPos.current = { x: e.clientX, y: e.clientY };
          const bar = (e.target as HTMLElement).closest(".day-bar") as HTMLElement | null;
          const pid = bar?.dataset.personId;
          const t0 = bar?.dataset.t0;
          barHit.current = pid != null && t0 != null ? { id: Number(pid), t0: Number(t0) } : null;
          seekFromClientX(e.clientX);
        }}
        onPointerMove={(e) => {
          if (!dragging.current) return;
          e.preventDefault();
          seekFromClientX(e.clientX);
        }}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onDragStart={(e) => e.preventDefault()}
      >
        <div className="day-axis-row">
          <div className="day-lane-label" aria-hidden />
          <div className="day-axis" ref={plotRef}>
            {Array.from({ length: 11 }).map((_, i) => {
              const frac = i / 10;
              const t = timeBounds.minT + frac * timeBounds.span;
              return (
                <div key={i} className="day-axis-tick" style={{ left: `${frac * 100}%` }}>
                  <i />
                  {formatShortTime(t)}
                </div>
              );
            })}
          </div>
        </div>
        <div className="day-playhead-layer" aria-hidden>
          <div className="day-lane-label" />
          <div className="day-playhead-track">
            <div className="day-playhead" style={{ left: `${playheadPct}%` }} />
          </div>
        </div>
        <div className="day-lanes">
          {cameras.map((sess) => {
            const hasSel =
              selectedPersonId != null &&
              persons.some(
                (p) => p.person_id === selectedPersonId && p.tracks.some((t) => t.camera === sess.camera),
              );
            return (
              <div key={sess.key} className={`day-lane ${hasSel ? "has-sel" : ""}`}>
                <div className="day-lane-label">{sess.camera}</div>
                <div className="day-lane-track">
                  {persons.map((person) => {
                    const selected = person.person_id === selectedPersonId;
                    const noFocus = selectedPersonId == null;
                    return person.tracks
                      .filter((t) => t.camera === sess.camera)
                      .map((t) => {
                        const leftPct = ((t.t0 - timeBounds.minT) / timeBounds.span) * 100;
                        const widthPct = Math.max(0.45, ((t.t1 - t.t0) / timeBounds.span) * 100);
                        return (
                          <div
                            key={t.uid}
                            role="button"
                            tabIndex={-1}
                            data-person-id={person.person_id}
                            data-t0={t.t0}
                            className={`day-bar ${noFocus ? "" : selected ? "on" : "dim"}`}
                            style={{
                              left: `${leftPct}%`,
                              width: `${widthPct}%`,
                              background: colorForTrackId(t.track_id),
                            }}
                            title={`${person.label} (${sess.camera} #${t.track_id}) ${formatShortTime(t.t0)}–${formatShortTime(t.t1)}`}
                          >
                            {selected && !noFocus ? `T${t.track_id}` : ""}
                          </div>
                        );
                      });
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
});

export function DayAnalysisPanel({ selectedDay }: { selectedDay: string }) {
  const [dayData, setDayData] = useState<DayLinksData | null>(null);
  const [loading, setLoading] = useState(false);
  const [runningSolver, setRunningSolver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cameraTracking, setCameraTracking] = useState<Record<string, TrackingData>>({});

  const [filterMode, setFilterMode] = useState<FilterMode>("multi_cam");
  const [sortMode, setSortMode] = useState<SortMode>("person_id");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedPersonId, setSelectedPersonId] = useState<number | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<DayTransitionEdge | null>(null);
  const [rightTab, setRightTab] = useState<RightTab>("edges");

  const [currentDaySec, setCurrentDaySec] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [zoom, setZoom] = useState(1);

  const playerRefs = useRef<Record<string, TrackingPlayerHandle | null>>({});
  const currentDaySecRef = useRef(0);
  const isPlayingRef = useRef(false);
  const scrubbingRef = useRef(false);
  const playbackSpeedRef = useRef(1);
  const timeBoundsRef = useRef<TimeBounds>({ minT: 0, maxT: 300, span: 300 });
  const seekToDaySecRef = useRef<(sec: number, playAfter?: boolean) => void>(() => {});
  const syncPlayersRef = useRef<(t: number, play?: boolean) => void>(() => {});

  const timeBounds = useMemo((): TimeBounds => {
    if (!dayData?.persons?.length) return { minT: 0, maxT: 300, span: 300 };
    let minT = Infinity;
    let maxT = -Infinity;
    for (const p of dayData.persons) {
      if (p.tracks?.length) {
        for (const t of p.tracks) {
          if (t.t0 < minT) minT = t.t0;
          if (t.t1 > maxT) maxT = t.t1;
        }
      } else {
        if (p.t0 < minT) minT = p.t0;
        if (p.t1 > maxT) maxT = p.t1;
      }
    }
    if (!Number.isFinite(minT)) minT = 0;
    if (!Number.isFinite(maxT) || maxT <= minT) maxT = minT + 300;
    return { minT, maxT, span: Math.max(1e-6, maxT - minT) };
  }, [dayData?.persons]);

  timeBoundsRef.current = timeBounds;
  isPlayingRef.current = isPlaying;
  playbackSpeedRef.current = playbackSpeed;

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
          setCurrentDaySec(first.t0);
          currentDaySecRef.current = first.t0;
        } else {
          setSelectedPersonId(null);
          setCurrentDaySec(0);
          currentDaySecRef.current = 0;
        }
        setSelectedEdge(null);
        setIsPlaying(false);

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

  const syncPlayersToTime = useCallback((tDay: number, playAfter = false) => {
    const sessions = dayData?.camera_sessions;
    if (!sessions) return;
    const minT = timeBoundsRef.current.minT;
    for (const sess of sessions) {
      const handle = playerRefs.current[sess.key];
      if (!handle) continue;
      const t0Abs = sess.t0_abs ?? minT;
      const duration = sess.duration_sec ?? 300;
      if (tDay < t0Abs || tDay > t0Abs + duration) {
        handle.pause();
        continue;
      }
      handle.setPlaybackRate(playbackSpeedRef.current);
      if (!playAfter) handle.pause();
      handle.seekToGlobal(Math.max(0, tDay - t0Abs), playAfter);
    }
  }, [dayData?.camera_sessions]);

  const seekToDaySec = useCallback(
    (targetSec: number, playAfter?: boolean) => {
      const { minT, maxT } = timeBoundsRef.current;
      const clamped = Math.max(minT, Math.min(maxT, targetSec));
      currentDaySecRef.current = clamped;
      setCurrentDaySec(clamped);
      syncPlayersToTime(clamped, playAfter ?? isPlayingRef.current);
    },
    [syncPlayersToTime],
  );

  seekToDaySecRef.current = seekToDaySec;
  syncPlayersRef.current = syncPlayersToTime;

  useEffect(() => {
    if (!isPlaying) return;
    const id = window.setInterval(() => {
      if (scrubbingRef.current) return;
      syncPlayersRef.current(currentDaySecRef.current, true);
    }, 1000);
    return () => window.clearInterval(id);
  }, [isPlaying]);

  useEffect(() => {
    if (!isPlaying) return;
    for (const handle of Object.values(playerRefs.current)) {
      handle?.setPlaybackRate(playbackSpeed);
    }
  }, [isPlaying, playbackSpeed]);

  useEffect(() => {
    if (!isPlaying) return;
    let animId = 0;
    let lastTs = performance.now();
    let lastUi = 0;

    const loop = (now: number) => {
      if (scrubbingRef.current) {
        lastTs = now;
        animId = requestAnimationFrame(loop);
        return;
      }
      const dt = ((now - lastTs) / 1000) * playbackSpeedRef.current;
      lastTs = now;
      const bounds = timeBoundsRef.current;
      let next = currentDaySecRef.current + dt;
      if (next >= bounds.maxT) {
        next = bounds.maxT;
        currentDaySecRef.current = next;
        setCurrentDaySec(next);
        setIsPlaying(false);
        syncPlayersRef.current(next, false);
        return;
      }
      currentDaySecRef.current = next;
      if (now - lastUi >= UI_HZ_MS) {
        lastUi = now;
        setCurrentDaySec(next);
      }
      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [isPlaying]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes((e.target as HTMLElement)?.tagName)) return;
      if (e.code === "Space") {
        e.preventDefault();
        setIsPlaying((p) => {
          const next = !p;
          syncPlayersRef.current(currentDaySecRef.current, next);
          return next;
        });
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        seekToDaySecRef.current(currentDaySecRef.current - (e.shiftKey ? 1 : 1 / 25));
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        seekToDaySecRef.current(currentDaySecRef.current + (e.shiftKey ? 1 : 1 / 25));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

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
      if (filterMode === "pass1" || filterMode === "pass2" || filterMode === "pass4") {
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
              <span className="pass-pill pass1">P1 {stats.pass1_merges ?? 0}</span>
              <span className="pass-pill pass2">P2 {stats.pass2_merges ?? 0}</span>
              <span className="pass-pill pass4">P4 {stats.pass4_merges ?? 0}</span>
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
            <div className="day-toolbar">
              <div className="day-toolbar-group">
                <button
                  type="button"
                  className={`day-btn day-btn-play ${isPlaying ? "is-playing" : "day-btn-primary"}`}
                  onClick={() => {
                    const next = !isPlaying;
                    setIsPlaying(next);
                    syncPlayersToTime(currentDaySecRef.current, next);
                  }}
                  title="Пробел"
                >
                  {isPlaying ? "Пауза" : "Пуск"}
                </button>
                <button type="button" className="day-btn" onClick={() => seekToDaySec(currentDaySecRef.current - 1 / 25)} title="←">
                  −1к
                </button>
                <button type="button" className="day-btn" onClick={() => seekToDaySec(currentDaySecRef.current + 1 / 25)} title="→">
                  +1к
                </button>
                {[0.5, 1, 2, 4].map((spd) => (
                  <button
                    key={spd}
                    type="button"
                    className={`day-btn day-speed ${playbackSpeed === spd ? "on" : ""}`}
                    onClick={() => setPlaybackSpeed(spd)}
                  >
                    {spd}×
                  </button>
                ))}
              </div>
              <span className="day-clock">{formatSecToTimeOfDay(currentDaySec)}</span>
              <span className="day-clock-span">
                {formatShortTime(timeBounds.minT)}–{formatShortTime(timeBounds.maxT)} ({formatDuration(timeBounds.span)})
              </span>
              <div className="day-toolbar-group right">
                {selectedPerson && (
                  <>
                    <button
                      type="button"
                      className="day-btn"
                      onClick={() => {
                        setSelectedPersonId(null);
                        setSelectedEdge(null);
                      }}
                    >
                      Все персоны
                    </button>
                    {personEdges.length > 0 && (
                      <>
                        <button type="button" className="day-btn" onClick={() => jumpToTransition("prev")}>
                          ← Склейка
                        </button>
                        <button type="button" className="day-btn" onClick={() => jumpToTransition("next")}>
                          Склейка →
                        </button>
                      </>
                    )}
                  </>
                )}
                <button type="button" className="day-btn" onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}>
                  −
                </button>
                <span className="day-clock-span">{Math.round(zoom * 100)}%</span>
                <button type="button" className="day-btn" onClick={() => setZoom((z) => Math.min(4, z + 0.25))}>
                  +
                </button>
                <button type="button" className="day-btn" onClick={() => setZoom(1)}>
                  100%
                </button>
              </div>
            </div>

            <DayTimeline
              cameras={cameraSessionsList}
              persons={timelinePersons}
              selectedPersonId={selectedPersonId}
              timeBounds={timeBounds}
              zoom={zoom}
              currentDaySec={currentDaySec}
              onSeek={seekToDaySec}
              onScrubbing={(active) => {
                scrubbingRef.current = active;
              }}
              onSelectPerson={(id, t0) => {
                setSelectedPersonId(id);
                setSelectedEdge(null);
                setRightTab("edges");
                seekToDaySec(t0);
              }}
            />

            <div
              className="day-cameras"
              style={{
                gridTemplateColumns: cameraSessionsList.length > 1 ? "repeat(auto-fit, minmax(240px, 1fr))" : "1fr",
              }}
            >
              {cameraSessionsList.map((sess) => {
                const t0Abs = sess.t0_abs ?? timeBounds.minT;
                const duration = sess.duration_sec ?? 300;
                const isActiveNow = currentDaySec >= t0Abs && currentDaySec <= t0Abs + duration;
                const localSec = Math.max(0, currentDaySec - t0Abs);
                const focusIds = focusTrackIdsByCamera[sess.camera] ?? [];
                const isPersonInCamera = focusIds.length > 0 && isActiveNow;
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
                        ref={(r) => {
                          playerRefs.current[sess.key] = r;
                        }}
                        videoUrl={sess.parts[0]?.videoUrl ?? null}
                        tracking={cameraTracking[sess.key] ?? null}
                        sessionParts={sess.parts.length ? sess.parts : null}
                        showLabels={true}
                        showTrails={true}
                        trailLength={25}
                        groupByTrack={dayData?.track_to_person?.[sess.key] ?? {}}
                        focusTrackIds={selectedPerson ? focusIds : null}
                        hideUnfocused={selectedPerson != null}
                        compact={true}
                      />
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
          />
        </div>
      )}
    </div>
  );
}

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { colorForTrackId, fetchTrackingJson, formatDuration } from "../utils";
import { TrackingPlayer, type TrackingPlayerHandle } from "./TrackingPlayer";
import type { MediaSession } from "../session";
import type { TrackingData } from "../types";

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
  has_face: boolean;
  has_reid: boolean;
  best_face_score: number;
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
  face?: number | null;
  face_scores?: Record<string, number> | null;
  pose_face?: number | null;
  reid?: number | null;
  motion: number;
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
  best_face_crop: string | null;
  face_crops: string[];
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
  face_models?: string[];
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

type FilterMode = "all" | "multi_cam" | "with_face" | "pass0" | "pass1" | "pass2" | "pass4" | "solo";
type SortMode = "person_id" | "tracks" | "span" | "time";
type RightTab = "edges" | "tracks" | "inspector";

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

function badgeClass(score: number): string {
  if (score >= 0.85) return "badge-score high";
  if (score >= 0.70) return "badge-score mid";
  return "badge-score low";
}

function passBadge(pass?: number) {
  switch (pass) {
    case 0:
      return <span className="pass-pill pass0">Pass 0 · Direct</span>;
    case 1:
      return <span className="pass-pill pass1">Pass 1 · Hungarian</span>;
    case 2:
      return <span className="pass-pill pass2">Pass 2 · Chain</span>;
    case 4:
      return <span className="pass-pill pass4">Pass 4 · Handover</span>;
    default:
      return <span className="pass-pill pass-other">Link</span>;
  }
}

export function DayAnalysisPanel({ selectedDay }: { selectedDay: string }) {
  const [dayData, setDayData] = useState<DayLinksData | null>(null);
  const [loading, setLoading] = useState(false);
  const [runningSolver, setRunningSolver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Детекции tracking.json по каждой сессии
  const [cameraTracking, setCameraTracking] = useState<Record<string, TrackingData>>({});

  // Фильтры и выбор персоны
  const [filterMode, setFilterMode] = useState<FilterMode>("all");
  const [sortMode, setSortMode] = useState<SortMode>("person_id");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedPersonId, setSelectedPersonId] = useState<number | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<DayTransitionEdge | null>(null);
  const [rightTab, setRightTab] = useState<RightTab>("edges");

  // Мастер-таймлайн и управление воспроизведением
  const [currentDaySec, setCurrentDaySec] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [zoom, setZoom] = useState<number>(1.0);

  const timelineContainerRef = useRef<HTMLDivElement>(null);
  const playerRefs = useRef<Record<string, TrackingPlayerHandle | null>>({});
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({});

  // Границы времени дня
  const timeBounds = useMemo(() => {
    if (!dayData?.persons || dayData.persons.length === 0) {
      return { minT: 0, maxT: 300, span: 300 };
    }
    let minT = Infinity;
    let maxT = -Infinity;
    for (const p of dayData.persons) {
      if (p.t0 < minT) minT = p.t0;
      if (p.t1 > maxT) maxT = p.t1;
    }
    if (!Number.isFinite(minT)) minT = 0;
    if (!Number.isFinite(maxT) || maxT <= minT) maxT = minT + 300;
    return { minT, maxT, span: Math.max(1, maxT - minT) };
  }, [dayData?.persons]);

  // Загрузка метаданных дня и tracking.json по камерам
  useEffect(() => {
    if (!selectedDay) return;
    let isCancelled = false;

    const loadDayMeta = async () => {
      try {
        setLoading(true);
        setError(null);
        const res = await fetch(`/api/day/meta?day=${encodeURIComponent(selectedDay)}`);
        const json = (await res.json()) as DayLinksData;
        if (isCancelled) return;

        setDayData(json);
        if (json.persons && json.persons.length > 0) {
          setSelectedPersonId(json.persons[0].person_id);
          setCurrentDaySec(json.persons[0].t0);
        } else {
          setSelectedPersonId(null);
          setCurrentDaySec(0);
        }
        setSelectedEdge(null);
        setIsPlaying(false);

        // Загрузка tracking.json для каждой сессии дня через fetchTrackingJson
        if (json.camera_sessions && json.camera_sessions.length > 0) {
          const loadedTracks: Record<string, TrackingData> = {};
          await Promise.all(
            json.camera_sessions.map(async (sess) => {
              if (sess.jsonUrl) {
                try {
                  const trData = await fetchTrackingJson(sess.jsonUrl);
                  loadedTracks[sess.key] = trData;
                } catch {
                  // Игнорируем отсутствие трекинга
                }
              }
            }),
          );
          if (!isCancelled) {
            setCameraTracking(loadedTracks);
          }
        }
      } catch (e) {
        if (!isCancelled) setError(`Ошибка загрузки дня ${selectedDay}: ${e}`);
      } finally {
        if (!isCancelled) setLoading(false);
      }
    };

    loadDayMeta();
    return () => {
      isCancelled = true;
    };
  }, [selectedDay]);

  // Синхронизация всех плееров при смене currentDaySec или isPlaying
  const syncPlayersToTime = useCallback(
    (tDay: number, playAfter = false) => {
      if (!dayData?.camera_sessions) return;
      for (const sess of dayData.camera_sessions) {
        const t0_abs = sess.t0_abs ?? timeBounds.minT;
        const tLocal = Math.max(0, tDay - t0_abs);
        const handle = playerRefs.current[sess.key];
        if (handle) {
          handle.seekToGlobal(tLocal, playAfter);
        }
      }
    },
    [dayData?.camera_sessions, timeBounds.minT],
  );

  const seekToDaySec = useCallback(
    (targetSec: number, playAfter?: boolean) => {
      const clamped = Math.max(timeBounds.minT, Math.min(timeBounds.maxT, targetSec));
      setCurrentDaySec(clamped);
      syncPlayersToTime(clamped, playAfter ?? isPlaying);
    },
    [timeBounds.minT, timeBounds.maxT, syncPlayersToTime, isPlaying],
  );

  // Мастер RAF цикл воспроизведения
  useEffect(() => {
    if (!isPlaying) return;
    let animId = 0;
    let lastTs = performance.now();

    const loop = (now: number) => {
      const dt = ((now - lastTs) / 1000) * playbackSpeed;
      lastTs = now;

      setCurrentDaySec((prev) => {
        const next = prev + dt;
        if (next >= timeBounds.maxT) {
          setIsPlaying(false);
          syncPlayersToTime(timeBounds.maxT, false);
          return timeBounds.maxT;
        }
        return next;
      });

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [isPlaying, playbackSpeed, timeBounds.maxT, syncPlayersToTime]);

  // Горячие клавиши (Space, Arrows)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Игнорируем если фокус в инпуте
      if (["INPUT", "TEXTAREA", "SELECT"].includes((e.target as HTMLElement)?.tagName)) return;

      if (e.code === "Space") {
        e.preventDefault();
        setIsPlaying((p) => {
          const next = !p;
          syncPlayersToTime(currentDaySec, next);
          return next;
        });
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        const step = e.shiftKey ? 1.0 : 1 / 25;
        seekToDaySec(currentDaySec - step);
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        const step = e.shiftKey ? 1.0 : 1 / 25;
        seekToDaySec(currentDaySec + step);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [currentDaySec, seekToDaySec, syncPlayersToTime]);

  // Запуск склейки дня
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

  // Выбранная персона
  const selectedPerson = useMemo(() => {
    if (!dayData?.persons || selectedPersonId == null) return null;
    return dayData.persons.find((p) => p.person_id === selectedPersonId) || null;
  }, [dayData?.persons, selectedPersonId]);

  // Ребра склеек выбранной персоны
  const personEdges = useMemo(() => {
    if (!selectedPerson || !dayData?.edges) return [];
    const uids = new Set(selectedPerson.tracks.map((t) => t.uid));
    return dayData.edges.filter((e) => uids.has(e.from) && uids.has(e.to));
  }, [selectedPerson, dayData?.edges]);

  // Маппинг треков выбранной персоны по камерам: cameraName -> track_id[]
  const focusTrackIdsByCamera = useMemo(() => {
    const map: Record<string, number[]> = {};
    if (!selectedPerson) return map;
    for (const tr of selectedPerson.tracks) {
      if (!map[tr.camera]) map[tr.camera] = [];
      map[tr.camera]!.push(tr.track_id);
    }
    return map;
  }, [selectedPerson]);

  // Фильтрация и сортировка списка персон
  const displayPersons = useMemo(() => {
    if (!dayData?.persons) return [];
    let list = dayData.persons.filter((p) => {
      if (filterMode === "multi_cam" && p.n_cameras < 2) return false;
      if (filterMode === "with_face" && !p.best_face_crop) return false;
      if (filterMode === "solo" && (p.n_cameras > 1 || p.n_tracks > 1)) return false;
      if (filterMode === "pass0") {
        const uids = new Set(p.tracks.map((t) => t.uid));
        const hasP0 = dayData.edges.some((e) => e.pass === 0 && uids.has(e.from) && uids.has(e.to));
        if (!hasP0) return false;
      }
      if (filterMode === "pass1") {
        const uids = new Set(p.tracks.map((t) => t.uid));
        const hasP1 = dayData.edges.some((e) => e.pass === 1 && uids.has(e.from) && uids.has(e.to));
        if (!hasP1) return false;
      }
      if (filterMode === "pass2") {
        const uids = new Set(p.tracks.map((t) => t.uid));
        const hasP2 = dayData.edges.some((e) => e.pass === 2 && uids.has(e.from) && uids.has(e.to));
        if (!hasP2) return false;
      }
      if (filterMode === "pass4") {
        const uids = new Set(p.tracks.map((t) => t.uid));
        const hasP4 = dayData.edges.some((e) => e.pass === 4 && uids.has(e.from) && uids.has(e.to));
        if (!hasP4) return false;
      }
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        const matchesId = String(p.person_id).includes(q) || p.label.toLowerCase().includes(q);
        const matchesCam = p.cameras.some((c) => c.toLowerCase().includes(q));
        const matchesTrack = p.tracks.some((t) => String(t.track_id).includes(q) || t.uid.toLowerCase().includes(q));
        if (!matchesId && !matchesCam && !matchesTrack) return false;
      }
      return true;
    });

    list = [...list].sort((a, b) => {
      if (sortMode === "person_id") return a.person_id - b.person_id;
      if (sortMode === "tracks") return b.n_tracks - a.n_tracks;
      if (sortMode === "span") return b.duration_sec - a.duration_sec;
      if (sortMode === "time") return a.t0 - b.t0;
      return 0;
    });

    return list;
  }, [dayData?.persons, dayData?.edges, filterMode, sortMode, searchQuery]);

  const cameraSessionsList = useMemo(() => {
    return dayData?.camera_sessions || [];
  }, [dayData?.camera_sessions]);

  // Обработка клика/скруббинга по таймлайну
  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(1, clickX / rect.width));
    const targetSec = timeBounds.minT + pct * timeBounds.span;
    seekToDaySec(targetSec);
  };

  // Переход к следующей/предыдущей склейке персоны
  const jumpToTransition = (direction: "prev" | "next") => {
    if (!personEdges.length || !selectedPerson) return;
    const sortedEdges = [...personEdges].sort((a, b) => {
      const trA = selectedPerson.tracks.find((t) => t.uid === a.from);
      const trB = selectedPerson.tracks.find((t) => t.uid === b.from);
      return (trA?.t1 ?? 0) - (trB?.t1 ?? 0);
    });

    if (direction === "next") {
      const nextEdge = sortedEdges.find((e) => {
        const tr = selectedPerson.tracks.find((t) => t.uid === e.from);
        return (tr?.t1 ?? 0) > currentDaySec + 0.5;
      }) || sortedEdges[0];
      if (nextEdge) {
        setSelectedEdge(nextEdge);
        const tr = selectedPerson.tracks.find((t) => t.uid === nextEdge.from);
        if (tr) seekToDaySec(tr.t1);
      }
    } else {
      const prevEdge = [...sortedEdges].reverse().find((e) => {
        const tr = selectedPerson.tracks.find((t) => t.uid === e.from);
        return (tr?.t1 ?? 0) < currentDaySec - 0.5;
      }) || sortedEdges[sortedEdges.length - 1];
      if (prevEdge) {
        setSelectedEdge(prevEdge);
        const tr = selectedPerson.tracks.find((t) => t.uid === prevEdge.from);
        if (tr) seekToDaySec(tr.t1);
      }
    }
  };

  return (
    <div className="merge-inspect-panel">
      {/* 1. Верхний Summary Bar */}
      <div className="merge-inspect-summary">
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <strong style={{ color: "var(--ink)" }}>День:</strong>
          <span
            style={{
              fontFamily: "var(--mono)",
              fontSize: 12,
              fontWeight: 700,
              color: "var(--accent, #0d6efd)",
              background: "rgba(13, 110, 253, 0.08)",
              padding: "2px 8px",
              borderRadius: 4,
              border: "1px solid rgba(13, 110, 253, 0.2)",
            }}
          >
            {dayData?.day || selectedDay}
          </span>
        </div>

        <span>
          Сессии: <strong>{dayData?.sessions?.join(", ") || "—"}</strong>
        </span>

        {dayData?.stats && (
          <>
            <span>
              Глобальных персон: <strong>{dayData.stats.n_persons ?? 0}</strong>
            </span>
            <span style={{ color: "var(--accent)" }}>
              Мультикамерных: <strong>{dayData.stats.n_multi_cam_persons ?? 0}</strong>
            </span>
            <span>
              Склеек: <strong>{dayData.stats.n_merges_total ?? 0}</strong> (
              <span className="pass-pill pass0">P0: {dayData.stats.pass0_merges ?? 0}</span>{" "}
              <span className="pass-pill pass1">P1: {dayData.stats.pass1_merges ?? 0}</span>{" "}
              <span className="pass-pill pass2">P2: {dayData.stats.pass2_merges ?? 0}</span>{" "}
              <span className="pass-pill pass4">P4: {dayData.stats.pass4_merges ?? 0}</span>)
            </span>
          </>
        )}

        <button
          type="button"
          className="merge-inspect-face-btn"
          style={{
            marginLeft: "auto",
            background: "var(--accent, #0d6efd)",
            color: "#fff",
            fontWeight: 600,
            padding: "3px 10px",
            borderRadius: 4,
          }}
          onClick={handleRunDayLink}
          disabled={runningSolver || !selectedDay}
        >
          {runningSolver ? "⏳ Выполняется склейка..." : "⚡ Запустить склейку дня"}
        </button>

        {loading && <span style={{ color: "var(--muted)" }}>Загрузка данных...</span>}
      </div>

      {error && (
        <div className="merge-inspect-stale" style={{ borderColor: "#e53e3e", background: "#fff5f5", color: "#c53030" }}>
          {error}
        </div>
      )}

      {/* 2. Основной 3-колоночный Grid Layout */}
      <div className="merge-inspect-grid">
        {/* ЛЕВАЯ КОЛОНКА: Поиск, фильтры и список персон */}
        <section className="merge-inspect-list" aria-label="Список персон">
          <input
            type="search"
            className="merge-inspect-search"
            placeholder="Поиск по ID, треку, камере…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />

          <div className="merge-inspect-filters">
            <button
              type="button"
              className={filterMode === "all" ? "on" : ""}
              onClick={() => setFilterMode("all")}
            >
              Все ({dayData?.persons?.length ?? 0})
            </button>
            <button
              type="button"
              className={filterMode === "multi_cam" ? "on" : ""}
              onClick={() => setFilterMode("multi_cam")}
            >
              Мультикам ({dayData?.stats?.n_multi_cam_persons ?? 0})
            </button>
            <button
              type="button"
              className={filterMode === "with_face" ? "on" : ""}
              onClick={() => setFilterMode("with_face")}
            >
              С лицом
            </button>
            <button
              type="button"
              className={filterMode === "pass0" ? "on" : ""}
              onClick={() => setFilterMode("pass0")}
            >
              Pass 0
            </button>
            <button
              type="button"
              className={filterMode === "pass1" ? "on" : ""}
              onClick={() => setFilterMode("pass1")}
            >
              Pass 1
            </button>
            <button
              type="button"
              className={filterMode === "solo" ? "on" : ""}
              onClick={() => setFilterMode("solo")}
            >
              Соло ({dayData?.stats?.n_solo_persons ?? 0})
            </button>
          </div>

          <div className="merge-inspect-sort">
            <span>Сортировка:</span>
            <select value={sortMode} onChange={(e) => setSortMode(e.target.value as SortMode)}>
              <option value="person_id">По ID персоны</option>
              <option value="tracks">По числу треков</option>
              <option value="span">По длительности</option>
              <option value="time">По времени дня</option>
            </select>
          </div>

          <ul className="merge-inspect-rows">
            {displayPersons.map((p) => {
              const isSelected = p.person_id === selectedPersonId;
              const pColor = colorForTrackId(p.person_id);

              return (
                <li key={p.person_id}>
                  <button
                    type="button"
                    className={`merge-inspect-row ${isSelected ? "on" : ""}`}
                    onClick={() => {
                      setSelectedPersonId(p.person_id);
                      setSelectedEdge(null);
                      setRightTab("edges");
                      seekToDaySec(p.t0);
                    }}
                  >
                    {/* Кроп лица */}
                    <div className="merge-inspect-face-wrap">
                      <div className="merge-inspect-face-box" style={{ borderColor: pColor }}>
                        {p.best_face_crop ? (
                          <img src={p.best_face_crop} alt={p.label} loading="lazy" />
                        ) : (
                          <div
                            style={{
                              width: "100%",
                              height: "100%",
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              background: pColor,
                              color: "#fff",
                              fontWeight: 700,
                              fontSize: 14,
                            }}
                          >
                            #{p.person_id}
                          </div>
                        )}
                        <span className="merge-inspect-face-tid">#{p.person_id}</span>
                      </div>
                    </div>

                    {/* Мета-информация строки */}
                    <div className="merge-inspect-row-content">
                      <div className="merge-inspect-row-head">
                        <div className="merge-inspect-row-title">
                          <strong>{p.label}</strong>
                          {p.n_cameras > 1 && (
                            <span
                              style={{
                                fontSize: 9.5,
                                fontWeight: 700,
                                padding: "1px 4px",
                                borderRadius: 3,
                                background: "rgba(13, 110, 253, 0.12)",
                                color: "var(--accent)",
                              }}
                            >
                              {p.n_cameras} кам.
                            </span>
                          )}
                        </div>

                        <span className={badgeClass(p.best_face_crop ? 0.9 : 0.7)}>
                          {p.tracks.length} тр.
                        </span>
                      </div>

                      <div className="merge-inspect-row-meta">
                        <span>🕒 {formatShortTime(p.t0)}–{formatShortTime(p.t1)}</span>
                        <span>{formatDuration(p.duration_sec)}</span>
                      </div>

                      <div style={{ fontSize: 10, color: "#0f6e56", fontWeight: 600 }}>
                        {p.cameras.join(" ➔ ")}
                      </div>

                      <div className="merge-inspect-ids">
                        {p.tracks.map((t) => (
                          <span
                            key={t.uid}
                            style={{
                              borderLeft: `2px solid ${colorForTrackId(t.track_id)}`,
                              paddingLeft: 3,
                            }}
                          >
                            {t.camera.replace("Camera_", "C")} #{t.track_id}
                          </span>
                        ))}
                      </div>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        </section>

        {/* ЦЕНТРАЛЬНАЯ КОЛОНКА: Профессиональный таймлайн + Синхронизированные плееры TrackingPlayer */}
        <section className="merge-inspect-center" aria-label="Таймлайн и видео дня" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {/* Тулбар управления воспроизведением дня */}
          <div className="merge-inspect-center-toolbar" style={{ flexWrap: "wrap", gap: 8 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              {/* Play / Pause */}
              <button
                type="button"
                className="merge-inspect-face-btn"
                style={{
                  background: isPlaying ? "#e53e3e" : "var(--accent, #0d6efd)",
                  color: "#fff",
                  padding: "4px 10px",
                  borderRadius: 4,
                  fontSize: 12,
                  fontWeight: 700,
                }}
                onClick={() => {
                  const next = !isPlaying;
                  setIsPlaying(next);
                  syncPlayersToTime(currentDaySec, next);
                }}
                title="Воспроизведение / Пауза (Пробел)"
              >
                {isPlaying ? "⏸ Пауза" : "▶ Пуск"}
              </button>

              {/* Step Back / Forward */}
              <button
                type="button"
                className="merge-inspect-face-btn"
                onClick={() => seekToDaySec(currentDaySec - 1 / 25)}
                title="Назад на 1 кадр (←)"
              >
                ⏮ -1к
              </button>
              <button
                type="button"
                className="merge-inspect-face-btn"
                onClick={() => seekToDaySec(currentDaySec + 1 / 25)}
                title="Вперед на 1 кадр (→)"
              >
                +1к ⏭
              </button>

              {/* Скорость */}
              <div style={{ display: "flex", alignItems: "center", gap: 2, marginLeft: 4 }}>
                {[0.5, 1.0, 2.0, 4.0].map((spd) => (
                  <button
                    key={spd}
                    type="button"
                    className={`merge-inspect-face-btn ${playbackSpeed === spd ? "on" : ""}`}
                    style={{
                      padding: "2px 5px",
                      fontSize: 10,
                      fontWeight: playbackSpeed === spd ? 700 : 400,
                    }}
                    onClick={() => setPlaybackSpeed(spd)}
                  >
                    {spd}x
                  </button>
                ))}
              </div>
            </div>

            {/* Часы суток и позиция */}
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 12,
                  fontWeight: 700,
                  color: "var(--ink)",
                  background: "rgba(0,0,0,0.05)",
                  padding: "2px 6px",
                  borderRadius: 4,
                }}
              >
                🕒 {formatSecToTimeOfDay(currentDaySec)}
              </span>
              <span style={{ fontSize: 10, color: "var(--muted)" }}>
                из {formatShortTime(timeBounds.maxT)} ({formatDuration(timeBounds.span)})
              </span>
            </div>

            {/* Кнопки перехода по склейкам и сброс выбора */}
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: "auto" }}>
              {selectedPerson && (
                <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <button
                    type="button"
                    className="merge-inspect-face-btn"
                    style={{ background: "rgba(13, 110, 253, 0.1)", color: "var(--accent)", fontWeight: 700 }}
                    onClick={() => {
                      setSelectedPersonId(null);
                      setSelectedEdge(null);
                    }}
                    title="Показать все персоны на таймлайне"
                  >
                    ✕ Все персоны
                  </button>

                  {personEdges.length > 0 && (
                    <>
                      <button
                        type="button"
                        className="merge-inspect-face-btn"
                        onClick={() => jumpToTransition("prev")}
                        title="К предыдущей склейке"
                      >
                        ← Склейка
                      </button>
                      <button
                        type="button"
                        className="merge-inspect-face-btn"
                        onClick={() => jumpToTransition("next")}
                        title="К следующей склейке"
                      >
                        Склейка →
                      </button>
                    </>
                  )}
                </div>
              )}

              {/* Зум таймлайна */}
              <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
                <button
                  type="button"
                  className="merge-inspect-face-btn"
                  onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
                >
                  −
                </button>
                <span style={{ fontSize: 10, fontFamily: "var(--mono)", minWidth: 28, textAlign: "center" }}>
                  {Math.round(zoom * 100)}%
                </span>
                <button
                  type="button"
                  className="merge-inspect-face-btn"
                  onClick={() => setZoom((z) => Math.min(4.0, z + 0.25))}
                >
                  +
                </button>
                <button
                  type="button"
                  className="merge-inspect-face-btn"
                  onClick={() => setZoom(1.0)}
                >
                  100%
                </button>
              </div>
            </div>
          </div>

          {/* Таймлайн с красным курсором воспроизведения */}
          <div
            ref={timelineContainerRef}
            style={{
              overflowX: "auto",
              overflowY: "hidden",
              position: "relative",
              flex: "0 0 auto",
              maxHeight: 220,
              background: "var(--panel, #fff)",
              border: "1px solid var(--line)",
              borderRadius: 6,
              padding: "6px 12px 14px",
            }}
          >
            <div
              style={{ width: `${Math.max(100, 100 * zoom)}%`, position: "relative", cursor: "crosshair" }}
              onClick={handleTimelineClick}
            >
              {/* Шкала времени суток */}
              <div className="merge-inspect-axis" style={{ marginBottom: 8, height: 20 }}>
                {Array.from({ length: 11 }).map((_, i) => {
                  const frac = i / 10;
                  const t = timeBounds.minT + frac * timeBounds.span;
                  return (
                    <div
                      key={i}
                      style={{
                        position: "absolute",
                        left: `${frac * 100}%`,
                        transform: "translateX(-50%)",
                        textAlign: "center",
                        fontSize: 9,
                        fontFamily: "var(--mono)",
                        color: "var(--muted)",
                      }}
                    >
                      <div style={{ width: 1, height: 5, background: "var(--line)", margin: "0 auto 2px" }} />
                      {formatShortTime(t)}
                    </div>
                  );
                })}
              </div>

              {/* Красный курсор воспроизведения (Playhead needle) */}
              {(() => {
                const playheadPct = ((currentDaySec - timeBounds.minT) / timeBounds.span) * 100;
                return (
                  <div
                    style={{
                      position: "absolute",
                      left: `${playheadPct}%`,
                      top: 0,
                      bottom: 0,
                      width: 2,
                      background: "#ff4d4d",
                      zIndex: 20,
                      pointerEvents: "none",
                      boxShadow: "0 0 4px rgba(255, 77, 77, 0.8)",
                    }}
                  >
                    <div
                      style={{
                        position: "absolute",
                        top: 0,
                        left: -4,
                        width: 0,
                        height: 0,
                        borderLeft: "5px solid transparent",
                        borderRight: "5px solid transparent",
                        borderTop: "6px solid #ff4d4d",
                      }}
                    />
                  </div>
                );
              })()}

              {/* Дорожки по камерам дня */}
              <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 4 }}>
                {cameraSessionsList.map((sess) => {
                  const camName = sess.camera;
                  // При выборе персоны показываем только треки входящие в эту персону
                  const personsToRender = selectedPerson
                    ? [selectedPerson]
                    : dayData?.persons || [];

                  return (
                    <div
                      key={sess.key}
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: 2,
                        background: "rgba(0, 0, 0, 0.02)",
                        border: "1px solid var(--line)",
                        borderRadius: 4,
                        padding: "3px 6px",
                      }}
                    >
                      <div style={{ fontSize: 10, fontWeight: 700, color: "var(--ink)", fontFamily: "var(--mono)" }}>
                        {camName}
                      </div>

                      <div style={{ position: "relative", height: 26 }}>
                        {personsToRender.map((person) => {
                          const isPersonSelected = person.person_id === selectedPersonId;
                          const pColor = colorForTrackId(person.person_id);
                          const camTracks = person.tracks.filter((t) => t.camera === camName);

                          return camTracks.map((t) => {
                            const leftPct = ((t.t0 - timeBounds.minT) / timeBounds.span) * 100;
                            const widthPct = Math.max(0.4, ((t.t1 - t.t0) / timeBounds.span) * 100);

                            return (
                              <div
                                key={t.uid}
                                className={`merge-inspect-track ${isPersonSelected ? "on" : ""}`}
                                style={{
                                  position: "absolute",
                                  left: `${leftPct}%`,
                                  width: `${widthPct}%`,
                                  top: 1,
                                  height: 22,
                                  borderRadius: 3,
                                  background: pColor,
                                  opacity: 1,
                                  color: "#fff",
                                  display: "flex",
                                  alignItems: "center",
                                  padding: "0 4px",
                                  fontSize: 9.5,
                                  fontWeight: 700,
                                  cursor: "pointer",
                                  border: isPersonSelected ? "2px solid #000" : "1px solid rgba(255,255,255,0.4)",
                                  boxShadow: isPersonSelected ? "0 0 6px rgba(0,0,0,0.4)" : "none",
                                  overflow: "hidden",
                                  whiteSpace: "nowrap",
                                  zIndex: isPersonSelected ? 5 : 1,
                                }}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setSelectedPersonId(person.person_id);
                                  setSelectedEdge(null);
                                  seekToDaySec(t.t0);
                                }}
                                title={`${person.label} (${camName} #${t.track_id}): ${formatShortTime(t.t0)} - ${formatShortTime(t.t1)} (${formatDuration(t.t1 - t.t0)})`}
                              >
                                P#{person.person_id} · T{t.track_id} ({formatDuration(t.t1 - t.t0)})
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

          {/* Синхронизированный мультикамерный видеовывод на базе существующего TrackingPlayer */}
          <div
            style={{
              flex: "1 1 auto",
              minHeight: 280,
              display: "grid",
              gridTemplateColumns: cameraSessionsList.length > 1 ? "repeat(auto-fit, minmax(280px, 1fr))" : "1fr",
              gap: 8,
              background: "var(--panel, #fff)",
              border: "1px solid var(--line)",
              borderRadius: 6,
              padding: 8,
            }}
          >
            {cameraSessionsList.map((sess) => {
              const t0_abs = sess.t0_abs ?? timeBounds.minT;
              const duration = sess.duration_sec ?? 300;
              const isActiveNow = currentDaySec >= t0_abs && currentDaySec <= t0_abs + duration;
              const localSec = Math.max(0, currentDaySec - t0_abs);

              // Треки выбранной персоны для этой камеры
              const focusIds = focusTrackIdsByCamera[sess.camera] ?? [];
              const isPersonInCamera = focusIds.length > 0 && isActiveNow;

              return (
                <div
                  key={sess.key}
                  className={`day-camera-card ${isPersonInCamera ? "has-person" : ""}`}
                >
                  {/* Шапка камеры */}
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      padding: "4px 8px",
                      background: isPersonInCamera ? "rgba(13, 110, 253, 0.08)" : "rgba(0,0,0,0.03)",
                      borderBottom: "1px solid var(--line)",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ fontSize: 11, fontWeight: 700, color: "var(--ink)", fontFamily: "var(--mono)" }}>
                        {sess.camera}
                      </span>
                      {isActiveNow ? (
                        <span style={{ fontSize: 9.5, color: "#2e7d32", fontWeight: 700 }}>🟢 Активна</span>
                      ) : (
                        <span style={{ fontSize: 9.5, color: "var(--muted)" }}>⚪ Вне диапазона</span>
                      )}
                    </div>

                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      {selectedPerson && isPersonInCamera && (
                        <span
                          style={{
                            fontSize: 9.5,
                            fontWeight: 700,
                            background: "var(--accent)",
                            color: "#fff",
                            padding: "1px 5px",
                            borderRadius: 3,
                          }}
                        >
                          👁 {selectedPerson.label} (T#{focusIds.join(", T#")})
                        </span>
                      )}
                      <span style={{ fontSize: 10, fontFamily: "var(--mono)", color: "var(--muted)" }}>
                        {formatDuration(localSec)}
                      </span>
                    </div>
                  </div>

                  {/* Тело плеера TrackingPlayer */}
                  <div style={{ flex: "1 1 auto", minHeight: 180, position: "relative", background: "var(--surface, #fff)" }}>
                    <TrackingPlayer
                      ref={(r) => {
                        playerRefs.current[sess.key] = r;
                      }}
                      videoRef={{ current: videoRefs.current[sess.key] ?? null }}
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

        {/* ПРАВАЯ КОЛОНКА: Детальный инспектор склейки и персоны */}
        <section className="merge-inspect-why" aria-label="Детали персоны и склейки">
          {selectedPerson ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {/* Шапка персоны */}
              <div
                style={{
                  display: "flex",
                  gap: 10,
                  background: "var(--surface, #fff)",
                  border: "1px solid var(--line)",
                  borderRadius: 6,
                  padding: 8,
                  alignItems: "center",
                }}
              >
                <div
                  className="merge-inspect-face-box"
                  style={{
                    width: 56,
                    height: 64,
                    borderColor: colorForTrackId(selectedPerson.person_id),
                    flexShrink: 0,
                  }}
                >
                  {selectedPerson.best_face_crop ? (
                    <img src={selectedPerson.best_face_crop} alt={selectedPerson.label} />
                  ) : (
                    <div
                      style={{
                        width: "100%",
                        height: "100%",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        background: colorForTrackId(selectedPerson.person_id),
                        color: "#fff",
                        fontWeight: 700,
                        fontSize: 15,
                      }}
                    >
                      #{selectedPerson.person_id}
                    </div>
                  )}
                  <span className="merge-inspect-face-tid">#{selectedPerson.person_id}</span>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: 2, flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <strong style={{ fontSize: 12.5, color: "var(--ink)" }}>{selectedPerson.label}</strong>
                    <span className={badgeClass(0.9)}>{selectedPerson.n_tracks} треков</span>
                  </div>

                  <div style={{ fontSize: 10.5, color: "var(--muted)", fontFamily: "var(--mono)" }}>
                    {formatShortTime(selectedPerson.t0)}–{formatShortTime(selectedPerson.t1)} ({formatDuration(selectedPerson.duration_sec)})
                  </div>

                  <div style={{ fontSize: 10.5, color: "#0f6e56", fontWeight: 600 }}>
                    {selectedPerson.cameras.join(" ➔ ")}
                  </div>
                </div>
              </div>

              {/* Галерея кропов лица персоны */}
              {selectedPerson.face_crops.length > 0 && (
                <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                  <div style={{ fontSize: 9.5, fontWeight: 700, color: "var(--muted)", textTransform: "uppercase" }}>
                    Кропы лиц InsightFace:
                  </div>
                  <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                    {selectedPerson.face_crops.map((url, i) => (
                      <a key={i} href={url} target="_blank" rel="noreferrer" className="merge-inspect-face-box" style={{ width: 38, height: 44 }}>
                        <img src={url} alt="face" />
                      </a>
                    ))}
                  </div>
                </div>
              )}

              {/* Табы правой панели */}
              <div className="merge-inspect-why-tabs">
                <button
                  type="button"
                  className={rightTab === "edges" ? "on" : ""}
                  onClick={() => setRightTab("edges")}
                >
                  Склейки ({personEdges.length})
                </button>
                <button
                  type="button"
                  className={rightTab === "tracks" ? "on" : ""}
                  onClick={() => setRightTab("tracks")}
                >
                  Все треки ({selectedPerson.tracks.length})
                </button>
                {selectedEdge && (
                  <button
                    type="button"
                    className={rightTab === "inspector" ? "on" : ""}
                    onClick={() => setRightTab("inspector")}
                  >
                    Инспектор пары
                  </button>
                )}
              </div>

              {/* Содержимое вкладки: Склейки */}
              {rightTab === "edges" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {personEdges.length === 0 ? (
                    <div className="merge-inspect-empty">
                      Одиночный трек — межкамерных склеек не производилось.
                    </div>
                  ) : (
                    personEdges.map((edge, idx) => {
                      const isEdgeSelected = selectedEdge?.from === edge.from && selectedEdge?.to === edge.to;
                      return (
                        <div
                          key={idx}
                          className={`merge-why-card ${isEdgeSelected ? "on" : ""}`}
                          style={{
                            border: isEdgeSelected ? "1.5px solid var(--accent)" : "1px solid var(--line)",
                            background: isEdgeSelected ? "rgba(13, 110, 253, 0.04)" : "var(--surface, #fff)",
                            borderRadius: 6,
                            padding: 6,
                            cursor: "pointer",
                            display: "flex",
                            flexDirection: "column",
                            gap: 4,
                          }}
                          onClick={() => {
                            setSelectedEdge(edge);
                            setRightTab("inspector");
                            const trFrom = selectedPerson.tracks.find((t) => t.uid === edge.from);
                            if (trFrom) seekToDaySec(trFrom.t1);
                          }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <div style={{ fontSize: 10.5, fontWeight: 700, color: "var(--ink)", fontFamily: "var(--mono)" }}>
                              {edge.from_camera} #{edge.from_track} ➔ {edge.to_camera} #{edge.to_track}
                            </div>
                            {passBadge(edge.pass)}
                          </div>

                          <div style={{ display: "flex", flexWrap: "wrap", gap: 4, fontSize: 9.5 }}>
                            <span className={badgeClass(edge.score)}>Скор: {formatScore(edge.score)}</span>
                            {edge.face != null && <span className="pass-pill">Лицо: {formatScore(edge.face)}</span>}
                            {edge.reid != null && <span className="pass-pill">ReID: {formatScore(edge.reid)}</span>}
                            <span className="pass-pill">Δd: {edge.dist_m.toFixed(1)}м</span>
                            <span className="pass-pill">v: {edge.speed_mps.toFixed(1)}м/с</span>
                            <span className="pass-pill">Δt: {edge.gap_sec.toFixed(1)}с</span>
                          </div>

                          <div style={{ fontSize: 9.5, color: "var(--muted)", fontFamily: "var(--mono)" }}>
                            {edge.reason}
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              )}

              {/* Содержимое вкладки: Инспектор пары */}
              {rightTab === "inspector" && selectedEdge && (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <strong style={{ fontSize: 11.5, color: "var(--ink)" }}>
                      Инспектор: {selectedEdge.from_camera} #{selectedEdge.from_track} ➔ {selectedEdge.to_camera} #{selectedEdge.to_track}
                    </strong>
                    {passBadge(selectedEdge.pass)}
                  </div>

                  {/* 2-колоночное попарное сравнение ДО и ПОСЛЕ */}
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                    {/* ДО */}
                    <div
                      style={{ background: "var(--surface, #fff)", border: "1px solid var(--line)", borderRadius: 6, padding: 6, cursor: "pointer" }}
                      onClick={() => {
                        const tr = selectedPerson.tracks.find((t) => t.uid === selectedEdge.from);
                        if (tr) seekToDaySec(tr.t1);
                      }}
                      title="Кликните для перехода к точке выхода"
                    >
                      <div style={{ fontSize: 10.5, fontWeight: 700, color: "var(--ink)", marginBottom: 3 }}>
                        [ДО] {selectedEdge.from_camera} #{selectedEdge.from_track}
                      </div>
                      {(() => {
                        const tFrom = selectedPerson.tracks.find((t) => t.uid === selectedEdge.from);
                        return (
                          <div style={{ display: "flex", flexDirection: "column", gap: 3, fontSize: 9.5, fontFamily: "var(--mono)" }}>
                            {tFrom?.crops[0] && (
                              <img src={tFrom.crops[0]} alt="crop" style={{ width: 44, height: 50, objectFit: "cover", borderRadius: 4, border: "1px solid var(--line)" }} />
                            )}
                            <div>Конец: {formatShortTime(tFrom?.t1 ?? 0)}</div>
                            <div>Точка: {tFrom?.p1 ? `(${tFrom.p1[0].toFixed(0)}, ${tFrom.p1[1].toFixed(0)})` : "—"}</div>
                          </div>
                        );
                      })()}
                    </div>

                    {/* ПОСЛЕ */}
                    <div
                      style={{ background: "var(--surface, #fff)", border: "1px solid var(--line)", borderRadius: 6, padding: 6, cursor: "pointer" }}
                      onClick={() => {
                        const tr = selectedPerson.tracks.find((t) => t.uid === selectedEdge.to);
                        if (tr) seekToDaySec(tr.t0);
                      }}
                      title="Кликните для перехода к точке входа"
                    >
                      <div style={{ fontSize: 10.5, fontWeight: 700, color: "var(--ink)", marginBottom: 3 }}>
                        [ПОСЛЕ] {selectedEdge.to_camera} #{selectedEdge.to_track}
                      </div>
                      {(() => {
                        const tTo = selectedPerson.tracks.find((t) => t.uid === selectedEdge.to);
                        return (
                          <div style={{ display: "flex", flexDirection: "column", gap: 3, fontSize: 9.5, fontFamily: "var(--mono)" }}>
                            {tTo?.crops[0] && (
                              <img src={tTo.crops[0]} alt="crop" style={{ width: 44, height: 50, objectFit: "cover", borderRadius: 4, border: "1px solid var(--line)" }} />
                            )}
                            <div>Старт: {formatShortTime(tTo?.t0 ?? 0)}</div>
                            <div>Точка: {tTo?.p0 ? `(${tTo.p0[0].toFixed(0)}, ${tTo.p0[1].toFixed(0)})` : "—"}</div>
                          </div>
                        );
                      })()}
                    </div>
                  </div>

                  {/* Метрики склейки */}
                  <table className="merge-why-table" style={{ width: "100%", fontSize: 9.5, borderCollapse: "collapse" }}>
                    <tbody>
                      <tr>
                        <td><strong>Комбинированный скор:</strong></td>
                        <td><span className={badgeClass(selectedEdge.score)}>{formatScore(selectedEdge.score)}</span></td>
                      </tr>
                      <tr>
                        <td>Лицо сходство (cos):</td>
                        <td><strong>{formatScore(selectedEdge.face)}</strong></td>
                      </tr>
                      <tr>
                        <td>ReID тела (cos):</td>
                        <td><strong>{formatScore(selectedEdge.reid)}</strong></td>
                      </tr>
                      <tr>
                        <td>Расстояние между точками:</td>
                        <td><strong>{selectedEdge.dist_m.toFixed(2)} м</strong></td>
                      </tr>
                      <tr>
                        <td>Скорость перемещения:</td>
                        <td><strong>{selectedEdge.speed_mps.toFixed(2)} м/с</strong></td>
                      </tr>
                      <tr>
                        <td>Временной зазор Δt:</td>
                        <td><strong>{selectedEdge.gap_sec.toFixed(2)} с</strong></td>
                      </tr>
                      <tr>
                        <td>Причина солвера:</td>
                        <td style={{ fontSize: 9, color: "var(--muted)" }}>{selectedEdge.reason}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}

              {/* Содержимое вкладки: Все треки */}
              {rightTab === "tracks" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <table className="merge-why-table" style={{ width: "100%", fontSize: 9.5, borderCollapse: "collapse" }}>
                    <thead>
                      <tr style={{ background: "rgba(0,0,0,0.04)", textAlign: "left" }}>
                        <th style={{ padding: 3 }}>Камера</th>
                        <th style={{ padding: 3 }}>ID</th>
                        <th style={{ padding: 3 }}>Интервал</th>
                        <th style={{ padding: 3 }}>Кадров</th>
                        <th style={{ padding: 3 }}>Лицо</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedPerson.tracks.map((t) => (
                        <tr
                          key={t.uid}
                          style={{ borderBottom: "1px solid var(--line)", cursor: "pointer" }}
                          onClick={() => seekToDaySec(t.t0)}
                          title="Кликните для перехода к началу трека"
                        >
                          <td style={{ padding: 3 }}><strong>{t.camera}</strong></td>
                          <td style={{ padding: 3 }}>#{t.track_id}</td>
                          <td style={{ padding: 3 }}>{formatShortTime(t.t0)}–{formatShortTime(t.t1)}</td>
                          <td style={{ padding: 3 }}>{t.n_frames}</td>
                          <td style={{ padding: 3 }}>
                            {t.has_face ? <span style={{ color: "#0f6e56", fontWeight: 700 }}>✓ {formatScore(t.best_face_score)}</span> : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ) : (
            <div className="merge-inspect-empty">
              Выберите персону в левом списке или на таймлайне.
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

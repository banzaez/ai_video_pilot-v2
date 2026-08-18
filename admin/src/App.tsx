import { useEffect, useMemo, useRef, useState } from "react";
import { FloatingVideoWindow } from "./components/FloatingVideoWindow";
import { MapCalibratePanel } from "./components/MapCalibratePanel";
import { MapFloorView } from "./components/MapFloorView";
import { MergeInspectPanel } from "./components/MergeInspectPanel";
import { DayAnalysisPanel } from "./components/DayAnalysisPanel";
import { PipelineJobsPanel } from "./components/PipelineJobsPanel";
import { TrackingPlayer, type TrackingPlayerHandle } from "./components/TrackingPlayer";
import { TrackingSidebar } from "./components/TrackingSidebar";
import type { HomographyDoc } from "./homography";
import type { FeetDoc, TrackingData } from "./types";
import {
  clampFloatVideo,
  FLOAT_PLAYER_CONTROLS_H,
  VIDEO_ASPECT,
  cameraKeyFromVideo,
  defaultFloatMap,
  fetchCounters,
  fetchFeetJson,
  fetchHomography,
  fetchMapsConfig,
  fetchMediaSessions,
  fetchMediaMeta,
  fetchTrackingJson,
  feetJsonUrl,
  formatBytes,
  formatDuration,
  formatTs,
  loadPrefs,
  savePrefs,
  type FloatVideoGeom,
  type CropShot,
  type FaceShot,
  type SimilarHit,
  type MergeTimeline,
  type MergeTimelinePair,
  type VideoInfo,
  type ViewerPrefs,
  type PipelineStaleReport,
} from "./utils";
import { sessionDays, sessionsForDay, formatSessionDay, cameraLabel, sessionLabel, type MediaSession, type SessionPart } from "./session";
import "./App.css";
import type { MapCameraMark, MapCalibPoint } from "./components/MapFloorView";
import { normalizeBodyCalib } from "./feet";
import type { CountersDoc } from "./counters";
import { emptyCountersDoc, normalizeCountersDoc } from "./counters";
import { normalizePlacement } from "./homography";

type AppTab = "tracks" | "merge" | "day" | "map" | "pipeline";

export default function App() {
  const initial = useMemo(() => loadPrefs(), []);
  const [sessions, setSessions] = useState<MediaSession[]>([]);
  const [librarySession, setLibrarySession] = useState<string>(
    initial.librarySession ?? initial.libraryVideo ?? "",
  );
  const [sessionParts, setSessionParts] = useState<SessionPart[]>([]);
  const days = useMemo(() => sessionDays(sessions), [sessions]);
  const selectedDay = useMemo(() => {
    const cur = sessions.find((s) => s.key === librarySession);
    return cur?.day ?? days[0] ?? "";
  }, [sessions, librarySession, days]);
  const camerasForDay = useMemo(
    () => (selectedDay ? sessionsForDay(sessions, selectedDay) : []),
    [sessions, selectedDay],
  );
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoName, setVideoName] = useState<string>("");
  const [tracking, setTracking] = useState<TrackingData | null>(null);
  const [feetDoc, setFeetDoc] = useState<FeetDoc | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showVideo, setShowVideo] = useState(initial.showVideo ?? true);
  const [showMap, setShowMap] = useState(initial.showMap);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [cropUrls, setCropUrls] = useState<Record<string, CropShot[]>>({});
  const [faceUrls, setFaceUrls] = useState<Record<string, FaceShot[]>>({});
  const [faceUrlsByModel, setFaceUrlsByModel] = useState<Record<string, Record<string, FaceShot[]>>>({});
  const [faceModels, setFaceModels] = useState<string[]>([]);
  const [cameraLink, setCameraLink] = useState<{
    face_models?: string[];
    edges?: MergeTimelinePair[];
    candidate_edges?: MergeTimelinePair[];
  } | null>(null);
  const [similarByTrack, setSimilarByTrack] = useState<Record<string, SimilarHit[]>>({});
  const [mergeByTrack, setMergeByTrack] = useState<Record<string, SimilarHit[]>>({});
  const [mergeTimeline, setMergeTimeline] = useState<MergeTimeline | null>(null);
  const [pipeline, setPipeline] = useState<PipelineStaleReport | null>(null);
  const [floatVideo, setFloatVideo] = useState<FloatVideoGeom>(() => clampFloatVideo(initial.floatVideo));
  const [floatMap, setFloatMap] = useState<FloatVideoGeom>(() =>
    clampFloatVideo(initial.floatMap ?? defaultFloatMap(), 4800 / 3200),
  );
  const [tab, setTab] = useState<AppTab>("tracks");
  const [mapDirty, setMapDirty] = useState(false);
  const [mergeFocusIds, setMergeFocusIds] = useState<number[] | null>(null);
  const [homography, setHomography] = useState<HomographyDoc | null>(null);
  const [floorplanUrl, setFloorplanUrl] = useState("grid");
  const [mapCameras, setMapCameras] = useState<MapCameraMark[]>([]);
  const [allCalibPoints, setAllCalibPoints] = useState<MapCalibPoint[]>([]);
  const [counters, setCounters] = useState<CountersDoc | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const playerRef = useRef<TrackingPlayerHandle>(null);
  const floatVideoRef = useRef(floatVideo);
  floatVideoRef.current = floatVideo;
  const floatMapRef = useRef(floatMap);
  floatMapRef.current = floatMap;
  const showVideoRef = useRef(showVideo);
  showVideoRef.current = showVideo;
  const showMapRef = useRef(showMap);
  showMapRef.current = showMap;

  const highlightIds = useMemo(() => {
    if (selectedTrackId == null) return [];
    const extra = [
      ...(similarByTrack[String(selectedTrackId)] ?? []),
      ...(mergeByTrack[String(selectedTrackId)] ?? []),
    ].map((s) => s.track_id);
    return [selectedTrackId, ...extra];
  }, [selectedTrackId, similarByTrack, mergeByTrack]);

  const playerFocusTrackIds = useMemo(() => {
    if (tab === "merge") return mergeFocusIds;
    return null;
  }, [tab, mergeFocusIds]);

  const overlayHighlightIds = useMemo(() => {
    if (tab === "merge") {
      const focus = mergeFocusIds;
      if (focus?.length) {
        if (selectedTrackId != null && focus.includes(selectedTrackId)) {
          const extra = [
            ...(similarByTrack[String(selectedTrackId)] ?? []),
            ...(mergeByTrack[String(selectedTrackId)] ?? []),
          ]
            .map((s) => s.track_id)
            .filter((id) => focus.includes(id));
          return [selectedTrackId, ...extra];
        }
        return focus;
      }
      if (selectedTrackId != null) return highlightIds;
      return [];
    }
    return highlightIds;
  }, [tab, mergeFocusIds, highlightIds, selectedTrackId, similarByTrack, mergeByTrack]);

  useEffect(() => {
    if (tab !== "merge") setMergeFocusIds(null);
  }, [tab]);

  const groupByTrack = useMemo(() => {
    const map: Record<number, number> = {};
    if (mergeTimeline) {
      for (const t of mergeTimeline.tracks) {
        if (typeof t.group_id === "number") {
          map[t.track_id] = t.group_id;
          if (typeof t.global_id === "number") map[t.global_id] = t.group_id;
        }
      }
      return map;
    }
    for (const [tid, hits] of Object.entries(mergeByTrack)) {
      const g = hits[0]?.group_id;
      if (typeof g !== "number") continue;
      const id = Number(tid);
      if (Number.isFinite(id)) map[id] = g;
      for (const h of hits) map[h.track_id] = g;
    }
    return map;
  }, [mergeByTrack, mergeTimeline]);

  const cropUrlsByGlobal = useMemo(() => {
    if (!mergeTimeline?.tracks.length) return cropUrls;
    const buckets = new Map<number, CropShot[]>();
    for (const t of mergeTimeline.tracks) {
      const gid = t.global_id ?? t.track_id;
      const shots = cropUrls[String(t.track_id)] ?? [];
      const list = buckets.get(gid) ?? [];
      list.push(...shots);
      buckets.set(gid, list);
    }
    const out: Record<string, CropShot[]> = {};
    for (const [gid, shots] of buckets) {
      out[String(gid)] = [...shots].sort(
        (a, b) => (a.frame ?? 0) - (b.frame ?? 0) || a.rank - b.rank,
      );
    }
    return out;
  }, [cropUrls, mergeTimeline]);

  const currentSec = useMemo(() => {
    const fps = tracking?.fps || videoInfo?.fps || 25;
    return fps > 0 ? currentFrame / fps : 0;
  }, [currentFrame, tracking?.fps, videoInfo?.fps]);

  const videoAspect = useMemo(() => {
    const w = videoInfo?.width || tracking?.width || 0;
    const h = videoInfo?.height || tracking?.height || 0;
    return w > 0 && h > 0 ? w / h : VIDEO_ASPECT;
  }, [videoInfo?.width, videoInfo?.height, tracking?.width, tracking?.height]);

  const mapAspect = useMemo(() => {
    const sz = homography?.map_size;
    if (sz && sz[0] > 0 && sz[1] > 0) return sz[0] / sz[1];
    return 4800 / 3200;
  }, [homography?.map_size]);

  const cameraKey = useMemo(
    () => cameraKeyFromVideo(videoName, videoInfo?.parsed?.camera_index),
    [videoName, videoInfo?.parsed?.camera_index],
  );

  const imageSize = useMemo((): [number, number] | null => {
    const w = videoInfo?.width || tracking?.width;
    const h = videoInfo?.height || tracking?.height;
    if (w && h) return [w, h];
    return null;
  }, [videoInfo?.width, videoInfo?.height, tracking?.width, tracking?.height]);

  useEffect(() => {
    setSelectedTrackId(null);
  }, [tracking]);

  function persistPrefs(
    videoGeom = floatVideoRef.current,
    mapGeom = floatMapRef.current,
    videoVisible = showVideoRef.current,
    mapVisible = showMapRef.current,
  ) {
    const prefs: ViewerPrefs = {
      libraryVideo: librarySession || null,
      librarySession: librarySession || null,
      floatVideo: videoGeom,
      floatVideoMinimized: false,
      showVideo: videoVisible,
      showMap: mapVisible,
      floatMap: mapGeom,
      floatMapMinimized: false,
    };
    savePrefs(prefs);
  }

  useEffect(() => {
    persistPrefs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [librarySession, showVideo, showMap]);

  function commitFloatVideo(geom: FloatVideoGeom) {
    const next = clampFloatVideo(geom, videoAspect, 280, FLOAT_PLAYER_CONTROLS_H);
    floatVideoRef.current = next;
    setFloatVideo(next);
    persistPrefs(next);
  }

  useEffect(() => {
    const next = clampFloatVideo(floatVideoRef.current, videoAspect, 280, FLOAT_PLAYER_CONTROLS_H);
    floatVideoRef.current = next;
    setFloatVideo(next);
  }, [videoAspect]);

  useEffect(() => {
    const next = clampFloatVideo(floatMapRef.current, mapAspect);
    floatMapRef.current = next;
    setFloatMap(next);
  }, [mapAspect]);

  function commitFloatMap(geom: FloatVideoGeom) {
    const next = clampFloatVideo(geom, mapAspect);
    floatMapRef.current = next;
    setFloatMap(next);
    persistPrefs();
  }

  function toggleShowVideo(next: boolean) {
    showVideoRef.current = next;
    setShowVideo(next);
    persistPrefs();
  }

  function toggleShowMap(next: boolean) {
    showMapRef.current = next;
    setShowMap(next);
    persistPrefs();
  }

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const doc = await fetchHomography(cameraKey);
        if (cancelled) return;
        setHomography({
          ...doc,
          placement: normalizePlacement(doc.placement),
          body_calib: normalizeBodyCalib(doc.body_calib) ?? undefined,
        });
        if (doc.floorplan === "grid" || !doc.floorplan || /\.svg$/i.test(doc.floorplan)) {
          setFloorplanUrl("grid");
        } else if (doc.floorplan) {
          setFloorplanUrl(`/maps/${encodeURIComponent(doc.floorplan)}`);
        }
      } catch {
        if (!cancelled) setHomography(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [cameraKey]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const raw = await fetchCounters();
        if (cancelled) return;
        setCounters(normalizeCountersDoc(raw));
      } catch {
        if (!cancelled) setCounters(emptyCountersDoc());
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [cameraKey]);

  function requestTab(next: AppTab) {
    if (tab === "map" && next !== "map" && mapDirty) {
      const ok = window.confirm("Есть несохранённые изменения на Карте.\nУйти без сохранения?");
      if (!ok) return;
    }
    setTab(next);
  }
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const cfg = await fetchMapsConfig();
        if (cancelled) return;
        const marks: MapCameraMark[] = [];
        const points: MapCalibPoint[] = [];
        for (const cam of cfg.cameras) {
          const pl = normalizePlacement(cam.placement);
          if (pl) marks.push({ key: cam.key, placement: pl });
          for (const p of cam.map_points ?? []) {
            points.push({ key: cam.key, index: p.index, map: p.map });
          }
        }
        setMapCameras(marks);
        setAllCalibPoints(points);
      } catch {
        if (!cancelled) {
          setMapCameras([]);
          setAllCalibPoints([]);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [cameraKey, homography?.updated_at, tab]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const items = await fetchMediaSessions();
        if (cancelled) return;
        setSessions(items);

        const remembered = initial.librarySession ?? initial.libraryVideo;
        const pick =
          (remembered && items.find((i) => i.key === remembered)) ||
          items.find((i) => i.hasJson) ||
          items[0];

        if (pick) {
          await selectSession(pick, false);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Ошибка загрузки библиотеки");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function selectSession(session: MediaSession, clearError = true) {
    if (clearError) setError(null);
    setSelectedTrackId(null);
    setLibrarySession(session.key);
    setSessionParts(session.parts);
    setVideoUrl(session.parts[0]?.videoUrl ?? null);
    setVideoName(sessionLabel(session));
    try {
      const meta = await fetchMediaMeta(session.key, { session: true });
      setVideoInfo(meta.info);
      setCropUrls(meta.crops);
      setFaceUrls(meta.faces ?? {});
      setFaceUrlsByModel(meta.facesByModel ?? {});
      setFaceModels(meta.faceModels ?? []);
      setCameraLink(meta.cameraLink ?? null);
      setSimilarByTrack(meta.similar);
      setMergeByTrack(meta.merge);
      setMergeTimeline(meta.mergeTimeline);
      setPipeline(meta.pipeline);
    } catch {
      setVideoInfo(null);
      setCropUrls({});
      setFaceUrls({});
      setFaceUrlsByModel({});
      setFaceModels([]);
      setCameraLink(null);
      setSimilarByTrack({});
      setMergeByTrack({});
      setMergeTimeline(null);
      setPipeline(null);
    }

    if (session.hasJson) {
      try {
        const data = await fetchTrackingJson(session.jsonUrl);
        setTracking(data);
        const feet = await fetchFeetJson(feetJsonUrl(session.jsonUrl));
        setFeetDoc(feet);
      } catch (e) {
        setTracking(null);
        setFeetDoc(null);
        setError(e instanceof Error ? e.message : "Не удалось загрузить JSON");
      }
    } else {
      setTracking(null);
      setFeetDoc(null);
      setError(`Для session «${session.key}» нет результатов в data/results`);
    }
  }

  async function onLibraryChange(key: string) {
    const item = sessions.find((i) => i.key === key);
    if (!item) return;
    await selectSession(item);
  }

  async function onDayChange(day: string) {
    const cams = sessionsForDay(sessions, day);
    if (!cams.length) return;
    const currentCam = sessions.find((s) => s.key === librarySession)?.camera_index;
    const pick =
      (currentCam != null ? cams.find((s) => s.camera_index === currentCam) : undefined) ?? cams[0]!;
    await selectSession(pick);
  }

  function handleSeekToSec(sec: number) {
    // Через плеер: partAtTime + смена куска + seek после loadeddata
    if (playerRef.current) {
      playerRef.current.seekToGlobal(sec, true);
      return;
    }
    if (videoRef.current) {
      videoRef.current.currentTime = sec;
      videoRef.current.play().catch(() => {});
    }
  }

  return (
    <>
      <div className="app">
        <header className="topbar">
          <div className="topbar-brand">
            <span className="brand">AI Video Pilot</span>
          </div>

          <div className="topbar-session">
            <label className="field field-inline field-day">
              <span>День</span>
              <select
                value={selectedDay}
                disabled={!days.length}
                onChange={(e) => void onDayChange(e.target.value)}
              >
                {!days.length ? (
                  <option value="">пусто</option>
                ) : (
                  days.map((day) => (
                    <option key={day} value={day}>
                      {formatSessionDay(day)}
                    </option>
                  ))
                )}
              </select>
            </label>
            <label className="field field-inline field-camera">
              <span>Камера</span>
              <select
                value={librarySession}
                disabled={!camerasForDay.length}
                onChange={(e) => void onLibraryChange(e.target.value)}
              >
                {!camerasForDay.length ? (
                  <option value="">—</option>
                ) : (
                  camerasForDay.map((item) => (
                    <option key={item.key} value={item.key}>
                      {cameraLabel(item)}
                    </option>
                  ))
                )}
              </select>
            </label>
          </div>

          <div className="topbar-toggles">
          <label className="toggle">
            <input
              type="checkbox"
              checked={showVideo}
              onChange={(e) => toggleShowVideo(e.target.checked)}
            />
            Видео
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={showMap}
              onChange={(e) => toggleShowMap(e.target.checked)}
            />
            Карта
          </label>
        </div>
        </header>

        {error && <p className="error">{error}</p>}

        {pipeline?.stale?.length ? (
          <div className="stale-banner">
            <p>
              Устарели: <strong>{pipeline.stale.join(" → ")}</strong>
            </p>
            {pipeline.cli && (
              <code
                className="stale-cli"
                title="Скопируйте в терминал"
                onClick={() => navigator.clipboard?.writeText(pipeline.cli!).catch(() => {})}
              >
                {pipeline.cli}
              </code>
            )}
          </div>
        ) : null}

        {(videoInfo || tracking) && (
          <section className="meta">
            {videoInfo && (
              <>
                <a className="meta-link" href={videoInfo.url} target="_blank" rel="noreferrer">
                  {videoInfo.name}
                </a>
                {videoInfo.parsed?.ok && videoInfo.parsed.camera && (
                  <span>{videoInfo.parsed.camera}</span>
                )}
                {videoInfo.parsed?.ip && (
                  <span>
                    {videoInfo.parsed.ip}
                    {videoInfo.parsed.peer_ip ? ` → ${videoInfo.parsed.peer_ip}` : ""}
                  </span>
                )}
                {videoInfo.parsed?.started_at && (
                  <span>
                    {formatTs(videoInfo.parsed.started_at)}
                    {videoInfo.parsed.ended_at ? ` – ${formatTs(videoInfo.parsed.ended_at)}` : ""}
                  </span>
                )}
                {videoInfo.parsed?.recording_id && <span>id {videoInfo.parsed.recording_id}</span>}
                <span>
                  {videoInfo.width}×{videoInfo.height}
                </span>
                <span>{videoInfo.fps} fps</span>
                <span>{videoInfo.frame_count} кадров</span>
                <span>{formatDuration(videoInfo.duration_sec)}</span>
                <span>{formatBytes(videoInfo.size_bytes)}</span>
                {videoInfo.codec && <span>{videoInfo.codec}</span>}
              </>
            )}
            {!videoInfo && tracking && (
              <>
                <span>
                  {tracking.width}×{tracking.height}
                </span>
                <span>{tracking.fps} fps</span>
                <span>{tracking.frame_count} frames</span>
              </>
            )}
            {tracking?.detect_every_n != null && tracking.detect_every_n > 1 && (
              <span>every {tracking.detect_every_n}</span>
            )}
            {selectedTrackId != null && <span className="tag accent">sel #{selectedTrackId}</span>}
          </section>
        )}

        <nav className="app-tabs" aria-label="Разделы">
          <button type="button" className={tab === "tracks" ? "on" : ""} onClick={() => requestTab("tracks")}>
            Треки
          </button>
          <button type="button" className={tab === "merge" ? "on" : ""} onClick={() => requestTab("merge")}>
            Склейки
            {mergeTimeline?.groups.length ? <em>{mergeTimeline.groups.length}</em> : null}
          </button>
          <button type="button" className={tab === "day" ? "on" : ""} onClick={() => requestTab("day")}>
            ДЕНЬ
          </button>
          <button type="button" className={tab === "map" ? "on" : ""} onClick={() => requestTab("map")}>
            Карта
            {homography?.H ? <em>H</em> : null}
          </button>
          <button type="button" className={tab === "pipeline" ? "on" : ""} onClick={() => requestTab("pipeline")}>
            Пайплайн
            {pipeline?.stale?.length ? <em>!</em> : null}
          </button>
        </nav>

        <main className="main-content">
          {tab === "tracks" ? (
            <div className="tracks-under-video">
              <TrackingSidebar
                tracking={tracking}
                currentFrame={currentFrame}
                selectedTrackId={selectedTrackId}
                onSelectTrackId={setSelectedTrackId}
                onSeekToSec={handleSeekToSec}
                cropUrls={cropUrlsByGlobal}
                similarByTrack={similarByTrack}
                mergeByTrack={mergeByTrack}
              />
            </div>
          ) : tab === "merge" ? (
            <MergeInspectPanel
              activeVideo={librarySession}
              mergeTimeline={mergeTimeline}
              cropUrls={cropUrls}
              faceUrls={faceUrls}
              faceUrlsByModel={faceUrlsByModel}
              faceModels={faceModels}
              cameraLink={cameraLink}
              similarByTrack={similarByTrack}
              pipeline={pipeline}
              tracking={tracking}
              currentSec={currentSec}
              currentFrame={currentFrame}
              selectedTrackId={selectedTrackId}
              onSelectTrackId={setSelectedTrackId}
              onSeekToSec={handleSeekToSec}
              onFocusTracks={setMergeFocusIds}
            />
          ) : tab === "day" ? (
            <DayAnalysisPanel selectedDay={selectedDay} />
          ) : tab === "pipeline" ? (
            <PipelineJobsPanel
              librarySession={librarySession}
              onPipelineUpdate={setPipeline}
            />
          ) : (
            <MapCalibratePanel
              videoName={videoName || librarySession}
              videoUrl={videoUrl}
              cameraIndex={videoInfo?.parsed?.camera_index}
              imageSize={imageSize}
              videoRef={videoRef}
              onHomographyChange={setHomography}
              onFloorplanChange={setFloorplanUrl}
              onCountersChange={setCounters}
              onDirtyChange={setMapDirty}
            />
          )}
        </main>
      </div>
      <div style={{ display: showVideo ? "contents" : "none" }}>
        <FloatingVideoWindow
          title={videoName || "нет файла"}
          label="Видео"
          geom={floatVideo}
          allowMinimize={false}
          onClose={() => toggleShowVideo(false)}
          onGeomChange={setFloatVideo}
          onGeomCommit={commitFloatVideo}
          aspect={videoAspect}
          extraHeight={FLOAT_PLAYER_CONTROLS_H}
        >
          <TrackingPlayer
            ref={playerRef}
            videoUrl={videoUrl}
            tracking={tracking}
            sessionParts={sessionParts.length ? sessionParts : null}
            showLabels
            showTrails
            trailLength={30}
            onFrameChange={setCurrentFrame}
            videoRef={videoRef}
            highlightIds={overlayHighlightIds}
            groupByTrack={groupByTrack}
            mergeTimeline={mergeTimeline}
            focusTrackIds={playerFocusTrackIds}
            compact={false}
          />
        </FloatingVideoWindow>
      </div>
      {showMap && (
        <FloatingVideoWindow
          title={`cam ${cameraKey}`}
          label="Карта"
          geom={floatMap}
          allowMinimize={false}
          onClose={() => toggleShowMap(false)}
          onGeomChange={setFloatMap}
          onGeomCommit={commitFloatMap}
          aspect={mapAspect}
        >
          <MapFloorView
            floorplanUrl={floorplanUrl}
            homography={homography}
            tracking={tracking}
            videoRef={videoRef}
            currentFrame={currentFrame}
            showTrails
            focusTrackIds={playerFocusTrackIds}
            groupByTrack={groupByTrack}
            mergeTimeline={mergeTimeline}
            camerasOnMap={mapCameras}
            allCalibPoints={allCalibPoints}
            activeCameraKey={cameraKey}
            counters={counters}
            feetDoc={feetDoc}
            compact={false}
          />
        </FloatingVideoWindow>
      )}
    </>
  );
}

import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { TrackingData } from "../types";
import {
  bboxWh,
  buildTrackKeyframes,
  colorForTrackId,
  detectionsAtFrame,
  formatDuration,
  resolveDetectEveryN,
  type CropShot,
  type SimilarHit,
} from "../utils";

type TrackSummary = {
  id: number;
  firstFrame: number;
  lastFrame: number;
  firstSec: number;
  lastSec: number;
  durationSec: number;
  totalDetections: number;
  avgConfidence: number;
  maxConfidence: number;
  bboxMinW: number;
  bboxMinH: number;
  bboxAvgW: number;
  bboxAvgH: number;
  bboxMaxW: number;
  bboxMaxH: number;
};

type Props = {
  tracking: TrackingData | null;
  currentFrame: number;
  selectedTrackId?: number | null;
  onSelectTrackId?: (trackId: number | null) => void;
  onSeekToSec?: (sec: number) => void;
  cropUrls?: Record<string, CropShot[]>;
  similarByTrack?: Record<string, SimilarHit[]>;
  mergeByTrack?: Record<string, SimilarHit[]>;
};

export function TrackingSidebar({
  tracking,
  currentFrame,
  selectedTrackId = null,
  onSelectTrackId,
  onSeekToSec,
  cropUrls = {},
  similarByTrack = {},
  mergeByTrack = {},
}: Props) {
  const [showOnlyActive, setShowOnlyActive] = useState(false);
  const [query, setQuery] = useState("");

  const trackSummaries = useMemo<TrackSummary[]>(() => {
    if (!tracking?.frames) return [];

    const map = new Map<
      number,
      {
        id: number;
        firstFrame: number;
        lastFrame: number;
        firstSec: number;
        lastSec: number;
        count: number;
        sumConf: number;
        maxConf: number;
        sumW: number;
        sumH: number;
        minArea: number;
        minW: number;
        minH: number;
        maxArea: number;
        maxW: number;
        maxH: number;
      }
    >();

    for (const frame of tracking.frames) {
      for (const det of frame.detections) {
        const { w, h, area } = bboxWh(det.bbox);
        let entry = map.get(det.track_id);
        if (!entry) {
          entry = {
            id: det.track_id,
            firstFrame: frame.frame_index,
            lastFrame: frame.frame_index,
            firstSec: frame.timestamp_sec,
            lastSec: frame.timestamp_sec,
            count: 0,
            sumConf: 0,
            maxConf: 0,
            sumW: 0,
            sumH: 0,
            minArea: Infinity,
            minW: 0,
            minH: 0,
            maxArea: -1,
            maxW: 0,
            maxH: 0,
          };
          map.set(det.track_id, entry);
        }
        entry.lastFrame = frame.frame_index;
        entry.lastSec = frame.timestamp_sec;
        entry.count += 1;
        entry.sumConf += det.confidence;
        entry.sumW += w;
        entry.sumH += h;
        if (det.confidence > entry.maxConf) {
          entry.maxConf = det.confidence;
        }
        if (area < entry.minArea) {
          entry.minArea = area;
          entry.minW = w;
          entry.minH = h;
        }
        if (area > entry.maxArea) {
          entry.maxArea = area;
          entry.maxW = w;
          entry.maxH = h;
        }
      }
    }

    return Array.from(map.values())
      .map((item) => ({
        id: item.id,
        firstFrame: item.firstFrame,
        lastFrame: item.lastFrame,
        firstSec: item.firstSec,
        lastSec: item.lastSec,
        durationSec: Number((item.lastSec - item.firstSec).toFixed(2)),
        totalDetections: item.count,
        avgConfidence: Number((item.sumConf / item.count).toFixed(2)),
        maxConfidence: Number(item.maxConf.toFixed(2)),
        bboxMinW: Math.round(item.minW),
        bboxMinH: Math.round(item.minH),
        bboxAvgW: Math.round(item.sumW / item.count),
        bboxAvgH: Math.round(item.sumH / item.count),
        bboxMaxW: Math.round(item.maxW),
        bboxMaxH: Math.round(item.maxH),
      }))
      .sort((a, b) => a.id - b.id);
  }, [tracking]);

  const keyframes = useMemo(
    () => (tracking ? buildTrackKeyframes(tracking) : null),
    [tracking],
  );
  const detectEveryN = useMemo(
    () => (tracking ? resolveDetectEveryN(tracking) : 1),
    [tracking],
  );

  const activeTrackIds = useMemo<Set<number>>(() => {
    if (!keyframes) return new Set();
    return new Set(
      detectionsAtFrame(keyframes, currentFrame, detectEveryN).map((d) => d.track_id),
    );
  }, [keyframes, currentFrame, detectEveryN]);

  const visibleTrackSummaries = useMemo(() => {
    let list = trackSummaries;
    if (showOnlyActive) {
      list = list.filter((t) => activeTrackIds.has(t.id));
    }
    const q = query.trim();
    if (q) {
      list = list.filter((t) => String(t.id).includes(q));
    }
    return list;
  }, [trackSummaries, showOnlyActive, activeTrackIds, query]);

  const { inFrame, rest } = useMemo(() => {
    const inFrame: TrackSummary[] = [];
    const rest: TrackSummary[] = [];
    for (const t of visibleTrackSummaries) {
      if (activeTrackIds.has(t.id)) inFrame.push(t);
      else rest.push(t);
    }
    return { inFrame, rest };
  }, [visibleTrackSummaries, activeTrackIds]);

  function selectTrack(track: TrackSummary) {
    if (selectedTrackId === track.id) {
      onSelectTrackId?.(null);
      return;
    }
    onSelectTrackId?.(track.id);
  }

  function gotoTrack(track: TrackSummary, sec?: number) {
    onSelectTrackId?.(track.id);
    onSeekToSec?.(sec ?? track.firstSec);
  }

  function renderCard(t: TrackSummary) {
    return (
      <TrackCard
        key={t.id}
        track={t}
        isActive={activeTrackIds.has(t.id)}
        isSelected={selectedTrackId === t.id}
        similar={similarByTrack[String(t.id)] ?? []}
        merge={mergeByTrack[String(t.id)] ?? []}
        cropByTrack={cropUrls}
        onSelect={() => selectTrack(t)}
        onPickTrack={(id) => onSelectTrackId?.(id)}
        onGoto={() => gotoTrack(t)}
      />
    );
  }

  function renderSection(title: string, tracks: TrackSummary[]) {
    if (!tracks.length) return null;
    return (
      <section className="track-section">
        <h4>
          {title}
          <span>{tracks.length}</span>
        </h4>
        <div className="track-grid">{tracks.map(renderCard)}</div>
      </section>
    );
  }

  if (!tracking) {
    return (
      <aside className="sidebar empty">
        <p>Выберите видео с результатами, чтобы увидеть статистику по каждому объекту.</p>
      </aside>
    );
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-toolbar">
        <input
          type="search"
          className="track-search"
          placeholder="фильтр #id"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <label className="toggle sidebar-toggle">
          <input
            type="checkbox"
            checked={showOnlyActive}
            onChange={(e) => setShowOnlyActive(e.target.checked)}
          />
          активные ({activeTrackIds.size})
        </label>
        <span className="badge">
          {visibleTrackSummaries.length}/{trackSummaries.length}
        </span>
      </div>

      <div className="track-list">
        {visibleTrackSummaries.length === 0 ? (
          <p className="no-tracks">
            {showOnlyActive || query ? "Нет треков по фильтру" : "Нет треков в tracking JSON"}
          </p>
        ) : (
          <>
            {renderSection("В кадре", inFrame)}
            {!showOnlyActive && renderSection("Остальные", rest)}
          </>
        )}
      </div>
    </aside>
  );
}

function formatScore(score: number | null): string {
  return score == null || !Number.isFinite(score) ? "—" : score.toFixed(2);
}

function PopMetrics({ hit, hideOverall }: { hit: SimilarHit; hideOverall?: boolean }) {
  const distUnit = hit.space === "map" ? "м" : "px";
  const motion =
    hit.motion != null
      ? `${formatScore(hit.motion)}${hit.dist != null ? ` (${hit.dist.toFixed(1)}${distUnit})` : ""}`
      : hit.dist != null
        ? `${hit.dist.toFixed(1)}${distUnit}`
        : null;
  const items: { label: string; value: string }[] = [];
  if (!hideOverall) items.push({ label: "combo", value: formatScore(hit.score) });
  if (hit.reid != null) items.push({ label: "reid", value: formatScore(hit.reid) });
  if (motion) items.push({ label: "motion", value: motion });
  if (hit.size != null) items.push({ label: "size", value: formatScore(hit.size) });
  if (hit.gap != null) items.push({ label: "Δt", value: `${hit.gap.toFixed(1)}с` });
  if (!items.length) return null;
  return (
    <div className="pop-metrics">
      {items.map(({ label, value }) => (
        <span key={label} className={`pop-metric${label === "combo" ? " overall" : ""}`}>
          <em>{label}</em> {value}
        </span>
      ))}
    </div>
  );
}

function CropStrip({
  shots,
  trackId,
  onReady,
}: {
  shots: CropShot[];
  trackId: number;
  onReady?: () => void;
}) {
  if (!shots.length) return <p className="similar-pop-empty">Нет кропов</p>;
  return (
    <div className="pop-crops">
      {shots.map((s) => (
        <div key={`${s.rank}-${s.frame ?? 0}`} className="pop-crop">
          <img
            src={s.url}
            alt={`#${trackId} k${s.rank}`}
            onLoad={onReady}
          />
          <span>
            k{s.rank} {formatScore(s.score)}
          </span>
        </div>
      ))}
    </div>
  );
}

function SimilarHitChip({
  hit,
  shots,
  onPick,
  variant = "similar",
}: {
  hit: SimilarHit;
  shots: CropShot[];
  onPick: () => void;
  variant?: "similar" | "merge";
}) {
  const btnRef = useRef<HTMLButtonElement>(null);
  const popRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [geom, setGeom] = useState({ left: 0, top: 0 });
  const [ready, setReady] = useState(0);
  const color = colorForTrackId(hit.track_id);
  const span =
    hit.t0 != null || hit.t1 != null
      ? `${formatDuration(hit.t0)} – ${formatDuration(hit.t1)}`
      : null;

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
  }, [open, shots.length, ready, variant]);

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        className={`crop-score similar-hit ${variant === "merge" ? "merge-hit" : ""} ${open ? "on" : ""}`}
        onMouseEnter={() => {
          const br = btnRef.current?.getBoundingClientRect();
          if (br) setGeom({ left: br.left, top: br.bottom + 8 });
          setOpen(true);
        }}
        onMouseLeave={() => setOpen(false)}
        onClick={(e) => {
          e.stopPropagation();
          onPick();
        }}
      >
        #{hit.track_id} {formatScore(hit.score)}
      </button>
      {open &&
        createPortal(
          <div
            ref={popRef}
            className={`similar-pop ${variant === "merge" ? "merge-pop" : "sim-pop"}`}
            style={{ left: geom.left, top: geom.top }}
          >
            {variant === "merge" ? (
              <>
                <div className="similar-pop-head">
                  <span className="merge-badge">
                    {typeof hit.group_id === "number" ? `объединение #${hit.group_id}` : "объединение"}
                  </span>
                  <span className="track-id" style={{ backgroundColor: color }}>
                    #{hit.track_id}
                  </span>
                  <span className="crop-pop-title">{formatScore(hit.score)}</span>
                </div>
                {hit.reason ? <p className="merge-pop-reason">{hit.reason}</p> : null}
                <div className="merge-pop-body">
                  <div className="merge-pop-side">
                    <PopMetrics hit={hit} hideOverall />
                    <div className="pop-meta-line">
                      {span ? <span>{span}</span> : null}
                    </div>
                  </div>
                  <CropStrip
                    shots={shots}
                    trackId={hit.track_id}
                    onReady={() => setReady((n) => n + 1)}
                  />
                </div>
              </>
            ) : (
              <>
                <div className="similar-pop-head">
                  <span className="similar-badge">похож на</span>
                  <span className="track-id" style={{ backgroundColor: color }}>
                    #{hit.track_id}
                  </span>
                  <span className="crop-pop-title">{formatScore(hit.score)}</span>
                  {span ? <span className="pop-span">{span}</span> : null}
                </div>
                <CropStrip
                  shots={shots}
                  trackId={hit.track_id}
                  onReady={() => setReady((n) => n + 1)}
                />
                <PopMetrics hit={hit} />
              </>
            )}
          </div>,
          document.body,
        )}
    </>
  );
}

function TrackCard({
  track: t,
  isActive,
  isSelected,
  similar,
  merge,
  cropByTrack,
  onSelect,
  onPickTrack,
  onGoto,
}: {
  track: TrackSummary;
  isActive: boolean;
  isSelected: boolean;
  similar: SimilarHit[];
  merge: SimilarHit[];
  cropByTrack: Record<string, CropShot[]>;
  onSelect: () => void;
  onPickTrack: (trackId: number) => void;
  onGoto: () => void;
}) {
  const color = colorForTrackId(t.id);
  const groupId = typeof merge[0]?.group_id === "number" ? merge[0].group_id : null;

  return (
    <div
      className={`track-card ${isActive ? "active" : ""} ${isSelected ? "selected" : ""}`}
      data-track-id={t.id}
      onClick={onSelect}
    >
      <div className="track-card-main">
        <div className="track-card-header">
          <div className="track-badge-group">
            <span className="track-id" style={{ backgroundColor: color }}>
              #{t.id}
            </span>
            {isActive && <span className="active-dot" title="В текущем кадре" />}
            {similar.length > 0 && (
              <span className="similar-badge" title="похожие треки">
                похож {similar.length}
              </span>
            )}
            {merge.length > 0 && (
              <span className="merge-badge" title="объединение LLM">
                {groupId != null ? `g${groupId} · ${merge.length}` : `LLM ${merge.length}`}
              </span>
            )}
            <span className="duration">
              {t.durationSec}s · {t.totalDetections}
            </span>
          </div>
        </div>

        <div className="track-card-section">
          <span className="track-card-section-label">время</span>
          <div className="track-card-compact-body">
            <span>
              {t.firstSec.toFixed(1)}–{t.lastSec.toFixed(1)}s
            </span>
            <span className="conf">{(t.avgConfidence * 100).toFixed(0)}%</span>
          </div>
        </div>

        <div className="track-card-section">
          <span className="track-card-section-label">bbox px</span>
          <div className="track-card-bbox" title="min / avg / max">
            <div className="bbox-head">
              <span />
              <span>min</span>
              <span>avg</span>
              <span>max</span>
            </div>
            <div className="bbox-row">
              <span>W</span>
              <span>{t.bboxMinW}</span>
              <span>{t.bboxAvgW}</span>
              <span>{t.bboxMaxW}</span>
            </div>
            <div className="bbox-row">
              <span>H</span>
              <span>{t.bboxMinH}</span>
              <span>{t.bboxAvgH}</span>
              <span>{t.bboxMaxH}</span>
            </div>
          </div>
        </div>

        {similar.length > 0 && (
          <div className="track-card-section">
            <span className="similar-badge">похож на</span>
            <div className="similar-tracks">
              {similar.map((hit) => (
                <SimilarHitChip
                  key={`sim-${hit.track_id}`}
                  hit={hit}
                  shots={cropByTrack[String(hit.track_id)] ?? []}
                  onPick={() => {
                    onPickTrack(hit.track_id);
                    document
                      .querySelector(`[data-track-id="${hit.track_id}"]`)
                      ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
                  }}
                />
              ))}
            </div>
          </div>
        )}

        {merge.length > 0 && (
          <div className="track-card-section merge-section">
            <span className="merge-badge">
              {groupId != null ? `объединение #${groupId}` : "объединение"}
            </span>
            <div className="similar-tracks">
              {merge.map((hit) => (
                <SimilarHitChip
                  key={`mg-${hit.track_id}`}
                  hit={hit}
                  shots={cropByTrack[String(hit.track_id)] ?? []}
                  variant="merge"
                  onPick={() => {
                    onPickTrack(hit.track_id);
                    document
                      .querySelector(`[data-track-id="${hit.track_id}"]`)
                      ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
                  }}
                />
              ))}
            </div>
          </div>
        )}

        <div className="track-goto-slot">
          <button
            type="button"
            className="track-goto"
            onClick={(e) => {
              e.stopPropagation();
              onGoto();
            }}
          >
            Перейти
          </button>
        </div>
      </div>
    </div>
  );
}

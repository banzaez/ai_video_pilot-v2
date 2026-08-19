import { memo, useCallback, useEffect, useRef, type PointerEvent as ReactPointerEvent } from "react";
import type { TimeBounds, TimelineLane, TimelineSegment } from "./types";

type Props<T> = {
  lanes: TimelineLane<T>[];
  bounds: TimeBounds;
  currentSec: number;
  zoom: number;
  formatTick: (sec: number) => string;
  onSeek: (sec: number) => void;
  onSelect?: (seg: TimelineSegment<T>) => void;
  onShiftSelect?: (seg: TimelineSegment<T>) => void;
  onScrubbing?: (active: boolean) => void;
  onLaneClick?: (lane: TimelineLane<T>) => void;
  tickCount?: number;
};

function PlayheadTimelineInner<T>({
  lanes,
  bounds,
  currentSec,
  zoom,
  formatTick,
  onSeek,
  onSelect,
  onShiftSelect,
  onScrubbing,
  onLaneClick,
  tickCount = 11,
}: Props<T>) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const downPos = useRef({ x: 0, y: 0 });
  const barHit = useRef<TimelineSegment<T> | null>(null);
  const segById = useRef(new Map<string, TimelineSegment<T>>());

  segById.current = new Map(lanes.flatMap((lane) => lane.segments.map((s) => [s.id, s] as const)));

  const seekFromClientX = useCallback(
    (clientX: number) => {
      const plot = plotRef.current;
      if (!plot) return;
      const rect = plot.getBoundingClientRect();
      const pct = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(1, rect.width)));
      onSeek(bounds.minT + pct * bounds.span);
    },
    [onSeek, bounds.minT, bounds.span],
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
        onSelect?.(barHit.current);
      } else {
        seekFromClientX(e.clientX);
      }
      barHit.current = null;
    },
    [onSelect, onScrubbing, seekFromClientX],
  );

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (!e.shiftKey) return;
      if (el.scrollWidth <= el.clientWidth + 1) return;
      const raw = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
      if (raw === 0) return;
      const scale =
        e.deltaMode === WheelEvent.DOM_DELTA_LINE
          ? 16
          : e.deltaMode === WheelEvent.DOM_DELTA_PAGE
            ? el.clientWidth
            : 1;
      e.preventDefault();
      el.scrollLeft += raw * scale;
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  useEffect(() => {
    if (dragging.current || zoom <= 1) return;
    const scroller = scrollRef.current;
    const plot = plotRef.current;
    if (!scroller || !plot || bounds.span <= 0) return;
    const pct = (currentSec - bounds.minT) / bounds.span;
    const plotRect = plot.getBoundingClientRect();
    const scrollRect = scroller.getBoundingClientRect();
    const x = plotRect.left - scrollRect.left + scroller.scrollLeft + pct * plot.offsetWidth;
    const view = scroller.clientWidth;
    const sl = scroller.scrollLeft;
    if (x < sl + 48 || x > sl + view - 48) {
      scroller.scrollLeft = Math.max(0, x - view * 0.45);
    }
  }, [currentSec, zoom, bounds.minT, bounds.span]);

  const playheadPct = bounds.span > 0 ? ((currentSec - bounds.minT) / bounds.span) * 100 : 0;

  return (
    <div className="playhead-timeline" ref={scrollRef}>
      <div
        className="playhead-timeline-inner"
        style={{ width: `${Math.max(100, 100 * zoom)}%` }}
        onPointerDown={(e) => {
          if (e.button !== 0) return;
          const bar = (e.target as HTMLElement).closest(".playhead-bar") as HTMLElement | null;
          const seg = bar?.dataset.segId ? segById.current.get(bar.dataset.segId) ?? null : null;
          if (e.shiftKey && seg) {
            e.preventDefault();
            onShiftSelect?.(seg);
            return;
          }
          e.preventDefault();
          e.currentTarget.setPointerCapture(e.pointerId);
          e.currentTarget.classList.add("is-dragging");
          dragging.current = true;
          onScrubbing?.(true);
          downPos.current = { x: e.clientX, y: e.clientY };
          barHit.current = seg;
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
        <div className="playhead-axis-row">
          <div className="playhead-lane-label" aria-hidden />
          <div className="playhead-axis" ref={plotRef}>
            {Array.from({ length: tickCount }).map((_, i) => {
              const frac = i / Math.max(1, tickCount - 1);
              const t = bounds.minT + frac * bounds.span;
              return (
                <div key={i} className="playhead-axis-tick" style={{ left: `${frac * 100}%` }}>
                  <i />
                  {formatTick(t)}
                </div>
              );
            })}
          </div>
        </div>
        <div className="playhead-needle-layer" aria-hidden>
          <div className="playhead-lane-label" />
          <div className="playhead-needle-track">
            <div className="playhead-needle" style={{ left: `${playheadPct}%` }} />
          </div>
        </div>
        <div className="playhead-lanes">
          {lanes.map((lane) => (
            <div key={lane.id} className={`playhead-lane ${lane.highlight ? "has-sel" : ""}`}>
              <div
                className="playhead-lane-label"
                title={lane.label}
                onPointerDown={(e) => {
                  if (!onLaneClick) return;
                  e.stopPropagation();
                  e.preventDefault();
                  onLaneClick(lane);
                }}
              >
                {lane.label}
              </div>
              <div className="playhead-lane-track">
                {lane.segments.map((seg) => {
                  const leftPct = ((seg.t0 - bounds.minT) / bounds.span) * 100;
                  const widthPct = Math.max(0.45, ((seg.t1 - seg.t0) / bounds.span) * 100);
                  const cls = seg.dimmed ? "dim" : seg.selected ? "on" : "";
                  return (
                    <div
                      key={seg.id}
                      role="button"
                      tabIndex={-1}
                      data-seg-id={seg.id}
                      className={`playhead-bar ${cls}`}
                      style={{
                        left: `${leftPct}%`,
                        width: `${widthPct}%`,
                        background: seg.color,
                      }}
                      title={seg.title}
                    >
                      {seg.selected ? seg.label ?? "" : ""}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export const PlayheadTimeline = memo(PlayheadTimelineInner) as typeof PlayheadTimelineInner;

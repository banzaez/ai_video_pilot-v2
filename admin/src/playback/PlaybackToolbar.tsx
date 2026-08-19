import type { ReactNode } from "react";
import { formatDuration } from "../utils";
import type { TimeBounds } from "./types";
import type { PlaybackClock } from "./usePlaybackClock";

type Props = {
  clock: PlaybackClock;
  bounds: TimeBounds;
  formatCurrent: (sec: number) => string;
  formatBound?: (sec: number) => string;
  extras?: ReactNode;
};

export function PlaybackToolbar({ clock, bounds, formatCurrent, formatBound, extras }: Props) {
  const bound = formatBound ?? formatCurrent;
  return (
    <div className="playhead-toolbar">
      <div className="playhead-toolbar-group">
        <button
          type="button"
          className={`playhead-btn playhead-btn-play ${clock.isPlaying ? "is-playing" : "playhead-btn-primary"}`}
          onClick={clock.togglePlay}
          title="Пробел"
        >
          {clock.isPlaying ? "Пауза" : "Пуск"}
        </button>
        <button type="button" className="playhead-btn" onClick={() => clock.step(-clock.frameSec)} title="←">
          −1к
        </button>
        <button type="button" className="playhead-btn" onClick={() => clock.step(clock.frameSec)} title="→">
          +1к
        </button>
        {clock.rates.map((spd) => (
          <button
            key={spd}
            type="button"
            className={`playhead-btn playhead-speed ${clock.rate === spd ? "on" : ""}`}
            onClick={() => clock.setRate(spd)}
          >
            {spd}×
          </button>
        ))}
      </div>
      <span className="playhead-clock">{formatCurrent(clock.currentSec)}</span>
      <span className="playhead-clock-span">
        {bound(bounds.minT)}–{bound(bounds.maxT)} ({formatDuration(bounds.span)})
      </span>
      <div className="playhead-toolbar-group right">
        {extras}
        <button type="button" className="playhead-btn" onClick={() => clock.setZoom((z) => Math.max(0.5, z - 0.25))}>
          −
        </button>
        <span className="playhead-clock-span">{Math.round(clock.zoom * 100)}%</span>
        <button type="button" className="playhead-btn" onClick={() => clock.setZoom((z) => Math.min(4, z + 0.25))}>
          +
        </button>
        <button type="button" className="playhead-btn" onClick={() => clock.setZoom(1)}>
          100%
        </button>
      </div>
    </div>
  );
}

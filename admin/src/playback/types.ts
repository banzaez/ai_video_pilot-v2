/** Общий контракт медиа для часов воспроизведения (день / склейки / сессия). */

export type TimeBounds = {
  minT: number;
  maxT: number;
  span: number;
};

export type PlaybackApplyMode = "hard" | "soft";

/** Адаптер к плееру(ам): часы не знают про камеры и куски сессии. */
export type PlaybackSink = {
  sampleTime: () => number | null;
  apply: (tSec: number, play: boolean, mode: PlaybackApplyMode) => void;
  setRate: (rate: number) => void;
};

export type TimelineSegment<T = unknown> = {
  id: string;
  t0: number;
  t1: number;
  color: string;
  label?: string;
  title?: string;
  selected?: boolean;
  dimmed?: boolean;
  data: T;
};

export type TimelineLane<T = unknown> = {
  id: string;
  label: string;
  highlight?: boolean;
  segments: TimelineSegment<T>[];
};

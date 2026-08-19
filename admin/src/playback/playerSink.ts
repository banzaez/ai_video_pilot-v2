import type { PlaybackSink } from "./types";

/** Минимальный контракт плеера сессии (TrackingPlayerHandle и аналоги). */
export type PlaybackPlayer = {
  seekToGlobal: (tSec: number, playAfter?: boolean) => void;
  setPlaybackRate: (rate: number) => void;
  play: () => void;
  pause: () => void;
  getGlobalSec: () => number | null;
  paused: () => boolean;
};

const DEFAULT_DRIFT_SEC = 0.45;
const SEEK_RETRY_FRAMES = 12;

type Options = {
  driftSec?: number;
  /** Запомнить время (например startAtSec при монтировании плавающего окна). */
  onTime?: (tSec: number) => void;
  /** Показать плеер, если он размонтирован (плавающее окно). */
  ensureVisible?: () => void;
};

function withPlayer(
  getPlayer: () => PlaybackPlayer | null | undefined,
  fn: (player: PlaybackPlayer) => void,
  retries = SEEK_RETRY_FRAMES,
) {
  const player = getPlayer();
  if (player) {
    fn(player);
    return;
  }
  if (retries <= 0) return;
  requestAnimationFrame(() => withPlayer(getPlayer, fn, retries - 1));
}

/** Sink для одного видео: seek/play/rate без знания о вкладке. */
export function createPlayerSink(
  getPlayer: () => PlaybackPlayer | null | undefined,
  options?: Options,
): PlaybackSink {
  const drift = options?.driftSec ?? DEFAULT_DRIFT_SEC;
  let lastRate: number | null = null;

  return {
    sampleTime: () => getPlayer()?.getGlobalSec() ?? null,
    apply: (t, play, mode) => {
      options?.onTime?.(t);
      if (play) options?.ensureVisible?.();
      withPlayer(getPlayer, (player) => {
        if (lastRate != null) player.setPlaybackRate(lastRate);
        if (mode === "hard") {
          if (!play) player.pause();
          player.seekToGlobal(t, play);
          return;
        }
        if (!play) {
          player.pause();
          return;
        }
        const g = player.getGlobalSec();
        if (g == null || Math.abs(g - t) > drift) {
          player.seekToGlobal(t, true);
        } else if (player.paused()) {
          player.play();
        }
      });
    },
    setRate: (rate) => {
      lastRate = rate;
      getPlayer()?.setPlaybackRate(rate);
    },
  };
}

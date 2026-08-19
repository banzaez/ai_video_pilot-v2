import { useCallback, useEffect, useRef, useState } from "react";
import { clampTime } from "./time";
import type { PlaybackSink, TimeBounds } from "./types";

const UI_HZ_MS = 100;
const DEFAULT_RATES = [0.5, 1, 2, 4];

export type PlaybackHotkeys = {
  space?: boolean;
  arrows?: boolean;
};

type Options = {
  bounds: TimeBounds;
  sink: PlaybackSink;
  defaultRate?: number;
  frameSec?: number;
  hotkeys?: boolean | PlaybackHotkeys;
  enabled?: boolean;
};

export function usePlaybackClock({
  bounds,
  sink,
  defaultRate = 2,
  frameSec = 1 / 25,
  hotkeys = true,
  enabled = true,
}: Options) {
  const [currentSec, setCurrentSec] = useState(bounds.minT);
  const [isPlaying, setIsPlaying] = useState(false);
  const [rate, setRateState] = useState(defaultRate);
  const [zoom, setZoom] = useState(1);

  const currentSecRef = useRef(currentSec);
  const isPlayingRef = useRef(isPlaying);
  const rateRef = useRef(rate);
  const boundsRef = useRef(bounds);
  const sinkRef = useRef(sink);
  const scrubbingRef = useRef(false);
  const seekGraceUntilRef = useRef<number>(0);
  const seekRef = useRef<(t: number, playAfter?: boolean) => void>(() => {});

  currentSecRef.current = currentSec;
  isPlayingRef.current = isPlaying;
  rateRef.current = rate;
  boundsRef.current = bounds;
  sinkRef.current = sink;

  const seek = useCallback((targetSec: number, playAfter?: boolean) => {
    const b = boundsRef.current;
    const clamped = clampTime(targetSec, b);
    seekGraceUntilRef.current = performance.now() + 600;
    currentSecRef.current = clamped;
    setCurrentSec(clamped);
    const play = playAfter ?? isPlayingRef.current;
    sinkRef.current.apply(clamped, play, "hard");
  }, []);

  seekRef.current = seek;

  const play = useCallback(() => {
    setIsPlaying(true);
    seekGraceUntilRef.current = performance.now() + 600;
    sinkRef.current.setRate(rateRef.current);
    sinkRef.current.apply(currentSecRef.current, true, "hard");
  }, []);

  const pause = useCallback(() => {
    setIsPlaying(false);
    sinkRef.current.apply(currentSecRef.current, false, "hard");
  }, []);

  const togglePlay = useCallback(() => {
    if (isPlayingRef.current) pause();
    else play();
  }, [pause, play]);

  const setRate = useCallback((next: number) => {
    if (!Number.isFinite(next) || next <= 0) return;
    rateRef.current = next;
    setRateState(next);
    sinkRef.current.setRate(next);
  }, []);

  const step = useCallback(
    (deltaSec: number) => {
      seek(currentSecRef.current + deltaSec, false);
    },
    [seek],
  );

  const setScrubbing = useCallback((active: boolean) => {
    scrubbingRef.current = active;
    if (!active) {
      seekGraceUntilRef.current = performance.now() + 600;
    }
  }, []);

  const noteExternalTime = useCallback((t: number) => {
    if (isPlayingRef.current || scrubbingRef.current) return;
    const clamped = clampTime(t, boundsRef.current);
    if (Math.abs(clamped - currentSecRef.current) < 0.04) return;
    currentSecRef.current = clamped;
    setCurrentSec(clamped);
  }, []);

  useEffect(() => {
    if (!enabled || !isPlaying) return;
    sinkRef.current.setRate(rate);
  }, [enabled, isPlaying, rate]);

  useEffect(() => {
    if (!enabled || !isPlaying) return;
    let animId = 0;
    let lastTs = performance.now();
    let lastUi = 0;

    const loop = (now: number) => {
      if (scrubbingRef.current) {
        lastTs = now;
        animId = requestAnimationFrame(loop);
        return;
      }
      const dt = Math.max(0, (now - lastTs) / 1000) * rateRef.current;
      lastTs = now;
      let next = currentSecRef.current + dt;

      // Во время и сразу после сика/клика видео еще выполняет перемотку.
      // Игнорируем устаревшие позиции с плееров, чтобы бегунок не прыгал назад.
      if (now >= seekGraceUntilRef.current) {
        const sampled = sinkRef.current.sampleTime();
        if (sampled != null && Number.isFinite(sampled)) {
          const drift = sampled - next;
          if (Math.abs(drift) > 0.15 && Math.abs(drift) < 2.0) {
            next += drift * 0.12; // Мягкая сходимость без прыжков
          } else if (Math.abs(drift) >= 2.0) {
            next = sampled;
          }
        }
      }

      const b = boundsRef.current;
      if (next >= b.maxT) {
        next = b.maxT;
        currentSecRef.current = next;
        setCurrentSec(next);
        setIsPlaying(false);
        sinkRef.current.apply(next, false, "hard");
        return;
      }
      currentSecRef.current = next;
      if (now - lastUi >= UI_HZ_MS) {
        lastUi = now;
        setCurrentSec(next);
        sinkRef.current.apply(next, true, "soft");
      }
      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [enabled, isPlaying]);

  const keys = typeof hotkeys === "boolean" ? { space: hotkeys, arrows: hotkeys } : hotkeys;

  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes((e.target as HTMLElement)?.tagName)) return;
      if (keys.space && e.code === "Space") {
        e.preventDefault();
        if (isPlayingRef.current) pause();
        else play();
      } else if (keys.arrows && e.code === "ArrowLeft") {
        e.preventDefault();
        seekRef.current(currentSecRef.current - (e.shiftKey ? 1 : frameSec), false);
      } else if (keys.arrows && e.code === "ArrowRight") {
        e.preventDefault();
        seekRef.current(currentSecRef.current + (e.shiftKey ? 1 : frameSec), false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [enabled, frameSec, keys.arrows, keys.space, pause, play]);

  return {
    currentSec,
    currentSecRef,
    isPlaying,
    rate,
    zoom,
    rates: DEFAULT_RATES,
    frameSec,
    setRate,
    setZoom,
    play,
    pause,
    togglePlay,
    seek,
    step,
    setScrubbing,
    noteExternalTime,
  };
}

export type PlaybackClock = ReturnType<typeof usePlaybackClock>;

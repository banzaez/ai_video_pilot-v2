/**
 * Общий playback-слой админки.
 *
 * - PlaybackSink — адаптер к медиа (одна сессия или несколько камер дня)
 * - usePlaybackClock — часы: play/pause, скорость, зум, seek, Space/стрелки
 * - createPlayerSink — sink для одного TrackingPlayer
 * - PlaybackToolbar / PlayheadTimeline — один UI на День и Склейки
 */
export type { PlaybackApplyMode, PlaybackSink, TimeBounds, TimelineLane, TimelineSegment } from "./types";
export type { PlaybackPlayer } from "./playerSink";
export type { PlaybackClock, PlaybackHotkeys } from "./usePlaybackClock";
export { clampTime, formatDurationClock, formatHms, formatTimeOfDay, makeTimeBounds } from "./time";
export { createPlayerSink } from "./playerSink";
export { usePlaybackClock } from "./usePlaybackClock";
export { PlaybackToolbar } from "./PlaybackToolbar";
export { PlayheadTimeline } from "./PlayheadTimeline";

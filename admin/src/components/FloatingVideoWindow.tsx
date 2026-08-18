import { useEffect, useRef, useState, type PointerEvent, type ReactNode } from "react";
import {
  clampFloatVideo,
  FLOAT_BAR_H,
  floatHeightForWidth,
  FLOAT_MINI_W,
  VIDEO_ASPECT,
  type FloatVideoGeom,
} from "../utils";

const HANDLES = ["n", "s", "e", "w", "ne", "nw", "se", "sw"] as const;
type Handle = (typeof HANDLES)[number];

type Drag = {
  kind: "move" | Handle;
  px: number;
  py: number;
  start: FloatVideoGeom;
  moved: boolean;
};

type Props = {
  title: string;
  /** Подпись в шапке окна (по умолчанию «Видео») */
  label?: string;
  geom: FloatVideoGeom;
  minimized?: boolean;
  onMinimizedChange?: (minimized: boolean) => void;
  /** false — без кнопки свернуть (для видео) */
  allowMinimize?: boolean;
  onClose?: () => void;
  onGeomChange: (geom: FloatVideoGeom) => void;
  onGeomCommit: (geom: FloatVideoGeom) => void;
  aspect?: number;
  extraHeight?: number;
  children: ReactNode;
};

function resizeProportional(
  kind: Handle,
  start: FloatVideoGeom,
  dx: number,
  dy: number,
  aspect: number,
  extraH = 0,
): FloatVideoGeom {
  const right = start.x + start.w;
  const bottom = start.y + start.h;
  const startVideoH = Math.max(10, start.h - FLOAT_BAR_H - extraH);

  let w = start.w;

  if (kind.length === 2) {
    const dw = kind.includes("e") ? dx : -dx;
    const dh = kind.includes("s") ? dy : -dy;
    const sx = (start.w + dw) / start.w;
    const sy = (startVideoH + dh) / startVideoH;
    const scale = Math.abs(sx - 1) >= Math.abs(sy - 1) ? sx : sy;
    w = start.w * Math.max(0.1, scale);
  } else if (kind === "e") {
    w = start.w + dx;
  } else if (kind === "w") {
    w = start.w - dx;
  } else if (kind === "s") {
    const nextVideoH = startVideoH + dy;
    w = Math.max(10, nextVideoH) * aspect;
  } else if (kind === "n") {
    const nextVideoH = startVideoH - dy;
    w = Math.max(10, nextVideoH) * aspect;
  }

  w = Math.max(FLOAT_MINI_W, w);
  const h = floatHeightForWidth(w, aspect, extraH);

  let x = start.x;
  let y = start.y;

  if (kind.includes("w")) x = right - w;
  if (kind.includes("n")) y = bottom - h;
  if (kind === "n" || kind === "s") x = start.x + (start.w - w) / 2;
  if (kind === "e" || kind === "w") y = start.y + (start.h - h) / 2;

  return { x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) };
}

function displayGeom(geom: FloatVideoGeom, minimized: boolean, aspect: number, extraH = 0): FloatVideoGeom {
  if (!minimized) return geom;
  const w = FLOAT_MINI_W;
  const h = floatHeightForWidth(w, aspect, extraH);
  return clampFloatVideo({ ...geom, w, h }, aspect, FLOAT_MINI_W, extraH);
}

export function FloatingVideoWindow({
  title,
  label = "Видео",
  geom,
  minimized = false,
  onMinimizedChange,
  allowMinimize = true,
  onClose,
  onGeomChange,
  onGeomCommit,
  aspect = VIDEO_ASPECT,
  extraHeight = 0,
  children,
}: Props) {
  const canMinimize = allowMinimize && !!onMinimizedChange;
  const isMin = canMinimize && minimized;

  const live = useRef(geom);
  live.current = geom;
  const drag = useRef<Drag | null>(null);
  const movedRef = useRef(false);
  const minimizedRef = useRef(isMin);
  minimizedRef.current = isMin;
  const onChangeRef = useRef(onGeomChange);
  const onCommitRef = useRef(onGeomCommit);
  const aspectRef = useRef(aspect);
  const extraHRef = useRef(extraHeight);
  onChangeRef.current = onGeomChange;
  onCommitRef.current = onGeomCommit;
  aspectRef.current = aspect;
  extraHRef.current = extraHeight;
  const [active, setActive] = useState(false);

  useEffect(() => {
    function apply(next: FloatVideoGeom) {
      let clamped: FloatVideoGeom;
      if (minimizedRef.current) {
        const mini = clampFloatVideo(
          {
            x: next.x,
            y: next.y,
            w: FLOAT_MINI_W,
            h: floatHeightForWidth(FLOAT_MINI_W, aspectRef.current, extraHRef.current),
          },
          aspectRef.current,
          FLOAT_MINI_W,
          extraHRef.current,
        );
        clamped = { ...live.current, x: mini.x, y: mini.y };
      } else {
        clamped = clampFloatVideo(next, aspectRef.current, 280, extraHRef.current);
      }
      live.current = clamped;
      onChangeRef.current(clamped);
      return clamped;
    }

    function onMove(e: globalThis.PointerEvent) {
      const d = drag.current;
      if (!d) return;
      const dx = e.clientX - d.px;
      const dy = e.clientY - d.py;
      if (Math.abs(dx) + Math.abs(dy) > 3) {
        d.moved = true;
        movedRef.current = true;
      }
      if (d.kind === "move") {
        apply({ ...d.start, x: d.start.x + dx, y: d.start.y + dy });
        return;
      }
      apply(resizeProportional(d.kind, d.start, dx, dy, aspectRef.current, extraHRef.current));
    }

    function onUp() {
      if (!drag.current) return;
      drag.current = null;
      setActive(false);
      onCommitRef.current(live.current);
    }

    function onWinResize() {
      const next = apply(live.current);
      onCommitRef.current(next);
    }

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    window.addEventListener("resize", onWinResize);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      window.removeEventListener("resize", onWinResize);
    };
  }, []);

  function startDrag(kind: "move" | Handle, e: PointerEvent<HTMLElement>) {
    if (e.button !== 0) return;
    if (isMin && kind !== "move") return;
    e.preventDefault();
    e.stopPropagation();
    movedRef.current = false;
    drag.current = { kind, px: e.clientX, py: e.clientY, start: live.current, moved: false };
    setActive(true);
  }

  const shown = displayGeom(geom, isMin, aspect, extraHeight);

  return (
    <div
      className={`float-video${active ? " is-active" : ""}${isMin ? " is-minimized" : ""}`}
      style={{ left: shown.x, top: shown.y, width: shown.w, height: shown.h }}
      onClick={
        isMin
          ? () => {
              if (movedRef.current) {
                movedRef.current = false;
                return;
              }
              onMinimizedChange?.(false);
            }
          : undefined
      }
    >
      <div className="float-video-bar" onPointerDown={(e) => startDrag("move", e)}>
        <span className="float-video-grip" aria-hidden>
          ⋮⋮
        </span>
        <strong>{label}</strong>
        <em>{title}</em>
        <div className="float-win-actions">
          {canMinimize && (
            <button
              type="button"
              className="float-win-btn"
              title={isMin ? "Развернуть" : "Свернуть"}
              aria-label={isMin ? "Развернуть" : "Свернуть"}
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                onMinimizedChange?.(!isMin);
              }}
            >
              {isMin ? (
                <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden>
                  <path
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    d="M3 9.5V13h3.5M13 6.5V3H9.5M3 13l4-4M13 3l-4 4"
                  />
                </svg>
              ) : (
                <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden>
                  <path fill="none" stroke="currentColor" strokeWidth="1.8" d="M3.5 8.5h9" />
                </svg>
              )}
            </button>
          )}
          {onClose && (
            <button
              type="button"
              className="float-win-btn"
              title="Закрыть"
              aria-label="Закрыть"
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                onClose();
              }}
            >
              <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden>
                <path fill="none" stroke="currentColor" strokeWidth="1.8" d="M4 4l8 8M12 4l-8 8" />
              </svg>
            </button>
          )}
        </div>
      </div>
      <div className="float-video-body">{children}</div>
      {!isMin &&
        HANDLES.map((h) => (
          <div key={h} className={`float-resize ${h}`} onPointerDown={(e) => startDrag(h, e)} />
        ))}
    </div>
  );
}

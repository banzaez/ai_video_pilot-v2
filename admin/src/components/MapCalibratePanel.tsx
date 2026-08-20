import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import {
  applyHomography,
  autoCalibrate,
  computeHomography,
  colorForCameraKey,
  drawCameraPlacement,
  emptyHomographyDoc,
  estimatePlacementFromHomography,
  invertHomography,
  normalizePlacement,
  reprojectionErrors,
  rmsError,
  leaveOneOutRms,
  H_LOO_MIN_PAIRS,
  yawFromPoints,
  type CameraPlacement,
  type HomographyDoc,
  type HomoPair,
  type Mat3,
  type Pt,
} from "../homography";
import {
  countersFingerprint,
  emptyCountersDoc,
  newCounterId,
  normalizeCountersDoc,
  reprojectCountersFromImage,
  type CounterPoly,
  type CountersDoc,
} from "../counters";
import {
  GRID_FLOORPLAN,
  MAP_SIZE,
  METER_PX,
  deconflictLabelPositions,
  distancesToWalls,
  drawFloorGrid,
  gridLabel,
  isGridFloorplan,
  offsetAlongSegment,
  snapToGrid,
  tilesBetween,
} from "../mapGrid";
import {
  fitRayPose,
  normalizeCameraPose,
  rayPairStats,
  rayToGroundMap,
} from "../cameraPose";
import {
  cameraKeyFromVideo,
  fetchCounters,
  fetchHomography,
  fetchMapsConfig,
  runFeetApi,
  saveCounters,
  saveHomography,
  type MapsConfig,
} from "../utils";

type Props = {
  videoName: string;
  sessionKey?: string;
  videoUrl?: string | null;
  cameraIndex?: number | null;
  imageSize: [number, number] | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  onHomographyChange?: (doc: HomographyDoc | null) => void;
  onFloorplanChange?: (url: string) => void;
  onCountersChange?: (doc: CountersDoc | null) => void;
  /** Есть несохранённые изменения (H и/или прилавки) — для confirm ухода с вкладки */
  onDirtyChange?: (dirty: boolean) => void;
  onFeetReload?: () => Promise<void> | void;
};

type Mode = "pairs" | "test" | "place" | "count" | "draw";
type ViewState = { scale: number; tx: number; ty: number; rot: number };
type PaneSize = { width: number; height: number };
const EMPTY_PANE: PaneSize = { width: 0, height: 0 };

const IDENTITY_VIEW: ViewState = { scale: 1, tx: 0, ty: 0, rot: 0 };
const DEFAULT_FOV = 70;

const MODE_HINT: Record<Mode, string> = {
  pairs: "Кадр → план · тяните точки · Del — удалить",
  test: "Клик на кадре или плане — H (оранж.) и 3D-луч (син.) · сравните расхождение",
  place: "Шаг 2–3: поставьте камеру на плане, затем подберите высоту и наклон",
  count: "Клики по кадру — номера 1, 2, 3…",
  draw: "Клики ≥3 · Enter — на план · тяните вершины",
};



const MOD_KEY =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad|iPod/i.test(navigator.platform || navigator.userAgent)
    ? "⌘"
    : "Ctrl";

function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="map-calib-kbd">{children}</kbd>;
}

function normRot(deg: number): number {
  return ((Math.round(deg / 45) * 45) % 360 + 360) % 360;
}

function paneWorldStyle(
  natW: number,
  natH: number,
  view: ViewState,
  pane: PaneSize,
): CSSProperties {
  if (natW < 1 || natH < 1 || pane.width < 2 || pane.height < 2) {
    return {
      position: "absolute",
      left: 0,
      top: 0,
      width: "100%",
      height: "100%",
      opacity: 0,
      pointerEvents: "none",
    };
  }
  const fit = Math.min(pane.width / natW, pane.height / natH);
  const s = Math.max(0.4, view.scale);
  const dw = natW * fit * s;
  const dh = natH * fit * s;
  const rot = view.rot || 0;
  return {
    position: "absolute",
    left: (pane.width - dw) / 2 + view.tx,
    top: (pane.height - dh) / 2 + view.ty,
    width: dw,
    height: dh,
    pointerEvents: "none",
    transform: rot ? `rotate(${rot}deg)` : undefined,
    transformOrigin: "50% 50%",
  };
}

function usePaneSize(ref: React.RefObject<HTMLDivElement | null>, layoutTick: number): PaneSize {
  const [size, setSize] = useState<PaneSize>(EMPTY_PANE);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) {
      setSize(EMPTY_PANE);
      return;
    }
    const update = () => {
      const rect = el.getBoundingClientRect();
      const width = Math.round(rect.width);
      const height = Math.round(rect.height);
      setSize((prev) => (prev.width === width && prev.height === height ? prev : { width, height }));
    };
    update();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(update) : null;
    ro?.observe(el);
    window.addEventListener("resize", update);
    return () => {
      ro?.disconnect();
      window.removeEventListener("resize", update);
    };
  }, [ref, layoutTick]);

  return size;
}

/** Масштаб HUD-маркеров: сохраняем комфортный читаемый размер при zoom in (не уменьшаем до микроскопического),
 * а при сильном zoom out (scale < 1) плавно сжимаем, чтобы точки не перекрывали весь план. */
function markerHudScale(viewScale: number): number {
  if (viewScale >= 1) {
    // При приближении оставляем стабильный читаемый размер 1.0 (при экстремальном 8x можно слегка 0.9)
    return Math.max(0.9, 1.0 - (viewScale - 1) * 0.015);
  }
  // При отдалении (scale < 1) уменьшаем пропорционально, не давая закрывать всю карту
  return Math.max(0.55, Math.min(1.0, Math.pow(viewScale, 0.6)));
}

function defaultPlacement(pt: Pt, yaw = 0, fov = DEFAULT_FOV): CameraPlacement {
  return { position: pt, yaw_deg: yaw, fov_deg: fov };
}

async function captureFirstFrame(
  videoUrl: string,
  signal?: AbortSignal,
): Promise<{ dataUrl: string; w: number; h: number; sec: number }> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const v = document.createElement("video");
    v.muted = true;
    v.playsInline = true;
    v.preload = "auto";
    v.crossOrigin = "anonymous";
    let settled = false;
    const cleanup = () => {
      signal?.removeEventListener("abort", onAbort);
      v.removeAttribute("src");
      v.load();
    };
    const fail = (msg: string, err?: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(err ?? new Error(msg));
    };
    const onAbort = () => fail("aborted", new DOMException("Aborted", "AbortError"));
    const done = () => {
      if (settled) return;
      if (signal?.aborted) {
        onAbort();
        return;
      }
      if (!v.videoWidth) {
        fail("пустой кадр");
        return;
      }
      settled = true;
      const c = document.createElement("canvas");
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      const ctx = c.getContext("2d");
      if (!ctx) {
        cleanup();
        fail("canvas");
        return;
      }
      ctx.drawImage(v, 0, 0);
      const dataUrl = c.toDataURL("image/jpeg", 0.9);
      const sec = v.currentTime;
      cleanup();
      resolve({ dataUrl, w: c.width, h: c.height, sec });
    };
    signal?.addEventListener("abort", onAbort);
    v.addEventListener("error", () => fail("не удалось загрузить видео"));
    v.addEventListener("loadeddata", () => {
      try {
        v.currentTime = 0.04;
      } catch {
        done();
      }
    });
    v.addEventListener("seeked", done);
    v.src = videoUrl;
  });
}

function clientToImage(
  wrap: HTMLDivElement,
  natW: number,
  natH: number,
  view: ViewState,
  clientX: number,
  clientY: number,
): Pt | null {
  const rect = wrap.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2 || natW < 1 || natH < 1) return null;
  // Клик только внутри вьюпорта панели (можно за краем кадра/плана в letterbox).
  if (
    clientX < rect.left ||
    clientY < rect.top ||
    clientX > rect.right ||
    clientY > rect.bottom
  ) {
    return null;
  }
  const fit = Math.min(rect.width / natW, rect.height / natH);
  const s = Math.max(0.4, view.scale);
  const dw = natW * fit * s;
  const dh = natH * fit * s;
  const ox = (rect.width - dw) / 2 + view.tx;
  const oy = (rect.height - dh) / 2 + view.ty;
  const lx = clientX - rect.left;
  const ly = clientY - rect.top;
  const cx = ox + dw / 2;
  const cy = oy + dh / 2;
  let localX = lx - cx;
  let localY = ly - cy;
  const rot = view.rot || 0;
  if (rot) {
    const rad = (-rot * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);
    const rx = localX * cos - localY * sin;
    const ry = localX * sin + localY * cos;
    localX = rx;
    localY = ry;
  }
  const x = ((localX + dw / 2) / dw) * natW;
  const y = ((localY + dh / 2) / dh) * natH;
  // Без clamp: точки за границами кадра/плана допустимы (vanishing / вынос).
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return [x, y];
}

function ptPct(pt: Pt, natW: number, natH: number, opts?: { interactive?: boolean }): CSSProperties {
  // Якорь 0×0 без transform: дочерний translate(-50%,-50%) центрирует маркер,
  // а поворот плана на родителе крутит и сетку, и точки вместе.
  return {
    position: "absolute",
    left: `${(pt[0] / natW) * 100}%`,
    top: `${(pt[1] / natH) * 100}%`,
    width: 0,
    height: 0,
    pointerEvents: opts?.interactive === false ? "none" : "auto",
  };
}

function cloneDoc(d: HomographyDoc): HomographyDoc {
  return JSON.parse(JSON.stringify(d)) as HomographyDoc;
}

function docFingerprint(d: HomographyDoc): string {
  return JSON.stringify({
    floorplan: d.floorplan,
    pairs: d.pairs,
    H: d.H,
    placement: d.placement,
    map_size: d.map_size,
    image_size: d.image_size,
  });
}

export function MapCalibratePanel({
  videoName,
  sessionKey,
  videoUrl = null,
  cameraIndex,
  imageSize,
  videoRef,
  onHomographyChange,
  onFloorplanChange,
  onCountersChange,
  onDirtyChange,
  onFeetReload,
}: Props) {
  const initialCameraKey = useMemo(
    () => cameraKeyFromVideo(videoName, cameraIndex),
    [videoName, cameraIndex],
  );
  const [activeCameraKey, setActiveCameraKey] = useState<string>(initialCameraKey);
  useEffect(() => {
    setActiveCameraKey(initialCameraKey);
  }, [initialCameraKey]);
  const cameraKey = activeCameraKey;
  const [cfg, setCfg] = useState<MapsConfig | null>(null);
  const [doc, setDoc] = useState<HomographyDoc | null>(null);
  const [floorplan, setFloorplan] = useState(GRID_FLOORPLAN);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [isRecalculatingFeet, setIsRecalculatingFeet] = useState(false);
  const [dirty, setDirty] = useState(false);
  const savedFpRef = useRef<string>("");
  const [counters, setCounters] = useState<CountersDoc>(() => emptyCountersDoc());
  const [countersDirty, setCountersDirty] = useState(false);
  const savedCountersFpRef = useRef<string>("");
  const [draftPts, setDraftPts] = useState<Pt[]>([]);
  const [draftSide, setDraftSide] = useState<"image" | "map" | null>(null);
  const [selectedCounterId, setSelectedCounterId] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("pairs");
  const [pendingImage, setPendingImage] = useState<Pt | null>(null);
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [frameSize, setFrameSize] = useState<[number, number] | null>(null);
  const [, setFrameSec] = useState<number | null>(null);
  const [hoverPair, setHoverPair] = useState<number | null>(null);
  const [selectedPair, setSelectedPair] = useState<number | null>(null);
  const [showReproj, setShowReproj] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [tileMarks, setTileMarks] = useState<Pt[]>([]);
  /** Точка, для которой рисуем расстояния до соседей (при drag на плане) */
  const [measurePt, setMeasurePt] = useState<Pt | null>(null);
  const [measureExcludeId, setMeasureExcludeId] = useState<string | null>(null);
  const [testImagePt, setTestImagePt] = useState<Pt | null>(null);
  const [testMapPt, setTestMapPt] = useState<Pt | null>(null);
  const [fitMoveCamera, setFitMoveCamera] = useState(true);
  const [imgView, setImgView] = useState<ViewState>(IDENTITY_VIEW);
  const [mapView, setMapView] = useState<ViewState>(IDENTITY_VIEW);
  const [mapNat, setMapNat] = useState<[number, number]>(MAP_SIZE);
  const [history, setHistory] = useState<HomographyDoc[]>([]);
  const [future, setFuture] = useState<HomographyDoc[]>([]);
  const [layoutTick, setLayoutTick] = useState(0);
  const bumpLayout = () => setLayoutTick((n) => n + 1);

  const imageWrapRef = useRef<HTMLDivElement>(null);
  const mapWrapRef = useRef<HTMLDivElement>(null);
  const mapImgRef = useRef<HTMLImageElement>(null);
  const dragRef = useRef<{
    pair: number;
    side: "image" | "map";
  } | null>(null);
  /** Снимок doc до начала drag пары/камеры — для Undo (во время drag doc уже изменён). */
  const historyBeforeEditRef = useRef<HomographyDoc | null>(null);
  const counterDragRef = useRef<{
    id: string;
    side: "image" | "map";
    index: number;
    moved: boolean;
  } | null>(null);
  const suppressClickRef = useRef(false);
  const placeDragRef = useRef<{
    kind: "move" | "aim" | "place";
    origin: CameraPlacement;
  } | null>(null);
  const panRef = useRef<{
    side: "image" | "map";
    x: number;
    y: number;
    view: ViewState;
  } | null>(null);
  const mapCamCanvasRef = useRef<HTMLCanvasElement>(null);
  const mapGridCanvasRef = useRef<HTMLCanvasElement>(null);

  const floorplanUrl = isGridFloorplan(floorplan) ? "" : `/maps/${encodeURIComponent(floorplan)}`;
  const useGrid = isGridFloorplan(floorplan);
  const iw = frameSize?.[0] || imageSize?.[0] || 1920;
  const ih = frameSize?.[1] || imageSize?.[1] || 1080;

  const imagePaneSize = usePaneSize(imageWrapRef, layoutTick);
  const mapPaneSize = usePaneSize(mapWrapRef, layoutTick);
  const imageWorldStyle = useMemo(
    () => paneWorldStyle(iw, ih, imgView, imagePaneSize),
    [iw, ih, imgView, imagePaneSize],
  );
  const mapWorldStyle = useMemo(
    () => paneWorldStyle(mapNat[0], mapNat[1], mapView, mapPaneSize),
    [mapNat, mapView, mapPaneSize],
  );

  function mapPoint(raw: Pt): Pt {
    return snapToGrid(raw);
  }

  const onHomoRef = useRef(onHomographyChange);
  onHomoRef.current = onHomographyChange;
  const onFloorRef = useRef(onFloorplanChange);
  onFloorRef.current = onFloorplanChange;
  const onCountersRef = useRef(onCountersChange);
  onCountersRef.current = onCountersChange;
  const onDirtyRef = useRef(onDirtyChange);
  onDirtyRef.current = onDirtyChange;
  const docRef = useRef<HomographyDoc | null>(null);
  docRef.current = doc;
  const historyRef = useRef(history);
  historyRef.current = history;
  const futureRef = useRef(future);
  futureRef.current = future;
  const countersRef = useRef(counters);
  countersRef.current = counters;
  const dirtyRef = useRef(dirty);
  dirtyRef.current = dirty;
  const countersDirtyRef = useRef(countersDirty);
  countersDirtyRef.current = countersDirty;

  useEffect(() => {
    onDirtyRef.current?.(dirty || countersDirty);
    return () => {
      onDirtyRef.current?.(false);
    };
  }, [dirty, countersDirty]);

  function markCountersClean(next: CountersDoc) {
    savedCountersFpRef.current = countersFingerprint(next);
    setCountersDirty(false);
  }

  function publishCounters(next: CountersDoc) {
    setCounters(next);
    countersRef.current = next;
    setCountersDirty(countersFingerprint(next) !== savedCountersFpRef.current);
    onCountersRef.current?.(next);
  }

  function moveCounterVertex(id: string, side: "image" | "map", index: number, raw: Pt) {
    // Вершины прилавков — свободно, без snap к сетке
    const pt: Pt = [raw[0], raw[1]];
    setCounters((prev) => {
      const cur = prev.counters.find((c) => c.id === id);
      if (!cur) return prev;

      let nextCounter: CounterPoly = cur;
      if (side === "map") {
        const map = cur.map.map((p, i) => (i === index ? pt : ([...p] as Pt)));
        // План — единственный источник истины: сбрасываем все контуры кадров,
        // иначе H другой камеры снова перезапишет map.
        nextCounter = { ...cur, map, image_by_camera: undefined };
      } else {
        let img = cur.image_by_camera?.[cameraKey]
          ? cur.image_by_camera[cameraKey]!.map((p) => [...p] as Pt)
          : null;
        if (!img || img.length !== cur.map.length) {
          if (Hinv) {
            const projected: Pt[] = [];
            for (const p of cur.map) {
              const im = applyHomography(Hinv, p[0], p[1]);
              if (!im) {
                projected.length = 0;
                break;
              }
              projected.push(im);
            }
            img = projected.length === cur.map.length ? projected : cur.map.map(() => [...pt] as Pt);
          } else {
            img = cur.map.map(() => [...pt] as Pt);
          }
        }
        img[index] = pt;
        let map = cur.map.map((p) => [...p] as Pt);
        if (H) {
          const m = applyHomography(H, pt[0], pt[1]);
          if (m) map[index] = m;
        }
        nextCounter = {
          ...cur,
          map,
          image_by_camera: { ...cur.image_by_camera, [cameraKey]: img },
        };
      }

      const next: CountersDoc = {
        ...prev,
        counters: prev.counters.map((c) => (c.id === id ? nextCounter : c)),
      };
      countersRef.current = next;
      const dirtyNow = countersFingerprint(next) !== savedCountersFpRef.current;
      queueMicrotask(() => {
        setCountersDirty(dirtyNow);
        onCountersRef.current?.(next);
      });
      return next;
    });
    if (side === "map") {
      setMeasurePt(pt);
      setMeasureExcludeId(null);
    }
  }

  const publish = useCallback((next: HomographyDoc, pushHistory = true) => {
    if (pushHistory && docRef.current) {
      setHistory((h) => {
        const nextH = [...h.slice(-39), cloneDoc(docRef.current!)];
        historyRef.current = nextH;
        return nextH;
      });
      futureRef.current = [];
      setFuture([]);
    }
    docRef.current = next;
    setDoc(next);
    onHomoRef.current?.(next);
    setDirty(docFingerprint(next) !== savedFpRef.current);
    setCfg((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        cameras: prev.cameras.map((c) =>
          c.key === next.camera_key
            ? {
                ...c,
                pairs: next.pairs.length,
                hasH: Array.isArray(next.H) && next.H.length === 9,
                hasPlacement: !!normalizePlacement(next.placement),
                placement: normalizePlacement(next.placement),
                map_points: next.pairs.map((p, index) => ({ index, map: p.map as [number, number] })),
              }
            : c,
        ),
      };
    });
  }, []);

  function beginHistoryEdit(snapshot?: HomographyDoc | null) {
    if (historyBeforeEditRef.current) return;
    const src = snapshot ?? docRef.current;
    if (src) historyBeforeEditRef.current = cloneDoc(src);
  }

  /** Зафиксировать жест (drag): в историю — снимок ДО правки, текущий doc уже на экране. */
  function commitHistoryEdit(finalDoc: HomographyDoc) {
    const before = historyBeforeEditRef.current;
    historyBeforeEditRef.current = null;
    if (before && docFingerprint(before) !== docFingerprint(finalDoc)) {
      const nextH = [...historyRef.current.slice(-39), before];
      historyRef.current = nextH;
      setHistory(nextH);
      futureRef.current = [];
      setFuture([]);
    }
    publish(finalDoc, false);
  }

  function syncDirty(next: HomographyDoc | null) {
    if (!next) {
      setDirty(false);
      return;
    }
    setDirty(docFingerprint(next) !== savedFpRef.current);
  }

  function markClean(next: HomographyDoc) {
    savedFpRef.current = docFingerprint(next);
    setDirty(false);
  }

  const load = useCallback(async (isStale: () => boolean) => {
    try {
      const [maps, homo, cntRaw] = await Promise.all([
        fetchMapsConfig(),
        fetchHomography(cameraKey),
        fetchCounters().catch(() => emptyCountersDoc()),
      ]);
      if (isStale()) return;
      setCfg(maps);
      // По умолчанию procedural-сетка; старые floorplan.svg тоже переводим на grid
      const rawFp = homo.floorplan || maps.floorplan || GRID_FLOORPLAN;
      const fp =
        isGridFloorplan(rawFp) || /\.svg$/i.test(rawFp) || rawFp === "floorplan.png"
          ? GRID_FLOORPLAN
          : rawFp;
      setFloorplan(fp);
      if (isGridFloorplan(fp)) setMapNat(MAP_SIZE);
      const normalized: HomographyDoc = {
        ...emptyHomographyDoc(cameraKey, fp),
        ...homo,
        camera_key: cameraKey,
        floorplan: fp,
        map_size: isGridFloorplan(fp) ? MAP_SIZE : homo.map_size,
        placement: normalizePlacement(homo.placement),
      };
      docRef.current = normalized;
      setDoc(normalized);
      setHistory([]);
      setFuture([]);
      historyBeforeEditRef.current = null;
      markClean(normalized);
      const cnt = normalizeCountersDoc(cntRaw, fp);
      countersRef.current = cnt;
      setCounters(cnt);
      markCountersClean(cnt);
      onCountersRef.current?.(cnt);
      setDraftPts([]);
      setDraftSide(null);
      onFloorRef.current?.(isGridFloorplan(fp) ? "grid" : `/maps/${encodeURIComponent(fp)}`);
      onHomoRef.current?.(normalized);
      setError(null);
      lastHFpRef.current = normalized.H ? normalized.H.map((n) => n.toFixed(8)).join(",") : null;
      hBaselineSkipRef.current = true;
      const pl = normalized.placement;
      setStatus(
        `Камера ${cameraKey} · ${normalized.pairs.length} пар${normalized.H ? " · H" : ""}${
          pl ? ` · на плане ${pl.yaw_deg.toFixed(0)}°` : ""
        } · прилавков ${cnt.counters.length}`,
      );
    } catch (e) {
      if (isStale()) return;
      setError(e instanceof Error ? e.message : "Ошибка загрузки");
    }
  }, [cameraKey]);

  useEffect(() => {
    let cancelled = false;
    void load(() => cancelled);
    return () => {
      cancelled = true;
    };
  }, [load]);

  const applyFrame = useCallback((dataUrl: string, w: number, h: number, sec: number) => {
    setFrameUrl(dataUrl);
    setFrameSize([w, h]);
    setFrameSec(sec);
    setImgView(IDENTITY_VIEW);
    bumpLayout();
  }, []);

  const loadFirstFrame = useCallback(
    async (signal?: AbortSignal) => {
      const url = videoUrl;
      if (!url) {
        setStatus("Нет видео для этой камеры");
        return;
      }
      try {
        setStatus(`Загрузка 1-го кадра cam ${cameraKey}…`);
        const frame = await captureFirstFrame(url, signal);
        if (signal?.aborted) return;
        applyFrame(frame.dataUrl, frame.w, frame.h, frame.sec);
        setStatus(`Cam ${cameraKey} · первый кадр ${frame.w}×${frame.h}`);
      } catch (e) {
        if (signal?.aborted || (e instanceof DOMException && e.name === "AbortError")) return;
        const video = videoRef.current;
        if (video && video.videoWidth) {
          const c = document.createElement("canvas");
          c.width = video.videoWidth;
          c.height = video.videoHeight;
          c.getContext("2d")?.drawImage(video, 0, 0);
          applyFrame(c.toDataURL("image/jpeg", 0.9), c.width, c.height, video.currentTime);
          setStatus(`Cam ${cameraKey} · кадр из плеера`);
          return;
        }
        setError(e instanceof Error ? e.message : "Не удалось взять кадр");
      }
    },
    [videoUrl, cameraKey, videoRef, applyFrame],
  );

  useEffect(() => {
    const ac = new AbortController();
    void loadFirstFrame(ac.signal);
    return () => {
      ac.abort();
    };
  }, [loadFirstFrame]);

  const H = useMemo(() => {
    if (!doc) return null;
    // Для err/RMS/ghost всегда фит по текущим парам (не устаревший doc.H).
    if (doc.pairs.length >= 4) return computeHomography(doc.pairs);
    return null;
  }, [doc]);

  const errors = useMemo(() => reprojectionErrors(doc?.pairs ?? [], H), [doc?.pairs, H]);
  const rms = useMemo(() => rmsError(errors), [errors]);
  const hLooRms = useMemo(() => leaveOneOutRms(doc?.pairs ?? []), [doc?.pairs]);
  const Hinv = useMemo(() => (H ? invertHomography(H) : null), [H]);
  /** Порог «плохой» пары на плане: ~15 см (план 80 px = 0.5 м). */
  const errBadPx = useMemo(() => (useGrid ? METER_PX * 0.15 : 12), [useGrid]);
  const rmsOkPx = useMemo(() => (useGrid ? METER_PX * 0.08 : 8), [useGrid]);
  const rmsBadPx = useMemo(() => (useGrid ? METER_PX * 0.2 : 15), [useGrid]);

  const calibImageSize = useMemo((): [number, number] | null => {
    const cur = doc;
    if (!cur) return null;
    const fromDoc = cur.image_size as [number, number] | null | undefined;
    if (fromDoc && fromDoc[0] > 0 && fromDoc[1] > 0) return fromDoc;
    return frameSize ?? imageSize ?? null;
  }, [doc, frameSize, imageSize]);

  const rayStats = useMemo(() => {
    if (!doc?.placement || !doc.pairs.length || !calibImageSize) return null;
    const pose = normalizeCameraPose(doc.placement);
    if (!pose) return null;
    return rayPairStats(pose, doc.pairs, calibImageSize);
  }, [doc?.placement, doc?.pairs, calibImageSize]);

  const testRayMapPt = useMemo(() => {
    if (!testImagePt || !doc?.placement || !calibImageSize) return null;
    const pose = normalizeCameraPose(doc.placement);
    if (!pose) return null;
    return rayToGroundMap(testImagePt[0], testImagePt[1], pose, calibImageSize, { torsoHeightM: 0 });
  }, [testImagePt, doc?.placement, calibImageSize]);

  const testHvsRayGap = useMemo(() => {
    if (!testMapPt || !testRayMapPt) return null;
    return Math.hypot(testMapPt[0] - testRayMapPt[0], testMapPt[1] - testRayMapPt[1]);
  }, [testMapPt, testRayMapPt]);

  const qualityStatus = useMemo(() => {
    const n = doc?.pairs.length ?? 0;
    if (n < 4) {
      return {
        cls: "is-bad",
        badge: "Мало точек (<4)",
        hint: "Задайте минимум 4 пары точек «кадр ↔ план» (рекомендуется 6–10 точек, распределенных по полу).",
      };
    }
    const curRmsCm = rms != null ? (useGrid ? (rms / METER_PX) * 100 : rms) : 999;
    const looCm = hLooRms != null ? (useGrid ? (hLooRms / METER_PX) * 100 : hLooRms) : null;

    if (n < 6) {
      if (curRmsCm <= 25) {
        return {
          cls: "is-good",
          badge: "H построена (хорошо)",
          hint: "Гомография стабильна. Можно переходить к Шагу 3 (3D-Камера) или добавить 1–2 точки для LOO.",
        };
      }
      return {
        cls: "is-warn",
        badge: "Базовая H (нужно ≥6 точек)",
        hint: `H построена. Добавьте еще ${6 - n} пар(ы) для честной проверки устойчивости калибровки (LOO).`,
      };
    }

    // Для n >= 6 оцениваем LOO и RMS
    if (looCm != null && (looCm <= 24 || (curRmsCm <= 14 && looCm <= 32))) {
      return {
        cls: "is-good",
        badge: "Отличное",
        hint: "Калибровка высокоточная и устойчивая. Переходите к Шагу 3 (3D-Камера).",
      };
    }
    if (looCm != null && (looCm <= 48 || curRmsCm <= 28)) {
      return {
        cls: "is-good",
        badge: "Хорошее (рабочее)",
        hint: "Точность достаточна для стабильного трекинга и позиционирования людей на карте.",
      };
    }
    if (looCm != null && (looCm <= 75 || curRmsCm <= 42)) {
      return {
        cls: "is-warn",
        badge: "Приемлемое",
        hint: "Погрешность в пределах нормы для дальних зон зала. Проверьте крайние точки при необходимости.",
      };
    }
    return {
      cls: "is-bad",
      badge: "Проверьте пары",
      hint: "Обнаружена высокая погрешность. Обратите внимание на точки с красной ошибкой в списке ниже.",
    };
  }, [doc?.pairs.length, hLooRms, rms, useGrid]);



  function formatErrPx(err: number): string {
    if (useGrid) {
      const cm = (err / METER_PX) * 100;
      return `${err.toFixed(1)} · ${cm.toFixed(0)} см`;
    }
    return err.toFixed(2);
  }

  const hBaselineSkipRef = useRef(true);
  const lastHFpRef = useRef<string | null>(null);

  useEffect(() => {
    hBaselineSkipRef.current = true;
    lastHFpRef.current = null;
  }, [cameraKey]);

  useEffect(() => {
    if (!H) return;
    const fp = H.map((n) => n.toFixed(8)).join(",");
    if (hBaselineSkipRef.current) {
      hBaselineSkipRef.current = false;
      lastHFpRef.current = fp;
      return;
    }
    if (lastHFpRef.current === fp) return;
    lastHFpRef.current = fp;

    setCounters((prev) => {
      const { doc: next, updated } = reprojectCountersFromImage(prev, cameraKey, H);
      if (!updated) return prev;
      countersRef.current = next;
      const dirtyNow = countersFingerprint(next) !== savedCountersFpRef.current;
      queueMicrotask(() => {
        setCountersDirty(dirtyNow);
        onCountersRef.current?.(next);
        setStatus(`Прилавки: пересчитано ${updated} по новой H (кадр → план)`);
      });
      return next;
    });
  }, [H, cameraKey]);

  function withRecomputed(base: HomographyDoc, pairs: HomoPair[]): HomographyDoc {
    const nextH = pairs.length >= 4 ? computeHomography(pairs) : null;
    return {
      ...base,
      camera_key: cameraKey,
      floorplan,
      image_size: frameSize ?? imageSize,
      map_size: mapNat,
      pairs,
      H: nextH,
      placement: normalizePlacement(base.placement),
    };
  }

  function withPlacement(base: HomographyDoc, placement: CameraPlacement | null): HomographyDoc {
    return {
      ...withRecomputed(base, base.pairs),
      placement: normalizePlacement(placement),
    };
  }

  function refreshFrame() {
    void loadFirstFrame();
  }

  function captureFromPlayer() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) {
      setStatus("Нет кадра в плеере — включите видео и поставьте на нужную секунду");
      return;
    }
    const c = document.createElement("canvas");
    c.width = video.videoWidth;
    c.height = video.videoHeight;
    c.getContext("2d")?.drawImage(video, 0, 0);
    applyFrame(c.toDataURL("image/jpeg", 0.9), c.width, c.height, video.currentTime);
    setStatus(`Cam ${cameraKey} · кадр из плеера @ ${video.currentTime.toFixed(1)} с`);
  }

  function undo() {
    const hist = historyRef.current;
    const cur = docRef.current;
    if (!hist.length || !cur) return;
    const prev = cloneDoc(hist[hist.length - 1]!);
    const nextHist = hist.slice(0, -1);
    const nextFut = [cloneDoc(cur), ...futureRef.current].slice(0, 40);
    historyRef.current = nextHist;
    futureRef.current = nextFut;
    setHistory(nextHist);
    setFuture(nextFut);
    historyBeforeEditRef.current = null;
    publish(prev, false);
    setStatus("Undo");
  }

  function redo() {
    const fut = futureRef.current;
    const cur = docRef.current;
    if (!fut.length || !cur) return;
    const next = cloneDoc(fut[0]!);
    const nextFut = fut.slice(1);
    const nextHist = [...historyRef.current, cloneDoc(cur)];
    historyRef.current = nextHist;
    futureRef.current = nextFut;
    setFuture(nextFut);
    setHistory(nextHist);
    historyBeforeEditRef.current = null;
    publish(next, false);
    setStatus("Redo");
  }

  useEffect(() => {
    const imgEl = imageWrapRef.current;
    const mapEl = mapWrapRef.current;
    const onImg = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (!imgEl) return;
      const rect = imgEl.getBoundingClientRect();
      const ox = e.clientX - rect.left - rect.width / 2;
      const oy = e.clientY - rect.top - rect.height / 2;
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      setImgView((v) => {
        const nextScale = Math.min(8, Math.max(0.4, v.scale * factor));
        const k = nextScale / v.scale;
        return {
          ...v,
          scale: nextScale,
          tx: ox - (ox - v.tx) * k,
          ty: oy - (oy - v.ty) * k,
        };
      });
    };
    const onMap = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (!mapEl) return;
      const rect = mapEl.getBoundingClientRect();
      const ox = e.clientX - rect.left - rect.width / 2;
      const oy = e.clientY - rect.top - rect.height / 2;
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      setMapView((v) => {
        const nextScale = Math.min(8, Math.max(0.4, v.scale * factor));
        const k = nextScale / v.scale;
        return {
          ...v,
          scale: nextScale,
          tx: ox - (ox - v.tx) * k,
          ty: oy - (oy - v.ty) * k,
        };
      });
    };
    imgEl?.addEventListener("wheel", onImg, { passive: false });
    mapEl?.addEventListener("wheel", onMap, { passive: false });
    return () => {
      imgEl?.removeEventListener("wheel", onImg);
      mapEl?.removeEventListener("wheel", onMap);
    };
  }, [frameUrl, floorplan, layoutTick]);

  function onPanePointerDown(side: "image" | "map", e: React.PointerEvent) {
    if (e.button === 1 || e.altKey || e.buttons === 4) {
      e.preventDefault();
      panRef.current = {
        side,
        x: e.clientX,
        y: e.clientY,
        view: side === "image" ? imgView : mapView,
      };
      (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
      return;
    }
  }

  function onPanePointerMove(e: React.PointerEvent) {
    const pan = panRef.current;
    if (pan) {
      const dx = e.clientX - pan.x;
      const dy = e.clientY - pan.y;
      const next = { ...pan.view, tx: pan.view.tx + dx, ty: pan.view.ty + dy };
      if (pan.side === "image") setImgView(next);
      else setMapView(next);
      return;
    }
    const cdrag = counterDragRef.current;
    if (cdrag) {
      const wrap = cdrag.side === "image" ? imageWrapRef.current : mapWrapRef.current;
      if (!wrap) return;
      const natW = cdrag.side === "image" ? iw : mapNat[0];
      const natH = cdrag.side === "image" ? ih : mapNat[1];
      const view = cdrag.side === "image" ? imgView : mapView;
      const raw = clientToImage(wrap, natW, natH, view, e.clientX, e.clientY);
      if (!raw) return;
      cdrag.moved = true;
      if (cdrag.id === "__draft__") {
        setDraftPts((prev) => prev.map((q, j) => (j === cdrag.index ? raw : q)));
      } else {
        moveCounterVertex(cdrag.id, cdrag.side, cdrag.index, raw);
      }
      return;
    }
    const place = placeDragRef.current;
    if (place && doc) {
      const wrap = mapWrapRef.current;
      if (!wrap) return;
      const raw = clientToImage(wrap, mapNat[0], mapNat[1], mapView, e.clientX, e.clientY);
      if (!raw) return;
      let nextPl: CameraPlacement;
      if (place.kind === "move") {
        nextPl = { ...place.origin, position: mapPoint(raw) };
        place.origin = nextPl;
        setMeasurePt(nextPl.position);
        setMeasureExcludeId(`cam:${cameraKey}`);
      } else {
        // aim / place: позиция фиксирована, крутим yaw (без snap направления)
        nextPl = {
          ...place.origin,
          yaw_deg: yawFromPoints(place.origin.position, raw),
        };
        if (place.kind === "place") {
          setMeasurePt(place.origin.position);
          setMeasureExcludeId(`cam:${cameraKey}`);
        } else {
          setMeasurePt(null);
        }
      }
      const next = withPlacement(doc, nextPl);
      docRef.current = next;
      setDoc(next);
      onHomoRef.current?.(next);
      syncDirty(next);
      return;
    }
    const drag = dragRef.current;
    if (!drag || !doc) return;
    const wrap = drag.side === "image" ? imageWrapRef.current : mapWrapRef.current;
    if (!wrap) return;
    const natW = drag.side === "image" ? iw : mapNat[0];
    const natH = drag.side === "image" ? ih : mapNat[1];
    const view = drag.side === "image" ? imgView : mapView;
    const raw = clientToImage(wrap, natW, natH, view, e.clientX, e.clientY);
    if (!raw) return;
    const pt = drag.side === "map" ? mapPoint(raw) : raw;
    if (drag.side === "map") {
      setMeasurePt(pt);
      setMeasureExcludeId(`${cameraKey}:${drag.pair}`);
    }
    const pairs = doc.pairs.map((p, i) => {
      if (i !== drag.pair) return p;
      return drag.side === "image" ? { ...p, image: pt } : { ...p, map: pt };
    });
    const next = withRecomputed(doc, pairs);
    docRef.current = next;
    setDoc(next);
    onHomoRef.current?.(next);
    syncDirty(next);
  }

  function onPanePointerUp() {
    if (counterDragRef.current) {
      if (counterDragRef.current.moved) {
        suppressClickRef.current = true;
        setStatus("Вершина прилавка сдвинута — сохраните");
      }
      counterDragRef.current = null;
      setMeasurePt(null);
      setMeasureExcludeId(null);
      return;
    }
    if (placeDragRef.current && docRef.current) {
      const finalDoc = docRef.current;
      commitHistoryEdit(cloneDoc(finalDoc));
      placeDragRef.current = null;
      setMeasurePt(null);
      setMeasureExcludeId(null);
      setStatus(
        finalDoc.placement
          ? `Камера: (${finalDoc.placement.position[0].toFixed(0)}, ${finalDoc.placement.position[1].toFixed(0)}) · ${finalDoc.placement.yaw_deg.toFixed(0)}° · FOV ${finalDoc.placement.fov_deg}°`
          : "Камера",
      );
      return;
    }
    if (dragRef.current && docRef.current) {
      commitHistoryEdit(cloneDoc(docRef.current));
    } else {
      historyBeforeEditRef.current = null;
    }
    dragRef.current = null;
    panRef.current = null;
    setMeasurePt(null);
    setMeasureExcludeId(null);
  }

  function onImageClick(e: React.MouseEvent) {
    if (e.button !== 0 || e.altKey) return;
    if (suppressClickRef.current) {
      suppressClickRef.current = false;
      return;
    }
    if (panRef.current || dragRef.current || placeDragRef.current || counterDragRef.current) return;
    if (mode === "place") return;
    const wrap = imageWrapRef.current;
    if (!wrap || !frameUrl) {
      setStatus("Дождитесь загрузки первого кадра");
      return;
    }
    const pt = clientToImage(wrap, iw, ih, imgView, e.clientX, e.clientY);
    if (!pt) {
      setStatus("Клик вне кадра");
      return;
    }

    if (mode === "count") {
      setTileMarks((prev) => {
        const next = [...prev, pt];
        queueMicrotask(() => setStatus(`Плитка #${next.length} на кадре`));
        return next;
      });
      bumpLayout();
      return;
    }
    if (mode === "draw") {
      addDraftPoint("image", pt);
      return;
    }
    if (mode === "test") {
      setTestImagePt(pt);
      const hMap = H ? applyHomography(H, pt[0], pt[1]) : null;
      setTestMapPt(hMap);
      if (!H) {
        setStatus("Нужна H (≥4 пар)");
        return;
      }
      const pose = doc?.placement ? normalizeCameraPose(doc.placement) : null;
      const imgSz = calibImageSize;
      const rayMap =
        pose && imgSz ? rayToGroundMap(pt[0], pt[1], pose, imgSz, { torsoHeightM: 0 }) : null;
      const gap = hMap && rayMap ? Math.hypot(hMap[0] - rayMap[0], hMap[1] - rayMap[1]) : null;
      if (gap != null) {
        const cm = useGrid ? ((gap / METER_PX) * 100).toFixed(0) : null;
        setStatus(
          cm != null
            ? `Тест: H ↔ 3D ${formatErrPx(gap)} (${cm} см) — подгоните «Подобрать 3D»`
            : `Тест: H ↔ 3D ${gap.toFixed(0)} px — подгоните «Подобрать 3D»`,
        );
      } else {
        setStatus("Тест: кадр → план (нет placement для 3D)");
      }
      return;
    }
    if (mode !== "pairs") return;
    if (pendingImage) return;
    setPendingImage(pt);
    setStatus(`Точка кадра (${pt[0].toFixed(0)}, ${pt[1].toFixed(0)}) — кликните соответствие на плане`);
  }

  function hitPlacement(pt: Pt, pl: CameraPlacement, mapW: number): "move" | "aim" | null {
    const [x, y] = pl.position;
    const rBody = Math.max(14, mapW / 70);
    const range = Math.max(120, mapW * 0.09);
    const yaw = (pl.yaw_deg * Math.PI) / 180;
    const tip: Pt = [x + Math.cos(yaw) * range * 0.85, y + Math.sin(yaw) * range * 0.85];
    if (Math.hypot(pt[0] - tip[0], pt[1] - tip[1]) < rBody * 1.2) return "aim";
    if (Math.hypot(pt[0] - x, pt[1] - y) < rBody) return "move";
    return null;
  }

  function onMapPointerDown(e: React.PointerEvent) {
    onPanePointerDown("map", e);
    if (panRef.current || mode !== "place" || e.button !== 0 || e.altKey) return;
    const wrap = mapWrapRef.current;
    if (!wrap) return;
    const raw = clientToImage(wrap, mapNat[0], mapNat[1], mapView, e.clientX, e.clientY);
    if (!raw) return;
    const pt = mapPoint(raw);
    e.preventDefault();
    e.stopPropagation();
    const base = doc ?? emptyHomographyDoc(cameraKey, floorplan);
    const cur = normalizePlacement(base.placement);
    if (cur) {
      const hit = hitPlacement(raw, cur, mapNat[0]); // hit-test по сырому курсору
      if (hit === "move") {
        beginHistoryEdit(base);
        placeDragRef.current = { kind: "move", origin: { ...cur, position: [...cur.position] as Pt } };
        setMeasurePt(cur.position);
        setMeasureExcludeId(`cam:${cameraKey}`);
        (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
        setStatus("Перемещение камеры…");
        return;
      }
      if (hit === "aim") {
        beginHistoryEdit(base);
        placeDragRef.current = { kind: "aim", origin: { ...cur } };
        (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
        setStatus("Поворот направления…");
        return;
      }
    }
    beginHistoryEdit(base);
    const nextPl = defaultPlacement(pt, cur?.yaw_deg ?? 0, cur?.fov_deg ?? DEFAULT_FOV);
    const next = withPlacement(base, nextPl);
    docRef.current = next;
    setDoc(next);
    onHomoRef.current?.(next);
    syncDirty(next);
    placeDragRef.current = { kind: "place", origin: { ...nextPl } };
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    setStatus(`Камера @ ${gridLabel(pt)} — тяните направление`);
  }

  function onMapClick(e: React.MouseEvent) {
    if (e.button !== 0 || e.altKey) return;
    if (suppressClickRef.current) {
      suppressClickRef.current = false;
      return;
    }
    if (panRef.current || dragRef.current || placeDragRef.current || counterDragRef.current) return;
    if (mode === "place") return; // жест через pointerdown/up
    const wrap = mapWrapRef.current;
    if (!wrap) return;
    const raw = clientToImage(wrap, mapNat[0], mapNat[1], mapView, e.clientX, e.clientY);
    if (!raw) return;

    if (mode === "draw") {
      addDraftPoint("map", raw);
      return;
    }
    const pt = mapPoint(raw);

    if (mode === "test") {
      setTestMapPt(pt);
      const imgPt = Hinv ? applyHomography(Hinv, pt[0], pt[1]) : null;
      setTestImagePt(imgPt);
      if (!Hinv || !imgPt) {
        setStatus("Нужна H (≥4 пар)");
        return;
      }
      const pose = doc?.placement ? normalizeCameraPose(doc.placement) : null;
      const imgSz = calibImageSize;
      const rayMap =
        pose && imgSz ? rayToGroundMap(imgPt[0], imgPt[1], pose, imgSz, { torsoHeightM: 0 }) : null;
      const gap = rayMap ? Math.hypot(pt[0] - rayMap[0], pt[1] - rayMap[1]) : null;
      if (gap != null) {
        const cm = useGrid ? ((gap / METER_PX) * 100).toFixed(0) : null;
        setStatus(
          cm != null
            ? `Тест: H ↔ 3D ${formatErrPx(gap)} (${cm} см)`
            : `Тест: H ↔ 3D ${gap.toFixed(0)} px`,
        );
      } else {
        setStatus(`Тест: план ${gridLabel(pt)} → кадр`);
      }
      return;
    }
    if (mode !== "pairs") return;
    if (!pendingImage) {
      setStatus("Сначала кликните точку на кадре (или тяните существующую пару)");
      return;
    }
    const pair: HomoPair = { image: pendingImage, map: pt };
    const base = doc ?? emptyHomographyDoc(cameraKey, floorplan);
    const next = withRecomputed(base, [...base.pairs, pair]);
    publish(next);
    setPendingImage(null);
    setSelectedPair(next.pairs.length - 1);
    setStatus(`Пара #${next.pairs.length} · ${gridLabel(pt)}`);
  }

  function startDragPair(i: number, side: "image" | "map", e: React.PointerEvent) {
    if (mode !== "pairs" || e.button !== 0 || e.altKey) return;
    e.stopPropagation();
    e.preventDefault();
    beginHistoryEdit();
    dragRef.current = { pair: i, side };
    setSelectedPair(i);
    if (side === "map" && doc?.pairs[i]) {
      setMeasurePt(doc.pairs[i]!.map);
      setMeasureExcludeId(`${cameraKey}:${i}`);
    }
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
  }

  function removePair(i: number) {
    if (!doc) return;
    const pairs = doc.pairs.filter((_, idx) => idx !== i);
    publish(withRecomputed(doc, pairs));
    setSelectedPair(null);
    setHoverPair(null);
  }

  function clearPlacement() {
    if (!doc) return;
    publish(withPlacement(doc, null));
    setStatus("Размещение камеры снято");
  }

  function setFov(fov: number) {
    if (!doc?.placement) return;
    publish(withPlacement(doc, { ...doc.placement, fov_deg: fov }));
  }

  function setHeightM(height_m: number) {
    if (!doc?.placement) return;
    publish(withPlacement(doc, { ...doc.placement, height_m }));
  }

  function setPitchDeg(pitch_deg: number) {
    if (!doc?.placement) return;
    publish(withPlacement(doc, { ...doc.placement, pitch_deg }));
  }

  function runEstimateRayPose() {
    const cur = docRef.current;
    if (!cur?.placement || !cur.pairs.length) {
      setStatus("3D: нужны placement и пары калибровки");
      return;
    }
    const imgSz: [number, number] | null =
      (cur.image_size as [number, number] | null) ?? frameSize ?? imageSize;
    if (!imgSz) {
      setStatus("3D: нет размера кадра");
      return;
    }
    const pose = normalizeCameraPose(cur.placement);
    if (!pose) return;
    const before = pose.position;
    const est = fitRayPose(pose, cur.pairs, imgSz, { fitPose: fitMoveCamera });
    if (!est) {
      setStatus("3D: не удалось подобрать pose");
      return;
    }
    const shift = Math.hypot(est.position[0] - before[0], est.position[1] - before[1]);
    let dYaw = ((est.yaw_deg - pose.yaw_deg) % 360 + 360) % 360;
    if (dYaw > 180) dYaw -= 360;
    publish(
      withPlacement(cur, {
        ...cur.placement,
        height_m: est.height_m,
        pitch_deg: est.pitch_deg,
        fov_deg: est.fov_deg,
        yaw_deg: est.yaw_deg,
        position: est.position,
      }),
    );
    const cover = `${est.projected}/${est.total}`;
    setStatus(
      `3D: h ${est.height_m.toFixed(2)} м · pitch ${est.pitch_deg.toFixed(0)}° · FOV ${est.fov_deg.toFixed(0)}° · yaw ${dYaw >= 0 ? "+" : ""}${dYaw.toFixed(1)}° · сдвиг ${shift.toFixed(0)} px · RMS ${est.rmsPx.toFixed(0)} px (${cover})`,
    );
  }

  function runAutoCalibrate() {
    const cur = docRef.current;
    if (!cur || cur.pairs.length < 4) {
      setStatus("Авто: нужно ≥4 пар");
      return;
    }
    const imgSz: [number, number] | null =
      frameSize ?? imageSize ?? (cur.image_size as [number, number] | null);
    const maxErrPx = useGrid ? METER_PX * 0.2 : 40; // ~20 см на сетке
    const result = autoCalibrate(cur.pairs, imgSz, {
      snapMap: useGrid ? snapToGrid : undefined,
      maxErrPx,
      minPairs: 4,
      fov_deg: cur.placement?.fov_deg ?? DEFAULT_FOV,
    });
    if (!result.H || result.pairs.length < 4) {
      setStatus("Авто: не удалось получить H — проверьте пары");
      return;
    }
    const next = withPlacement(
      withRecomputed(cur, result.pairs),
      result.placement ?? cur.placement,
    );
    publish(next);
    setSelectedPair(null);
    const rmsB = result.rmsBefore != null ? result.rmsBefore.toFixed(1) : "—";
    const rmsA = result.rmsAfter != null ? result.rmsAfter.toFixed(1) : "—";
    setStatus(
      `Авто: snap ${result.snapped}, убрано ${result.removed}, пар ${result.pairs.length}, RMS ${rmsB}→${rmsA} px` +
        (result.placement ? " · камера на плане" : ""),
    );
  }

  function runAutoPlaceFromH() {
    const cur = docRef.current;
    const liveH =
      cur && cur.pairs.length >= 4
        ? computeHomography(cur.pairs)
        : cur?.H && cur.H.length === 9
          ? (cur.H as Mat3)
          : H;
    if (!cur || !liveH) {
      setStatus("План: нужна H (≥4 пар калибровки)");
      return;
    }
    const imgSz: [number, number] | null =
      frameSize ?? imageSize ?? (cur.image_size as [number, number] | null);
    if (!imgSz) {
      setStatus("План: нет размера кадра");
      return;
    }
    const pl = estimatePlacementFromHomography(liveH, imgSz, {
      fov_deg: cur.placement?.fov_deg ?? DEFAULT_FOV,
      snap: useGrid ? snapToGrid : undefined,
    });
    if (!pl) {
      setStatus("План: не удалось оценить позицию камеры");
      return;
    }
    publish(withPlacement(cur, pl));
    setStatus(
      `План: (${pl.position[0].toFixed(0)}, ${pl.position[1].toFixed(0)}) · yaw ${pl.yaw_deg.toFixed(0)}°`,
    );
  }

  function addDraftPoint(side: "image" | "map", pt: Pt) {
    if (draftSide && draftSide !== side) {
      setStatus(`Черновик на ${draftSide === "map" ? "плане" : "кадре"} — замкните или Esc`);
      return;
    }
    setDraftSide(side);
    setDraftPts((prev) => {
      const next = [...prev, pt];
      setStatus(`Прилавок: вершина ${next.length} (${side === "map" ? "план" : "кадр"}) · Enter — замкнуть`);
      return next;
    });
  }

  function undoDraftVertex() {
    setDraftPts((prev) => {
      const next = prev.slice(0, -1);
      if (!next.length) setDraftSide(null);
      setStatus(next.length ? `Вершин: ${next.length}` : "Черновик очищен");
      return next;
    });
  }

  function finishCounterDraft() {
    const countersNow = countersRef.current;
    if (draftPts.length < 3 || !draftSide) {
      setStatus("Нужно ≥3 вершины");
      return;
    }
    let mapPts: Pt[];
    let imagePts: Pt[] | null = null;

    if (draftSide === "map") {
      mapPts = draftPts.map((p) => [...p] as Pt);
    } else {
      imagePts = draftPts;
      if (!H) {
        setStatus("Для переноса с кадра на план нужна H (≥4 пар). Сначала калибровка.");
        return;
      }
      const projected: Pt[] = [];
      for (const p of draftPts) {
        const m = applyHomography(H, p[0], p[1]);
        if (!m) {
          setStatus("Не удалось спроецировать точку на план");
          return;
        }
        projected.push(m);
      }
      mapPts = projected;
    }

    const id = newCounterId(countersNow.counters);
    const counter: CounterPoly = {
      id,
      name: `Прилавок ${countersNow.counters.length + 1}`,
      map: mapPts,
      ...(imagePts
        ? { image_by_camera: { [cameraKey]: imagePts } }
        : {}),
    };
    const next: CountersDoc = {
      ...countersNow,
      floorplan,
      map_size: mapNat,
      counters: [...countersNow.counters, counter],
    };
    publishCounters(next);
    setSelectedCounterId(id);
    setDraftPts([]);
    setDraftSide(null);
    setStatus(
      `Добавлен ${counter.name} (${mapPts.length} вершин)${imagePts ? " · кадр+план" : " · план"} — сохраните`,
    );
  }

  function deleteSelectedCounter() {
    if (!selectedCounterId) return;
    const countersNow = countersRef.current;
    const next: CountersDoc = {
      ...countersNow,
      counters: countersNow.counters.filter((c) => c.id !== selectedCounterId),
    };
    publishCounters(next);
    setSelectedCounterId(null);
    setStatus("Прилавок удалён");
  }

  async function persistCounters() {
    try {
      const payload: CountersDoc = {
        ...countersRef.current,
        floorplan,
        map_size: mapNat,
      };
      const saved = normalizeCountersDoc(await saveCounters(payload), floorplan);
      countersRef.current = saved;
      setCounters(saved);
      markCountersClean(saved);
      onCountersRef.current?.(saved);
      setStatus(`Сохранено data/maps/counters.json · ${saved.counters.length} прилавков`);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Ошибка сохранения прилавков");
    }
  }

  async function persistAll() {
    let savedAny = false;
    if (countersDirtyRef.current || countersRef.current) {
      await persistCounters();
      savedAny = true;
    }
    if (dirtyRef.current || docRef.current) {
      await persist();
      savedAny = true;
    }
    if (savedAny) {
      setDirty(false);
      setCountersDirty(false);
      onDirtyRef.current?.(false);
    }
  }

  async function triggerFeetRecalc() {
    setIsRecalculatingFeet(true);
    setStatus("Запущен пересчёт stage feet через API...");
    try {
      const targetSession = sessionKey || videoName;
      const res = await runFeetApi({ session: targetSession, camera: cameraKey });
      if (res.success) {
        setStatus(`Калибровка сохранена · stage feet успешно пересчитан через API (${new Date().toLocaleTimeString()})`);
        await onFeetReload?.();
      } else {
        setStatus(`Калибровка сохранена · ошибка API feet: ${res.error || res.output || "неизвестно"}`);
      }
    } catch (err) {
      setStatus(`Калибровка сохранена · сбой запроса API: ${String(err)}`);
    } finally {
      setIsRecalculatingFeet(false);
    }
  }

  async function persist() {
    if (!docRef.current) return;
    try {
      const payload = withRecomputed(docRef.current, docRef.current.pairs);
      const saved = await saveHomography(cameraKey, payload);
      docRef.current = saved;
      setDoc(saved);
      onHomoRef.current?.(saved);
      markClean(saved);
      setError(null);
      // обновить статусы камер в полоске
      try {
        const maps = await fetchMapsConfig();
        setCfg(maps);
      } catch {
        /* ignore */
      }
      await triggerFeetRecalc();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Ошибка сохранения");
    }
  }

  function exportJson() {
    if (!doc) return;
    const blob = new Blob([JSON.stringify(withRecomputed(doc, doc.pairs), null, 2)], {
      type: "application/json",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `homography_${cameraKey}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function importJson(file: File) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result)) as HomographyDoc;
        if (!Array.isArray(parsed.pairs)) throw new Error("нет pairs");
        const next = withRecomputed(
          {
            ...emptyHomographyDoc(cameraKey, floorplan),
            ...parsed,
            camera_key: cameraKey,
            floorplan,
            placement: normalizePlacement(parsed.placement),
          },
          parsed.pairs,
        );
        publish(next);
        setStatus(`Импортировано ${next.pairs.length} пар`);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Плохой JSON");
      }
    };
    reader.readAsText(file);
  }

  useEffect(() => {
    const onResize = () => bumpLayout();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    bumpLayout();
  }, [imgView, mapView, frameUrl, floorplan]);

  useEffect(() => {
    const canvas = mapGridCanvasRef.current;
    if (!canvas || !useGrid) return;
    const [mw, mh] = MAP_SIZE;
    if (canvas.width !== mw || canvas.height !== mh) {
      canvas.width = mw;
      canvas.height = mh;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    drawFloorGrid(ctx, mw, mh);
    bumpLayout();
  }, [useGrid, floorplan]);

  useEffect(() => {
    const canvas = mapCamCanvasRef.current;
    if (!canvas) return;
    const [mw, mh] = mapNat;
    if (canvas.width !== mw || canvas.height !== mh) {
      canvas.width = mw;
      canvas.height = mh;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, mw, mh);

    const active = normalizePlacement(doc?.placement);
    for (const cam of cfg?.cameras ?? []) {
      if (cam.key === cameraKey) continue;
      const pl = normalizePlacement(cam.placement);
      if (!pl) continue;
      drawCameraPlacement(ctx, pl, `cam ${cam.key}`, {
        active: false,
        dimmed: true,
        mapW: mw,
        cameraKey: cam.key,
        body: false,
        label: false,
      });
    }
    if (active) {
      drawCameraPlacement(ctx, active, `cam ${cameraKey}`, {
        active: true,
        mapW: mw,
        cameraKey,
        body: false,
        label: false,
      });
    }
  }, [cfg, doc?.placement, cameraKey, mapNat]);

  function rotateMap(delta: number) {
    setMapView((v) => {
      const rot = normRot((v.rot || 0) + delta);
      setStatus(`План: поворот ${rot}°`);
      return { ...v, rot };
    });
    bumpLayout();
  }

  useEffect(() => {
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if (!dirtyRef.current && !countersDirtyRef.current) return;
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, []);

  const keyHandlersRef = useRef({
    mode: mode as Mode,
    draftPtsLen: 0,
    tileMarksLen: 0,
    selectedCounterId: null as string | null,
    selectedPair: null as number | null,
    undo: () => {},
    redo: () => {},
    persistAll: async () => {},
    finishCounterDraft: () => {},
    deleteSelectedCounter: () => {},
    undoDraftVertex: () => {},
    removePair: (_i: number) => {},
    rotateMap: (_d: number) => {},
    refreshFrame: () => {},
    captureFromPlayer: () => {},
  });
  keyHandlersRef.current = {
    mode,
    draftPtsLen: draftPts.length,
    tileMarksLen: tileMarks.length,
    selectedCounterId,
    selectedPair,
    undo,
    redo,
    persistAll,
    finishCounterDraft,
    deleteSelectedCounter,
    undoDraftVertex,
    removePair,
    rotateMap,
    refreshFrame,
    captureFromPlayer,
  };

  useEffect(() => {
    function switchMode(id: Mode) {
      setMode(id);
      setPendingImage(null);
      setStatus(MODE_HINT[id]);
    }
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement)
        return;
      const h = keyHandlersRef.current;
      const mod = e.metaKey || e.ctrlKey;
      const key = e.key.toLowerCase();

      if (mod && key === "z") {
        e.preventDefault();
        if (e.shiftKey) h.redo();
        else h.undo();
        return;
      }
      if (mod && key === "y") {
        e.preventDefault();
        h.redo();
        return;
      }
      if (mod && key === "s") {
        e.preventDefault();
        void h.persistAll();
        return;
      }
      if (e.key === "Escape") {
        setPendingImage(null);
        setTestImagePt(null);
        setTestMapPt(null);
        if (h.mode === "draw" && h.draftPtsLen) {
          setDraftPts([]);
          setDraftSide(null);
          setStatus("Черновик прилавка сброшен");
          return;
        }
        if (h.mode === "count" && h.tileMarksLen) {
          setTileMarks((m) => m.slice(0, -1));
          setStatus("Убрана последняя плитка");
          return;
        }
        setStatus("Сброс режима ввода");
        return;
      }
      if (e.key === "Enter" && h.mode === "draw") {
        e.preventDefault();
        h.finishCounterDraft();
        return;
      }
      if (e.key === "Delete" || e.key === "Backspace") {
        if (h.mode === "draw" && h.draftPtsLen) {
          e.preventDefault();
          h.undoDraftVertex();
          return;
        }
        if (h.mode === "draw" && h.selectedCounterId) {
          e.preventDefault();
          h.deleteSelectedCounter();
          return;
        }
        if (h.mode === "count" && h.tileMarksLen) {
          e.preventDefault();
          setTileMarks((m) => m.slice(0, -1));
          setStatus("Убрана последняя плитка");
          return;
        }
        if (h.selectedPair != null) {
          e.preventDefault();
          h.removePair(h.selectedPair);
        }
        return;
      }
      if (mod || e.altKey) return;

      if (e.key === "1") {
        e.preventDefault();
        switchMode("pairs");
      } else if (e.key === "2") {
        e.preventDefault();
        switchMode("test");
      } else if (e.key === "3") {
        e.preventDefault();
        switchMode("place");
      } else if (e.key === "4") {
        e.preventDefault();
        switchMode("count");
      } else if (e.key === "5") {
        e.preventDefault();
        switchMode("draw");
      } else if (key === "r") {
        e.preventDefault();
        setImgView(IDENTITY_VIEW);
        setMapView(IDENTITY_VIEW);
        setStatus("Вид сброшен");
      } else if (key === "f") {
        e.preventDefault();
        if (e.shiftKey) h.refreshFrame();
        else h.captureFromPlayer();
      } else if (e.key === "[") {
        e.preventDefault();
        h.rotateMap(-45);
      } else if (e.key === "]") {
        e.preventDefault();
        h.rotateMap(45);
      } else if (key === "g" && h.mode === "pairs") {
        e.preventDefault();
        setShowGrid((v) => !v);
      } else if (key === "p" && h.mode === "pairs") {
        e.preventDefault();
        setShowReproj((v) => !v);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  /** Режимы, где точки калибровки не трогаем мышью (но всё равно рисуем). */
  const dotsPassive = mode === "count" || mode === "test" || mode === "place" || mode === "draw";

  function renderDots(side: "image" | "map", natW: number, natH: number) {
    if (natW < 1 || natH < 1) return null;
    const rot = side === "map" ? mapView.rot || 0 : 0;
    const hud = markerHudScale(side === "image" ? imgView.scale : mapView.scale);
    const hudStyle = {
      ["--dot-hud" as string]: String(hud),
      ["--dot-rot" as string]: `${-rot}deg`,
    };

    if (side === "image") {
      if (!doc) return null;
      return doc.pairs.map((p, i) => {
        const err = errors[i]?.errPx;
        const hot = hoverPair === i || selectedPair === i;
        const bad = Number.isFinite(err) && err! > errBadPx;
        const passive = dotsPassive;
        const common = {
          key: `image-${i}`,
          className: `map-calib-dot${hot ? " is-hot" : ""}${bad ? " is-bad" : ""}${mode === "count" ? " is-dim" : ""}`,
          anchor: ptPct(p.image, natW, natH, { interactive: !passive }),
          title: Number.isFinite(err)
            ? `cam ${cameraKey} · #${i + 1} err ${err!.toFixed(1)} px`
            : `cam ${cameraKey} · #${i + 1}`,
        };
        if (passive) {
          return (
            <span
              key={common.key}
              style={{ ...common.anchor, pointerEvents: "auto" }}
              title={common.title}
              onMouseEnter={() => setHoverPair(i)}
              onMouseLeave={() => setHoverPair(null)}
            >
              <span
                className={common.className}
                style={{ ...hudStyle, background: colorForCameraKey(cameraKey) }}
              >
                {cameraKey}·{i + 1}
              </span>
            </span>
          );
        }
        return (
          <span key={common.key} style={common.anchor} title={common.title}>
            <button
              type="button"
              className={common.className}
              style={{
                ...hudStyle,
                cursor: mode === "pairs" ? "grab" : "pointer",
                background: colorForCameraKey(cameraKey),
              }}
              onMouseEnter={() => setHoverPair(i)}
              onMouseLeave={() => setHoverPair(null)}
              onClick={(e) => {
                if (e.button !== 0 || e.altKey) return;
                e.stopPropagation();
                setSelectedPair(i);
              }}
              onPointerDown={(e) => startDragPair(i, side, e)}
            >
              {cameraKey}·{i + 1}
            </button>
          </span>
        );
      });
    }

    // План: точки всех камер — всегда
    const nodes: ReactNode[] = [];
    for (const cam of cfg?.cameras ?? []) {
      if (cam.key === cameraKey) continue;
      const color = colorForCameraKey(cam.key);
      for (const p of cam.map_points ?? []) {
        nodes.push(
          <span
            key={`map-${cam.key}-${p.index}`}
            style={ptPct(p.map, natW, natH, { interactive: false })}
            title={`cam ${cam.key} · #${p.index + 1}`}
          >
            <span className="map-calib-dot is-other" style={{ ...hudStyle, background: color }}>
              {cam.key}·{p.index + 1}
            </span>
          </span>,
        );
      }
    }
    if (doc) {
      const color = colorForCameraKey(cameraKey);
      const mapPassive = mode !== "pairs";
      doc.pairs.forEach((p, i) => {
        const err = errors[i]?.errPx;
        const hot = hoverPair === i || selectedPair === i;
        const bad = Number.isFinite(err) && err! > errBadPx;
        const title = Number.isFinite(err)
          ? `cam ${cameraKey} · #${i + 1} err ${err!.toFixed(1)} px`
          : `cam ${cameraKey} · #${i + 1}`;
        if (mapPassive) {
          nodes.push(
            <span
              key={`map-${cameraKey}-${i}`}
              style={{ ...ptPct(p.map, natW, natH, { interactive: false }), pointerEvents: "auto" }}
              title={title}
              onMouseEnter={() => setHoverPair(i)}
              onMouseLeave={() => setHoverPair(null)}
            >
              <span
                className={`map-calib-dot${hot ? " is-hot" : ""}${bad ? " is-bad" : ""}`}
                style={{ ...hudStyle, background: color }}
              >
                {cameraKey}·{i + 1}
              </span>
            </span>,
          );
          return;
        }
        nodes.push(
          <span key={`map-${cameraKey}-${i}`} style={ptPct(p.map, natW, natH)} title={title}>
            <button
              type="button"
              className={`map-calib-dot${hot ? " is-hot" : ""}${bad ? " is-bad" : ""}`}
              style={{
                ...hudStyle,
                cursor: "grab",
                background: color,
              }}
              onMouseEnter={() => setHoverPair(i)}
              onMouseLeave={() => setHoverPair(null)}
              onClick={(e) => {
                if (e.button !== 0 || e.altKey) return;
                e.stopPropagation();
                setSelectedPair(i);
              }}
              onPointerDown={(e) => startDragPair(i, "map", e)}
            >
              {cameraKey}·{i + 1}
            </button>
          </span>,
        );
      });
    }
    return nodes;
  }

  /** Иконки размещения камер на плане — всегда. */
  function renderCameraPins(natW: number, natH: number) {
    if (natW < 1 || natH < 1) return null;
    type Pin = { key: string; pl: NonNullable<ReturnType<typeof normalizePlacement>>; active: boolean };
    const pins: Pin[] = [];
    for (const cam of cfg?.cameras ?? []) {
      if (cam.key === cameraKey) continue;
      const pl = normalizePlacement(cam.placement);
      if (!pl) continue;
      pins.push({ key: cam.key, pl, active: false });
    }
    const live = normalizePlacement(doc?.placement);
    if (live) pins.push({ key: cameraKey, pl: live, active: true });

    // Горизонтальный развод подписей (в CSS px), чтобы не наезжали друг на друга.
    // По вертикали всегда над основанием — не трогаем (иначе налезают на точку).
    const labelDx = pins.map(() => 0);
    const nearPx = Math.max(50, Math.min(natW, natH) * 0.02);
    for (let iter = 0; iter < 10; iter++) {
      for (let i = 0; i < pins.length; i++) {
        for (let j = i + 1; j < pins.length; j++) {
          const a = pins[i]!.pl.position;
          const b = pins[j]!.pl.position;
          if (Math.hypot(b[0] - a[0], b[1] - a[1]) > nearPx * 2.5) continue;
          const gap = labelDx[j]! - labelDx[i]!;
          const need = 78;
          if (Math.abs(gap) >= need) continue;
          const push = (need - Math.abs(gap || 0.01)) / 2;
          const dir = gap >= 0 ? 1 : -1;
          labelDx[i]! -= dir * push;
          labelDx[j]! += dir * push;
        }
      }
    }

    const rot = mapView.rot || 0;
    return pins.map(({ key, pl, active }, i) => {
      const color = colorForCameraKey(key);
      return (
        <span
          key={`pin-${key}`}
          style={ptPct(pl.position, natW, natH, { interactive: false })}
          title={`cam ${key} · ${pl.yaw_deg.toFixed(0)}° · FOV ${pl.fov_deg}°`}
        >
          <span
            className={`map-calib-cam-pin${active ? " is-active" : " is-dim"}`}
            style={{
              ["--cam-color" as string]: color,
              transform: `translate(-50%, -50%) rotate(${pl.yaw_deg}deg)`,
            }}
          >
            <i className="map-calib-cam-pin-body" />
            <i className="map-calib-cam-pin-dir" />
          </span>
          <b
            className={`map-calib-cam-pin-label${active ? " is-active" : ""}`}
            style={{
              ["--cam-color" as string]: color,
              ["--label-dx" as string]: `${labelDx[i]!.toFixed(1)}px`,
              ["--pin-rot" as string]: `${-rot}deg`,
            }}
          >
            cam {key}
          </b>
        </span>
      );
    });
  }

  function polyToSvgPoints(pts: Pt[]): string {
    return pts.map(([x, y]) => `${x},${y}`).join(" ");
  }

  function renderCounterLayer(side: "image" | "map", natW: number, natH: number) {
    if (natW < 1) return null;
    const draft = draftSide === side ? draftPts : [];
    const viewScale = Math.max(0.4, side === "image" ? imgView.scale : mapView.scale);
    // В SVG-координатах: меньше при zoom, чтобы на экране не раздувались
    const r = Math.max(3.5, natW / 220 / viewScale);
    const strokeW = Math.max(1.2, natW / 900 / viewScale);
    const editable = mode === "draw";
    return (
      <svg
        className="map-calib-counters"
        viewBox={`0 0 ${natW} ${natH}`}
        preserveAspectRatio="none"
        style={{ pointerEvents: "none" }}
      >
        {counters.counters.map((c) => {
          let pts: Pt[] | null = null;
          let vertsEditable = false;
          if (side === "map") {
            pts = c.map;
            vertsEditable = editable;
          } else {
            pts = c.image_by_camera?.[cameraKey] ?? null;
            if (pts) vertsEditable = editable;
            else if (Hinv && c.map.length >= 3) {
              const projected: Pt[] = [];
              for (const p of c.map) {
                const im = applyHomography(Hinv, p[0], p[1]);
                if (!im) {
                  projected.length = 0;
                  break;
                }
                projected.push(im);
              }
              if (projected.length >= 3) {
                pts = projected;
                vertsEditable = editable;
              }
            }
          }
          if (!pts || pts.length < 3) return null;
          const hot = selectedCounterId === c.id;
          const showVerts = editable && (hot || !selectedCounterId);
          return (
            <g key={c.id}>
              <polygon
                points={polyToSvgPoints(pts)}
                className={`map-calib-counter-poly${hot ? " is-hot" : ""}`}
                style={{ pointerEvents: "auto", cursor: "pointer", strokeWidth: strokeW * (hot ? 1.4 : 1) }}
                onClick={(e) => {
                  if (e.button !== 0 || e.altKey) return;
                  e.stopPropagation();
                  setSelectedCounterId(c.id);
                  setStatus(
                    `${c.name} · тяните серые вершины${side === "map" ? " (свободно)" : ""}`,
                  );
                }}
              />
              {showVerts &&
                pts.map((p, i) => (
                  <circle
                    key={`${c.id}-v-${i}`}
                    cx={p[0]}
                    cy={p[1]}
                    r={r}
                    className={`map-calib-counter-vert${hot ? " is-hot" : ""}`}
                    style={{
                      pointerEvents: vertsEditable ? "auto" : "none",
                      cursor: vertsEditable ? "grab" : "inherit",
                      strokeWidth: strokeW,
                    }}
                    onPointerDown={(e) => {
                      if (!vertsEditable || e.button !== 0 || e.altKey) return;
                      e.stopPropagation();
                      e.preventDefault();
                      setSelectedCounterId(c.id);
                      counterDragRef.current = {
                        id: c.id,
                        side,
                        index: i,
                        moved: false,
                      };
                      const wrap = side === "image" ? imageWrapRef.current : mapWrapRef.current;
                      wrap?.setPointerCapture?.(e.pointerId);
                      setStatus(`Тяните вершину ${i + 1} · ${c.name}`);
                    }}
                  />
                ))}
            </g>
          );
        })}
        {draft.length >= 2 && (
          <polyline
            points={polyToSvgPoints(draft)}
            className="map-calib-counter-draft"
            style={{ strokeWidth: strokeW }}
          />
        )}
        {draft.map((p, i) => (
          <circle
            key={`d-${i}`}
            cx={p[0]}
            cy={p[1]}
            r={r}
            className="map-calib-counter-vert is-draft"
            style={{ pointerEvents: "auto", cursor: "grab", strokeWidth: strokeW }}
            onPointerDown={(e) => {
              if (e.button !== 0 || e.altKey) return;
              e.stopPropagation();
              e.preventDefault();
              counterDragRef.current = {
                id: "__draft__",
                side,
                index: i,
                moved: false,
              };
              const wrap = side === "image" ? imageWrapRef.current : mapWrapRef.current;
              wrap?.setPointerCapture?.(e.pointerId);
            }}
          />
        ))}
      </svg>
    );
  }

  function renderGhosts(natW: number, natH: number) {
    if (!showReproj || !H || !doc || natW < 1) return null;
    return errors.map((er) => {
      if (!er.projected) return null;
      return (
        <span key={`g-${er.index}`} style={ptPct(er.projected, natW, natH)} title={`reproj #${er.index + 1} · ${formatErrPx(er.errPx)}`}>
          <span className="map-calib-ghost" />
        </span>
      );
    });
  }

  function renderWarpGrid(natW: number, natH: number) {
    if (!showGrid || !H || !frameUrl || natW < 1) return null;
    const pts: Pt[] = [];
    for (let yi = 0; yi <= 4; yi++) {
      for (let xi = 0; xi <= 4; xi++) {
        const x = (iw * xi) / 4;
        const y = (ih * yi) / 4;
        const m = applyHomography(H, x, y);
        if (m) pts.push(m);
      }
    }
    return pts.map((p, i) => (
      <span key={`w-${i}`} style={ptPct(p, natW, natH)}>
        <span className="map-calib-grid-pt" />
      </span>
    ));
  }

  function allNeighborMapPoints(): { id: string; camKey: string; label: string; map: Pt }[] {
    const out: { id: string; camKey: string; label: string; map: Pt }[] = [];
    for (const cam of cfg?.cameras ?? []) {
      if (cam.key === cameraKey) continue;
      for (const p of cam.map_points ?? []) {
        out.push({
          id: `${cam.key}:${p.index}`,
          camKey: cam.key,
          label: `${cam.key}·${p.index + 1}`,
          map: p.map,
        });
      }
      const pl = normalizePlacement(cam.placement);
      if (pl) {
        out.push({ id: `cam:${cam.key}`, camKey: cam.key, label: `cam ${cam.key}`, map: pl.position });
      }
    }
    if (doc) {
      doc.pairs.forEach((p, i) => {
        out.push({
          id: `${cameraKey}:${i}`,
          camKey: cameraKey,
          label: `${cameraKey}·${i + 1}`,
          map: p.map,
        });
      });
      const pl = normalizePlacement(doc.placement);
      if (pl) {
        out.push({
          id: `cam:${cameraKey}`,
          camKey: cameraKey,
          label: `cam ${cameraKey}`,
          map: pl.position,
        });
      }
    }
    return out;
  }

  function renderMeasureLinks(natW: number, natH: number) {
    if (!measurePt || natW < 1) return null;
    const neighbors = allNeighborMapPoints()
      .filter((n) => n.id !== measureExcludeId)
      .map((n) => ({ ...n, dist: tilesBetween(measurePt, n.map) }))
      .filter((n) => n.dist.tiles > 1e-6)
      .sort((a, b) => a.dist.tiles - b.dist.tiles)
      .slice(0, 3);

    const walls = distancesToWalls(measurePt).filter((w) => w.tiles > 0);

    if (!neighbors.length && !walls.length) return null;

    type Card = {
      key: string;
      pos: Pt;
      className: string;
      color?: string;
      title: string;
      main: string;
      sub: string;
    };

    const cards: Card[] = [];
    walls.forEach((w, i) => {
      const t = 0.55 + (i % 3) * 0.12;
      const side = ((i % 2 === 0 ? 1 : -1) * (36 + i * 14));
      cards.push({
        key: `WM-${w.id}`,
        pos: offsetAlongSegment(measurePt, w.foot, t, side),
        className: "map-calib-measure-label is-wall",
        title: `${w.name}: ${w.tiles} пл · ${w.meters.toFixed(1)} м`,
        main: `${w.tiles} пл`,
        sub: w.name,
      });
    });
    neighbors.forEach((n, i) => {
      const t = 0.32 + i * 0.18;
      const side = (i - (neighbors.length - 1) / 2) * 48;
      const short =
        n.dist.dx === 0 || n.dist.dy === 0
          ? `${Math.round(n.dist.tiles)} пл`
          : `${n.dist.tiles.toFixed(1)} пл`;
      cards.push({
        key: `M-${n.id}`,
        pos: offsetAlongSegment(measurePt, n.map, t, side),
        className: "map-calib-measure-label",
        color: colorForCameraKey(n.camKey),
        title: `${n.dist.label} → ${n.label}`,
        main: short,
        sub: n.label,
      });
    });

    // ~ширина/высота карточки в px плана при типичном zoom (подписи ~90×36 CSS ≈ 120–160 план-px)
    const spread = deconflictLabelPositions(
      cards.map((c) => c.pos),
      Math.max(110, Math.min(natW, natH) * 0.035),
    );

    return (
      <>
        <svg className="map-calib-measure" viewBox={`0 0 ${natW} ${natH}`} preserveAspectRatio="none">
          {walls.map((w) => (
            <line
              key={`WL-${w.id}`}
              x1={measurePt[0]}
              y1={measurePt[1]}
              x2={w.foot[0]}
              y2={w.foot[1]}
              className="map-calib-measure-line is-wall"
            />
          ))}
          {neighbors.map((n) => {
            const color = colorForCameraKey(n.camKey);
            return (
              <line
                key={`L-${n.id}`}
                x1={measurePt[0]}
                y1={measurePt[1]}
                x2={n.map[0]}
                y2={n.map[1]}
                className="map-calib-measure-line"
                style={{ stroke: color }}
              />
            );
          })}
        </svg>
        {cards.map((c, i) => (
          <span key={c.key} style={ptPct(spread[i]!, natW, natH)} title={c.title}>
            <span
              className={c.className}
              style={{
                ...(c.color ? { ["--measure-color" as string]: c.color } : {}),
                ["--measure-rot" as string]: `${-(mapView.rot || 0)}deg`,
              }}
            >
              {c.main}
              <em>{c.sub}</em>
            </span>
          </span>
        ))}
      </>
    );
  }

  const [mainTab, setMainTab] = useState<"calibrate" | "counters">("calibrate");
  const [activeStep, setActiveStep] = useState<number>(2); // 1: Frame/Floor, 2: Pairs H, 3: 3D Camera, 4: Validation Test

  return (
    <div className="map-calib-v2">
      {/* TOP HEADER */}
      <header className="map-calib-topbar">
        <div className="map-calib-topbar-left">
          <div className="map-calib-cam-badge">
            <span
              className="map-calib-cam-dot"
              style={{ background: colorForCameraKey(cameraKey) }}
            />
            <strong>Камера {cameraKey}</strong>
          </div>

          <div className="map-calib-main-tabs" role="tablist">
            <button
              type="button"
              className={mainTab === "calibrate" ? "is-active" : ""}
              onClick={() => {
                setMainTab("calibrate");
                setMode("pairs");
              }}
            >
              📐 Калибровка камеры
            </button>
            <button
              type="button"
              className={mainTab === "counters" ? "is-active" : ""}
              onClick={() => {
                setMainTab("counters");
                setMode("draw");
              }}
            >
              🏪 Зоны и прилавки
              {counters.counters.length ? ` (${counters.counters.length})` : ""}
            </button>
          </div>
        </div>

        <div className="map-calib-topbar-right">
          <div className="map-calib-save-status">
            {dirty || countersDirty ? (
              <span className="map-calib-dirty-pill" title="Есть несохранённые изменения">
                ● Не сохранено
              </span>
            ) : (
              <span className="map-calib-clean-pill">✓ Сохранено</span>
            )}
          </div>

          <button
            type="button"
            className="map-calib-btn-feet"
            onClick={() => void triggerFeetRecalc()}
            disabled={isRecalculatingFeet}
            title="Пересчитать stage feet на бэкенде через API"
          >
            {isRecalculatingFeet ? "Пересчёт..." : "⚡ Feet API"}
          </button>

          <button
            type="button"
            className={`map-calib-btn-save primary${dirty || countersDirty ? " needs-save" : ""}`}
            onClick={() => void persistAll()}
            disabled={(!doc && !countersDirty) || isRecalculatingFeet}
            title={`Сохранить (${MOD_KEY}+S)`}
          >
            {isRecalculatingFeet
              ? "Сохранение..."
              : dirty || countersDirty
                ? "Сохранить *"
                : "Сохранить"}
            <Kbd>{MOD_KEY}+S</Kbd>
          </button>

          <div className="map-calib-more-actions">
            <button
              type="button"
              className="map-calib-btn-sub"
              onClick={exportJson}
              disabled={!doc}
              title="Экспорт калибровки в JSON"
            >
              Export
            </button>
            <label className="map-calib-file-btn" title="Импорт калибровки из JSON">
              Import
              <input
                type="file"
                accept="application/json,.json"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) importJson(f);
                  e.target.value = "";
                }}
              />
            </label>
          </div>
        </div>
      </header>

      {/* ERROR BANNER */}
      {error && <div className="map-calib-error-banner">{error}</div>}

      {/* MAIN WORKSPACE: 2 COLUMNS */}
      <div className="map-calib-workspace">
        {/* LEFT: CANVASES (Кадр ↔ План) */}
        <div className="map-calib-canvases">
          {/* PANE 1: IMAGE (Кадр видео) */}
          <div className="map-calib-pane">
            <div className="map-calib-pane-header">
              <div className="map-calib-pane-title">
                <strong>Кадр видео</strong>
                <span className="map-calib-pane-meta">
                  {iw}×{ih}
                  {mode === "pairs" && pendingImage ? " · ждём клик на плане" : ""}
                  {mode === "count" ? ` · плитки (${tileMarks.length})` : ""}
                  {mode === "draw" ? " · прилавки" : ""}
                </span>
              </div>
              <div className="map-calib-pane-hud">
                <button
                  type="button"
                  onClick={captureFromPlayer}
                  title="Взять текущий кадр из видеоплеера · F"
                >
                  Из плеера <Kbd>F</Kbd>
                </button>
                <button
                  type="button"
                  onClick={refreshFrame}
                  title="Загрузить 1-й кадр видео · ⇧F"
                >
                  1-й кадр <Kbd>⇧F</Kbd>
                </button>
                <button
                  type="button"
                  onClick={() => setImgView(IDENTITY_VIEW)}
                  title="Сбросить зум и центрировать · R"
                >
                  1:1 <Kbd>R</Kbd>
                </button>
              </div>
            </div>

            <div
              ref={imageWrapRef}
              className={`map-calib-stage${mode === "pairs" && !pendingImage ? " is-armed" : ""}${mode === "test" ? " is-test" : ""}${mode === "count" ? " is-count" : ""}${mode === "draw" ? " is-draw" : ""}`}
              onClick={onImageClick}
              onPointerDown={(e) => onPanePointerDown("image", e)}
              onPointerMove={onPanePointerMove}
              onPointerUp={onPanePointerUp}
              onPointerCancel={onPanePointerUp}
            >
              {frameUrl ? (
                <div style={imageWorldStyle}>
                  <img src={frameUrl} alt="Кадр" draggable={false} onLoad={bumpLayout} />
                  {renderCounterLayer("image", iw, ih)}
                  {renderDots("image", iw, ih)}
                  {tileMarks.map((pt, i) => (
                    <span
                      key={`tile-${i}`}
                      style={ptPct(pt, iw, ih, { interactive: false })}
                      title={`плитка ${i + 1}`}
                    >
                      <span className="map-calib-tile-mark">{i + 1}</span>
                    </span>
                  ))}
                  {pendingImage && (
                    <span style={ptPct(pendingImage, iw, ih, { interactive: false })}>
                      <span
                        className="map-calib-dot pending"
                        style={{
                          ["--dot-hud" as string]: String(markerHudScale(imgView.scale)),
                        }}
                      >
                        +
                      </span>
                    </span>
                  )}
                  {testImagePt && (
                    <span style={ptPct(testImagePt, iw, ih, { interactive: false })}>
                      <span className="map-calib-test-h" title="точка на кадре" />
                    </span>
                  )}
                </div>
              ) : (
                <p className="merge-tl-empty">Загрузка кадра камеры…</p>
              )}
            </div>
          </div>

          {/* PANE 2: MAP (План магазина) */}
          <div className="map-calib-pane">
            <div className="map-calib-pane-header">
              <div className="map-calib-pane-title">
                <strong>План магазина</strong>
                <span className="map-calib-pane-meta">
                  {useGrid ? "Сетка 0.5 м" : floorplan}
                  {mapView.rot ? ` · ${mapView.rot}°` : ""}
                </span>
              </div>
              <div className="map-calib-pane-hud">
                <button
                  type="button"
                  onClick={() => rotateMap(-45)}
                  title="Повернуть план −45° · ["
                >
                  ↶ 45° <Kbd>[</Kbd>
                </button>
                <button
                  type="button"
                  onClick={() => rotateMap(45)}
                  title="Повернуть план +45° · ]"
                >
                  ↷ 45° <Kbd>]</Kbd>
                </button>
                <button
                  type="button"
                  onClick={() => setMapView(IDENTITY_VIEW)}
                  title="Сбросить зум плана"
                >
                  1:1
                </button>
                <label
                  className="map-calib-hud-toggle"
                  title="Призраки H: куда точки кадра ложатся на план · P"
                >
                  <input
                    type="checkbox"
                    checked={showReproj}
                    onChange={(e) => setShowReproj(e.target.checked)}
                  />
                  Призраки <Kbd>P</Kbd>
                </label>
                <label
                  className="map-calib-hud-toggle"
                  title="Сетка кадра, перенесённая на план через H · G"
                >
                  <input
                    type="checkbox"
                    checked={showGrid}
                    onChange={(e) => setShowGrid(e.target.checked)}
                  />
                  Сетка H <Kbd>G</Kbd>
                </label>
              </div>
            </div>

            <div
              ref={mapWrapRef}
              className={`map-calib-stage${mode === "pairs" && pendingImage ? " is-armed" : ""}${mode === "test" ? " is-test" : ""}${mode === "place" ? " is-place" : ""}${mode === "draw" ? " is-draw" : ""}`}
              onClick={onMapClick}
              onPointerDown={onMapPointerDown}
              onPointerMove={onPanePointerMove}
              onPointerUp={onPanePointerUp}
              onPointerCancel={onPanePointerUp}
            >
              <div style={mapWorldStyle}>
                {useGrid ? (
                  <canvas ref={mapGridCanvasRef} className="map-calib-floor-canvas" />
                ) : (
                  <img
                    ref={mapImgRef}
                    src={floorplanUrl}
                    alt="План"
                    draggable={false}
                    onLoad={(e) => {
                      const img = e.currentTarget;
                      setMapNat([img.naturalWidth, img.naturalHeight]);
                      bumpLayout();
                    }}
                  />
                )}
                <canvas ref={mapCamCanvasRef} className="map-calib-cam-canvas" />
                {renderCounterLayer("map", mapNat[0], mapNat[1])}
                {renderMeasureLinks(mapNat[0], mapNat[1])}
                {renderWarpGrid(mapNat[0], mapNat[1])}
                {renderGhosts(mapNat[0], mapNat[1])}
                {renderDots("map", mapNat[0], mapNat[1])}
                {renderCameraPins(mapNat[0], mapNat[1])}
                {mode === "test" &&
                  testMapPt &&
                  testRayMapPt &&
                  testHvsRayGap != null &&
                  testHvsRayGap > 1 && (
                    <svg
                      className="map-calib-test-gap-svg"
                      viewBox={`0 0 ${mapNat[0]} ${mapNat[1]}`}
                      preserveAspectRatio="none"
                    >
                      <line
                        x1={testMapPt[0]}
                        y1={testMapPt[1]}
                        x2={testRayMapPt[0]}
                        y2={testRayMapPt[1]}
                        stroke="rgba(196, 92, 38, 0.75)"
                        strokeWidth={Math.max(2, mapNat[0] / 800)}
                        strokeDasharray="10 8"
                      />
                    </svg>
                  )}
                {testMapPt && (
                  <span
                    style={ptPct(testMapPt, mapNat[0], mapNat[1], { interactive: false })}
                  >
                    <span className="map-calib-test-h" title="H (гомография)" />
                  </span>
                )}
                {mode === "test" && testRayMapPt && (
                  <span
                    style={ptPct(testRayMapPt, mapNat[0], mapNat[1], { interactive: false })}
                  >
                    <span className="map-calib-test-ray" title="3D-луч (ноги)" />
                  </span>
                )}
              </div>

              {cfg?.cameras && cfg.cameras.length > 0 && (
                <div className="map-pane-cam-legend" title="Камеры проекта и их цвета на плане">
                  <span className="map-pane-cam-legend-title">Камеры:</span>
                  {cfg.cameras.map((c) => {
                    const active = c.key === cameraKey;
                    const color = colorForCameraKey(c.key);
                    return (
                      <span
                        key={c.key}
                        className={`map-pane-cam-legend-item${active ? " is-active" : ""}`}
                      >
                        <span className="map-floor-cam-dot" style={{ background: color }} />
                        <span>cam {c.key}</span>
                      </span>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* RIGHT: INSPECTOR SIDEBAR */}
        <aside className="map-calib-sidebar">
          {mainTab === "calibrate" ? (
            <>
              {/* QUALITY STATUS BANNER */}
              <div className={`map-calib-quality-card ${qualityStatus.cls}`}>
                <div className="map-calib-quality-head">
                  <span className="map-calib-quality-title">Качество калибровки</span>
                  <span className="map-calib-quality-badge">{qualityStatus.badge}</span>
                </div>

                <div className="map-calib-quality-metrics">
                  <div className="map-calib-q-metric">
                    <label>Пары H</label>
                    <strong>
                      {doc?.pairs.length ?? 0}
                      {(doc?.pairs.length ?? 0) < H_LOO_MIN_PAIRS
                        ? ` / ${H_LOO_MIN_PAIRS}`
                        : ""}
                    </strong>
                  </div>
                  <div className="map-calib-q-metric">
                    <label>Ошибка RMS</label>
                    <strong>
                      {rms != null
                        ? useGrid
                        ? `${((rms / METER_PX) * 100).toFixed(0)} см`
                        : `${rms.toFixed(1)} px`
                        : "—"}
                    </strong>
                  </div>
                  <div className="map-calib-q-metric">
                    <label>Честная LOO</label>
                    <strong>
                      {hLooRms != null
                        ? useGrid
                        ? `${((hLooRms / METER_PX) * 100).toFixed(0)} см`
                        : `${hLooRms.toFixed(1)} px`
                        : (doc?.pairs.length ?? 0) < 6
                        ? `≥6 пар`
                        : "—"}
                    </strong>
                  </div>
                  <div className="map-calib-q-metric">
                    <label>3D-луч</label>
                    <strong>
                      {rayStats != null
                        ? useGrid
                        ? `${((rayStats.rmsPx / METER_PX) * 100).toFixed(0)} см`
                        : `${rayStats.rmsPx.toFixed(1)} px`
                        : "—"}
                    </strong>
                  </div>
                </div>

                <p className="map-calib-quality-hint">{qualityStatus.hint}</p>
              </div>

              {/* STEP 1: FRAME & FLOORPLAN */}
              <div className={`map-calib-section${activeStep === 1 ? " is-open" : ""}`}>
                <div
                  className="map-calib-section-head"
                  onClick={() => setActiveStep((s) => (s === 1 ? 0 : 1))}
                >
                  <div className="map-calib-step-num">1</div>
                  <div className="map-calib-section-title">
                    <strong>Кадр и подложка</strong>
                    <small>{useGrid ? "Сетка 0.5 м" : floorplan}</small>
                  </div>
                  <span className="map-calib-step-arrow">{activeStep === 1 ? "▲" : "▼"}</span>
                </div>

                {activeStep === 1 && (
                  <div className="map-calib-section-body">
                    <div className="map-calib-control-row">
                      <label className="map-calib-label">Подложка плана</label>
                      <select
                        className="map-calib-select-input"
                        value={floorplan}
                        onChange={(e) => {
                          const v = e.target.value;
                          setFloorplan(v);
                          if (isGridFloorplan(v)) {
                            setMapNat(MAP_SIZE);
                            onFloorRef.current?.("grid");
                          } else {
                            onFloorRef.current?.(`/maps/${encodeURIComponent(v)}`);
                          }
                          setDoc((prev) => {
                            if (!prev) return prev;
                            const next = {
                              ...prev,
                              floorplan: v,
                              map_size: isGridFloorplan(v) ? MAP_SIZE : prev.map_size,
                            };
                            syncDirty(next);
                            return next;
                          });
                          setMapView(IDENTITY_VIEW);
                        }}
                      >
                        <option value={GRID_FLOORPLAN}>Сетка 0.5 м (по умолчанию)</option>
                        {(cfg?.floorplans ?? [])
                          .filter((f) => !isGridFloorplan(f.name))
                          .map((f) => (
                            <option key={f.name} value={f.name}>
                              {f.name}
                            </option>
                          ))}
                      </select>
                    </div>

                    <div className="map-calib-btn-row">
                      <button
                        type="button"
                        className="map-calib-btn"
                        onClick={captureFromPlayer}
                        title="Взять текущую секунду видео из плеера (F)"
                      >
                        Кадр из плеера <Kbd>F</Kbd>
                      </button>
                      <button
                        type="button"
                        className="map-calib-btn"
                        onClick={refreshFrame}
                        title="Перезагрузить первый кадр (⇧F)"
                      >
                        1-й кадр <Kbd>⇧F</Kbd>
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* STEP 2: HOMOGRAPHY PAIRS */}
              <div className={`map-calib-section is-pairs${activeStep === 2 ? " is-open" : ""}`}>
                <div
                  className="map-calib-section-head"
                  onClick={() => {
                    setActiveStep((s) => (s === 2 ? 0 : 2));
                    setMode("pairs");
                  }}
                >
                  <div className="map-calib-step-num">2</div>
                  <div className="map-calib-section-title">
                    <strong>Пары точек H</strong>
                    <small>
                      {doc?.pairs.length ?? 0} пар · клик на кадре → клик на плане
                    </small>
                  </div>
                  <span className="map-calib-step-arrow">{activeStep === 2 ? "▲" : "▼"}</span>
                </div>

                {activeStep === 2 && (
                  <div className="map-calib-section-body">
                    <div className="map-calib-btn-row">
                      <button
                        type="button"
                        className={`map-calib-btn${mode === "pairs" ? " is-active" : ""}`}
                        onClick={() => {
                          setMode("pairs");
                          setStatus(MODE_HINT.pairs);
                        }}
                      >
                        Режим точек <Kbd>1</Kbd>
                      </button>
                      <button
                        type="button"
                        className="map-calib-btn primary"
                        disabled={!doc || doc.pairs.length < 4}
                        title="Snap к сетке + авто-коррекция выбросов"
                        onClick={runAutoCalibrate}
                      >
                        Автокалибровка
                      </button>
                      <button
                        type="button"
                        className="map-calib-btn"
                        onClick={undo}
                        disabled={!history.length}
                        title={`Отменить (${MOD_KEY}+Z)`}
                      >
                        Undo <Kbd>{MOD_KEY}+Z</Kbd>
                      </button>
                      <button
                        type="button"
                        className="map-calib-btn"
                        onClick={redo}
                        disabled={!future.length}
                        title={`Повторить (${MOD_KEY}+⇧Z)`}
                      >
                        Redo
                      </button>
                    </div>

                    {/* LIST OF PAIRS */}
                    <div className="map-calib-pairs-list">
                      {doc?.pairs.length ? (
                        doc.pairs.map((p, i) => {
                          const err = errors[i]?.errPx;
                          const isBad = Number.isFinite(err) && err! > errBadPx;
                          const isHot = selectedPair === i || hoverPair === i;

                          return (
                            <div
                              key={i}
                              className={`map-calib-pair-item${isHot ? " is-hot" : ""}${isBad ? " is-bad" : ""}`}
                              onMouseEnter={() => setHoverPair(i)}
                              onMouseLeave={() => setHoverPair(null)}
                              onClick={() => setSelectedPair(i)}
                            >
                              <div className="map-calib-pair-index">#{i + 1}</div>
                              <div className="map-calib-pair-coords">
                                <span>
                                  Кадр: {p.image[0].toFixed(0)}, {p.image[1].toFixed(0)}
                                </span>
                                <span>
                                  План: {p.map[0].toFixed(0)}, {p.map[1].toFixed(0)}
                                </span>
                              </div>
                              <div
                                className={`map-calib-pair-err${isBad ? " is-bad" : " is-ok"}`}
                                title={
                                  Number.isFinite(err)
                                    ? `Ошибка репроекции: ${formatErrPx(err!)}`
                                    : "Требуется ≥4 пар"
                                }
                              >
                                {Number.isFinite(err)
                                  ? useGrid
                                    ? `${((err! / METER_PX) * 100).toFixed(0)} см`
                                    : `${err!.toFixed(1)} px`
                                  : "—"}
                              </div>
                              <button
                                type="button"
                                className="map-calib-pair-del"
                                title="Удалить пару (Del)"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  removePair(i);
                                }}
                              >
                                ✕
                              </button>
                            </div>
                          );
                        })
                      ) : (
                        <div className="map-calib-empty-pairs">
                          Кликните на объекте на <strong>кадре</strong>, затем на его
                          положении на <strong>плане</strong>.
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* STEP 3: 3D CAMERA POSE */}
              <div className={`map-calib-section${activeStep === 3 ? " is-open" : ""}`}>
                <div
                  className="map-calib-section-head"
                  onClick={() => {
                    setActiveStep((s) => (s === 3 ? 0 : 3));
                    setMode("place");
                  }}
                >
                  <div className="map-calib-step-num">3</div>
                  <div className="map-calib-section-title">
                    <strong>3D-Камера и лучи</strong>
                    <small>
                      {doc?.placement
                        ? `${doc.placement.height_m?.toFixed(1) ?? "3.0"}м · ${doc.placement.pitch_deg?.toFixed(0) ?? "35"}° · yaw ${doc.placement.yaw_deg.toFixed(0)}°`
                        : "Камера не установлена"}
                    </small>
                  </div>
                  <span className="map-calib-step-arrow">{activeStep === 3 ? "▲" : "▼"}</span>
                </div>

                {activeStep === 3 && (
                  <div className="map-calib-section-body">
                    <div className="map-calib-btn-row">
                      <button
                        type="button"
                        className="map-calib-btn primary"
                        disabled={!H && !(doc?.pairs && doc.pairs.length >= 4)}
                        title="Поставить камеру и повернуть по гомографии H"
                        onClick={runAutoPlaceFromH}
                      >
                        Поставить из H
                      </button>
                      <button
                        type="button"
                        className="map-calib-btn primary"
                        disabled={!doc?.placement || !(doc?.pairs && doc.pairs.length >= 2)}
                        title="Оптимизировать высоту, наклон и FOV по парам"
                        onClick={runEstimateRayPose}
                      >
                        Подобрать 3D
                      </button>
                    </div>

                    <div className="map-calib-inputs-grid">
                      <label className="map-calib-field">
                        <span>Высота, м</span>
                        <input
                          type="number"
                          min={0.5}
                          max={8}
                          step={0.1}
                          value={doc?.placement?.height_m ?? 3}
                          disabled={!doc?.placement}
                          onChange={(e) => setHeightM(Number(e.target.value) || 3)}
                        />
                      </label>
                      <label className="map-calib-field">
                        <span>Наклон, °</span>
                        <input
                          type="number"
                          min={0}
                          max={89}
                          step={1}
                          value={doc?.placement?.pitch_deg ?? 35}
                          disabled={!doc?.placement}
                          onChange={(e) => setPitchDeg(Number(e.target.value) || 35)}
                        />
                      </label>
                      <label className="map-calib-field">
                        <span>Угол FOV, °</span>
                        <input
                          type="number"
                          min={20}
                          max={160}
                          step={5}
                          value={doc?.placement?.fov_deg ?? DEFAULT_FOV}
                          disabled={!doc?.placement}
                          onChange={(e) => setFov(Number(e.target.value) || DEFAULT_FOV)}
                        />
                      </label>
                    </div>

                    <div className="map-calib-checkbox-row">
                      <label className="map-calib-checkbox-label">
                        <input
                          type="checkbox"
                          checked={fitMoveCamera}
                          onChange={(e) => setFitMoveCamera(e.target.checked)}
                        />
                        Разрешить сдвиг позиции камеры при 3D-подборе
                      </label>
                    </div>

                    {doc?.placement && (
                      <button
                        type="button"
                        className="map-calib-btn-danger"
                        onClick={clearPlacement}
                      >
                        Снять камеру с плана
                      </button>
                    )}
                  </div>
                )}
              </div>

              {/* STEP 4: TEST & VALIDATION */}
              <div className={`map-calib-section${activeStep === 4 ? " is-open" : ""}`}>
                <div
                  className="map-calib-section-head"
                  onClick={() => {
                    setActiveStep((s) => (s === 4 ? 0 : 4));
                    setMode("test");
                  }}
                >
                  <div className="map-calib-step-num">4</div>
                  <div className="map-calib-section-title">
                    <strong>Проверка и тест</strong>
                    <small>
                      {mode === "test"
                        ? testHvsRayGap != null
                          ? `Невязка: ${((testHvsRayGap / METER_PX) * 100).toFixed(0)} см`
                          : "Кликните на кадре или плане"
                        : "Интерактивный тест лучей"}
                    </small>
                  </div>
                  <span className="map-calib-step-arrow">{activeStep === 4 ? "▲" : "▼"}</span>
                </div>

                {activeStep === 4 && (
                  <div className="map-calib-section-body">
                    <button
                      type="button"
                      className={`map-calib-btn is-full${mode === "test" ? " is-active" : ""}`}
                      onClick={() => {
                        setMode("test");
                        setStatus(MODE_HINT.test);
                      }}
                    >
                      {mode === "test" ? "● Режим теста включен" : "Включить клик-тест H vs 3D-луч"}
                    </button>

                    <p className="map-calib-text-muted">
                      Кликните в любую точку пола на кадре: оранжевая метка — проекция H, синяя —
                      3D-луч. Расстояние между ними показывает точность совпадения.
                    </p>

                    {testHvsRayGap != null && (
                      <div className="map-calib-test-result">
                        <span>Невязка H ↔ 3D:</span>
                        <strong
                          className={
                            testHvsRayGap < rmsOkPx
                              ? "is-ok"
                              : testHvsRayGap > rmsBadPx
                                ? "is-bad"
                                : ""
                          }
                        >
                          {useGrid
                            ? `${((testHvsRayGap / METER_PX) * 100).toFixed(0)} см (${testHvsRayGap.toFixed(1)} px)`
                            : `${testHvsRayGap.toFixed(1)} px`}
                        </strong>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* UTILITY: TILE COUNTER */}
              <div className="map-calib-section is-util">
                <div
                  className="map-calib-section-head"
                  onClick={() => {
                    setMode(mode === "count" ? "pairs" : "count");
                  }}
                >
                  <div className="map-calib-step-num">📏</div>
                  <div className="map-calib-section-title">
                    <strong>Счётчик и линейка плиток</strong>
                    <small>
                      {tileMarks.length ? `Отмечено ${tileMarks.length} пл` : "Клик по кадру 1,2,3…"}
                    </small>
                  </div>
                  <span className="map-calib-step-arrow">
                    {mode === "count" ? "ВКЛ" : "ВЫКЛ"}
                  </span>
                </div>

                {mode === "count" && (
                  <div className="map-calib-section-body">
                    <div className="map-calib-btn-row">
                      <button
                        type="button"
                        className="map-calib-btn"
                        onClick={() => setTileMarks((m) => m.slice(0, -1))}
                        disabled={!tileMarks.length}
                        title="Убрать последнюю плитку (⌫)"
                      >
                        −1 <Kbd>⌫</Kbd>
                      </button>
                      <button
                        type="button"
                        className="map-calib-btn"
                        onClick={() => setTileMarks([])}
                        disabled={!tileMarks.length}
                      >
                        Сбросить
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </>
          ) : (
            /* COUNTERS AND ZONES SUBTAB */
            <div className="map-calib-counters-panel">
              <div className="map-calib-counters-head">
                <strong>Зоны и прилавки зала</strong>
                <p className="map-calib-text-muted">
                  Разметка торговых зон, касс и витрин. Кликните ≥3 точки по кадру или плану, затем
                  нажмите <Kbd>Enter</Kbd> для замыкания полигона.
                </p>
              </div>

              <div className="map-calib-btn-row">
                <button
                  type="button"
                  className="map-calib-btn primary"
                  disabled={draftPts.length < 3}
                  onClick={finishCounterDraft}
                  title="Замкнуть полигон прилавка (Enter)"
                >
                  Замкнуть → план <Kbd>Enter</Kbd>
                </button>
                <button
                  type="button"
                  className="map-calib-btn"
                  disabled={!draftPts.length}
                  onClick={undoDraftVertex}
                  title="Удалить последнюю вершину черновика (⌫)"
                >
                  −вершина <Kbd>⌫</Kbd>
                </button>
                <button
                  type="button"
                  className="map-calib-btn-danger"
                  disabled={!selectedCounterId}
                  onClick={deleteSelectedCounter}
                  title="Удалить выбранную зону (Del)"
                >
                  Удалить <Kbd>Del</Kbd>
                </button>
              </div>

              <div className="map-calib-counters-list">
                {counters.counters.length ? (
                  counters.counters.map((c, i) => {
                    const isSel = selectedCounterId === c.id;
                    return (
                      <div
                        key={c.id}
                        className={`map-calib-counter-card${isSel ? " is-selected" : ""}`}
                        onClick={() => setSelectedCounterId(c.id)}
                      >
                        <div className="map-calib-counter-info">
                          <strong>{c.name || `Прилавок #${i + 1}`}</strong>
                          <small>Вершин: {c.map.length}</small>
                        </div>
                        <button
                          type="button"
                          className="map-calib-pair-del"
                          title="Удалить прилавок"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedCounterId(c.id);
                            deleteSelectedCounter();
                          }}
                        >
                          ✕
                        </button>
                      </div>
                    );
                  })
                ) : (
                  <div className="map-calib-empty-pairs">
                    Нет размеченных зон. Начните кликать по углам прилавка на кадре или плане.
                  </div>
                )}
              </div>

              <button
                type="button"
                className={`map-calib-btn-save primary is-full${countersDirty ? " needs-save" : ""}`}
                onClick={() => void persistCounters()}
                disabled={!countersDirty}
              >
                {countersDirty ? "Сохранить прилавки *" : "Прилавки сохранены"}
              </button>
            </div>
          )}
        </aside>
      </div>

      {/* STATUS FOOTER BAR */}
      <footer className="map-calib-footer">
        <span className="map-calib-footer-status">{status || "Готов к работе"}</span>
        <span className="map-calib-footer-hotkeys">
          Горячие клавиши: <Kbd>1</Kbd> Точки · <Kbd>2</Kbd> Тест · <Kbd>3</Kbd> Камера ·{" "}
          <Kbd>F</Kbd> Кадр из плеера · <Kbd>R</Kbd> Сброс зума · <Kbd>{MOD_KEY}+S</Kbd> Сохранить
        </span>
      </footer>
    </div>
  );
}


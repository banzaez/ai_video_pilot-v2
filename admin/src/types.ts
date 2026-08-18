export type Detection = {
  track_id: number;
  confidence: number;
  bbox: [number, number, number, number];
};

export type TrackingFrame = {
  frame_index: number;
  timestamp_sec: number;
  detections: Detection[];
};

export type TrackingData = {
  fps: number;
  frame_count: number;
  width: number;
  height: number;
  /** Шаг детекции: 1 = каждый кадр, 3 = каждый 3-й. Старые JSON могут не иметь поля. */
  detect_every_n?: number;
  frames: TrackingFrame[];
};

export type TrackKeyframe = {
  frame: number;
  det: Detection;
};

export type FeetPoint = {
  track_id: number;
  map: [number, number];
  source?: string;
  confidence?: number;
};

export type FeetFrame = {
  frame_index: number;
  points: FeetPoint[];
};

export type FeetDoc = {
  stage?: string;
  camera_key?: string;
  image_size?: [number, number] | null;
  tracking_size?: [number, number] | null;
  map_size?: [number, number] | null;
  torso_height_m?: number;
  person_height_m?: number;
  calibration?: { fingerprint?: string };
  n_points?: number;
  frames: FeetFrame[];
};

import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import type { Plugin } from "vite";
import { camerasDir, mapsDir, projectRoot, resultsDir, videoDir } from "./config.js";
import {
  isInside,
  readArtifact,
  readJsonFile,
  readRequestBody,
  refMatches,
  resultsUrl,
  workFile,
} from "./common.js";
import { staleStagesReport } from "./stale.js";
import { cameraLinkFor, faceGalleryFor } from "./faces.js";
import { cropUrlsFor, findTrackletBbox, videoLocalFrame } from "./crops.js";
import { hitsByTrack, mergeHitsFor, mergeTimelineFor, similarFromTrackletLinks } from "./merges.js";
import { cameraHomoPath, mapsConfig } from "./maps.js";
import { discoverSessionsFromVideoDir } from "./sessions.js";
import { discoverDays, getDayLinksMeta, runDayLinkProcess } from "./day.js";

// Re-exports for backwards compatibility
export { readArtifact, refMatches, readJsonFile, workFile };
export { staleStagesReport } from "./stale.js";
export { faceGalleryFor, cameraLinkFor } from "./faces.js";
export { cropUrlsFor } from "./crops.js";
export { mergeHitsFor, mergeTimelineFor, similarFromTrackletLinks } from "./merges.js";
export { discoverDays, getDayLinksMeta, runDayLinkProcess } from "./day.js";

export function mediaLibraryPlugin(): Plugin {
  return {
    name: "media-library",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = req.url ?? "";

        if (url.startsWith("/api/day/list")) {
          try {
            const days = discoverDays();
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ days }));
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          }
          return;
        }

        if (url.startsWith("/api/day/meta")) {
          try {
            const u = new URL(url, "http://local");
            const dayParam = (u.searchParams.get("day") ?? "").trim();
            if (!dayParam) {
              res.statusCode = 400;
              res.end(JSON.stringify({ error: "Missing ?day= param" }));
              return;
            }
            const data = getDayLinksMeta(dayParam);
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(data));
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          }
          return;
        }

        if (url.startsWith("/api/day/run")) {
          if (req.method !== "POST") {
            res.statusCode = 405;
            res.end(JSON.stringify({ error: "Method not allowed" }));
            return;
          }
          try {
            const raw = await readRequestBody(req);
            const body = JSON.parse(raw || "{}") as { day?: string };
            if (!body.day) {
              res.statusCode = 400;
              res.end(JSON.stringify({ error: "Missing { day } in request body" }));
              return;
            }
            const result = await runDayLinkProcess(body.day);
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(result));
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          }
          return;
        }

        if (url.startsWith("/api/media/sessions")) {
          try {
            const sessions = discoverSessionsFromVideoDir();
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ sessions, dir: "data/video" }));
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          }
          return;
        }

        if (url.startsWith("/api/media/list")) {
          try {
            const files = fs.existsSync(videoDir) ? fs.readdirSync(videoDir) : [];
            const videos = files
              .filter((f) => /\.(mp4|webm|mov|mkv)$/i.test(f))
              .sort()
              .map((video) => {
                const base = video.replace(/\.[^.]+$/, "");
                const json = `${base}/tracking.json`;
                return {
                  video,
                  json,
                  hasJson: fs.existsSync(workFile(base, "tracking.json")),
                  videoUrl: `/media/${encodeURIComponent(video)}`,
                  jsonUrl: resultsUrl(base, "tracking.json"),
                  feetUrl: resultsUrl(base, "feet.json"),
                };
              });

            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ items: videos, dir: "data/video" }));
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          }
          return;
        }

        if (url.startsWith("/api/maps/config")) {
          try {
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(mapsConfig()));
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          }
          return;
        }

        if (url.startsWith("/api/maps/counters")) {
          try {
            const countersPath = path.join(mapsDir, "counters.json");
            if (req.method === "GET") {
              const data = readJsonFile(countersPath);
              res.setHeader("Content-Type", "application/json");
              res.end(
                JSON.stringify(
                  data ?? {
                    floorplan: mapsConfig().floorplan,
                    map_size: [4800, 3200],
                    counters: [],
                  },
                ),
              );
              return;
            }
            if (req.method === "PUT" || req.method === "POST") {
              const raw = await readRequestBody(req);
              const body = JSON.parse(raw) as Record<string, unknown>;
              fs.mkdirSync(mapsDir, { recursive: true });
              const countersIn = Array.isArray(body.counters) ? body.counters : [];
              const counters = countersIn
                .map((item, idx) => {
                  if (!item || typeof item !== "object") return null;
                  const c = item as Record<string, unknown>;
                  const mapPts = Array.isArray(c.map)
                    ? c.map
                        .filter((p) => Array.isArray(p) && (p as unknown[]).length >= 2)
                        .map((p) => [Number((p as number[])[0]), Number((p as number[])[1])])
                        .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
                    : [];
                  if (mapPts.length < 3) return null;
                  const image_by_camera: Record<string, number[][]> = {};
                  if (c.image_by_camera && typeof c.image_by_camera === "object") {
                    for (const [cam, pts] of Object.entries(c.image_by_camera as Record<string, unknown>)) {
                      if (!Array.isArray(pts)) continue;
                      const arr = pts
                        .filter((p) => Array.isArray(p) && (p as unknown[]).length >= 2)
                        .map((p) => [Number((p as number[])[0]), Number((p as number[])[1])])
                        .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
                      if (arr.length >= 3) image_by_camera[cam] = arr;
                    }
                  }
                  return {
                    id: typeof c.id === "string" && c.id ? c.id : `c${idx + 1}`,
                    name: typeof c.name === "string" && c.name ? c.name : `Прилавок ${idx + 1}`,
                    map: mapPts,
                    ...(Object.keys(image_by_camera).length ? { image_by_camera } : {}),
                  };
                })
                .filter(Boolean);
              const doc = {
                floorplan: typeof body.floorplan === "string" ? body.floorplan : mapsConfig().floorplan,
                map_size: Array.isArray(body.map_size) ? body.map_size : [4800, 3200],
                counters,
                updated_at: new Date().toISOString(),
              };
              fs.writeFileSync(countersPath, JSON.stringify(doc, null, 2), "utf8");
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify(doc));
              return;
            }
            res.statusCode = 405;
            res.end(JSON.stringify({ error: "method" }));
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          }
          return;
        }

        if (url.startsWith("/api/maps/homography")) {
          try {
            const u = new URL(url, "http://local");
            const key = (u.searchParams.get("camera") ?? "").trim();
            if (!key || key.includes("/") || key.includes("..") || key.includes("\\")) {
              res.statusCode = 400;
              res.end(JSON.stringify({ error: "bad camera" }));
              return;
            }
            if (req.method === "GET") {
              const fp = cameraHomoPath(key);
              const data = readJsonFile(fp);
              res.setHeader("Content-Type", "application/json");
              res.end(
                JSON.stringify(
                  data ?? {
                    camera_key: key,
                    floorplan: mapsConfig().floorplan,
                    image_size: null,
                    map_size: null,
                    pairs: [],
                    H: null,
                    placement: null,
                  },
                ),
              );
              return;
            }
            if (req.method === "PUT" || req.method === "POST") {
              const raw = await readRequestBody(req);
              const body = JSON.parse(raw) as Record<string, unknown>;
              fs.mkdirSync(camerasDir, { recursive: true });
              const plRaw = body.placement;
              let placement: {
                position: [number, number];
                yaw_deg: number;
                fov_deg: number;
                height_m?: number;
                pitch_deg?: number;
              } | null = null;
              if (plRaw && typeof plRaw === "object") {
                const pl = plRaw as {
                  position?: unknown;
                  yaw_deg?: unknown;
                  fov_deg?: unknown;
                  height_m?: unknown;
                  pitch_deg?: unknown;
                };
                if (Array.isArray(pl.position) && pl.position.length >= 2) {
                  const x = Number(pl.position[0]);
                  const y = Number(pl.position[1]);
                  if (Number.isFinite(x) && Number.isFinite(y)) {
                    const yaw = Number(pl.yaw_deg);
                    const fov = Number(pl.fov_deg);
                    const height = Number(pl.height_m);
                    const pitch = Number(pl.pitch_deg);
                    placement = {
                      position: [x, y],
                      yaw_deg: Number.isFinite(yaw) ? ((yaw % 360) + 360) % 360 : 0,
                      fov_deg: Number.isFinite(fov) ? Math.min(160, Math.max(20, fov)) : 70,
                      ...(Number.isFinite(height) ? { height_m: Math.max(0.5, height) } : {}),
                      ...(Number.isFinite(pitch) ? { pitch_deg: Math.min(89, Math.max(0, pitch)) } : {}),
                    };
                  }
                }
              }
              const existing = readJsonFile(cameraHomoPath(key)) as Record<string, unknown> | null;
              const doc: Record<string, unknown> = {
                camera_key: key,
                floorplan: typeof body.floorplan === "string" ? body.floorplan : mapsConfig().floorplan,
                image_size: body.image_size ?? null,
                map_size: body.map_size ?? null,
                pairs: Array.isArray(body.pairs) ? body.pairs : [],
                H: Array.isArray(body.H) && body.H.length === 9 ? body.H : null,
                placement,
                updated_at: new Date().toISOString(),
              };
              if (body.body_calib != null) {
                doc.body_calib = body.body_calib;
              } else if (existing?.body_calib != null) {
                doc.body_calib = existing.body_calib;
              }
              fs.writeFileSync(cameraHomoPath(key), JSON.stringify(doc, null, 2), "utf8");
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify(doc));
              return;
            }
            res.statusCode = 405;
            res.end(JSON.stringify({ error: "method" }));
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          }
          return;
        }

        if (url.startsWith("/maps/")) {
          const name = decodeURIComponent(url.slice("/maps/".length).split("?")[0] ?? "");
          const filePath = path.resolve(mapsDir, name);
          if (!isInside(mapsDir, filePath) || !fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
            res.statusCode = 404;
            res.end("Not found");
            return;
          }
          const ext = path.extname(filePath).toLowerCase();
          const types: Record<string, string> = {
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
          };
          res.setHeader("Content-Type", types[ext] ?? "application/octet-stream");
          res.setHeader("Cache-Control", "no-cache");
          fs.createReadStream(filePath).pipe(res);
          return;
        }

        if (url.startsWith("/api/media/meta")) {
          try {
            const q = new URL(url, "http://local");
            const session = (q.searchParams.get("session") ?? "").trim();
            const baseRaw = session || (q.searchParams.get("base") ?? "").trim();
            const base = baseRaw.trim();
            if (!base || base.includes("/") || base.includes("\\") || base.includes("..")) {
              res.statusCode = 400;
              res.end(JSON.stringify({ error: "bad base/session" }));
              return;
            }
            const fg = faceGalleryFor(base);
            res.setHeader("Content-Type", "application/json");
            res.end(
              JSON.stringify({
                info: readJsonFile(workFile(base, "info.json")),
                crops: cropUrlsFor(base),
                faces: fg.faces,
                facesByModel: fg.facesByModel,
                faceModels: fg.faceModels,
                cameraLink: cameraLinkFor(base),
                similar: similarFromTrackletLinks(base) ?? hitsByTrack(base, "link.json"),
                merge: mergeHitsFor(base),
                mergeTimeline: mergeTimelineFor(base),
                pipeline: staleStagesReport(base),
              }),
            );
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          }
          return;
        }

        if (url.startsWith("/api/crop/")) {
          try {
            const rawPath = url.slice("/api/crop/".length).split("?")[0];
            const slashIdx = rawPath.indexOf("/");
            if (slashIdx === -1) {
              res.statusCode = 400;
              res.end("Invalid crop path");
              return;
            }
            const base = decodeURIComponent(rawPath.slice(0, slashIdx)).trim();
            const filename = decodeURIComponent(rawPath.slice(slashIdx + 1)).trim();
            if (!base || !filename || base.includes("..") || filename.includes("..")) {
              res.statusCode = 400;
              res.end("Bad base/filename");
              return;
            }

            const cacheCropDir = path.join(projectRoot, "data", ".cache", "crops", base);
            const cachedFile = path.join(cacheCropDir, filename);
            const diskFile = path.join(resultsDir, base, "tracklet_crops", filename);

            const serveJpeg = (filePath: string) => {
              res.setHeader("Content-Type", "image/jpeg");
              res.setHeader("Cache-Control", "public, max-age=86400");
              fs.createReadStream(filePath).pipe(res);
            };

            if (fs.existsSync(diskFile) && fs.statSync(diskFile).size > 0) {
              serveJpeg(diskFile);
              return;
            }
            if (fs.existsSync(cachedFile) && fs.statSync(cachedFile).size > 0) {
              serveJpeg(cachedFile);
              return;
            }

            const m = /^tl_(\d+)_k(\d+)_f(\d+)\.(jpe?g|png|webp)$/i.exec(filename);
            if (!m) {
              res.statusCode = 404;
              res.end("Invalid crop filename format");
              return;
            }
            const trackletId = Number(m[1]);
            const targetFrame = Number(m[3]);

            const info = readJsonFile(workFile(base, "info.json")) as {
              parts?: Array<{ path?: string; frame_offset?: number; frame_count?: number }>;
            } | null;
            const { videoRelPath, frame0 } = videoLocalFrame(info, targetFrame);
            if (!videoRelPath) {
              res.statusCode = 404;
              res.end("Video not found in info.json");
              return;
            }
            const videoAbsPath = path.isAbsolute(videoRelPath) ? videoRelPath : path.join(projectRoot, videoRelPath);
            if (!fs.existsSync(videoAbsPath)) {
              res.statusCode = 404;
              res.end("Video file does not exist");
              return;
            }

            // Находим точный bbox детекции для этого треклета
            const bbox = findTrackletBbox(base, trackletId, targetFrame);

            if (!bbox || bbox.length !== 4) {
              res.statusCode = 404;
              res.end("BBox not found for frame");
              return;
            }

            fs.mkdirSync(cacheCropDir, { recursive: true });

            const pythonBin = path.join(projectRoot, "venv", "bin", "python");
            const pyScript = path.join(projectRoot, "app", "tools", "extract_crop.py");
            const child = spawn(pythonBin, [
              pyScript,
              "--video",
              videoAbsPath,
              "--frame",
              String(frame0),
              "--bbox",
              String(bbox[0]),
              String(bbox[1]),
              String(bbox[2]),
              String(bbox[3]),
              "--output",
              cachedFile,
            ]);

            let stderr = "";
            child.stderr?.on("data", (chunk) => {
              stderr += String(chunk);
            });
            let sent = false;
            const finish = (status: number, body: string) => {
              if (sent) return;
              sent = true;
              res.statusCode = status;
              res.end(body);
            };
            const timer = setTimeout(() => {
              child.kill("SIGKILL");
              finish(504, "Crop extraction timeout");
            }, 30000);
            child.on("error", (err) => {
              clearTimeout(timer);
              finish(500, `Crop spawn failed: ${err}`);
            });
            child.on("close", (code) => {
              clearTimeout(timer);
              if (sent) return;
              if (code === 0 && fs.existsSync(cachedFile) && fs.statSync(cachedFile).size > 0) {
                sent = true;
                res.setHeader("Content-Type", "image/jpeg");
                res.setHeader("Cache-Control", "public, max-age=86400");
                fs.createReadStream(cachedFile).pipe(res);
              } else {
                finish(500, `Crop extraction failed${stderr.trim() ? `: ${stderr.trim().slice(0, 400)}` : ""}`);
              }
            });
          } catch (err) {
            res.statusCode = 500;
            res.end(String(err));
          }
          return;
        }

        if (url.startsWith("/api/group_crop/")) {
          const rawPath = url.slice("/api/group_crop/".length).split("?")[0];
          const slashIdx = rawPath.indexOf("/");
          if (slashIdx === -1) {
            res.statusCode = 400;
            res.end("Invalid group crop path");
            return;
          }
          const gcBase = decodeURIComponent(rawPath.slice(0, slashIdx)).trim();
          const gcFile = decodeURIComponent(rawPath.slice(slashIdx + 1)).trim();
          if (!gcBase || !gcFile || gcBase.includes("..") || gcFile.includes("..")) {
            res.statusCode = 400;
            res.end("Bad base/filename");
            return;
          }
          const gcPath = path.resolve(resultsDir, gcBase, "day_group_crops", gcFile);
          if (!isInside(resultsDir, gcPath) || !fs.existsSync(gcPath) || !fs.statSync(gcPath).isFile()) {
            res.statusCode = 404;
            res.end("Group crop not found");
            return;
          }
          res.setHeader("Content-Type", "image/jpeg");
          res.setHeader("Cache-Control", "public, max-age=86400");
          fs.createReadStream(gcPath).pipe(res);
          return;
        }

        if (url.startsWith("/api/face_crop/")) {
          const rawPath = url.slice("/api/face_crop/".length).split("?")[0];
          const slashIdx = rawPath.indexOf("/");
          if (slashIdx === -1) {
            res.statusCode = 400;
            res.end("Invalid face crop path");
            return;
          }
          const fcBase = decodeURIComponent(rawPath.slice(0, slashIdx)).trim();
          const fcFile = decodeURIComponent(rawPath.slice(slashIdx + 1)).trim();
          if (!fcBase || !fcFile || fcBase.includes("..") || fcFile.includes("..")) {
            res.statusCode = 400;
            res.end("Bad base/filename");
            return;
          }
          const fcPath = path.resolve(resultsDir, fcBase, "face_crops", fcFile);
          if (!isInside(resultsDir, fcPath) || !fs.existsSync(fcPath) || !fs.statSync(fcPath).isFile()) {
            res.statusCode = 404;
            res.end("Face crop not found");
            return;
          }
          res.setHeader("Content-Type", "image/jpeg");
          res.setHeader("Cache-Control", "public, max-age=86400");
          fs.createReadStream(fcPath).pipe(res);
          return;
        }

        if (url.startsWith("/media/") || url.startsWith("/results/")) {
          const fromResults = url.startsWith("/results/");
          const prefix = fromResults ? "/results/" : "/media/";
          const root = fromResults ? resultsDir : videoDir;
          const name = decodeURIComponent(url.slice(prefix.length).split("?")[0] ?? "");
          const filePath = path.resolve(root, name);
          if (!isInside(root, filePath) || !fs.existsSync(filePath)) {
            res.statusCode = 404;
            res.end("Not found");
            return;
          }
          const stat = fs.statSync(filePath);
          if (!stat.isFile()) {
            res.statusCode = 404;
            res.end("Not found");
            return;
          }
          const size = stat.size;
          const ext = path.extname(filePath).toLowerCase();
          const types: Record<string, string> = {
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".json": "application/json",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
          };
          const contentType = types[ext] ?? "application/octet-stream";
          const range = req.headers.range;

          res.setHeader("Accept-Ranges", "bytes");
          res.setHeader("Content-Type", contentType);
          const isImg = [".jpg", ".jpeg", ".png", ".webp"].includes(ext);
          res.setHeader("Cache-Control", isImg ? "public, max-age=86400" : "no-cache");

          // Без Range браузер не умеет перематывать видео
          if (range && [".mp4", ".webm", ".mov", ".mkv"].includes(ext)) {
            const match = /bytes=(\d*)-(\d*)/.exec(range);
            if (!match) {
              res.statusCode = 416;
              res.setHeader("Content-Range", `bytes */${size}`);
              res.end();
              return;
            }

            let start = 0;
            let end = size - 1;

            if (!match[1] && match[2]) {
              // Запрос суффикса: bytes=-500000 (последние 500 КБ для moov-атома)
              const suffixLen = Number(match[2]);
              start = Math.max(0, size - suffixLen);
              end = size - 1;
            } else if (match[1] && !match[2]) {
              // Запрос от позиции: bytes=1000- (отдаем чанк 16MB для буфера)
              start = Number(match[1]);
              end = Math.min(start + 16 * 1024 * 1024 - 1, size - 1);
            } else if (match[1] && match[2]) {
              // Явный диапазон: bytes=1000-2000
              start = Number(match[1]);
              end = Math.min(Number(match[2]), size - 1);
            }

            if (Number.isNaN(start) || Number.isNaN(end) || start >= size || end < start) {
              res.statusCode = 416;
              res.setHeader("Content-Range", `bytes */${size}`);
              res.end();
              return;
            }

            res.statusCode = 206;
            res.setHeader("Content-Range", `bytes ${start}-${end}/${size}`);
            res.setHeader("Content-Length", String(end - start + 1));
            fs.createReadStream(filePath, { start, end }).pipe(res);
            return;
          }

          res.setHeader("Content-Length", String(size));
          fs.createReadStream(filePath).pipe(res);
          return;
        }

        next();
      });
    },
  };
}

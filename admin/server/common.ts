import fs from "node:fs";
import path from "node:path";
import type { IncomingMessage } from "node:http";
import { resultsDir } from "./config.js";

export function isInside(root: string, filePath: string): boolean {
  const base = path.resolve(root);
  const resolved = path.resolve(filePath);
  return resolved === base || resolved.startsWith(base + path.sep);
}

export function readJsonFile(fp: string): unknown | null {
  if (!fs.existsSync(fp)) return null;
  try {
    return JSON.parse(fs.readFileSync(fp, "utf8"));
  } catch {
    return null;
  }
}

export function workFile(base: string, name: string): string {
  return path.join(resultsDir, base, name);
}

export function resultsUrl(...parts: string[]): string {
  return `/results/${parts.map((p) => encodeURIComponent(p)).join("/")}`;
}

export function sessionFps(base: string): number {
  const info = readJsonFile(workFile(base, "info.json")) as { fps?: number } | null;
  return typeof info?.fps === "number" && info.fps > 0 ? info.fps : 25;
}

export function readRequestBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += String(chunk);
    });
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

export type ArtifactMeta = {
  stage?: string;
  file_version?: number;
  written_at?: string;
  inputs?: Record<string, { written_at?: string | null; file_version?: number; mtime?: number; size?: number }>;
};

export function readArtifact(fp: string): ArtifactMeta | null {
  const data = readJsonFile(fp);
  if (!data || typeof data !== "object") return null;
  const art = (data as Record<string, unknown>).artifact;
  return art && typeof art === "object" ? (art as ArtifactMeta) : null;
}

export function refMatches(
  recorded: ArtifactMeta["inputs"] extends infer I
    ? I extends Record<string, infer V>
      ? V
      : never
    : never,
  parentPath: string,
  parentArt: ArtifactMeta | null,
): boolean {
  if (!recorded) return false;
  if (parentArt?.written_at) {
    return (
      String(recorded.written_at ?? "") === String(parentArt.written_at) &&
      Number(recorded.file_version ?? 0) === Number(parentArt.file_version ?? 0)
    );
  }
  if (!fs.existsSync(parentPath)) return false;
  const st = fs.statSync(parentPath);
  if (recorded.mtime != null) {
    return Math.abs(Number(recorded.mtime) - st.mtimeMs / 1000) < 0.05 && Number(recorded.size ?? -1) === st.size;
  }
  return false;
}

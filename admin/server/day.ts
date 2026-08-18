import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { projectRoot, resultsDir } from "./config.js";
import { readJsonFile } from "./common.js";
import { discoverSessionsFromVideoDir } from "./sessions.js";

export interface DaySummary {
  day: string;
  day_clean: string;
  has_links: boolean;
  sessions: string[];
  cameras: string[];
  stats?: {
    n_persons?: number;
    n_multi_cam_persons?: number;
    n_merges_total?: number;
    pass0_merges?: number;
    pass1_merges?: number;
    pass2_merges?: number;
    pass4_merges?: number;
  };
}

export function discoverDays(): DaySummary[] {
  if (!fs.existsSync(resultsDir)) return [];
  const entries = fs.readdirSync(resultsDir);

  // Группировка сессий по дню
  const daysMap = new Map<string, { sessions: string[]; cameras: Set<string> }>();

  for (const entry of entries) {
    if (entry.startsWith(".") || entry.startsWith("_") || entry.startsWith("day_")) continue;
    const sessPath = path.join(resultsDir, entry);
    const infoPath = path.join(sessPath, "info.json");
    if (fs.existsSync(infoPath)) {
      const info = readJsonFile(infoPath) as { day?: string; camera?: string } | null;
      const dayClean = String(info?.day ?? entry.split("_")[1] ?? "").replace(/-/g, "");
      if (/^\d{8}$/.test(dayClean)) {
        if (!daysMap.has(dayClean)) {
          daysMap.set(dayClean, { sessions: [], cameras: new Set() });
        }
        const item = daysMap.get(dayClean)!;
        item.sessions.push(entry);
        if (info?.camera) item.cameras.add(info.camera);
      }
    }
  }

  const result: DaySummary[] = [];

  for (const [dayClean, data] of daysMap.entries()) {
    const dayDir = path.join(resultsDir, `day_${dayClean}`);
    const linksPath = path.join(dayDir, "day_links.json");
    const hasLinks = fs.existsSync(linksPath);
    let stats = undefined;
    let cameras = Array.from(data.cameras);

    if (hasLinks) {
      const linksDoc = readJsonFile(linksPath) as {
        stats?: DaySummary["stats"];
        cameras?: string[];
      } | null;
      if (linksDoc?.stats) stats = linksDoc.stats;
      if (linksDoc?.cameras) cameras = linksDoc.cameras;
    }

    const formattedDay = `${dayClean.slice(0, 4)}-${dayClean.slice(4, 6)}-${dayClean.slice(6, 8)}`;
    result.push({
      day: formattedDay,
      day_clean: dayClean,
      has_links: hasLinks,
      sessions: data.sessions.sort(),
      cameras: cameras.sort(),
      stats,
    });
  }

  return result.sort((a, b) => b.day_clean.localeCompare(a.day_clean));
}

export function getDayLinksMeta(dayClean: string) {
  const clean = dayClean.replace(/-/g, "").trim();
  const formattedDay = `${clean.slice(0, 4)}-${clean.slice(4, 6)}-${clean.slice(6, 8)}`;
  const dayDir = path.join(resultsDir, `day_${clean}`);
  const linksPath = path.join(dayDir, "day_links.json");

  // Получаем сессии дня через существующий сервис discoverSessionsFromVideoDir
  const allSessions = discoverSessionsFromVideoDir();
  const daySessions = allSessions
    .filter((s) => s.day === formattedDay || s.key.endsWith(`_${clean}`))
    .map((s) => {
      // Вычисляем t0_abs (секунды от полуночи)
      const infoPath = path.join(resultsDir, s.key, "info.json");
      let t0_abs = 0;
      if (fs.existsSync(infoPath)) {
        const info = readJsonFile(infoPath) as {
          started_at?: string;
          parsed?: { started_at?: string };
          parts?: Array<{ started_at?: string }>;
        } | null;
        const startedIso = info?.parsed?.started_at || info?.started_at || info?.parts?.[0]?.started_at;
        if (startedIso) {
          const match = /(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)/.exec(startedIso);
          if (match) {
            t0_abs = Number(match[1]) * 3600 + Number(match[2]) * 60 + Number(match[3]);
          }
        }
      }
      return {
        ...s,
        t0_abs,
      };
    });

  const sessionKeys = daySessions.map((s) => s.key);
  const camerasFromSessions = [
    ...new Set(daySessions.map((s) => s.camera).filter((c): c is string => Boolean(c))),
  ];

  if (!fs.existsSync(linksPath)) {
    return {
      day: formattedDay,
      day_clean: clean,
      has_links: false,
      persons: [],
      edges: [],
      candidate_edges: [],
      sessions: sessionKeys,
      cameras: camerasFromSessions,
      camera_sessions: daySessions,
      track_to_person: {},
      n_persons: 0,
      stats: {
        n_persons: 0,
        n_multi_cam_persons: 0,
        n_merges_total: 0,
      },
    };
  }

  const doc = readJsonFile(linksPath) as {
    persons?: Array<{
      person_id: number;
      tracks: Array<{ uid: string; session_key: string; track_id: number }>;
    }>;
    sessions?: string[];
    cameras?: string[];
  };

  // Построение быстрого маппинга track_id -> person_id по каждой сессии
  const trackToPerson: Record<string, Record<number, number>> = {};
  if (doc?.persons) {
    for (const p of doc.persons) {
      for (const tr of p.tracks) {
        if (!trackToPerson[tr.session_key]) {
          trackToPerson[tr.session_key] = {};
        }
        trackToPerson[tr.session_key]![tr.track_id] = p.person_id;
      }
    }
  }

  return {
    ...(doc as object),
    has_links: true,
    day: formattedDay,
    day_clean: clean,
    sessions: doc?.sessions?.length ? doc.sessions : sessionKeys,
    cameras: doc?.cameras?.length ? doc.cameras : camerasFromSessions,
    camera_sessions: daySessions,
    track_to_person: trackToPerson,
  };
}

export function runDayLinkProcess(dayClean: string): Promise<{ success: boolean; output: string }> {
  return new Promise((resolve) => {
    const clean = dayClean.replace(/-/g, "").trim();
    const pythonExe = path.join(projectRoot, "venv", "bin", "python");
    const args = ["-m", "app.main", "--input", `day:${clean}`, "--stage", "day_link"];

    const proc = spawn(pythonExe, args, {
      cwd: projectRoot,
      env: { ...process.env },
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (d) => {
      stdout += String(d);
    });

    proc.stderr.on("data", (d) => {
      stderr += String(d);
    });

    proc.on("close", (code) => {
      resolve({
        success: code === 0,
        output: (stdout + "\n" + stderr).trim(),
      });
    });

    proc.on("error", (err) => {
      resolve({
        success: false,
        output: String(err),
      });
    });
  });
}

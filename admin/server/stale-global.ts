import fs from "node:fs";
import path from "node:path";
import { resultsDir } from "./config.js";
import { readArtifact, refMatches } from "./media-handlers.js";

const GLOBAL_STAGE_ORDER = ["global_similar", "global_link", "global_merge"] as const;
const GLOBAL_STAGE_FILES: Record<(typeof GLOBAL_STAGE_ORDER)[number], string> = {
  global_similar: "global_similar.json",
  global_link: "global_link.json",
  global_merge: "global_merge.json",
};
const GLOBAL_STAGE_PARENT: Record<(typeof GLOBAL_STAGE_ORDER)[number], (typeof GLOBAL_STAGE_ORDER)[number] | null> = {
  global_similar: null,
  global_link: "global_similar",
  global_merge: "global_link",
};

function linkFingerprint(): { written_at?: string; mtime?: number } {
  let maxMtime = 0;
  let latestWritten: string | undefined;
  if (!fs.existsSync(resultsDir)) return {};
  for (const name of fs.readdirSync(resultsDir)) {
    if (name.startsWith("_")) continue;
    const link = path.join(resultsDir, name, "link.json");
    const merge = path.join(resultsDir, name, "merge.json");
    const fp = fs.existsSync(link) ? link : fs.existsSync(merge) ? merge : null;
    if (!fp) continue;
    const art = readArtifact(fp);
    const st = fs.statSync(fp);
    if (st.mtimeMs > maxMtime) maxMtime = st.mtimeMs;
    if (art?.written_at) latestWritten = art.written_at;
  }
  return { written_at: latestWritten, mtime: maxMtime ? maxMtime / 1000 : undefined };
}

export function globalStaleReport() {
  const gdir = path.join(resultsDir, "_global");
  const fp = linkFingerprint();
  const stages: Record<
    string,
    {
      stage: string;
      file: string;
      exists: boolean;
      file_version: number | null;
      written_at: string | null;
      size: number | null;
      mtime: string | null;
      stale: boolean;
      reason: string | null;
    }
  > = {};
  const stale: string[] = [];

  for (const stage of GLOBAL_STAGE_ORDER) {
    const file = GLOBAL_STAGE_FILES[stage];
    const artPath = path.join(gdir, file);
    const exists = fs.existsSync(artPath);
    const art = exists ? readArtifact(artPath) : null;
    const st = exists ? fs.statSync(artPath) : null;
    const entry = {
      stage,
      file,
      exists,
      file_version: art?.file_version ?? null,
      written_at: art?.written_at ?? null,
      size: st?.size ?? null,
      mtime: st ? new Date(st.mtimeMs).toISOString() : null,
      stale: false,
      reason: null as string | null,
    };
    if (exists && stage === "global_similar") {
      const recorded = art?.inputs?.link ?? art?.inputs?.merge;
      if (fp.written_at && recorded?.written_at && String(recorded.written_at) !== String(fp.written_at)) {
        entry.stale = true;
        entry.reason = "link.json новее — пересчитать global";
      } else if (fp.mtime && fs.statSync(artPath).mtimeMs / 1000 + 0.01 < fp.mtime) {
        entry.stale = true;
        entry.reason = "link.json новее (mtime)";
      }
    } else if (exists && stage !== "global_similar") {
      let parent = GLOBAL_STAGE_PARENT[stage];
      if (stage === "global_merge" && parent === "global_link") {
        const linkPath = path.join(gdir, GLOBAL_STAGE_FILES.global_link);
        if (!fs.existsSync(linkPath)) parent = "global_similar";
      }
      const parentName = parent ?? "global_similar";
      const parentPath = path.join(gdir, GLOBAL_STAGE_FILES[parentName]);
      if (!fs.existsSync(parentPath)) {
        entry.stale = true;
        entry.reason = `нет ${GLOBAL_STAGE_FILES[parentName]}`;
      } else {
        const parentArt = readArtifact(parentPath);
        const recorded = art?.inputs?.[parentName];
        if (recorded && refMatches(recorded, parentPath, parentArt)) {
          // ok
        } else if (art && fs.statSync(parentPath).mtimeMs > fs.statSync(artPath).mtimeMs + 10) {
          entry.stale = true;
          entry.reason = `${GLOBAL_STAGE_FILES[parentName]} новее (mtime)`;
        } else if (art) {
          entry.stale = true;
          entry.reason = `не совпадает с ${GLOBAL_STAGE_FILES[parentName]} — пересчитать`;
        }
      }
    }
    if (entry.stale) stale.push(stage);
    stages[stage] = entry;
  }

  if (stages.global_similar?.stale) {
    for (const child of ["global_link", "global_merge"] as const) {
      if (stages[child]?.exists && !stages[child].stale) {
        stages[child].stale = true;
        stages[child].reason = stages[child].reason || `устарел родитель global_similar`;
        stale.push(child);
      }
    }
  }
  if (stages.global_link?.stale && stages.global_merge?.exists && !stages.global_merge.stale) {
    stages.global_merge.stale = true;
    stages.global_merge.reason = stages.global_merge.reason || "устарел родитель global_link";
    stale.push("global_merge");
  }

  const staleOrdered = GLOBAL_STAGE_ORDER.filter((s) => stages[s]?.stale);
  const recompute_from = staleOrdered[0] ?? null;
  const recompute_to = staleOrdered.length ? staleOrdered[staleOrdered.length - 1]! : null;
  let cli: string | null = null;
  if (recompute_from && recompute_to) {
    cli =
      recompute_from === recompute_to
        ? `python -m app.main --stage ${recompute_from}`
        : `python -m app.main --from ${recompute_from} --to ${recompute_to}`;
  }

  const fileStats = (name: string) => {
    const fp = path.join(gdir, name);
    if (!fs.existsSync(fp) || !fs.statSync(fp).isFile()) {
      return { exists: false, size: null as number | null, mtime: null as string | null };
    }
    const st = fs.statSync(fp);
    return { exists: true, size: st.size, mtime: new Date(st.mtimeMs).toISOString() };
  };
  const reid = fileStats("reid.npz");
  const face = fileStats("face.npz");
  const extras = {
    global_similar: [
      {
        key: "reid",
        label: "reid.npz",
        path: "_global/reid.npz",
        kind: "file" as const,
        exists: reid.exists,
        files: null as number | null,
        size: reid.size,
        mtime: reid.mtime,
        note: reid.exists ? "эмбеддинги кропов (ReID)" : stages.global_similar?.exists ? "нет кэша ReID" : null,
      },
      {
        key: "face",
        label: "face.npz",
        path: "_global/face.npz",
        kind: "file" as const,
        exists: face.exists,
        files: null as number | null,
        size: face.size,
        mtime: face.mtime,
        note: face.exists ? "эмбеддинги лиц" : stages.global_similar?.exists ? "нет кэша лиц" : null,
      },
    ],
  };

  return { stages, stale: staleOrdered, recompute_from, recompute_to, cli, link: fp, extras };
}

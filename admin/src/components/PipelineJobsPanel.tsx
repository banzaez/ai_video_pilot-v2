import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createJob,
  fetchActiveJob,
  fetchJobLog,
  fetchJobs,
  fetchMediaMeta,
  restartJob,
  stopJob,
  type PipelineJob,
  type PipelineStageInfo,
  type PipelineStaleReport,
  type PipelineExtraInfo,
} from "../utils";

const STAGES = [
  "info",
  "detect",
  "tracklets",
  "tracklet_reid",
  "tracklet_link",
  "track",
  "pose",
  "feet",
  "camera_link",
  "all",
  "no_merge",
] as const;

const STAGE_LABELS: Record<string, string> = {
  info: "info",
  detect: "detect",
  tracklets: "tracklets · 2a",
  tracklet_reid: "tracklet_reid · 2b",
  tracklet_link: "tracklet_link · 2c",
  track: "track",
  pose: "pose",
  feet: "feet",
  camera_link: "camera_link · 5 (Pass 10)",
  all: "all · полный цикл",
  no_merge: "no_merge · алиас all",
};

const STAGE_GROUPS: { title: string; stages: readonly string[] }[] = [
  { title: "0 · info", stages: ["info"] },
  { title: "1 · detect", stages: ["detect"] },
  { title: "2 · треки", stages: ["tracklets", "tracklet_reid", "tracklet_link", "track"] },
  { title: "3 · карта", stages: ["pose", "feet"] },
  { title: "4 · камера / лица (Pass 10)", stages: ["camera_link"] },
];

const SESSION_STAGE_ORDER = STAGE_GROUPS.flatMap((g) => g.stages);

const RANGE_STAGES = STAGES.filter((s) => !["all", "no_merge"].includes(s));

type Mode = "stage" | "range";

type Props = {
  librarySession: string;
  onPipelineUpdate?: (pipeline: PipelineStaleReport | null) => void;
};

function statusClass(s: PipelineJob["status"]): string {
  if (s === "running" || s === "pending") return `is-${s}`;
  if (s === "done") return "is-done";
  if (s === "failed") return "is-failed";
  return "is-stopped";
}

function cmdPreview(job: PipelineJob): string {
  const parts = job.cmd.slice(job.cmd.indexOf("app.main"));
  return parts.length ? `python -m ${parts.join(" ")}` : job.cmd.join(" ");
}

function sessionInputArg(session: string): string {
  if (!session) return "";
  return session.startsWith("session:") ? session : `session:${session}`;
}

function previewCli(session: string, mode: Mode, stage: string, from: string, to: string): string {
  const parts = ["python", "-m", "app.main"];
  const input = sessionInputArg(session);
  if (input) parts.push("--input", input);
  if (mode === "stage") parts.push("--stage", stage);
  else parts.push("--from", from, "--to", to);
  return parts.join(" ");
}

function formatBytes(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatWhen(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function stageState(s: PipelineStageInfo | undefined): "ok" | "stale" | "missing" {
  if (!s?.exists) return "missing";
  if (s.stale) return "stale";
  return "ok";
}

function extraState(extra: PipelineExtraInfo): "ok" | "stale" | "missing" {
  if (!extra.exists) return "missing";
  if (extra.kind === "dir" && extra.files === 0) return "stale";
  return "ok";
}

function StagesTable({
  title,
  report,
  onPick,
}: {
  title: string;
  report: PipelineStaleReport | null;
  onPick: (stage: string) => void;
}) {
  return (
    <div className="pipeline-stages-block">
      <header>
        <strong>{title}</strong>
        {report?.stale?.length ? <em className="is-stale">устарели: {report.stale.join(" → ")}</em> : null}
      </header>
      <table className="pipeline-stages-table">
        <thead>
          <tr>
            <th>Стадия</th>
            <th>Файл</th>
            <th>Статус</th>
            <th>Размер</th>
            <th>Записан</th>
          </tr>
        </thead>
        <tbody>
          {STAGE_GROUPS.map((group) => (
            <Fragment key={group.title}>
              <tr className="pipeline-stage-group">
                <td colSpan={5}>{group.title}</td>
              </tr>
              {group.stages.map((name) => {
            const s = report?.stages?.[name];
            const state = stageState(s);
            const label = state === "ok" ? "готово" : state === "stale" ? "устарел" : "нет";
            const extras = report?.extras?.[name] ?? [];
            return (
              <Fragment key={name}>
                <tr className={`is-${state}`} title={s?.reason ?? undefined}>
                  <td>
                    <button type="button" className="pipeline-stage-pick" onClick={() => onPick(name)}>
                      {STAGE_LABELS[name] ?? name}
                    </button>
                  </td>
                  <td>
                    <code>{s?.file ?? "—"}</code>
                  </td>
                  <td>
                    <span className={`pipeline-stage-badge is-${state}`}>{label}</span>
                    {s?.reason ? <small className="pipeline-stage-reason">{s.reason}</small> : null}
                  </td>
                  <td>{formatBytes(s?.size)}</td>
                  <td>{formatWhen(s?.written_at ?? s?.mtime)}</td>
                </tr>
                {extras.map((extra) => {
                  const st = extraState(extra);
                  const badge = st === "ok" ? "есть" : st === "stale" ? "пусто" : "нет";
                  return (
                    <tr
                      key={`${name}-${extra.key}`}
                      className={`pipeline-stage-sub is-${st}`}
                      title={extra.note ?? undefined}
                    >
                      <td>
                        <span className="pipeline-stage-sub-label">↳ {extra.label}</span>
                      </td>
                      <td>
                        <code>{extra.path}</code>
                        <small className="pipeline-stage-sub-meta">
                          {extra.kind === "dir"
                            ? extra.exists
                              ? `${extra.files ?? 0} файл.`
                              : extra.note || "нет папки"
                            : extra.note || (extra.exists ? "файл" : "нет файла")}
                        </small>
                      </td>
                      <td>
                        <span className={`pipeline-stage-badge is-${st}`}>{badge}</span>
                      </td>
                      <td>{formatBytes(extra.size)}</td>
                      <td>{formatWhen(extra.mtime)}</td>
                    </tr>
                  );
                })}
              </Fragment>
            );
              })}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function PipelineJobsPanel({
  librarySession,
  onPipelineUpdate,
}: Props) {
  const [mode, setMode] = useState<Mode>("range");
  const [stage, setStage] = useState("track");
  const [from, setFrom] = useState("detect");
  const [to, setTo] = useState("track");
  const [jobs, setJobs] = useState<PipelineJob[]>([]);
  const [active, setActive] = useState<PipelineJob | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [logText, setLogText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [apiDown, setApiDown] = useState(false);
  const [sessionPipe, setSessionPipe] = useState<PipelineStaleReport | null>(null);
  const logOffset = useRef(0);
  const logBoxRef = useRef<HTMLPreElement>(null);
  const stickBottom = useRef(true);
  const prevActiveStatus = useRef<string | null>(null);

  const watchId = selectedId ?? active?.id ?? jobs[0]?.id ?? null;

  const refreshStages = useCallback(async () => {
    if (!librarySession) {
      setSessionPipe(null);
      onPipelineUpdate?.(null);
      return;
    }
    try {
      const meta = await fetchMediaMeta(librarySession, { session: true });
      setSessionPipe(meta.pipeline);
      onPipelineUpdate?.(meta.pipeline);
    } catch {
      setSessionPipe(null);
    }
  }, [librarySession, onPipelineUpdate]);

  const refreshList = useCallback(async () => {
    try {
      const [list, act] = await Promise.all([fetchJobs(40), fetchActiveJob()]);
      setJobs(list);
      setActive(act);
      setApiDown(false);
      setError(null);
      if (!selectedId && act) setSelectedId(act.id);
    } catch (e) {
      setApiDown(true);
      setError(e instanceof Error ? e.message : "Job API недоступен");
    }
  }, [selectedId]);

  useEffect(() => {
    void refreshList();
    const t = window.setInterval(() => void refreshList(), 2000);
    return () => window.clearInterval(t);
  }, [refreshList]);

  useEffect(() => {
    void refreshStages();
    const t = window.setInterval(() => void refreshStages(), 4000);
    return () => window.clearInterval(t);
  }, [refreshStages]);

  useEffect(() => {
    const st = active?.status ?? null;
    const prev = prevActiveStatus.current;
    prevActiveStatus.current = st;
    if (prev === "running" && (st === "done" || st === "failed" || st === "stopped" || st == null)) {
      void refreshStages();
    }
  }, [active?.status, refreshStages]);

  useEffect(() => {
    logOffset.current = 0;
    setLogText("");
  }, [watchId]);

  useEffect(() => {
    if (!watchId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const chunk = await fetchJobLog(watchId, logOffset.current);
        if (cancelled) return;
        if (chunk.text) {
          setLogText((prev) => prev + chunk.text);
          logOffset.current = chunk.offset;
        } else {
          logOffset.current = chunk.offset;
        }
      } catch {
        /* ignore poll errors */
      }
    };
    void tick();
    const watching = jobs.find((j) => j.id === watchId);
    const live = watching?.status === "running" || watching?.status === "pending" || active?.id === watchId;
    const t = window.setInterval(() => void tick(), live ? 500 : 2500);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, [watchId, jobs, active?.id]);

  useEffect(() => {
    const el = logBoxRef.current;
    if (!el || !stickBottom.current) return;
    el.scrollTop = el.scrollHeight;
  }, [logText]);

  function pickStage(name: string) {
    setMode("range");
    setFrom(name);
    setTo(name);
  }

  function applyRecompute(report: PipelineStaleReport | null) {
    if (!report?.recompute_from) return;
    if (report.recompute_from === report.recompute_to) {
      setMode("stage");
      setStage(report.recompute_from);
    } else {
      setMode("range");
      setFrom(report.recompute_from);
      setTo(report.recompute_to ?? report.recompute_from);
    }
  }

  async function onStart() {
    const input = sessionInputArg(librarySession);
    if (!input) {
      setError("Выберите session в шапке");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const body = mode === "stage" ? { input, stage } : { input, from, to };
      const job = await createJob(body);
      setSelectedId(job.id);
      await refreshList();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Не удалось запустить");
    } finally {
      setBusy(false);
    }
  }

  async function onStop() {
    const id = active?.id ?? watchId;
    if (!id) return;
    setBusy(true);
    try {
      await stopJob(id);
      await refreshList();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Не удалось остановить");
    } finally {
      setBusy(false);
    }
  }

  async function onRestart(id: string) {
    setBusy(true);
    setError(null);
    try {
      const job = await restartJob(id);
      setSelectedId(job.id);
      await refreshList();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Не удалось перезапустить");
    } finally {
      setBusy(false);
    }
  }

  const running = active?.status === "running" || active?.status === "pending";
  const doneCount = SESSION_STAGE_ORDER.filter((n) => stageState(sessionPipe?.stages?.[n]) === "ok").length;
  const launchCli = useMemo(
    () => previewCli(librarySession, mode, stage, from, to),
    [librarySession, mode, stage, from, to],
  );

  return (
    <div className="pipeline-panel">
      <p className="pipeline-hint">
        Запуск стадий через Job API (<code>python -m app.jobs</code> на :8765). Один job одновременно.
        {librarySession ? (
          <>
            {" "}
            Session <code>{librarySession}</code>: {doneCount}/{SESSION_STAGE_ORDER.length} стадий готовы.
          </>
        ) : null}
      </p>
      {apiDown && (
        <p className="people-banner people-banner-warn">
          Job API не отвечает. В другом терминале: <code>./venv/bin/python -m app.jobs</code>
        </p>
      )}
      {error && !apiDown && <p className="error">{error}</p>}

      <section className="pipeline-artifacts">
        <header>
          <strong>Артефакты</strong>
          <div className="pipeline-artifacts-actions">
            {sessionPipe?.recompute_from ? (
              <button type="button" className="pipeline-btn ghost" onClick={() => applyRecompute(sessionPipe)}>
                Подставить пересчёт
              </button>
            ) : null}
            <button type="button" className="pipeline-btn ghost" onClick={() => void refreshStages()}>
              Обновить
            </button>
          </div>
        </header>
        {!librarySession ? (
          <p className="pipeline-empty">Выберите session, чтобы увидеть стадии и файлы</p>
        ) : (
          <div className="pipeline-artifacts-grid">
            <StagesTable
              title={`Session ${librarySession}`}
              report={sessionPipe}
              onPick={pickStage}
            />
          </div>
        )}
      </section>

      <div className="pipeline-grid">
        <section className="pipeline-controls">
          <div className="pipeline-mode">
            <button type="button" className={mode === "range" ? "on" : ""} onClick={() => setMode("range")}>
              from → to
            </button>
            <button type="button" className={mode === "stage" ? "on" : ""} onClick={() => setMode("stage")}>
              stage
            </button>
          </div>

          {mode === "stage" ? (
            <label className="field">
              <span>Stage</span>
              <select value={stage} onChange={(e) => setStage(e.target.value)}>
                {STAGES.map((s) => (
                  <option key={s} value={s}>
                    {STAGE_LABELS[s] ?? s}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <div className="pipeline-range">
              <label className="field">
                <span>From</span>
                <select value={from} onChange={(e) => setFrom(e.target.value)}>
                  {RANGE_STAGES.map((s) => (
                    <option key={s} value={s}>
                      {STAGE_LABELS[s] ?? s}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>To</span>
                <select value={to} onChange={(e) => setTo(e.target.value)}>
                  {RANGE_STAGES.map((s) => (
                    <option key={s} value={s}>
                      {STAGE_LABELS[s] ?? s}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          )}

          <div className="pipeline-actions">
            <button type="button" className="pipeline-btn primary" disabled={busy || apiDown} onClick={() => void onStart()}>
              Start
            </button>
            <button type="button" className="pipeline-btn" disabled={busy || !running} onClick={() => void onStop()}>
              Stop
            </button>
            <button
              type="button"
              className="pipeline-btn"
              disabled={busy || !watchId}
              onClick={() => watchId && void onRestart(watchId)}
            >
              Restart
            </button>
          </div>

          <p className="pipeline-stages-cli">
            <button
              type="button"
              className="pipeline-btn ghost"
              onClick={() => navigator.clipboard?.writeText(launchCli).catch(() => {})}
              title="Скопировать"
            >
              CLI
            </button>
            <code
              title="Скопировать"
              role="button"
              tabIndex={0}
              onClick={() => navigator.clipboard?.writeText(launchCli).catch(() => {})}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  navigator.clipboard?.writeText(launchCli).catch(() => {});
                }
              }}
            >
              {launchCli}
            </code>
          </p>

          {active && (
            <p className="pipeline-active">
              Активный: <strong className={statusClass(active.status)}>{active.status}</strong>{" "}
              <code>{active.id}</code>
            </p>
          )}
        </section>

        <section className="pipeline-log-wrap">
          <header>
            <strong>Лог</strong>
            <em>{watchId ?? "—"}</em>
          </header>
          <pre
            ref={logBoxRef}
            className="pipeline-log"
            onScroll={(e) => {
              const el = e.currentTarget;
              stickBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
            }}
          >
            {logText || (watchId ? "…" : "Выберите job или запустите новый")}
          </pre>
        </section>

        <section className="pipeline-history">
          <header>
            <strong>История</strong>
            <button type="button" className="pipeline-btn ghost" onClick={() => void refreshList()}>
              Обновить
            </button>
          </header>
          <ul>
            {jobs.map((j) => (
              <li key={j.id} className={watchId === j.id ? "on" : undefined}>
                <button type="button" className="pipeline-job-row" onClick={() => setSelectedId(j.id)}>
                  <span className={`pipeline-status ${statusClass(j.status)}`}>{j.status}</span>
                  <span className="pipeline-job-meta">
                    <strong>{j.input}</strong>
                    <em title={cmdPreview(j)}>{cmdPreview(j)}</em>
                    <small>
                      {j.started_at || j.created_at}
                      {j.exit_code != null ? ` · exit ${j.exit_code}` : ""}
                    </small>
                  </span>
                </button>
              </li>
            ))}
            {!jobs.length && <li className="pipeline-empty">Пока нет запусков</li>}
          </ul>
        </section>
      </div>
    </div>
  );
}

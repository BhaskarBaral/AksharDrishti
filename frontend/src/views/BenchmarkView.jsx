import { useEffect, useState } from "react";
import { getEngines, getScripts, runBenchmark } from "../api.js";
import "./playground.css";
import "./demoScreens.css";

export default function BenchmarkView() {
  const [scripts, setScripts] = useState([]);
  const [script, setScript] = useState(null);
  const [loadError, setLoadError] = useState("");

  const [engines, setEngines] = useState([]);
  const [selectedEngines, setSelectedEngines] = useState(new Set());
  const [limit, setLimit] = useState(15);

  const [status, setStatus] = useState("idle"); // idle | running | done | error | no-engines
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getScripts()
      .then(({ scripts: list }) => {
        setScripts(list);
        if (list.length) setScript(list[0]);
      })
      .catch((err) => setLoadError(err.message));
  }, []);

  useEffect(() => {
    if (!script) return;
    setResult(null);
    setStatus("idle");
    getEngines(script)
      .then(({ engines: list }) => {
        setEngines(list);
        setSelectedEngines(new Set(list));
      })
      .catch(() => {
        setEngines([]);
        setSelectedEngines(new Set());
      });
  }, [script]);

  function toggleEngine(name) {
    const next = new Set(selectedEngines);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    setSelectedEngines(next);
  }

  async function run() {
    if (selectedEngines.size === 0) {
      setStatus("no-engines");
      return;
    }
    setStatus("running");
    setResult(null);
    try {
      const data = await runBenchmark(script, [...selectedEngines], limit);
      setResult(data);
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  if (loadError) {
    return <p style={{ color: "#A8452C", padding: "1.5rem 2rem" }}>Failed to load dataset: {loadError}</p>;
  }

  const bhashiniMissing = script && !engines.includes("bhashini-ocr");

  return (
    <main className="demo-root">
      <div className="s-head">
        <div>
          <h1 className="title">Benchmark</h1>
          <p className="desc">
            The same head-to-head comparison <code>rebenchmark_all.py</code> runs offline, live: pick a script,
            pick engines, and score them against real dataset ground truth. A remote engine like{" "}
            <code>bhashini-ocr</code> runs in this same table, next to your local ones.
          </p>
        </div>
      </div>

      <section className="pg-hero" style={{ marginTop: "1.25rem" }}>
        <div className="pg-script-pills">
          {scripts.map((s) => (
            <button key={s} className={s === script ? "active" : ""} onClick={() => setScript(s)}>
              {s}
            </button>
          ))}
        </div>

        <div style={{ marginTop: "1.1rem" }}>
          <span className="pg-label">Engines to benchmark</span>
          <div className="pg-engine-pills">
            {engines.length === 0 ? (
              <span className="empty">No engine supports this script</span>
            ) : (
              engines.map((name) => (
                <label key={name} className={selectedEngines.has(name) ? "active" : ""}>
                  <input type="checkbox" checked={selectedEngines.has(name)} onChange={() => toggleEngine(name)} />
                  {name}
                  {name.endsWith("-finetuned") && <span className="badge-finetuned">fine-tuned</span>}
                  {name === "bhashini-ocr" && <span className="badge-finetuned">remote</span>}
                </label>
              ))
            )}
          </div>
          {bhashiniMissing && (
            <p className="citizen-hint" style={{ marginTop: "0.6rem" }}>
              bhashini-ocr isn't listed here because BHASHINI_USER_ID / BHASHINI_ULCA_API_KEY / BHASHINI_PIPELINE_ID
              aren't configured on the server -- see app/engines/bhashini_ocr_engine.py.
            </p>
          )}
        </div>

        <div style={{ display: "flex", alignItems: "flex-end", gap: "1rem", marginTop: "1.1rem", flexWrap: "wrap" }}>
          <label style={{ display: "block" }}>
            <span className="pg-label" style={{ marginBottom: "0.35rem" }}>
              Samples to score
            </span>
            <input
              className="api-ctl"
              type="number"
              min={1}
              max={40}
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value) || 1)}
              style={{ width: "6rem" }}
            />
          </label>
          <button onClick={run} disabled={status === "running"}>
            {status === "running" ? "Running..." : "Run benchmark"}
          </button>
        </div>
      </section>

      <section className="pg-results">
        {status === "idle" && <div className="pg-empty">Run a benchmark to see aggregate CER per engine.</div>}
        {status === "no-engines" && <div className="pg-empty">Pick at least one engine above.</div>}
        {status === "running" && (
          <div className="pg-empty">
            <span className="spinner" /> Scoring {[...selectedEngines].length} engine(s) across up to {limit}{" "}
            samples...
          </div>
        )}
        {status === "error" && <div className="pg-empty">Error: {error}</div>}

        {status === "done" && result && (
          <>
            <div className="stats">
              <div className="stat">
                <div className="n">{result.sampleCount}</div>
                <div className="l">Samples scored</div>
              </div>
              <div className="stat">
                <div className="n hi">{result.summary[0]?.engine ?? "—"}</div>
                <div className="l">Best mean CER</div>
              </div>
              <div className="stat">
                <div className="n">{result.summary.length}</div>
                <div className="l">Engines compared</div>
              </div>
            </div>

            <div className="tbl-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Engine</th>
                    <th>Mean CER</th>
                    <th>Scored</th>
                  </tr>
                </thead>
                <tbody>
                  {result.summary.map((row) => {
                    const hasCer = row.meanCer != null;
                    const pct = hasCer ? Math.max(0, Math.min(100, (1 - row.meanCer) * 100)) : 0;
                    const color = !hasCer ? "var(--line)" : row.meanCer <= 0.3 ? "var(--hi)" : "var(--low)";
                    return (
                      <tr key={row.engine}>
                        <td className="fname">
                          {row.engine}
                          {row.engine === "bhashini-ocr" && <span className="badge-finetuned">remote</span>}
                        </td>
                        <td>
                          <span className="cbar">
                            <span style={{ width: `${pct}%`, background: color }} />
                          </span>
                          {hasCer ? row.meanCer.toFixed(3) : "no reference"}
                        </td>
                        <td>
                          {row.scored} / {row.total}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="demo-note">
              Lower CER is better. Each sample is scored on the field with the longest ground truth.
            </p>
          </>
        )}
      </section>
    </main>
  );
}

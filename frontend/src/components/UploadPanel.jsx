import { useEffect, useState } from "react";
import { getEngines, runOnUpload } from "../api.js";

export default function UploadPanel({ scripts }) {
  const [script, setScript] = useState(scripts[0] ?? "");
  const [engines, setEngines] = useState([]);
  const [selectedEngines, setSelectedEngines] = useState(new Set());
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | running | done | error | no-engines
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!script && scripts.length) setScript(scripts[0]);
  }, [scripts, script]);

  useEffect(() => {
    if (!script) return;
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
    setSelectedEngines((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) return;
    if (selectedEngines.size === 0) {
      setStatus("no-engines");
      return;
    }
    setStatus("running");
    try {
      const data = await runOnUpload(script, file, [...selectedEngines]);
      setResults(data.results);
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  return (
    <section id="upload-panel">
      <h2>4. Try your own image</h2>
      <form id="upload-form" onSubmit={handleSubmit}>
        <label htmlFor="upload-script">Script</label>
        <select id="upload-script" value={script} onChange={(e) => setScript(e.target.value)}>
          {scripts.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <div className="row">
          <span className="engine-select-label">Engines to run</span>
          <div className="engine-checkboxes">
            {engines.map((name) => (
              <label key={name}>
                <input
                  type="checkbox"
                  checked={selectedEngines.has(name)}
                  onChange={() => toggleEngine(name)}
                />
                {name}
              </label>
            ))}
          </div>
        </div>
        <input
          type="file"
          accept="image/*"
          required
          onChange={(e) => setFile(e.target.files[0] ?? null)}
        />
        <button type="submit">Run OCR</button>
      </form>

      {status !== "idle" && (
        <table id="upload-results-table">
          <thead>
            <tr>
              <th>Engine</th>
              <th>Hypothesis</th>
            </tr>
          </thead>
          <tbody>
            {status === "running" && (
              <tr>
                <td colSpan={2} style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
                  <span className="spinner" /> Running OCR...
                </td>
              </tr>
            )}
            {status === "no-engines" && (
              <tr>
                <td colSpan={2}>Select at least one engine to run.</td>
              </tr>
            )}
            {status === "error" && (
              <tr>
                <td colSpan={2}>Error: {error}</td>
              </tr>
            )}
            {status === "done" &&
              results.map((r, i) => (
                <tr key={i}>
                  <td>{r.engine}</td>
                  <td>{r.hypothesis}</td>
                </tr>
              ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

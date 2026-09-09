function EngineLabel({ name }) {
  const isFinetuned = name.endsWith("-finetuned");
  return (
    <>
      {name}
      {isFinetuned && <span className="badge-finetuned">fine-tuned</span>}
    </>
  );
}

export default function ResultsPanel({ engines, selectedEngines, onToggleEngine, status, result, error }) {
  return (
    <section id="results-panel">
      <h2>3. OCR result</h2>
      <div className="row">
        <span className="engine-select-label">Engines to run</span>
        <div className="engine-checkboxes">
          {engines.length === 0 ? (
            <span className="empty">No engine supports this script</span>
          ) : (
            engines.map((name) => (
              <label key={name}>
                <input
                  type="checkbox"
                  checked={selectedEngines.has(name)}
                  onChange={() => onToggleEngine(name)}
                />
                <EngineLabel name={name} />
              </label>
            ))
          )}
        </div>
      </div>

      {status !== "done" && (
        <div className="empty">
          {status === "running" && <span className="spinner" />}
          {status === "running"
            ? "Running OCR..."
            : status === "no-engines"
            ? "Select at least one engine to run."
            : status === "error"
            ? `Error: ${error}`
            : "Run OCR on a field to see engine output here."}
        </div>
      )}

      {status === "done" && result && (
        <>
          <table id="results-table">
            <thead>
              <tr>
                <th>Engine</th>
                <th>Hypothesis</th>
                <th>CER</th>
              </tr>
            </thead>
            <tbody>
              {result.results.map((r, i) => {
                const hasCer = r.cer !== null && r.cer !== undefined;
                const cerClass = hasCer ? (r.cer <= 0.3 ? "cer-good" : "cer-bad") : "cer-neutral";
                return (
                  <tr key={i}>
                    <td>
                      <EngineLabel name={r.engine} />
                    </td>
                    <td>{r.hypothesis}</td>
                    <td>
                      <span className={`cer-badge ${cerClass}`}>{hasCer ? r.cer.toFixed(3) : "-"}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div id="results-gt">
            <p>
              <strong>Ground truth:</strong> {result.groundTruth}
            </p>
          </div>
        </>
      )}
    </section>
  );
}

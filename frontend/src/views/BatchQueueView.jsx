import "./demoScreens.css";

const ROWS = [
  { name: "land-record-8842", ext: "pdf", lang: "Hindi", script: "Devanagari", pages: 4, confidence: 96, status: "done" },
  { name: "court-order-1190", ext: "pdf", lang: "Bengali", script: "Bengali", pages: 7, confidence: 74, status: "review" },
  { name: "pension-form-3301", ext: "jpg", lang: "Tamil", script: "Tamil", pages: 1, confidence: 89, status: "done" },
  { name: "nikah-nama-0455", ext: "pdf", lang: "Urdu", script: "Perso-Arabic", pages: 2, confidence: 68, status: "review" },
  { name: "survey-sheet-7720", ext: "tiff", lang: "Marathi", script: "Devanagari", pages: 3, confidence: null, status: "proc" },
  { name: "gazette-2026-88", ext: "pdf", lang: "Odia", script: "Odia", pages: 12, confidence: null, status: "queue" },
];

const STATUS_LABEL = { done: "Done", review: "Review", proc: "Processing", queue: "Queued" };

function confColor(c) {
  if (c == null) return "var(--line)";
  if (c >= 90) return "var(--hi)";
  if (c >= 75) return "var(--med)";
  return "var(--low)";
}

export default function BatchQueueView() {
  return (
    <main className="demo-root">
      <div className="s-head">
        <div>
          <h1 className="title">Batch queue</h1>
          <p className="desc">
            The screen for scale — digitizing thousands of records at once. Operators triage by status and
            confidence instead of opening documents one by one, and only the flagged ones need a human look.
            Built for the government-digitization use case.
          </p>
        </div>
        <div className="actions">
          <button className="outline">Upload batch</button>
          <button>Export approved</button>
        </div>
      </div>

      <div className="stats">
        <div className="stat"><div className="n">3,418</div><div className="l">Documents</div></div>
        <div className="stat"><div className="n low">247</div><div className="l">Awaiting review</div></div>
        <div className="stat"><div className="n hi">92.6%</div><div className="l">Avg confidence</div></div>
        <div className="stat"><div className="n">11</div><div className="l">Languages seen</div></div>
      </div>

      <div className="tbl-wrap">
        <table>
          <thead>
            <tr><th>Document</th><th>Language</th><th>Pages</th><th>Confidence</th><th>Status</th><th></th></tr>
          </thead>
          <tbody>
            {ROWS.map((r) => (
              <tr key={r.name}>
                <td className="fname">{r.name}<span className="fx">.{r.ext}</span></td>
                <td><div className="lang-cell">{r.lang}<span className="sc">{r.script}</span></div></td>
                <td>{r.pages}</td>
                <td>
                  <span className="cbar">
                    <span style={{ width: `${r.confidence ?? 0}%`, background: confColor(r.confidence) }} />
                  </span>
                  {r.confidence != null ? `${r.confidence}%` : "—"}
                </td>
                <td><span className={`pill ${r.status}`}>{STATUS_LABEL[r.status]}</span></td>
                <td><button className="outline">Open</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="demo-note">Sample data — this dashboard isn't wired to a live batch pipeline yet.</p>
    </main>
  );
}

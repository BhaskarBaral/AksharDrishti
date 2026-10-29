import { useEffect, useRef, useState } from "react";
import "./layout.css";

// Fetches live here rather than in api.js, which the OCR views own.
async function jsonFetch(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

const SCRIPTS = ["devanagari", "bengali", "tamil", "telugu", "urdu", "gujarati", "kannada", "malayalam", "odia", "punjabi"];
const RTL = new Set(["urdu"]);

// Region roles (app/layout_preserve.py's CLASS_GROUP) -> colour + label.
const GROUPS = {
  title: { color: "#273A6B", label: "Titles" },
  text: { color: "#4F6BB5", label: "Text" },
  list: { color: "#6A7FC1", label: "Lists" },
  table: { color: "#2F6B4A", label: "Tables" },
  caption: { color: "#5E8C6A", label: "Captions" },
  "form-field": { color: "#7A5BA8", label: "Form fields" },
  figure: { color: "#8B857C", label: "Figures / images" },
  "page-furniture": { color: "#A39E95", label: "Header / footer" },
  seal: { color: "#B03A2E", label: "Seals" },
  stamp: { color: "#D2691E", label: "Stamps" },
  signature: { color: "#A8781F", label: "Signatures" },
};
const GT_COLOR = "#C2185B";

function groupOf(cls) {
  if (["seal", "stamp", "signature"].includes(cls)) return cls;
  if (/title|headline/.test(cls)) return "title";
  if (/list|options/.test(cls)) return "list";
  if (/^table|table-of-contents/.test(cls)) return cls.endsWith("caption") ? "caption" : "table";
  if (/caption/.test(cls)) return "caption";
  if (["figure", "advertisement", "flag", "formula"].includes(cls)) return "figure";
  if (["header", "footer", "page-number", "folio", "jumpline"].includes(cls)) return "page-furniture";
  if (cls === "placeholder-text") return "form-field";
  return "text";
}

function download(name, type, content) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

// Standalone HTML that redraws the page: every region absolutely positioned
// at its detected place, carrying its OCR text -- the preserved layout.
function toHtml(result) {
  const { width, height, regions, script } = result;
  const blocks = regions
    .map((r) => {
      const [x0, y0, x1, y1] = r.bbox;
      const style = `left:${(x0 / width) * 100}%;top:${(y0 / height) * 100}%;width:${((x1 - x0) / width) * 100}%;height:${((y1 - y0) / height) * 100}%`;
      let inner;
      if (r.table?.length) {
        inner = `<table>${r.table.map((row) => `<tr>${row.map((c) => `<td>${escapeHtml(c)}</td>`).join("")}</tr>`).join("")}</table>`;
      } else if (["figure", "seal", "stamp", "signature"].includes(r.group)) {
        inner = `<em>[${escapeHtml(r.class)}]</em>`;
      } else {
        inner = escapeHtml(r.text || "").replace(/\n/g, "<br>");
      }
      return `<div class="r ${r.group}" style="${style}">${inner}</div>`;
    })
    .join("\n");
  return `<!doctype html><html><head><meta charset="utf-8"><title>Preserved layout</title><style>
body{margin:0;background:#eee;font-family:"Noto Sans","Noto Sans Devanagari","Noto Sans Bengali","Noto Sans Tamil","Noto Sans Telugu","Noto Nastaliq Urdu",sans-serif}
.page{position:relative;width:min(900px,96vw);aspect-ratio:${width}/${height};margin:16px auto;background:#fff;box-shadow:0 2px 8px #0002}
.r{position:absolute;overflow:hidden;font-size:11px;line-height:1.25;direction:${RTL.has(script) ? "rtl" : "ltr"}}
.title{font-weight:700;font-size:14px}.figure,.seal,.stamp,.signature{border:1px dashed #999;color:#777;display:flex;align-items:center;justify-content:center}
table{border-collapse:collapse;width:100%}td{border:1px solid #bbb;padding:1px 3px;font-size:10px}
</style></head><body><div class="page">${blocks}</div></body></html>`;
}

export default function LayoutView() {
  const [samples, setSamples] = useState([]);
  const [sampleGroups, setSampleGroups] = useState([]);
  const [modelName, setModelName] = useState("");
  const [selectedSample, setSelectedSample] = useState(null);
  const [script, setScript] = useState("devanagari");
  const [file, setFile] = useState(null);
  const [previewSrc, setPreviewSrc] = useState(null);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [result, setResult] = useState(null);
  const [view, setView] = useState("regions"); // regions | preserved | markdown
  const [showGroundTruth, setShowGroundTruth] = useState(false);
  const [hiddenGroups, setHiddenGroups] = useState(new Set());
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    jsonFetch("/api/layout/samples")
      .then(({ samples: list, groups }) => {
        setSamples(list);
        setSampleGroups(groups);
      })
      .catch(() => setSamples([]));
    jsonFetch("/api/layout/status").then((s) => setModelName(s.model)).catch(() => {});
  }, []);

  async function run(form, url) {
    setResult(null);
    setSelectedRegion(null);
    setShowGroundTruth(false);
    setStatus("running");
    const t0 = performance.now();
    try {
      setResult(await jsonFetch(url, { method: "POST", body: form }));
      setElapsed((performance.now() - t0) / 1000);
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  function pickSample(s) {
    setSelectedSample(s.id);
    setScript(s.script);
    setFile(null);
    setPreviewSrc(`/api/layout/samples/image?id=${encodeURIComponent(s.id)}`);
    setImgSize({ w: 0, h: 0 });
    const form = new FormData();
    form.set("id", s.id);
    form.set("script", s.script);
    run(form, "/api/layout/analyze-sample");
  }

  function analyzeFile(f, scr) {
    const form = new FormData();
    form.set("file", f);
    form.set("script", scr);
    run(form, "/api/layout/analyze-upload");
  }

  function handleFiles(fileList) {
    const f = fileList?.[0];
    if (!f) return;
    setSelectedSample(null);
    setFile(f);
    setPreviewSrc(URL.createObjectURL(f));
    setImgSize({ w: 0, h: 0 });
    analyzeFile(f, script);
  }

  function changeScript(scr) {
    setScript(scr);
    if (file) analyzeFile(file, scr);
    else if (selectedSample) {
      const form = new FormData();
      form.set("id", selectedSample);
      form.set("script", scr);
      run(form, "/api/layout/analyze-sample");
    }
  }

  function toggleGroup(g) {
    setHiddenGroups((prev) => {
      const next = new Set(prev);
      next.has(g) ? next.delete(g) : next.add(g);
      return next;
    });
  }

  // Overlay boxes are in the analysed image's pixels; the server reports its size.
  const W = result?.width || imgSize.w;
  const H = result?.height || imgSize.h;
  function pos([x0, y0, x1, y1]) {
    return {
      left: `${(x0 / W) * 100}%`,
      top: `${(y0 / H) * 100}%`,
      width: `${((x1 - x0) / W) * 100}%`,
      height: `${((y1 - y0) / H) * 100}%`,
    };
  }

  const regions = result?.regions ?? [];
  const overlay = showGroundTruth && result?.groundTruth
    ? result.groundTruth.map((r) => ({ ...r, group: groupOf(r.class), gt: true }))
    : regions;
  const counts = {};
  for (const r of overlay) counts[r.group] = (counts[r.group] ?? 0) + 1;
  const stem = (file?.name || selectedSample?.split("/").pop() || "page").replace(/\.[^.]+$/, "");

  return (
    <main className="demo-root">
      <div className="s-head">
        <div>
          <h1 className="title">Layout detection &amp; preservation</h1>
          <p className="desc">
            Detects titles, paragraphs, lists, tables, form fields, figures, headers/footers, seals, stamps and
            signatures, reads each region with the script's OCR engine, and rebuilds the page in reading order —
            tables as rows and columns — as a redrawn page, Markdown, HTML or JSON.
            {modelName && <> Model: <strong>{modelName}</strong>.</>}
          </p>
        </div>
      </div>

      <div className="ly-toolbar">
        <label className="ly-script">
          Script
          <select value={script} onChange={(e) => changeScript(e.target.value)}>
            {SCRIPTS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>
        <button className="primary" onClick={() => fileInputRef.current?.click()}>Upload a page…</button>
        <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={(e) => handleFiles(e.target.files)} />
      </div>
      {/* TestData/layout/<group>/ -- synthetic pages narrowed to the picked script. */}
      <div className="ly-sample-groups">
        {sampleGroups.map((g) => {
          const items = samples.filter((s) => s.group === g.id && (g.id !== "synthetic_labelled" || s.script === script));
          if (!items.length) return null;
          return (
            <div key={g.id} className="ly-sample-group">
              <span className="ly-sample-group-label">{g.label}</span>
              <div className="ly-samples">
                {items.map((s) => (
                  <button key={s.id} className={selectedSample === s.id ? "active" : ""} onClick={() => pickSample(s)}>
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {status === "done" && (
        <div className="ly-viewbar">
          <div className="ly-toggle">
            <button className={view === "regions" ? "active" : ""} onClick={() => setView("regions")}>Regions</button>
            <button className={view === "preserved" ? "active" : ""} onClick={() => setView("preserved")}>Preserved layout</button>
            <button className={view === "markdown" ? "active" : ""} onClick={() => setView("markdown")}>Markdown</button>
          </div>
          <div className="ly-exports">
            <button className="ghost" onClick={() => download(`${stem}.layout.json`, "application/json", JSON.stringify(result, null, 2))}>JSON</button>
            <button className="ghost" onClick={() => download(`${stem}.md`, "text/markdown", result.markdown)}>Markdown</button>
            <button className="ghost" onClick={() => download(`${stem}.html`, "text/html", toHtml(result))}>HTML</button>
          </div>
        </div>
      )}

      <section className="ly-stage">
        <div className="ly-page">
          {!previewSrc && <div className="ly-empty">Pick a sample above or upload a full page image.</div>}
          {previewSrc && view !== "markdown" && (
            <div className={`ly-frame${view === "preserved" ? " preserved" : ""}`}>
              <img
                src={previewSrc}
                alt="page"
                onLoad={(e) => setImgSize({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
              />
              {W > 0 && view === "regions" &&
                overlay
                  .filter((r) => !hiddenGroups.has(r.group))
                  .map((r, i) => (
                    <div
                      key={i}
                      className={`ly-box${selectedRegion === r.id ? " selected" : ""}`}
                      style={{
                        ...pos(r.bbox),
                        borderColor: r.gt ? GT_COLOR : GROUPS[r.group].color,
                        background: `${r.gt ? GT_COLOR : GROUPS[r.group].color}1a`,
                      }}
                      onClick={() => !r.gt && setSelectedRegion(r.id)}
                      title={r.text || r.class}
                    >
                      <span className="ly-box-label" style={{ background: r.gt ? GT_COLOR : GROUPS[r.group].color }}>
                        {r.class}{r.score != null ? ` ${(r.score * 100).toFixed(0)}%` : ""}
                      </span>
                    </div>
                  ))}
              {W > 0 && view === "preserved" && (
                <div className="ly-rebuilt" dir={RTL.has(result.script) ? "rtl" : "ltr"}>
                  {regions.map((r) => (
                    <div key={r.id} className={`ly-rb ly-rb-${r.group}`} style={pos(r.bbox)}>
                      {r.table?.length ? (
                        <table>
                          <tbody>
                            {r.table.map((row, i) => (
                              <tr key={i}>{row.map((c, j) => <td key={j}>{c}</td>)}</tr>
                            ))}
                          </tbody>
                        </table>
                      ) : ["figure", "seal", "stamp", "signature"].includes(r.group) ? (
                        <em>[{r.class}]</em>
                      ) : (
                        r.text
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          {view === "markdown" && result && (
            <pre className="ly-md" dir={RTL.has(result.script) ? "rtl" : "ltr"}>{result.markdown || "(no text found)"}</pre>
          )}
        </div>

        <aside className="ly-side">
          {status === "idle" && <div className="ly-empty">Results appear here.</div>}
          {status === "running" && (
            <div className="ly-empty">
              <span className="spinner" /> Detecting regions and reading text...
            </div>
          )}
          {status === "error" && <div className="ly-empty">Error: {error}</div>}
          {status === "done" && (
            <>
              <div className="ly-stats">
                <div><strong>{regions.length}</strong><span>regions</span></div>
                <div><strong>{regions.filter((r) => r.group === "table").length}</strong><span>tables</span></div>
                <div><strong>{elapsed != null ? `${elapsed.toFixed(1)}s` : "-"}</strong><span>round trip</span></div>
              </div>
              <p className="ly-note">OCR: {result.ocrEngine ?? "no page-level engine for this script"}</p>

              {result.groundTruth && view === "regions" && (
                <div className="ly-toggle">
                  <button className={!showGroundTruth ? "active" : ""} onClick={() => setShowGroundTruth(false)}>Model output</button>
                  <button className={showGroundTruth ? "active" : ""} onClick={() => setShowGroundTruth(true)}>Answer key</button>
                </div>
              )}

              <div className="ly-legend">
                {Object.entries(GROUPS).map(([g, { color, label }]) => (
                  <button
                    key={g}
                    className={`ly-legend-item${hiddenGroups.has(g) ? " off" : ""}`}
                    onClick={() => toggleGroup(g)}
                    disabled={!counts[g]}
                  >
                    <span className="ly-swatch" style={{ background: color }} />
                    <span className="ly-legend-name">{label}</span>
                    <span className="ly-legend-count">{counts[g] ?? 0}</span>
                  </button>
                ))}
              </div>

              <div className="ly-regions">
                {regions.map((r) => (
                  <div
                    key={r.id}
                    className={`ly-region${selectedRegion === r.id ? " selected" : ""}`}
                    onClick={() => setSelectedRegion(r.id)}
                  >
                    <span className="ly-swatch" style={{ background: GROUPS[r.group].color }} />
                    <div>
                      <div className="ly-region-class">{r.class}{r.score != null ? ` · ${(r.score * 100).toFixed(0)}%` : ""}</div>
                      {r.text && <div className="ly-region-text" dir="auto">{r.text}</div>}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </aside>
      </section>
    </main>
  );
}

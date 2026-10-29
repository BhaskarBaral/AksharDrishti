import { useEffect, useRef, useState } from "react";
import { getEngines, getPageSamples, pageSampleImageUrl, runPageUpload, createGroundTruth } from "../api.js";
import "./playground.css";

const REVIEW_THRESHOLD = 0.75;
const HIGH_THRESHOLD = 0.9;

const TEXT_TYPES = [
  { value: "computer-typed-text", label: "Printed" },
  { value: "handwritten-text", label: "Handwritten" },
];

function confTier(confidence) {
  if (confidence >= HIGH_THRESHOLD) return "high";
  if (confidence >= REVIEW_THRESHOLD) return "med";
  return "low";
}

// page CER is much noisier than a single field's (reading order, headers),
// so the bands are looser than Word level's 0.3 pass mark
function cerTier(cer) {
  if (cer <= 0.1) return "high";
  if (cer <= 0.3) return "med";
  return "low";
}

function splitKV(text) {
  const idx = text.indexOf(":");
  if (idx === -1 || idx === text.length - 1) return { key: null, value: text };
  const key = text.slice(0, idx).trim();
  const value = text.slice(idx + 1).trim();
  if (!key || !value) return { key: null, value: text };
  return { key, value };
}

const VIEW_META = {
  lines: {
    title: "Review workspace",
    desc: "The core operator screen for accuracy-critical work. Every line carries a confidence score, and low-confidence lines surface for correction before approval.",
    exportLabel: "Export JSON",
    approveLabel: "Approve page",
  },
  fields: {
    title: "Form field extraction",
    desc: "Pull structured fields out as key→value pairs instead of raw prose — each one independently scored and editable.",
    exportLabel: "Export CSV",
    approveLabel: "Confirm fields",
  },
  regions: {
    title: "Region overlay",
    desc: "Verification stays spatial: detected regions are drawn back onto the page, coloured by confidence, so you can see exactly where the model was unsure.",
    exportLabel: "Export JSON",
    approveLabel: "Approve page",
  },
};

export default function DocumentMode({ script }) {
  // page-capable engines, best first, per document type (server-ranked from
  // measured page CER -- see PAGE_ENGINE_PREFERENCE in ocr_router.py)
  const [pageEngines, setPageEngines] = useState({ printed: [], handwritten: [] });
  const [enginesLoaded, setEnginesLoaded] = useState(false);
  const [engine, setEngine] = useState(null);
  const [engineTouched, setEngineTouched] = useState(false); // operator picked one by hand
  const [imageTextType, setImageTextType] = useState("computer-typed-text");

  const [pageSamples, setPageSamples] = useState([]);
  const [activeSampleId, setActiveSampleId] = useState(null);
  const [groundTruth, setGroundTruth] = useState(null);

  const [previewSrc, setPreviewSrc] = useState(null);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  // one OCR result per engine for the current page, so switching engines
  // back and forth (or "Compare all engines") doesn't re-run anything
  const [resultsByEngine, setResultsByEngine] = useState({});
  const [selectedLine, setSelectedLine] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [runningLabel, setRunningLabel] = useState("");
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [copied, setCopied] = useState(false);

  const [currentFile, setCurrentFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [edits, setEdits] = useState({});
  const [editingIndex, setEditingIndex] = useState(null);
  const [panelView, setPanelView] = useState("lines"); // "lines" | "fields" | "regions"
  const [approved, setApproved] = useState(false);
  const [showAllBoxes, setShowAllBoxes] = useState(true);

  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState("");
  const [savedInfo, setSavedInfo] = useState(null); // {recordCount, savedTo}

  const fileInputRef = useRef(null);
  const requestRef = useRef(0); // drops responses for a page that's no longer loaded

  const allPageEngines = pageEngines.printed;
  const supported = allPageEngines.length > 0;

  function defaultEngineFor(textType, lists = pageEngines) {
    const list = textType === "handwritten-text" ? lists.handwritten : lists.printed;
    return list[0] ?? null;
  }

  function resetPage() {
    requestRef.current += 1;
    setPreviewSrc(null);
    setImgSize({ w: 0, h: 0 });
    setResultsByEngine({});
    setSelectedLine(null);
    setStatus("idle");
    setError("");
    setCurrentFile(null);
    setFileName("");
    setActiveSampleId(null);
    setGroundTruth(null);
    resetReview();
  }

  function resetReview() {
    setEdits({});
    setEditingIndex(null);
    setApproved(false);
    setSaving(false);
    setSaveError("");
    setSavedInfo(null);
  }

  useEffect(() => {
    resetPage();
    setEnginesLoaded(false);
    setEngine(null);
    setEngineTouched(false);
    setPageSamples([]);
    if (!script) return;

    getEngines(script)
      .then((data) => {
        const lists = { printed: data.pageEngines ?? [], handwritten: data.pageEnginesHandwritten ?? data.pageEngines ?? [] };
        setPageEngines(lists);
        setEngine(defaultEngineFor(imageTextType, lists));
      })
      .catch(() => setPageEngines({ printed: [], handwritten: [] }))
      .finally(() => setEnginesLoaded(true));

    getPageSamples(script)
      .then(({ samples }) => setPageSamples(samples))
      .catch(() => setPageSamples([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [script]);

  async function runPage(file, engineNames, gt) {
    const reqId = requestRef.current;
    setStatus("running");
    setRunningLabel(engineNames.length > 1 ? `Running ${engineNames.length} engines...` : `Running ${engineNames[0]}...`);
    try {
      const data = await runPageUpload(script, file, engineNames, gt);
      if (reqId !== requestRef.current) return;
      setResultsByEngine((prev) => {
        const next = { ...prev };
        data.results.forEach((r) => {
          next[r.engine] = r;
        });
        return next;
      });
      setStatus("done");
    } catch (err) {
      if (reqId !== requestRef.current) return;
      setError(err.message);
      setStatus("error");
    }
  }

  function loadFile(file, { gt = null, textType = null, sampleId = null, engineHint = null } = {}) {
    if (!supported || !file) return;
    resetPage();
    let nextType = imageTextType;
    if (textType) {
      nextType = textType;
      setImageTextType(textType);
    }
    let nextEngine = engine;
    if (!engineTouched || !nextEngine) {
      // a dataset sample knows which engine reads it best (see /page-samples)
      nextEngine = (engineHint && allPageEngines.includes(engineHint) && engineHint) || defaultEngineFor(nextType);
      setEngine(nextEngine);
    }
    setPreviewSrc(URL.createObjectURL(file));
    setCurrentFile(file);
    setFileName(file.name);
    setActiveSampleId(sampleId);
    setGroundTruth(gt);
    runPage(file, [nextEngine], gt);
  }

  function handleFiles(fileList) {
    loadFile(fileList?.[0]);
  }

  async function pickSample(s) {
    try {
      const res = await fetch(pageSampleImageUrl(script, s.id));
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const blob = await res.blob();
      const file = new File([blob], s.filename, { type: blob.type || "image/png" });
      loadFile(file, { gt: s.groundTruth, textType: s.imageTextType, sampleId: s.id, engineHint: s.recommendedEngine });
    } catch (err) {
      setError(`Couldn't load sample page: ${err.message}`);
      setStatus("error");
    }
  }

  function switchEngine(name) {
    if (name === engine || status === "running") return;
    setEngine(name);
    setSelectedLine(null);
    resetReview();
    if (currentFile && !resultsByEngine[name]) runPage(currentFile, [name], groundTruth);
    else if (currentFile) setStatus("done");
  }

  function pickEngine(name) {
    setEngineTouched(true);
    switchEngine(name);
  }

  function pickTextType(value) {
    setImageTextType(value);
    if (!engineTouched) {
      const next = defaultEngineFor(value);
      if (next) switchEngine(next);
    }
  }

  function compareAll() {
    if (!currentFile) return;
    const missing = allPageEngines.filter((n) => !resultsByEngine[n]);
    if (missing.length) runPage(currentFile, missing, groundTruth);
  }

  function handleImgLoad(e) {
    setImgSize({ w: e.target.naturalWidth, h: e.target.naturalHeight });
  }

  function boxStyle(bbox) {
    const xs = bbox.map((p) => p[0]);
    const ys = bbox.map((p) => p[1]);
    const x0 = Math.min(...xs);
    const x1 = Math.max(...xs);
    const y0 = Math.min(...ys);
    const y1 = Math.max(...ys);
    return {
      left: `${(x0 / imgSize.w) * 100}%`,
      top: `${(y0 / imgSize.h) * 100}%`,
      width: `${((x1 - x0) / imgSize.w) * 100}%`,
      height: `${((y1 - y0) / imgSize.h) * 100}%`,
    };
  }

  const current = engine ? resultsByEngine[engine] : null;
  const lines = current?.lines ?? [];
  // keep the current engine's result on screen while "Compare all engines" runs
  const showResult = status !== "error" && current;

  function textFor(i) {
    return edits[i] ?? lines[i].text;
  }

  function commitEdit(i, value) {
    setEdits((prev) => ({ ...prev, [i]: value }));
    setEditingIndex(null);
  }

  const avgConfidence = lines.length
    ? lines.reduce((sum, l) => sum + (l.confidence ?? 0), 0) / lines.length
    : null;
  const needsReviewCount = lines.filter((l) => l.confidence < REVIEW_THRESHOLD).length;
  const transcript = lines.map((_, i) => textFor(i)).join("\n");
  const comparedEngines = allPageEngines.filter((n) => resultsByEngine[n]);

  async function copyTranscript() {
    try {
      await navigator.clipboard.writeText(transcript);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard API unavailable in this context -- text is still on screen to select manually
    }
  }

  function exportJson() {
    const payload = {
      script,
      engine,
      fileName,
      approved,
      lines: lines.map((l, i) => ({ bbox: l.bbox, text: textFor(i), confidence: l.confidence })),
      text: transcript,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${(fileName || "document").replace(/\.[^.]+$/, "")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function reRun() {
    if (!currentFile || !engine) return;
    resetReview();
    setResultsByEngine((prev) => {
      const next = { ...prev };
      delete next[engine];
      return next;
    });
    runPage(currentFile, [engine], groundTruth);
  }

  async function saveGroundTruth() {
    if (!currentFile || !lines.length) return;
    setSaving(true);
    setSaveError("");
    setSavedInfo(null);
    try {
      const payload = lines.map((l, i) => ({
        bbox: l.bbox,
        text: textFor(i),
        confidence: l.confidence,
      }));
      const data = await createGroundTruth(script, currentFile, payload, engine, imageTextType);
      setSavedInfo(data);
    } catch (err) {
      setSaveError(err.message);
    } finally {
      setSaving(false);
    }
  }

  if (!enginesLoaded) {
    return (
      <div className="pg-empty">
        <span className="spinner" /> Loading page-level engines...
      </div>
    );
  }

  if (!supported) {
    return (
      <div className="pg-doc-disabled">
        No page-level engine supports <strong>{script}</strong> yet — none of the engines here has both a text
        detector and a recogniser for this script. Pick another script above, or use Word level for single-word crops.
      </div>
    );
  }

  const viewMeta = VIEW_META[panelView] ?? VIEW_META.lines;
  const activeSample = pageSamples.find((s) => s.id === activeSampleId);
  const recommendedEngine = activeSample?.recommendedEngine ?? defaultEngineFor(imageTextType);

  return (
    <>
      <section className="pg-doc-toolbar">
        <div>
          <span className="pg-label">Page engine</span>
          <div className="pg-engine-pills">
            {allPageEngines.map((name) => (
              <label key={name} className={name === engine ? "active" : ""}>
                <input
                  type="radio"
                  name="page-engine"
                  checked={name === engine}
                  disabled={status === "running"}
                  onChange={() => pickEngine(name)}
                />
                {name}
                {name.endsWith("-finetuned") && <span className="badge-finetuned">fine-tuned</span>}
                {name === recommendedEngine && <span className="badge-recommended">recommended</span>}
              </label>
            ))}
          </div>
        </div>
        <div>
          <span className="pg-label">Document type</span>
          <div className="pg-doc-view-toggle">
            {TEXT_TYPES.map((t) => (
              <button
                key={t.value}
                className={imageTextType === t.value ? "active" : ""}
                onClick={() => pickTextType(t.value)}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
        {allPageEngines.length > 1 && (
          <div className="pg-doc-toolbar-end">
            <button
              className="pg-doc-compare-btn"
              onClick={compareAll}
              disabled={!currentFile || status === "running" || comparedEngines.length === allPageEngines.length}
            >
              Compare all engines
            </button>
          </div>
        )}
      </section>

      {showResult && (
        <div className="pg-doc-meta">
          <div className="s-head pg-doc-shead">
            <div>
              <h1 className="title">{viewMeta.title}</h1>
              <p className="desc">{viewMeta.desc}</p>
            </div>
            <div className="pg-doc-actions">
              {panelView === "regions" && (
                <button className="ghost" onClick={() => setShowAllBoxes((v) => !v)}>
                  {showAllBoxes ? "Hide box labels" : "Show box labels"}
                </button>
              )}
              <button className="ghost" onClick={reRun}>Re-run OCR</button>
              <button onClick={exportJson}>{viewMeta.exportLabel}</button>
              <button className={approved ? "approved" : "primary"} onClick={() => setApproved(true)} disabled={approved}>
                {approved ? "Approved" : viewMeta.approveLabel}
              </button>
            </div>
          </div>
          <div className="pg-doc-tags">
            <span className="tag">{fileName}</span>
            <span className="tag">Script: {script}</span>
            <span className="tag">Engine: {engine}</span>
            {current.seconds != null && <span className="tag">{current.seconds.toFixed(1)}s</span>}
            <span className={`tag conf-${confTier(avgConfidence ?? 0)}`}>
              page avg {avgConfidence != null ? `${(avgConfidence * 100).toFixed(0)}%` : "-"}
            </span>
            {current.cer != null && (
              <span className={`tag conf-${cerTier(current.cer)}`}>page CER {current.cer.toFixed(3)} vs ground truth</span>
            )}
            {needsReviewCount > 0 && <span className="tag conf-low">{needsReviewCount} need review</span>}
          </div>

          <div className="pg-gt-bar">
            <span className="pg-gt-note">
              Saves each reviewed line as a ULCA record ({TEXT_TYPES.find((t) => t.value === imageTextType)?.label.toLowerCase()} text)
            </span>
            <button className="primary" onClick={saveGroundTruth} disabled={saving || !lines.length}>
              {saving ? "Saving..." : "Create Ground Truth & Record (ULCA)"}
            </button>
          </div>
          {saveError && <div className="pg-gt-error">Couldn't save: {saveError}</div>}
          {savedInfo && (
            <div className="pg-gt-success">
              Saved <strong>{savedInfo.recordCount}</strong> ULCA record{savedInfo.recordCount === 1 ? "" : "s"} +
              the image to <code>{savedInfo.savedTo}</code>
            </div>
          )}
        </div>
      )}

      <section className="pg-stage">
        <div
          className={`pg-dropzone${dragActive ? " drag" : ""}${previewSrc ? " has-preview" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            setDragActive(false);
          }}
          onDrop={(e) => {
            e.preventDefault();
            setDragActive(false);
            handleFiles(e.dataTransfer.files);
          }}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => {
              handleFiles(e.target.files);
              e.target.value = "";
            }}
          />
          {!previewSrc && (
            <div className="pg-dropzone-empty">
              <div className="pg-dropzone-icon">&#8679;</div>
              <p>Drag &amp; drop a full page image here, or click to browse</p>
            </div>
          )}
          {previewSrc && (
            <div className="pg-preview-frame">
              <img src={previewSrc} onLoad={handleImgLoad} alt="document preview" />
              {imgSize.w > 0 &&
                showResult &&
                lines.map((l, i) => (
                  <div
                    key={i}
                    className={`pg-doc-box${i === selectedLine ? " selected" : ""}${
                      l.confidence < REVIEW_THRESHOLD ? " needs-review" : ""
                    }`}
                    style={boxStyle(l.bbox)}
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedLine(i);
                    }}
                  >
                    {panelView === "regions" && showAllBoxes && (
                      <span className={`pg-doc-box-label conf-${confTier(l.confidence)}`}>
                        {(l.confidence * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                ))}
            </div>
          )}
        </div>

        <div className="pg-stage-side">
          {status === "idle" && <div className="pg-empty">Drop a page, or pick a sample page below, to detect its lines.</div>}
          {status === "running" && (
            <div className="pg-empty">
              <span className="spinner" /> {runningLabel} detecting and reading lines
            </div>
          )}
          {status === "error" && <div className="pg-empty">Error: {error}</div>}
          {showResult && current.error && (
            <div className="pg-doc-engine-error">
              <strong>{engine} failed on this page.</strong> {current.error}
              <span>Try another page engine above.</span>
            </div>
          )}
          {showResult && !current.error && lines.length === 0 && (
            <div className="pg-empty">{engine} found no text on this page. Try another page engine above.</div>
          )}
          {showResult && lines.length > 0 && (
            <>
              <div className="pg-doc-stats">
                <div>
                  <strong>{lines.length}</strong>
                  <span>lines detected</span>
                </div>
                <div>
                  <strong>{avgConfidence != null ? `${(avgConfidence * 100).toFixed(0)}%` : "-"}</strong>
                  <span>avg confidence</span>
                </div>
                {current.cer != null && (
                  <div>
                    <strong>{current.cer.toFixed(3)}</strong>
                    <span>page CER</span>
                  </div>
                )}
              </div>

              <div className="pg-doc-view-toggle">
                <button className={panelView === "lines" ? "active" : ""} onClick={() => setPanelView("lines")}>
                  Lines
                </button>
                <button className={panelView === "fields" ? "active" : ""} onClick={() => setPanelView("fields")}>
                  Form fields
                </button>
                <button className={panelView === "regions" ? "active" : ""} onClick={() => setPanelView("regions")}>
                  Regions
                </button>
              </div>

              {panelView === "lines" && (
                <div className="pg-doc-lines">
                  {lines.map((l, i) => (
                    <div
                      key={i}
                      className={`pg-doc-line${i === selectedLine ? " selected" : ""}${
                        l.confidence < REVIEW_THRESHOLD ? " needs-review" : ""
                      }`}
                      onClick={() => editingIndex !== i && setSelectedLine(i)}
                    >
                      <span className="pg-doc-line-idx">{i + 1}</span>
                      {editingIndex === i ? (
                        <input
                          autoFocus
                          className="pg-doc-line-input"
                          dir="auto"
                          defaultValue={textFor(i)}
                          onClick={(e) => e.stopPropagation()}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") commitEdit(i, e.currentTarget.value);
                            if (e.key === "Escape") setEditingIndex(null);
                          }}
                          onBlur={(e) => commitEdit(i, e.currentTarget.value)}
                        />
                      ) : (
                        <span className="pg-doc-line-text" dir="auto">{textFor(i) || <em>(empty)</em>}</span>
                      )}
                      <span className={`cer-badge conf-${confTier(l.confidence)}`}>
                        {(l.confidence * 100).toFixed(0)}%
                      </span>
                      <button
                        className="pg-doc-line-edit"
                        onClick={(e) => {
                          e.stopPropagation();
                          setEditingIndex(i);
                        }}
                      >
                        Edit
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {panelView === "fields" && (
                <div className="pg-kv">
                  <p className="pg-kv-note">Best-effort split on the first ":" in each line — not a trained form-field model.</p>
                  {lines.map((l, i) => {
                    const { key, value } = splitKV(textFor(i));
                    return (
                      <div key={i} className={`pg-kv-row${l.confidence < REVIEW_THRESHOLD ? " needs-review" : ""}`}>
                        <span className="pg-kv-key">{key ?? <em>(no key detected)</em>}</span>
                        <span className="pg-kv-value" dir="auto">{value}</span>
                        <span className={`cer-badge conf-${confTier(l.confidence)}`}>
                          {(l.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}

              {panelView === "regions" && (
                <div className="pg-doc-regions">
                  {lines.map((l, i) => (
                    <div
                      key={i}
                      className={`pg-doc-region${i === selectedLine ? " selected" : ""}${
                        l.confidence < REVIEW_THRESHOLD ? " needs-review" : ""
                      }`}
                      onClick={() => setSelectedLine(i)}
                    >
                      <span className="pg-doc-region-text" dir="auto">{textFor(i) || <em>(empty)</em>}</span>
                      <span className={`cer-badge conf-${confTier(l.confidence)}`}>
                        {(l.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          {pageSamples.length > 0 && (
            <div className="pg-sample-strip pg-doc-samples">
              <span className="pg-label">or try a sample page (scored against its ground truth)</span>
              <div className="pg-sample-thumbs">
                {pageSamples.map((s) => (
                  <button
                    key={s.id}
                    className={s.id === activeSampleId ? "active" : ""}
                    onClick={() => pickSample(s)}
                    title={s.filename}
                    disabled={status === "running"}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      {comparedEngines.length > 1 && (
        <section className="pg-doc-compare">
          <span className="pg-label">Engine comparison on this page</span>
          <table>
            <thead>
              <tr>
                <th>Engine</th>
                {groundTruth && <th>Page CER</th>}
                <th>Lines</th>
                <th>Avg confidence</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {comparedEngines.map((name) => {
                const r = resultsByEngine[name];
                const avg = r.lines.length ? r.lines.reduce((s, l) => s + (l.confidence ?? 0), 0) / r.lines.length : null;
                return (
                  <tr key={name} className={name === engine ? "active" : ""} onClick={() => pickEngine(name)}>
                    <td>{name}</td>
                    {groundTruth && (
                      <td>
                        {r.error ? (
                          <span className="cer-badge conf-low">error</span>
                        ) : r.cer != null ? (
                          <span className={`cer-badge conf-${cerTier(r.cer)}`}>{r.cer.toFixed(3)}</span>
                        ) : (
                          "-"
                        )}
                      </td>
                    )}
                    <td>{r.error ? "-" : r.lines.length}</td>
                    <td>{avg != null ? `${(avg * 100).toFixed(0)}%` : "-"}</td>
                    <td>{r.seconds != null ? `${r.seconds.toFixed(1)}s` : "-"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      )}

      {showResult && lines.length > 0 && (
        <section className="pg-doc-transcript">
          <div className="pg-doc-transcript-head">
            <span className="pg-label">Transcript (reading order)</span>
            <button onClick={copyTranscript}>{copied ? "Copied" : "Copy"}</button>
          </div>
          <pre dir="auto">{transcript}</pre>
          {groundTruth && (
            <>
              <span className="pg-label pg-doc-gt-label">Ground truth (dataset answer key)</span>
              <pre className="pg-doc-gt" dir="auto">{groundTruth}</pre>
            </>
          )}
        </section>
      )}
    </>
  );
}

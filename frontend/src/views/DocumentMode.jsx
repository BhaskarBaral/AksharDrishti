import { useEffect, useRef, useState } from "react";
import { runPageUpload } from "../api.js";
import "./playground.css";

const SUPPORTED_SCRIPTS = new Set(["devanagari", "tamil"]);
const REVIEW_THRESHOLD = 0.75;
const HIGH_THRESHOLD = 0.9;

function confTier(confidence) {
  if (confidence >= HIGH_THRESHOLD) return "high";
  if (confidence >= REVIEW_THRESHOLD) return "med";
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
  const [previewSrc, setPreviewSrc] = useState(null);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [lines, setLines] = useState([]);
  const [selectedLine, setSelectedLine] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | running | done | error
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

  const fileInputRef = useRef(null);
  const supported = SUPPORTED_SCRIPTS.has(script);

  useEffect(() => {
    setPreviewSrc(null);
    setImgSize({ w: 0, h: 0 });
    setLines([]);
    setSelectedLine(null);
    setStatus("idle");
    setError("");
    setCurrentFile(null);
    setFileName("");
    setEdits({});
    setEditingIndex(null);
    setApproved(false);
  }, [script]);

  async function runPage(file) {
    setStatus("running");
    try {
      const data = await runPageUpload(script, file, ["paddleocr"]);
      const result = data.results[0];
      setLines(result?.lines ?? []);
      setSelectedLine(null);
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  function handleFiles(fileList) {
    if (!supported) return;
    const file = fileList?.[0];
    if (!file) return;
    setPreviewSrc(URL.createObjectURL(file));
    setImgSize({ w: 0, h: 0 });
    setLines([]);
    setSelectedLine(null);
    setCurrentFile(file);
    setFileName(file.name);
    setEdits({});
    setEditingIndex(null);
    setApproved(false);
    runPage(file);
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
    if (currentFile) runPage(currentFile);
  }

  if (!supported) {
    return (
      <div className="pg-doc-disabled">
        Document mode currently supports Devanagari and Tamil only (PaddleOCR is the only engine with a
        text detector wired up here). Pick one of those scripts above to try it.
      </div>
    );
  }

  const viewMeta = VIEW_META[panelView] ?? VIEW_META.lines;

  return (
    <>
      {status === "done" && (
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
            <span className={`tag conf-${confTier(avgConfidence ?? 0)}`}>
              page avg {avgConfidence != null ? `${(avgConfidence * 100).toFixed(0)}%` : "-"}
            </span>
            {needsReviewCount > 0 && <span className="tag conf-low">{needsReviewCount} need review</span>}
          </div>
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
            onChange={(e) => handleFiles(e.target.files)}
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
          {status === "idle" && <div className="pg-empty">Drop a document to detect its lines.</div>}
          {status === "running" && (
            <div className="pg-empty">
              <span className="spinner" /> Detecting lines...
            </div>
          )}
          {status === "error" && <div className="pg-empty">Error: {error}</div>}
          {status === "done" && (
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
                          defaultValue={textFor(i)}
                          onClick={(e) => e.stopPropagation()}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") commitEdit(i, e.currentTarget.value);
                            if (e.key === "Escape") setEditingIndex(null);
                          }}
                          onBlur={(e) => commitEdit(i, e.currentTarget.value)}
                        />
                      ) : (
                        <span className="pg-doc-line-text">{textFor(i) || <em>(empty)</em>}</span>
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
                        <span className="pg-kv-value">{value}</span>
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
                      <span className="pg-doc-region-text">{textFor(i) || <em>(empty)</em>}</span>
                      <span className={`cer-badge conf-${confTier(l.confidence)}`}>
                        {(l.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </section>

      {status === "done" && (
        <section className="pg-doc-transcript">
          <div className="pg-doc-transcript-head">
            <span className="pg-label">Transcript (reading order)</span>
            <button onClick={copyTranscript}>{copied ? "Copied" : "Copy"}</button>
          </div>
          <pre>{transcript}</pre>
        </section>
      )}
    </>
  );
}

import { useEffect, useRef, useState } from "react";
import { runPageUpload } from "../api.js";
import "./playground.css";

const SUPPORTED_SCRIPTS = new Set(["devanagari", "tamil"]);

export default function DocumentMode({ script }) {
  const [previewSrc, setPreviewSrc] = useState(null);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [lines, setLines] = useState([]);
  const [selectedLine, setSelectedLine] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [copied, setCopied] = useState(false);

  const fileInputRef = useRef(null);
  const supported = SUPPORTED_SCRIPTS.has(script);

  useEffect(() => {
    setPreviewSrc(null);
    setImgSize({ w: 0, h: 0 });
    setLines([]);
    setSelectedLine(null);
    setStatus("idle");
    setError("");
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

  const avgConfidence = lines.length
    ? lines.reduce((sum, l) => sum + (l.confidence ?? 0), 0) / lines.length
    : null;
  const transcript = lines.map((l) => l.text).join("\n");

  async function copyTranscript() {
    try {
      await navigator.clipboard.writeText(transcript);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard API unavailable in this context -- text is still on screen to select manually
    }
  }

  if (!supported) {
    return (
      <div className="pg-doc-disabled">
        Document mode currently supports Devanagari and Tamil only (PaddleOCR is the only engine with a
        text detector wired up here). Pick one of those scripts above to try it.
      </div>
    );
  }

  return (
    <>
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
                    className={`pg-doc-box${i === selectedLine ? " selected" : ""}`}
                    style={boxStyle(l.bbox)}
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedLine(i);
                    }}
                  />
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
              <div className="pg-doc-lines">
                {lines.map((l, i) => (
                  <div
                    key={i}
                    className={`pg-doc-line${i === selectedLine ? " selected" : ""}`}
                    onClick={() => setSelectedLine(i)}
                  >
                    <span className="pg-doc-line-idx">{i + 1}</span>
                    <span className="pg-doc-line-text">{l.text || <em>(empty)</em>}</span>
                    <span className={`cer-badge ${l.confidence >= 0.7 ? "cer-good" : "cer-bad"}`}>
                      {(l.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
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

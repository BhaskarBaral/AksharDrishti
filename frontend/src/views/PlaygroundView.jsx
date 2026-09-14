import { useEffect, useRef, useState } from "react";
import {
  getScripts,
  getSamples,
  getRecords,
  getEngines,
  imageUrl,
  runOnSample,
  runOnUpload,
} from "../api.js";
import DocumentMode from "./DocumentMode.jsx";
import "./playground.css";

export default function PlaygroundView() {
  const [scripts, setScripts] = useState([]);
  const [script, setScript] = useState(null);
  const [loadError, setLoadError] = useState("");
  const [stageMode, setStageMode] = useState("field"); // "field" | "document"

  const [engines, setEngines] = useState([]);
  const [selectedEngines, setSelectedEngines] = useState(new Set());
  const [samples, setSamples] = useState([]);

  const [mode, setMode] = useState(null); // null | "upload" | "sample"
  const [previewSrc, setPreviewSrc] = useState(null);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [currentFile, setCurrentFile] = useState(null);

  const [sampleMeta, setSampleMeta] = useState(null); // {split, filename}
  const [sampleRecords, setSampleRecords] = useState([]);
  const [fieldIndex, setFieldIndex] = useState(null);
  const [fieldBBox, setFieldBBox] = useState(null);
  const [groundTruth, setGroundTruth] = useState(null);

  const [dragActive, setDragActive] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | running | done | error | no-engines
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");

  const fileInputRef = useRef(null);

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
    setMode(null);
    setPreviewSrc(null);
    setCurrentFile(null);
    setSampleMeta(null);
    setSampleRecords([]);
    setFieldIndex(null);
    setFieldBBox(null);
    setGroundTruth(null);
    setResults([]);
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

    getSamples(script)
      .then(({ samples: list }) => setSamples(list))
      .catch(() => setSamples([]));
  }, [script]);

  async function runUpload(file, engineSet) {
    if (engineSet.size === 0) {
      setStatus("no-engines");
      return;
    }
    setStatus("running");
    try {
      const data = await runOnUpload(script, file, [...engineSet]);
      setResults(data.results);
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  function handleFiles(fileList) {
    const file = fileList?.[0];
    if (!file) return;
    setMode("upload");
    setSampleMeta(null);
    setFieldBBox(null);
    setGroundTruth(null);
    setImgSize({ w: 0, h: 0 });
    setCurrentFile(file);
    setPreviewSrc(URL.createObjectURL(file));
    runUpload(file, selectedEngines);
  }

  async function runSampleField(split, filename, records, index, engineSet) {
    setFieldIndex(index);
    const rec = records[index];
    setFieldBBox(rec.boundingBox.vertices);
    setGroundTruth(rec.groundTruth);
    if (engineSet.size === 0) {
      setStatus("no-engines");
      return;
    }
    setStatus("running");
    try {
      const data = await runOnSample(script, split, filename, index, [...engineSet]);
      setResults(data.results);
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  async function pickSample(s) {
    setMode("sample");
    setCurrentFile(null);
    setSampleMeta(s);
    setImgSize({ w: 0, h: 0 });
    setPreviewSrc(imageUrl(script, s.split, s.filename));
    try {
      const { records: list } = await getRecords(script, s.split, s.filename);
      setSampleRecords(list);
      if (!list.length) return;
      let best = 0;
      let bestLen = -1;
      list.forEach((r, i) => {
        const len = (r.groundTruth || "").length;
        if (len > bestLen) {
          bestLen = len;
          best = i;
        }
      });
      runSampleField(s.split, s.filename, list, best, selectedEngines);
    } catch {
      setSampleRecords([]);
    }
  }

  function cycleField() {
    if (!sampleRecords.length || !sampleMeta) return;
    const next = (fieldIndex + 1) % sampleRecords.length;
    runSampleField(sampleMeta.split, sampleMeta.filename, sampleRecords, next, selectedEngines);
  }

  function toggleEngine(name) {
    const next = new Set(selectedEngines);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    setSelectedEngines(next);
    if (mode === "upload" && currentFile) {
      runUpload(currentFile, next);
    } else if (mode === "sample" && sampleMeta && sampleRecords.length) {
      runSampleField(sampleMeta.split, sampleMeta.filename, sampleRecords, fieldIndex, next);
    }
  }

  function handleImgLoad(e) {
    setImgSize({ w: e.target.naturalWidth, h: e.target.naturalHeight });
  }

  if (loadError) {
    return <p style={{ color: "#A8452C", padding: "1.5rem 2rem" }}>Failed to load dataset: {loadError}</p>;
  }

  const sortedResults = results.slice().sort((a, b) => {
    const ac = a.cer ?? Infinity;
    const bc = b.cer ?? Infinity;
    return ac - bc;
  });
  const hasWinner = mode === "sample" && sortedResults.length > 0 && sortedResults[0].cer != null;

  let boxStyle = null;
  if (fieldBBox && imgSize.w > 0 && imgSize.h > 0) {
    const xs = fieldBBox.map((v) => v.x);
    const ys = fieldBBox.map((v) => v.y);
    const x0 = Math.min(...xs);
    const x1 = Math.max(...xs);
    const y0 = Math.min(...ys);
    const y1 = Math.max(...ys);
    boxStyle = {
      left: `${(x0 / imgSize.w) * 100}%`,
      top: `${(y0 / imgSize.h) * 100}%`,
      width: `${((x1 - x0) / imgSize.w) * 100}%`,
      height: `${((y1 - y0) / imgSize.h) * 100}%`,
    };
  }

  return (
    <main className="pg-root">
      <section className="pg-hero">
        <div className="pg-hero-text">
          <h2>Drop an image. Watch every engine race.</h2>
          <p>Upload a photo of Indic text, or try a real dataset sample, and compare OCR engines side by side.</p>
        </div>
        <div className="pg-script-pills">
          {scripts.map((s) => (
            <button key={s} className={s === script ? "active" : ""} onClick={() => setScript(s)}>
              {s}
            </button>
          ))}
        </div>
        <div className="pg-stage-toggle">
          <button className={stageMode === "field" ? "active" : ""} onClick={() => setStageMode("field")}>
            Field
          </button>
          <button className={stageMode === "document" ? "active" : ""} onClick={() => setStageMode("document")}>
            Document
          </button>
        </div>
      </section>

      {stageMode === "document" ? (
        <DocumentMode script={script} />
      ) : (
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
              <p>Drag &amp; drop an image here, or click to browse</p>
            </div>
          )}
          {previewSrc && (
            <div className="pg-preview-frame">
              <img src={previewSrc} onLoad={handleImgLoad} alt="preview" />
              {boxStyle && <div className="pg-field-box" style={boxStyle} />}
            </div>
          )}
        </div>

        <div className="pg-stage-side">
          <div>
            <span className="pg-label">Engines to race</span>
            <div className="pg-engine-pills">
              {engines.length === 0 ? (
                <span className="empty">No engine supports this script</span>
              ) : (
                engines.map((name) => (
                  <label key={name} className={selectedEngines.has(name) ? "active" : ""}>
                    <input type="checkbox" checked={selectedEngines.has(name)} onChange={() => toggleEngine(name)} />
                    {name}
                    {name.endsWith("-finetuned") && <span className="badge-finetuned">fine-tuned</span>}
                  </label>
                ))
              )}
            </div>
          </div>

          {mode === "sample" && groundTruth != null && (
            <div className="pg-groundtruth">
              <span>Ground truth ({fieldIndex + 1}/{sampleRecords.length})</span>
              <strong>{groundTruth}</strong>
              {sampleRecords.length > 1 && <button onClick={cycleField}>Try another field &rarr;</button>}
            </div>
          )}

          <div className="pg-sample-strip">
            <span className="pg-label">or try a dataset sample</span>
            <div className="pg-sample-thumbs">
              {samples.slice(0, 24).map((s) => {
                const isActive = sampleMeta && sampleMeta.split === s.split && sampleMeta.filename === s.filename;
                return (
                  <button
                    key={`${s.split}/${s.filename}`}
                    className={isActive ? "active" : ""}
                    onClick={() => pickSample(s)}
                    title={s.filename}
                  >
                    {s.filename}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      <section className="pg-results">
        {status === "idle" && <div className="pg-empty">Results will appear here once you run an image.</div>}
        {status === "running" && (
          <div className="pg-empty">
            <span className="spinner" /> Racing engines...
          </div>
        )}
        {status === "no-engines" && <div className="pg-empty">Pick at least one engine above.</div>}
        {status === "error" && <div className="pg-empty">Error: {error}</div>}
        {status === "done" && (
          <div className="pg-card-grid">
            {sortedResults.map((r, i) => {
              const hasCer = r.cer !== null && r.cer !== undefined;
              const cerClass = hasCer ? (r.cer <= 0.3 ? "cer-good" : "cer-bad") : "cer-neutral";
              return (
                <div key={i} className={`pg-card${i === 0 && hasWinner ? " winner" : ""}`}>
                  {i === 0 && hasWinner && <span className="pg-winner-badge">Best match</span>}
                  <div className="pg-card-head">
                    <span className="pg-card-engine">{r.engine}</span>
                    {r.engine.endsWith("-finetuned") && <span className="badge-finetuned">fine-tuned</span>}
                  </div>
                  <p className="pg-card-hyp">{r.hypothesis || <em>(empty)</em>}</p>
                  <span className={`cer-badge ${cerClass}`}>{hasCer ? `${r.cer.toFixed(3)} CER` : "no reference"}</span>
                </div>
              );
            })}
          </div>
        )}
      </section>
        </>
      )}
    </main>
  );
}

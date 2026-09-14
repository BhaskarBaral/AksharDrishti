import { useEffect, useRef, useState } from "react";
import { getScripts, runPageUpload } from "../api.js";
import "./demoScreens.css";

// Same limitation as DocumentMode: PaddleOCR is the only engine with a text
// detector wired up, and only these two scripts have one trained.
const SUPPORTED_SCRIPTS = new Set(["devanagari", "tamil"]);

export default function CitizenUploadView() {
  const [scripts, setScripts] = useState([]);
  const [script, setScript] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [error, setError] = useState("");
  const [transcript, setTranscript] = useState("");
  const [copied, setCopied] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    getScripts()
      .then(({ scripts: list }) => {
        const supported = list.filter((s) => SUPPORTED_SCRIPTS.has(s));
        setScripts(supported);
        if (supported.length) setScript(supported[0]);
      })
      .catch(() => setScripts([]));
  }, []);

  async function handleFiles(fileList) {
    const file = fileList?.[0];
    if (!file || !script) return;
    setStatus("running");
    setError("");
    try {
      const data = await runPageUpload(script, file, ["paddleocr"]);
      const lines = data.results[0]?.lines ?? [];
      setTranscript(lines.map((l) => l.text).join("\n"));
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  async function copyText() {
    try {
      await navigator.clipboard.writeText(transcript);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard API unavailable in this context -- text is still on screen to select manually
    }
  }

  function downloadText() {
    const blob = new Blob([transcript], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "extracted-text.txt";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="demo-root">
      <div className="s-head">
        <div>
          <h1 className="title">Citizen upload</h1>
          <p className="desc">
            The public-facing flow — one document, no jargon, no confidence tables. Upload a page and get
            clean, copyable text back. This is the Bhashini-style citizen-service front door.
          </p>
        </div>
      </div>

      <div className="citizen">
        {scripts.length > 1 && (
          <div className="citizen-script-pick">
            <span className="pg-label">Language</span>
            <div className="pg-script-pills">
              {scripts.map((s) => (
                <button key={s} className={s === script ? "active" : ""} onClick={() => setScript(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        <div
          className={`citizen-drop${dragActive ? " drag" : ""}`}
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
          onClick={() => script && fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => handleFiles(e.target.files)}
          />
          <div className="citizen-drop-icon">&#8593;</div>
          <h3>Extract text from a document</h3>
          <p>Drop an image, or choose a file. Handwritten and printed pages both work.</p>
          <button
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
            disabled={!script}
          >
            Choose file
          </button>
          <div className="citizen-hint">
            {script ? `Script: ${script} · files are processed in-memory only` : "No supported script available"}
          </div>
        </div>

        {status === "running" && (
          <div className="citizen-status"><span className="spinner" /> Extracting text...</div>
        )}
        {status === "error" && <div className="citizen-status error">Error: {error}</div>}

        {status === "done" && (
          <div className="citizen-result">
            <div className="citizen-result-head">
              <strong>Extracted text</strong>
              <span className="tag conf-high">Detected: {script}</span>
            </div>
            <div className="citizen-result-body">{transcript || <em>(no text detected)</em>}</div>
            <div className="citizen-result-foot">
              <button className="outline" onClick={copyText}>{copied ? "Copied" : "Copy text"}</button>
              <button className="outline" onClick={downloadText}>Download</button>
              <button disabled title="Translation isn't wired up in this demo yet">
                Translate to English
              </button>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

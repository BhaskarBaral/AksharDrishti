import { useEffect, useMemo, useRef, useState } from "react";
import "./demoScreens.css";

// Talks to the public /v1 API (app/routers/api_v1_router.py) with plain
// fetch -- deliberately not via api.js, so what this screen sends is exactly
// what the generated curl command below sends.

const ENDPOINTS = [
  { id: "recognize", method: "POST", path: "/v1/ocr/recognize", upload: true, label: "OCR — recognize text" },
  { id: "layout", method: "POST", path: "/v1/layout/detect", upload: true, label: "Layout — detect regions" },
  { id: "scripts", method: "GET", path: "/v1/scripts", upload: false, label: "List scripts & engines" },
  { id: "health", method: "GET", path: "/v1/health", upload: false, label: "Health check" },
];

const FALLBACK_SCRIPTS = [
  "devanagari", "bengali", "tamil", "telugu", "urdu",
  "gujarati", "kannada", "malayalam", "odia", "punjabi",
].map((script) => ({ script, levels: ["word", "page"], engines: [], defaultEngine: {} }));

function shellQuote(s) {
  return `'${String(s).replace(/'/g, `'\\''`)}'`;
}

// Tokenize pretty-printed JSON into the api-code colour classes.
function highlightJson(text) {
  const out = [];
  const re = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;
  let last = 0;
  let m;
  let i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1]) {
      out.push(<span key={i++} className={m[2] ? "k" : "s"}>{m[1]}</span>);
      if (m[2]) out.push(m[2]);
    } else {
      out.push(<span key={i++} className="n">{m[0]}</span>);
    }
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export default function ApiPlaygroundView() {
  const [endpointId, setEndpointId] = useState("recognize");
  const [scripts, setScripts] = useState(FALLBACK_SCRIPTS);
  const [script, setScript] = useState("devanagari");
  const [level, setLevel] = useState("page");
  const [format, setFormat] = useState("ulca");
  const [engine, setEngine] = useState("");
  const [threshold, setThreshold] = useState("0.5");
  const [file, setFile] = useState(null);
  const [fileIsSample, setFileIsSample] = useState(false);
  const [loadingSample, setLoadingSample] = useState(false);
  const [sending, setSending] = useState(false);
  const [result, setResult] = useState(null); // {status, statusText, ms, body, isJson}
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);
  const fileInput = useRef(null);

  const endpoint = ENDPOINTS.find((e) => e.id === endpointId);
  const scriptInfo = scripts.find((s) => s.script === script) || FALLBACK_SCRIPTS[0];
  const enginesForLevel = (scriptInfo.engines || []).filter((e) => e.levels.includes(level));
  const defaultEngine = scriptInfo.defaultEngine?.[level];

  useEffect(() => {
    let alive = true;
    fetch("/v1/scripts")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data) => alive && data.scripts?.length && setScripts(data.scripts))
      .catch(() => {});
    return () => { alive = false; };
  }, []);

  // A picked engine that doesn't exist for the new script/level would 400.
  useEffect(() => {
    if (engine && !enginesForLevel.some((e) => e.name === engine)) setEngine("");
  }, [script, level, scripts]); // eslint-disable-line react-hooks/exhaustive-deps

  // A bundled sample tracks the script/level it was chosen for.
  useEffect(() => {
    if (fileIsSample) loadSample();
  }, [script, level, endpointId]); // eslint-disable-line react-hooks/exhaustive-deps

  async function loadSample() {
    const sampleLevel = endpointId === "layout" ? "page" : level;
    setLoadingSample(true);
    setError(null);
    try {
      const res = await fetch(`/v1/samples/${script}?level=${sampleLevel}`);
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `${res.status} ${res.statusText}`);
      const blob = await res.blob();
      const name = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(res.headers.get("content-disposition") || "")?.[1]
        || `sample_${script}.${blob.type.includes("png") ? "png" : "jpg"}`;
      setFile(new File([blob], decodeURIComponent(name), { type: blob.type }));
      setFileIsSample(true);
    } catch (e) {
      setError(`Couldn't load sample: ${e.message}`);
    } finally {
      setLoadingSample(false);
    }
  }

  function pickFile(e) {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setFileIsSample(false);
    }
    e.target.value = "";
  }

  function buildForm() {
    const form = new FormData();
    if (file) form.set("file", file);
    if (endpointId === "recognize") {
      form.set("script", script);
      form.set("level", level);
      form.set("format", format);
      if (engine) form.set("engine", engine);
    } else if (endpointId === "layout") {
      form.set("threshold", threshold);
    }
    return form;
  }

  const curl = useMemo(() => {
    const origin = window.location.origin;
    const parts = [`curl -X ${endpoint.method} ${shellQuote(origin + endpoint.path)}`];
    if (endpoint.upload) {
      parts.push(`-F ${shellQuote(`file=@${file ? file.name : "page.png"}`)}`);
      if (endpointId === "recognize") {
        parts.push(`-F script=${script}`, `-F level=${level}`, `-F format=${format}`);
        if (engine) parts.push(`-F ${shellQuote(`engine=${engine}`)}`);
      } else {
        parts.push(`-F ${shellQuote(`threshold=${threshold}`)}`);
      }
    }
    return parts.join(" \\\n  ");
  }, [endpoint, endpointId, file, script, level, format, engine, threshold]);

  async function send() {
    setSending(true);
    setError(null);
    setResult(null);
    const t0 = performance.now();
    try {
      const res = await fetch(endpoint.path, {
        method: endpoint.method,
        body: endpoint.method === "POST" ? buildForm() : undefined,
      });
      const raw = await res.text();
      const ms = performance.now() - t0;
      let body = raw;
      let isJson = false;
      try {
        body = JSON.stringify(JSON.parse(raw), null, 2);
        isJson = true;
      } catch {
        /* non-JSON body (e.g. proxy error page) -- show as-is */
      }
      setResult({ status: res.status, statusText: res.statusText, ms, body, isJson });
    } catch (e) {
      setError(`Request failed: ${e.message}`);
    } finally {
      setSending(false);
    }
  }

  async function copyCurl() {
    try {
      await navigator.clipboard.writeText(curl);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked -- the command is still selectable */
    }
  }

  const statusTier = !result ? null : result.status < 300 ? "conf-high" : result.status < 500 ? "conf-med" : "conf-low";
  const latency = result && (result.ms >= 1000 ? `${(result.ms / 1000).toFixed(2)}s` : `${Math.round(result.ms)}ms`);
  const missingFile = endpoint.upload && !file;

  return (
    <main className="demo-root">
      <div className="s-head">
        <div>
          <h1 className="title">API playground</h1>
          <p className="desc">
            The developer console — send a document, get ULCA-schema JSON back. This is what makes the tool
            deployment-ready and Bhashini-compatible: the response shape follows the locked ULCA schema, so
            integrators can wire it straight into the national stack.
          </p>
        </div>
      </div>

      <div className="api-pg">
        <div className="api-card">
          <div className="api-chead"><span>Request</span><span className="api-method">{endpoint.method}</span></div>
          <div className="api-cbody">
            <div className="api-field">
              <label htmlFor="api-endpoint">Endpoint</label>
              <select id="api-endpoint" className="api-ctl api-mono" value={endpointId}
                      onChange={(e) => { setEndpointId(e.target.value); setResult(null); }}>
                {ENDPOINTS.map((e) => (
                  <option key={e.id} value={e.id}>{e.method} {e.path} — {e.label}</option>
                ))}
              </select>
            </div>

            {endpointId === "recognize" && (
              <>
                <div className="api-row">
                  <div className="api-field">
                    <label htmlFor="api-script">Script</label>
                    <select id="api-script" className="api-ctl" value={script} onChange={(e) => setScript(e.target.value)}>
                      {scripts.map((s) => (
                        <option key={s.script} value={s.script}>
                          {s.script}{s.sourceLanguage ? ` (${s.sourceLanguage} · ${s.sourceScriptCode})` : ""}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="api-field">
                    <label htmlFor="api-level">Level</label>
                    <select id="api-level" className="api-ctl" value={level} onChange={(e) => setLevel(e.target.value)}>
                      <option value="page">page — detect every line</option>
                      <option value="word">word — image is one word/line crop</option>
                    </select>
                  </div>
                </div>
                <div className="api-row">
                  <div className="api-field">
                    <label htmlFor="api-format">Output format</label>
                    <select id="api-format" className="api-ctl" value={format} onChange={(e) => setFormat(e.target.value)}>
                      <option value="ulca">ULCA JSON records</option>
                      <option value="text">Plain text</option>
                    </select>
                  </div>
                  <div className="api-field">
                    <label htmlFor="api-engine">Engine</label>
                    <select id="api-engine" className="api-ctl" value={engine} onChange={(e) => setEngine(e.target.value)}>
                      <option value="">default{defaultEngine ? ` (${defaultEngine})` : ""}</option>
                      {enginesForLevel.map((e) => <option key={e.name} value={e.name}>{e.name}</option>)}
                    </select>
                  </div>
                </div>
              </>
            )}

            {endpointId === "layout" && (
              <div className="api-field">
                <label htmlFor="api-threshold">Score threshold</label>
                <input id="api-threshold" className="api-ctl" type="number" min="0" max="1" step="0.05"
                       value={threshold} onChange={(e) => setThreshold(e.target.value)} />
              </div>
            )}

            {endpoint.upload && (
              <div className="api-field">
                <label>Document</label>
                <div className="api-filepick">
                  <span className="api-filename" title={file?.name}>
                    {file ? `📄 ${file.name}` : "No file selected"}
                    {file && <span className="api-filemeta"> · {(file.size / 1024).toFixed(0)} KB{fileIsSample ? " · sample" : ""}</span>}
                  </span>
                  <button type="button" className="outline api-small" onClick={() => fileInput.current?.click()}>
                    Choose file
                  </button>
                  <button type="button" className="outline api-small" onClick={loadSample} disabled={loadingSample}>
                    {loadingSample ? "Loading…" : "Use sample"}
                  </button>
                  <input ref={fileInput} type="file" accept="image/png,image/jpeg,image/bmp,image/tiff"
                         hidden onChange={pickFile} />
                </div>
              </div>
            )}

            <button className="api-send" onClick={send} disabled={sending || missingFile}>
              {sending ? "Sending…" : missingFile ? "Choose a file or use the sample" : "Send request"}
            </button>
            {error && <p className="api-error">{error}</p>}

            <div className="api-field api-curl-field">
              <div className="api-curl-head">
                <label>curl</label>
                <button type="button" className="ghost api-small" onClick={copyCurl}>{copied ? "Copied" : "Copy"}</button>
              </div>
              <pre className="api-code api-curl">{curl}</pre>
            </div>
          </div>
        </div>

        <div className="api-card">
          <div className="api-chead">
            <span>Response</span>
            {result && (
              <span className={`tag ${statusTier}`}>
                {result.status} {result.statusText || ""} · {latency}
              </span>
            )}
          </div>
          <div className="api-cbody">
            {result ? (
              <pre className="api-code api-response">{result.isJson ? highlightJson(result.body) : result.body}</pre>
            ) : (
              <pre className="api-code api-response api-empty">
                <span className="c">
                  {sending
                    ? "// waiting for response… (first call per engine loads the model, can take ~10s)"
                    : "// send a request to see the live response here"}
                </span>
              </pre>
            )}
          </div>
        </div>
      </div>
      <p className="demo-note">
        Live calls against this server's public <code>/v1</code> API. ULCA records follow the project's locked
        ocr-record schema; per-line confidence is returned alongside them in <code>confidences</code>, since the
        ULCA schema has no confidence field.
      </p>
    </main>
  );
}

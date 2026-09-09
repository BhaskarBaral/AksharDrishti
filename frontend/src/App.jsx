import { useEffect, useState } from "react";
import { getScripts, getSamples, getRecords, getEngines, runOnSample } from "./api.js";
import DatasetPanel from "./components/DatasetPanel.jsx";
import SampleViewer from "./components/SampleViewer.jsx";
import ResultsPanel from "./components/ResultsPanel.jsx";
import UploadPanel from "./components/UploadPanel.jsx";

export default function App() {
  const [scripts, setScripts] = useState([]);
  const [script, setScript] = useState(null);
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null); // {split, filename}
  const [records, setRecords] = useState([]);
  const [selectedField, setSelectedField] = useState(null);

  const [engines, setEngines] = useState([]);
  const [selectedEngines, setSelectedEngines] = useState(new Set());
  const [status, setStatus] = useState("idle"); // idle | running | done | error | no-engines
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState("");

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
    setSelectedSample(null);
    setRecords([]);
    setSelectedField(null);
    setStatus("idle");
    setResult(null);

    getSamples(script)
      .then(({ samples: list }) => setSamples(list))
      .catch(() => setSamples([]));

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

  function selectSample(split, filename) {
    setSelectedSample({ split, filename });
    setSelectedField(null);
    setStatus("idle");
    setResult(null);
    getRecords(script, split, filename)
      .then(({ records: list }) => setRecords(list))
      .catch(() => setRecords([]));
  }

  function toggleEngine(name) {
    setSelectedEngines((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  async function runField(index) {
    setSelectedField(index);
    if (selectedEngines.size === 0) {
      setStatus("no-engines");
      return;
    }
    setStatus("running");
    try {
      const data = await runOnSample(
        script,
        selectedSample.split,
        selectedSample.filename,
        index,
        [...selectedEngines]
      );
      setResult(data);
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  if (loadError) {
    return <p style={{ color: "#e0616f", padding: "1.5rem 2rem" }}>Failed to load dataset: {loadError}</p>;
  }

  return (
    <>
      <header>
        <h1>AksharDrishti</h1>
        <p className="subtitle">
          Stage 3 prototype — browse the Stage 2 dataset and run OCR baselines against it
        </p>
      </header>
      <main>
        <DatasetPanel
          scripts={scripts}
          script={script}
          onScriptChange={setScript}
          samples={samples}
          selected={selectedSample}
          onSelectSample={selectSample}
        />
        <SampleViewer
          script={script}
          split={selectedSample?.split}
          filename={selectedSample?.filename}
          records={records}
          selectedField={selectedField}
          onRunField={runField}
        />
        <ResultsPanel
          engines={engines}
          selectedEngines={selectedEngines}
          onToggleEngine={toggleEngine}
          status={status}
          result={result}
          error={error}
        />
        <UploadPanel scripts={scripts} />
      </main>
    </>
  );
}

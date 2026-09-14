import { useState } from "react";
import ClassicView from "./views/ClassicView.jsx";
import PlaygroundView from "./views/PlaygroundView.jsx";
import BatchQueueView from "./views/BatchQueueView.jsx";
import CitizenUploadView from "./views/CitizenUploadView.jsx";
import ApiPlaygroundView from "./views/ApiPlaygroundView.jsx";
import BenchmarkView from "./views/BenchmarkView.jsx";
import yellowsenseMark from "./assets/yellowsense-mark.png";

const VIEWS = {
  playground: {
    label: "Playground",
    subtitle: "Drop an image or try a dataset sample and compare OCR engines instantly",
    Component: PlaygroundView,
  },
  classic: {
    label: "Dashboard",
    subtitle: "Stage 3 prototype — browse the Stage 2 dataset and run OCR baselines against it",
    Component: ClassicView,
  },
  batch: {
    label: "Batch queue",
    subtitle: "The screen for scale — triage thousands of digitized records by status and confidence",
    Component: BatchQueueView,
  },
  citizen: {
    label: "Citizen upload",
    subtitle: "The public-facing flow — one document in, clean copyable text out",
    Component: CitizenUploadView,
  },
  api: {
    label: "API playground",
    subtitle: "The developer console — send a document, get ULCA-schema JSON back",
    Component: ApiPlaygroundView,
  },
  benchmark: {
    label: "Benchmark",
    subtitle: "Score every engine — including the Bhashini-hosted OCR API — against real ground truth",
    Component: BenchmarkView,
  },
};

export default function App() {
  const [view, setView] = useState("playground");
  const { subtitle, Component } = VIEWS[view];

  return (
    <>
      <header>
        <div className="header-top">
          <div className="brand">
            <img src={yellowsenseMark} alt="Yellowsense Technologies" className="brand-logo" />
            <h1>AksharDrishti</h1>
          </div>
          <nav className="view-tabs">
            {Object.entries(VIEWS).map(([key, { label }]) => (
              <button key={key} className={view === key ? "active" : ""} onClick={() => setView(key)}>
                {label}
              </button>
            ))}
          </nav>
        </div>
        <p className="subtitle">{subtitle}</p>
      </header>
      <Component />
    </>
  );
}
